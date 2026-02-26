# Project Research Summary

**Project:** NJ Housing Price Predictor (QLoRA + ONNX + Lambda ML Pipeline)
**Domain:** QLoRA fine-tuned LLM — Colab training, ONNX export, AWS Lambda serving
**Researched:** 2026-02-26
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project implements an end-to-end ML pipeline that fine-tunes a small language model (Qwen2.5-0.5B) using QLoRA on NJ housing data, exports it to ONNX, and serves predictions via AWS Lambda behind API Gateway. The defining characteristic of this approach — using an LLM for tabular regression — is unusual but deliberate: the learning objective is to demonstrate the full QLoRA-to-production workflow, not to maximize prediction accuracy over simpler alternatives like XGBoost. Research confirms this pipeline is achievable within the stated constraints (Colab free tier, under 20 min training, Lambda cold starts under 10s) but only with precise execution of the merge-then-export workflow.

The recommended approach trains Qwen2.5-0.5B with 4-bit QLoRA (bitsandbytes + PEFT), evaluates on held-out NJ data, merges LoRA adapters into the base model at full precision, exports via HuggingFace Optimum to ONNX, and packages the result in a minimal Lambda container image with only onnxruntime (CPU) and a tokenizer. Infrastructure is managed with Terraform and deployment is automated via GitHub Actions using OIDC authentication. The entire architecture maps to 4 numbered Colab notebooks plus a lambda/ serving directory and a terraform/ IaC directory.

The primary risks cluster around three areas: (1) the QLoRA merge workflow — you must reload the base model in full precision before merging adapters, never export while still quantized; (2) the regression output format — the LLM must generate a parseable number, requiring a strict prompt template and robust output parsing defined before training begins; and (3) the Lambda container image — it must exclude PyTorch and bitsandbytes entirely, containing only inference-time dependencies to stay under 3GB uncompressed. Synthetic data quality is a secondary risk: NJ housing prices vary significantly by county, and naive synthetic generation without county-level price priors produces a model that generalizes poorly to real inputs.

---

## Key Findings

### Recommended Stack

The training stack runs entirely on Google Colab and is built around the HuggingFace ecosystem: `transformers 5.2.0` (use v5, not legacy v4), `peft 0.18.1` for QLoRA adapter injection, `bitsandbytes 0.49.2` for 4-bit quantization, `trl 0.29.0` (SFTTrainer) for the training loop, and `accelerate 1.12.0` for device placement. The export stack uses `optimum 2.1.0` (the transformers-5.x-compatible branch) to produce ONNX artifacts, validated with `onnxruntime 1.24.2` on Colab before deployment.

The Lambda serving stack is intentionally minimal: `onnxruntime 1.24.2` (CPU build, not GPU), `fastapi 0.133.1` + `mangum 0.21.0` for the ASGI handler, on Python 3.12. Infrastructure is Terraform 1.14.6 with AWS Provider 6.34.0. CI/CD uses GitHub Actions with OIDC-based AWS auth (no long-lived credentials). All versions were verified against PyPI on 2026-02-26.

**Core technologies:**
- `transformers 5.2.0`: Model loading, tokenization, Qwen2.5 support — use v5, v4 is legacy
- `peft 0.18.1`: QLoRA adapter injection and `merge_and_unload()` — canonical path for adapter training
- `bitsandbytes 0.49.2`: 4-bit NF4 quantization — v0.43+ supports CPU-only import (critical for Lambda safety)
- `trl 0.29.0 + accelerate 1.12.0`: SFTTrainer for supervised fine-tuning on text-format regression tasks
- `optimum 2.1.0`: HuggingFace ONNX exporter — handles merge-then-export workflow and dynamic axes
- `onnxruntime 1.24.2`: CPU inference in Lambda — same version used for Colab validation and Lambda serving
- `fastapi + mangum`: REST API handler — typed endpoints with ASGI-to-Lambda event bridging
- `Terraform 1.14.6`: Declarative IaC for Lambda + API Gateway — simpler than CDK or SAM for single-service scope

### Expected Features

The pipeline has a strict linear dependency chain: dataset generation feeds prompt formatting, which feeds QLoRA training, which feeds evaluation and ONNX export, which feeds Lambda containerization, which feeds Terraform provisioning. CI/CD wires the deployment end. Training is and remains intentionally manual on Colab. Every stage must succeed before the next begins.

**Must have (table stakes) — pipeline breaks without these:**
- Dataset generation with train/val/test split (80/10/10, stratified by price range)
- Natural language prompt formatting — 7 features as text; same template used in training data and Lambda handler
- QLoRA 4-bit training on Colab GPU (under 20 min target)
- MAE, RMSE, R², MAPE evaluation metrics on held-out test set
- Training loss and validation loss curves (low complexity, high diagnostic value)
- LoRA weight merging via two-step save/reload pattern
- ONNX export with numerical validation on Colab before containerization
- Lambda container image (onnxruntime CPU only, tokenizer, handler) pushed to ECR
- REST API endpoint via API Gateway v2 (HTTP API)
- Terraform IaC for Lambda + API Gateway + ECR + IAM
- GitHub Actions CI/CD for container build and deployment

**Should have (add during or immediately after core pipeline):**
- Prediction vs actual scatter plot and residual histogram
- Synthetic data augmentation using county-level NJ price statistics
- Per-zip-code error breakdown to surface geographic generalization gaps
- Input validation in API handler (400 errors with field-level messages)
- Cold start latency measurement (informs provisioned concurrency decision)
- Model card JSON artifact (base model, hyperparameters, metrics, export date)

**Defer to v2+:**
- Prediction confidence intervals (requires ensemble or calibration work — HIGH complexity)
- Prompt template versioning system (low risk in v1 since template is stable)
- Scheduled monthly retrain pipeline
- Multi-state support beyond NJ

### Architecture Approach

The architecture separates concerns across four numbered Colab notebooks (data prep, training, evaluation, ONNX export), a self-contained `lambda/` directory (handler, Dockerfile, minimal requirements), a flat `terraform/` directory (no modules needed for single-service scope), and `.github/workflows/` for CI/CD. Model artifacts are explicitly gitignored and stored in Google Drive or GitHub Releases — both the LoRA adapter weights and the ONNX files will exceed GitHub's 100MB file limit. The prompt formatting function is the single most critical shared boundary: defined once in `lambda/prompt_utils.py`, imported in notebooks and the Lambda handler. Any mismatch between training and inference prompt format causes silent accuracy degradation.

**Major components:**
1. `01_data_prep.ipynb` — load NJ housing CSV, generate synthetic records with county-level price priors, format as text prompts, split and save as JSONL
2. `02_train.ipynb` — load Qwen2.5-0.5B in 4-bit, apply QLoRA, train with SFTTrainer, save LoRA adapter to Drive
3. `03_evaluate.ipynb` — load adapter, run inference on test set, compute all 4 metrics, generate plots
4. `04_export.ipynb` — two-step merge (reload base in fp32, merge adapter, save merged), export to ONNX via optimum, validate numerically
5. `lambda/handler.py` — parse API Gateway event, format prompt, tokenize, run ONNX InferenceSession (module-level global), return JSON
6. `lambda/Dockerfile` — AWS Lambda Python 3.12 base image, bundles model.onnx + tokenizer, installs only inference deps
7. `terraform/` — Lambda function, API Gateway v2, ECR repository, IAM execution role
8. `.github/workflows/` — CI on PR (lint + test); deploy on git tag (build image, push ECR, terraform apply, smoke test)

### Critical Pitfalls

1. **QLoRA merge on still-quantized base model** — always use two-step pattern: save adapter after training, then reload base in fp32 on CPU (no `load_in_4bit`), load adapter on top, then call `merge_and_unload()`. Merging while the base is still quantized produces corrupted weights silently; the ONNX validation step will catch this if the atol check is run.

2. **LLM output unparseable as a number** — define the prompt template ending (e.g., `"Predicted price: $"`) and output parsing logic before writing training code. Validate parse success rate on 100+ training examples first. Token-based regression with strict prompt engineering is recommended over a custom regression head (simpler ONNX graph).

3. **Training dependencies in Lambda container** — Lambda container must contain only: `onnxruntime-cpu`, `transformers` (tokenizer only), `numpy`, `mangum`, `fastapi`. No PyTorch, no bitsandbytes. PyTorch adds 2-3GB, pushing image toward 10GB limit and drastically increasing cold start latency.

4. **Synthetic data with unrealistic price distribution** — NJ prices vary by county (Hudson/Bergen ~$600k median vs. Salem ~$200k). Generate prices as log-normal distributions with county-level multipliers. Use at least 30% real public data. Validate synthetic distribution against known NJ statistics before training. This pitfall has the highest recovery cost (full retrain required).

5. **ONNX InferenceSession recreated per request** — create the `InferenceSession` and tokenizer as module-level globals in the Lambda handler, initialized at cold start. Recreating per request adds 3-5 seconds of overhead on every warm invocation and negates the benefit of Lambda environment reuse.

---

## Implications for Roadmap

Based on the dependency chain in FEATURES.md and the build order in ARCHITECTURE.md, the pipeline has a strict sequential dependency structure. The suggested phases mirror the notebook numbering, which reflects the actual dependency order.

### Phase 1: Data Foundation

**Rationale:** Nothing else can proceed without training data. This phase also carries the highest risk of silent failures (synthetic data distribution mismatch) that only manifest later as poor model accuracy. Catching data quality issues in Phase 1 prevents expensive retraining in Phase 2.

**Delivers:** `train.jsonl`, `val.jsonl`, `test.jsonl` in the correct prompt format; validated price distribution against NJ county statistics; `lambda/prompt_utils.py` with `format_prompt()` defined

**Addresses:**
- Dataset generation with train/val/test split (stratified by price range)
- Natural language prompt formatting (define `format_prompt()` once in `lambda/prompt_utils.py`)
- Synthetic data augmentation using county-level NJ price statistics

**Avoids:**
- Synthetic data distribution mismatch pitfall (validate histogram against NJ county medians before splitting)
- Prompt template divergence anti-pattern (define once, import everywhere from day 1)

**Research flag:** Needs phase-specific research on public NJ housing datasets (data.gov, NJ Treasury property records, Kaggle NJ housing). Verify downloadable record count and schema before committing to synthetic augmentation ratio.

---

### Phase 2: QLoRA Training

**Rationale:** Training is the core learning objective of the project. It must be built on validated data from Phase 1. The regression output format (token parsing vs. regression head) is a training-time decision that cannot be changed without full retraining.

**Delivers:** Saved LoRA adapter checkpoint on Google Drive; training and validation loss curves; confirmed parse success rate on training outputs; evaluation metrics on test set

**Addresses:**
- QLoRA 4-bit setup with `prepare_model_for_kbit_training()` before `get_peft_model()` (order matters)
- Training loop on Colab GPU (under 20 min, gradient checkpointing enabled)
- Training loss and validation loss curves
- Drive checkpointing configured as first cell in the training notebook
- MAE/RMSE/R²/MAPE evaluation in `03_evaluate.ipynb`

**Avoids:**
- Quantized merge pitfall (save adapter in the correct format for Phase 3 merge)
- Colab session disconnect pitfall (Drive checkpointing + Drive-cached base model from first run)
- LLM output parse failure (validate strict prompt format on 100 samples before full training run)

**Research flag:** Standard QLoRA patterns are well-documented. Skip phase research. Verify `prepare_model_for_kbit_training()` behavior for Qwen2.5 architecture if training issues arise.

---

### Phase 3: Evaluation and ONNX Export

**Rationale:** Evaluation validates the model before export; export must validate before containerization. The ONNX numerical validation on Colab is the gate that prevents a broken model from reaching Lambda. Issues found here are far cheaper to fix than in a deployed Lambda.

**Delivers:** Full evaluation report (MAE, RMSE, R², MAPE on test set); prediction vs. actual scatter plot; residual histogram; `model.onnx` + tokenizer files validated at `atol=1e-3` against PyTorch outputs; model card JSON artifact

**Addresses:**
- LoRA weight merging (two-step: reload base in fp32 on CPU, then merge and save)
- ONNX export with correct task type (verify `feature-extraction` vs. `text-generation-with-past` for regression use case)
- Numerical ONNX validation before containerization
- Per-zip-code error breakdown (add if aggregate metrics are passing)
- Model card JSON artifact alongside ONNX export

**Avoids:**
- Exporting PeftModel directly to ONNX without merging first
- Wrong ONNX task type for regression output
- Deploying an unvalidated ONNX model to Lambda

**Research flag:** Needs phase-specific research to confirm correct `optimum-cli export onnx --task` flag for Qwen2.5 regression use case. ONNX opset compatibility with Qwen2.5 rotary position embeddings should be verified against current Optimum 2.x documentation before implementing the export step.

---

### Phase 4: Lambda Container and REST API

**Rationale:** Lambda serving is the final output surface. Requires a validated ONNX model from Phase 3. Container design decisions — minimal dependencies, module-level session initialization, memory configuration — directly determine cold start latency and reliability.

**Delivers:** Docker container image pushed to ECR; Lambda handler with global model load, input validation, and JSON response; container tested locally with `docker run --network none` (confirms tokenizer is bundled, not fetched at runtime)

**Addresses:**
- Lambda container with onnxruntime CPU, tokenizer, and handler only — no PyTorch, no bitsandbytes
- Module-level `InferenceSession` and tokenizer initialization (warm invocation target under 1s)
- Input validation on all 7 required fields (type and range checks, 400 errors with field-level messages)
- Lambda memory set to 3008MB minimum; timeout set to 30s
- Cold start latency measurement (log first invocation separately from warm invocations)

**Avoids:**
- Training dependencies in Lambda container (image bloat pitfall)
- Tokenizer loaded from HuggingFace Hub at Lambda runtime (bundle in image)
- InferenceSession recreated per request (performance trap)

**Research flag:** Standard Lambda container deployment patterns are well-documented. Skip phase research. Benchmark actual cold start times on the first deploy — provisioned concurrency decision depends on the measured result.

---

### Phase 5: Infrastructure and CI/CD

**Rationale:** Infrastructure must be provisioned before CI/CD can deploy to it. The ECR repository must exist before Terraform can create the Lambda function that references it. GitHub Actions is the last thing wired up — it orchestrates deployment but requires everything else to be in place.

**Delivers:** Terraform-managed Lambda + API Gateway v2 + ECR + IAM; Terraform S3 remote state backend; GitHub Actions deploy workflow (build image, push ECR, terraform apply on git tag); smoke test against live API endpoint

**Addresses:**
- Terraform resources: `aws_lambda_function`, `aws_api_gateway_v2_api`, `aws_ecr_repository`, `aws_iam_role`
- Terraform remote state in S3 (required for GitHub Actions CI to run `terraform apply` reliably)
- OIDC-based AWS auth in GitHub Actions (no long-lived credentials in Secrets)
- `terraform plan` in PR as approval gate; `terraform apply` only on merge to main
- CI workflow on PR: lint + test Lambda handler; deploy workflow on tag: build + ECR push + terraform apply + smoke test

**Avoids:**
- Lambda Terraform resource created before ECR image exists (use `depends_on` or provision ECR in a prior step)
- Hardcoded AWS credentials in Dockerfile or GitHub Secrets (use OIDC IAM role)
- `terraform apply` without a plan approval gate

**Research flag:** Terraform + GitHub Actions + OIDC are extremely well-documented industry standards. Skip phase research. The one setup prerequisite: create the S3 Terraform state bucket before the first CI run — this is the single step that blocks all of Phase 5.

---

### Phase Ordering Rationale

- Data before training: no training dataset = no model; synthetic data quality issues discovered here prevent expensive retraining later
- Prompt format defined in Phase 1: the `format_prompt()` function is used in both training data and Lambda handler; changing it after training requires full retraining
- Training before export: LoRA adapter must exist before merge-and-export; correct adapter saving format in Phase 2 gates Phase 3
- Evaluation before containerization: ONNX numerical validation on Colab is cheaper than debugging a broken model in a deployed Lambda
- Container before infrastructure: ECR image URI must exist before Terraform can reference it in `aws_lambda_function`; ECR repository is provisioned first
- Infrastructure before CI/CD: GitHub Actions deploys to existing infrastructure; Terraform S3 state backend must exist before CI can run `terraform apply`

### Research Flags

Phases needing deeper research during planning:

- **Phase 1 (Data Foundation):** Public NJ housing dataset sourcing is under-documented in this research. Verify data.gov NJ property sales schema, download process, and record count before committing to synthetic augmentation ratio. Synthetic data county-level price statistics should be validated against current NJ real estate data.

- **Phase 3 (Evaluation and ONNX Export):** Correct `optimum-cli export onnx --task` flag for Qwen2.5 regression is not definitively established from this research. Confirm whether `feature-extraction` or `text-generation-with-past` is the correct task for single-token price prediction, and validate ONNX opset compatibility with Qwen2.5 rotary position embeddings before writing the export step.

Phases with standard, well-documented patterns (skip phase research):

- **Phase 2 (QLoRA Training):** QLoRA training with PEFT + SFTTrainer is the canonical HuggingFace tutorial workflow.
- **Phase 4 (Lambda Container):** Lambda container deployment with onnxruntime is fully covered by AWS official documentation.
- **Phase 5 (Infrastructure and CI/CD):** Terraform + GitHub Actions + OIDC are industry-standard with thorough official documentation.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All library versions verified against PyPI index on 2026-02-26. Terraform version confirmed from local binary. Lambda Python 3.12 runtime availability from AWS docs (verify current state at deployment time). |
| Features | MEDIUM | Core pipeline features are well-established. Token regression vs. regression head trade-off confirmed from training knowledge; token approach recommended for simpler ONNX graph. Qwen2.5-specific ONNX behavior needs empirical verification. |
| Architecture | HIGH | Component structure and data flow verified against official HuggingFace Optimum docs, AWS Lambda docs, and PEFT docs. Build order matches documented dependency constraints. |
| Pitfalls | MEDIUM-HIGH | QLoRA merge workflow pitfall confirmed by PEFT official docs. ONNX task type pitfall confirmed by Optimum docs. Colab disconnect and container size pitfalls are well-known community patterns. Synthetic data distribution pitfall is domain-specific judgment call requiring NJ market data verification. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Qwen2.5 ONNX export task type:** The correct `--task` flag for regression-as-generation with Qwen2.5-0.5B needs empirical verification. Plan a short export smoke test (one sample forward pass) before committing the full export pipeline. Address in Phase 3 planning.

- **NJ public housing dataset availability:** data.gov and NJ Treasury datasets may have changed schema or record counts. Verify actual downloadable record count before deciding synthetic augmentation ratio. Address in Phase 1 planning.

- **Lambda cold start latency:** Actual cold start time with a 0.5B ONNX model is estimated at 5-15 seconds but not empirically confirmed. Measure on first deploy in Phase 4. Provisioned concurrency decision depends on this measurement.

- **bitsandbytes CPU-mode import safety:** The claim that bitsandbytes v0.43+ supports CPU-only import without CUDA is MEDIUM confidence. Mitigation is straightforward: exclude bitsandbytes from Lambda requirements.txt entirely since it is not needed for ONNX inference.

---

## Sources

### Primary (HIGH confidence)

- PyPI index (`pip3 index versions <package>`) — all library versions verified 2026-02-26
- HuggingFace Optimum ONNX Export documentation — export workflow, task types, dynamic axes
- HuggingFace PEFT Official Docs (quantization guide, troubleshooting, LoRA reference) — QLoRA merge pattern, `prepare_model_for_kbit_training()` ordering
- HuggingFace PEFT Model Merging documentation — `merge_and_unload()` API
- AWS Lambda quotas documentation — 10GB container limit, 900s timeout, 10GB memory ceiling
- AWS Lambda Python container image docs — container deployment workflow, ECR integration
- Terraform Registry API — AWS provider 6.34.0 confirmed
- Local Terraform binary — version 1.14.6 confirmed

### Secondary (MEDIUM confidence)

- Training knowledge of PEFT/HuggingFace QLoRA patterns — core pipeline flows, stable API since 2023
- Training knowledge of ONNX Runtime inference patterns — session management, CPU vs. GPU builds
- NJ county-level housing market statistics — used for synthetic data validation guidance
- Qwen2.5-0.5B model card — architecture details, tokenizer requirements (sentencepiece backend)

### Tertiary (LOW confidence — verify before implementation)

- bitsandbytes v0.43+ CPU-mode import behavior — corroborated by PyPI version history; verify at https://github.com/bitsandbytes-foundation/bitsandbytes/releases
- Lambda Python 3.13 runtime wheel availability — verify current state at https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html
- optimum 2.x as transformers 5.x-compatible branch — verify at https://huggingface.co/docs/optimum/

---

*Research completed: 2026-02-26*
*Ready for roadmap: yes*
