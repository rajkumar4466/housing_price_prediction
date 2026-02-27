# Roadmap: NJ Housing Price Predictor

## Overview

This pipeline moves in one strict sequence: curate NJ housing data and define the shared prompt format, fine-tune Qwen2.5-0.5B with QLoRA on Colab, evaluate the trained model and export to ONNX, package ONNX inference into a minimal Lambda container, then provision the cloud infrastructure and wire CI/CD. Each phase gates the next — there is no parallelism because the model artifact is the critical dependency throughout.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Foundation** - Build, validate, and format the NJ housing dataset that all downstream training depends on (completed 2026-02-26)
- [x] **Phase 2: QLoRA Training** - Fine-tune Qwen2.5-0.5B with 4-bit quantization on Colab GPU; LoRA adapter saved to Google Drive (completed 2026-02-27)
- [ ] **Phase 3: Evaluation and ONNX Export** - Validate model accuracy and produce a Colab-validated ONNX artifact ready for containerization
- [ ] **Phase 4: Lambda Container and REST API** - Package ONNX inference into a minimal, deployable Lambda container image
- [x] **Phase 5: Infrastructure and CI/CD** - Provision cloud infrastructure with Terraform and automate deployment with GitHub Actions (completed 2026-02-27)

## Phase Details

### Phase 1: Data Foundation
**Goal**: Training-ready NJ housing data exists with validated price distributions and a shared prompt format defined once for both training and inference
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. `train.jsonl`, `val.jsonl`, and `test.jsonl` exist with the correct 70/15/15 split across all 7 property features
  2. `lambda/prompt_utils.py` contains `format_prompt()` and can be imported by both notebooks and the Lambda handler without modification
  3. Synthetic records use county-level NJ price distributions (log-normal with county multipliers), and the price histogram matches known NJ county medians
  4. At least 30% of records derive from public NJ datasets (data.gov or equivalent), and the dataset schema is documented
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Create shared prompt module (lambda/prompt_utils.py) and project scaffold
- [x] 01-02-PLAN.md — Build data generation notebook (notebooks/01_data_prep.ipynb) and produce train/val/test JSONL splits

### Phase 2: QLoRA Training
**Goal**: A LoRA adapter checkpoint exists on Google Drive, trained from validated Phase 1 data, completing within the 20-minute Colab budget
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. `02_train.ipynb` runs end-to-end on Colab free tier GPU and finishes in under 20 minutes
  2. LoRA adapter weights are saved to Google Drive and can be reloaded in a fresh Colab session
  3. Training loss decreases across epochs and the loss curve plot is generated without errors
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md — Build QLoRA training notebook; train Qwen2.5-0.5B on 4,900 NJ housing records; adapter at Drive/housing_model/lora_adapter/ (final loss: 0.6514)

### Phase 3: Evaluation and ONNX Export
**Goal**: The model is evaluated against held-out test data, all 4 regression metrics are computed, and a numerically validated ONNX artifact is ready for containerization
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03, ONNX-01, ONNX-02, ONNX-03
**Success Criteria** (what must be TRUE):
  1. MAE, RMSE, R², and MAPE are computed on the held-out test set and printed in `03_evaluate.ipynb`
  2. A predicted-vs-actual scatter plot and a training loss curve are generated as image files by the evaluation notebook
  3. `model.onnx` and tokenizer files exist and pass numerical validation against PyTorch outputs at `atol=1e-3` on Colab
  4. The merge was performed via the two-step fp32 reload pattern (base reloaded without `load_in_4bit`, then `merge_and_unload()` called)
**Plans**: TBD

### Phase 4: Lambda Container and REST API
**Goal**: A Docker container image with ONNX Runtime (CPU), tokenizer, and handler is ready to push to ECR and returns a valid price prediction from a local test invocation
**Depends on**: Phase 3
**Requirements**: SERV-01, SERV-02, SERV-03
**Success Criteria** (what must be TRUE):
  1. A POST request with valid 7-feature JSON payload to the handler returns a predicted price in under 5 seconds during a warm invocation
  2. The container image contains no PyTorch or bitsandbytes — only `onnxruntime-cpu`, tokenizer files, `fastapi`, and `mangum`
  3. `docker run --network none` successfully runs a prediction (confirms tokenizer is bundled, not fetched at runtime)
  4. The `InferenceSession` and tokenizer are initialized as module-level globals (verified by a single cold-start log showing model load once per container lifetime)
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Create Lambda handler (handler.py), requirements.txt, and Dockerfile
- [ ] 04-02-PLAN.md -- Build Docker image and verify locally with RIE + offline test

### Phase 5: Infrastructure and CI/CD
**Goal**: Lambda and API Gateway are provisioned by Terraform, the ECR image is deployed, and a live API endpoint returns a price prediction on a smoke test
**Depends on**: Phase 4
**Requirements**: INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):
  1. `terraform apply` completes without errors and creates Lambda function, API Gateway v2 HTTP API, ECR repository, and IAM execution role
  2. A `curl` smoke test against the live API Gateway endpoint returns a predicted price with HTTP 200
  3. A GitHub Actions PR run shows lint + Lambda handler tests passing; a tagged commit triggers image build, ECR push, and `terraform apply` without manual steps
  4. Terraform state is stored in an S3 backend (no local `terraform.tfstate` committed to the repository)
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md -- Create Terraform IaC (versions.tf, main.tf, variables.tf, outputs.tf) for ECR, Lambda, API Gateway v2, IAM, OIDC, S3 backend
- [x] 05-02-PLAN.md -- Create GitHub Actions CI (ci.yml) and Deploy (deploy.yml) workflows

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Foundation | 2/2 | Complete   | 2026-02-27 |
| 2. QLoRA Training | 1/1 | Complete   | 2026-02-27 |
| 3. Evaluation and ONNX Export | 0/TBD | Not started | - |
| 4. Lambda Container and REST API | 1/2 | In Progress|  |
| 5. Infrastructure and CI/CD | 2/2 | Complete   | 2026-02-27 |
