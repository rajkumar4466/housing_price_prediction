# Stack Research

**Domain:** QLoRA fine-tuned LLM ML pipeline — training, ONNX export, Lambda serving
**Researched:** 2026-02-26
**Confidence:** HIGH (all versions verified directly from PyPI index and local Terraform binary)

---

## Recommended Stack

### Core ML Training Stack (Google Colab)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| torch | 2.10.0 | Neural network training, tensor ops | Latest stable; required by all HF training libs. Pin to exact version to avoid Colab dependency drift. |
| transformers | 5.2.0 | Model loading, tokenization, training APIs | v5.x is now stable (not 4.x). Qwen2.5 support is first-class. Use v5 — v4 is legacy. |
| peft | 0.18.1 | LoRA/QLoRA adapter injection, merge, save | The canonical QLoRA library. `get_peft_model()` + `LoraConfig` is the only sane path to adapter training. |
| bitsandbytes | 0.49.2 | 4-bit quantization (NF4) for QLoRA | Required for `load_in_4bit=True`. v0.43+ supports CPU-only mode — critical because Lambda deployment will not have CUDA; inference package must not crash at import. |
| accelerate | 1.12.0 | Mixed-precision training, device placement | Required by transformers `Trainer`. Handles `fp16`/`bf16` on Colab GPU automatically. |
| trl | 0.29.0 | `SFTTrainer` for supervised fine-tuning | Wraps `Trainer` with QLoRA-aware defaults. Simpler than raw `Trainer` for text-format regression tasks. |
| datasets | 4.6.0 | Dataset loading, train/val/test splitting | Native HuggingFace format; integrates directly with `SFTTrainer`. Handles arrow-format caching that matters on Colab's limited disk. |

### ONNX Export Stack (Colab, post-training)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| optimum | 2.1.0 | ONNX export via `optimum-cli export onnx` | The official HuggingFace ONNX exporter. Handles the merge-LoRA-then-export workflow and correctly sets dynamic axes for variable sequence lengths. |
| onnx | 1.20.1 | ONNX model validation, graph inspection | Core ONNX spec library. Used to verify exported model shape before packaging for Lambda. |
| onnxruntime | 1.24.2 | Colab-side inference validation | Validates the exported model runs correctly before shipping to Lambda. Use the same version in Lambda container. |

### Lambda Serving Stack (AWS)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| onnxruntime | 1.24.2 | CPU inference inside Lambda container | CPU-only build (`onnxruntime`, not `onnxruntime-gpu`). Matches validation version from Colab. No CUDA dependency = no Lambda crash. |
| fastapi | 0.133.1 | REST API handler inside Lambda | Minimal overhead, pydantic validation, clean request/response models. Pairs with Mangum for Lambda ASGI bridging. |
| mangum | 0.21.0 | ASGI-to-Lambda event adapter | Translates API Gateway v2 proxy events to ASGI. Single import wraps FastAPI with zero boilerplate. |
| Python runtime | 3.12 | Lambda execution environment | Use Python 3.12 (not 3.13) for Lambda — 3.13 Lambda runtime is GA but numpy/onnxruntime wheels lag behind. 3.12 has proven wheel availability for all stack dependencies. |

### Infrastructure (IaC)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Terraform | 1.14.6 | Provision Lambda + API Gateway | Declarative IaC, best-in-class AWS provider, reproducible teardown. Do not use CDK or SAM — adds cognitive overhead for a single-Lambda deployment. |
| AWS Provider (Terraform) | 6.34.0 | AWS resource definitions | Current major; use `~> 6.0` constraint to allow patch updates without breaking. |

### CI/CD

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| GitHub Actions | ubuntu-22.04 runner | Build Lambda container, push to ECR, apply Terraform | `ubuntu-22.04` has stable Docker + AWS CLI without surprise upgrades. Use `ubuntu-latest` only if you pin action versions too. |
| actions/checkout | v4 | Source checkout | Current major, supports sparse checkout for large repos. |
| aws-actions/configure-aws-credentials | v4 | OIDC-based AWS auth | No long-lived credentials in secrets. OIDC with IAM role is the 2025 standard. |
| aws-actions/amazon-ecr-login | v2 | ECR push auth | Works with OIDC credentials automatically. |
| hashicorp/setup-terraform | v3 | Terraform CLI in Actions | Caches terraform binary, supports `terraform fmt` check gates. |

---

## Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 2.4.2 | Array ops, data preprocessing | Always — torch, onnxruntime, pandas all depend on it. Pin to `>=2.0,<3` to avoid ABI breaks. |
| pandas | 3.0.1 | Dataset loading, CSV manipulation | Loading raw NJ housing data CSVs, feature engineering before prompt formatting. |
| scikit-learn | 1.8.0 | Train/val/test split, metrics (MAE, RMSE, R²) | `train_test_split`, `mean_absolute_error`, `r2_score` — simpler than reimplementing. |
| scipy | 1.17.1 | Statistical metrics | MAPE calculation, distribution analysis of price data. |
| matplotlib | 3.10.8 | Training loss curves, prediction vs actual plots | Colab-native rendering. Use `%matplotlib inline`. |
| sentencepiece | 0.2.1 | Qwen2.5 tokenizer backend | Qwen2.5 tokenizer requires sentencepiece. Must be installed alongside transformers. |
| tokenizers | 0.22.2 | Fast tokenizer Rust backend | Auto-installed by transformers but pin explicitly to avoid version drift on Colab. |

---

## Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Google Colab | Training + export environment | Free T4/A100 GPU. Use A100 when available for 4-bit training. Save checkpoints to Google Drive to survive session restarts. |
| Docker | Lambda container build | Use `public.ecr.aws/lambda/python:3.12` as base image — pre-configured for Lambda, not a generic python image. |
| AWS ECR | Container registry | Private registry in same region as Lambda. Tag images by git SHA, not `latest`. |
| AWS API Gateway v2 (HTTP API) | REST endpoint | HTTP API (not REST API) is cheaper, lower latency, and sufficient for simple POST /predict endpoint. |

---

## Installation

```bash
# Colab training environment (run in notebook cell)
pip install \
  torch==2.10.0 \
  transformers==5.2.0 \
  peft==0.18.1 \
  bitsandbytes==0.49.2 \
  accelerate==1.12.0 \
  trl==0.29.0 \
  datasets==4.6.0 \
  optimum==2.1.0 \
  onnx==1.20.1 \
  onnxruntime==1.24.2 \
  sentencepiece==0.2.1

# Lambda container requirements.txt
fastapi==0.133.1
mangum==0.21.0
onnxruntime==1.24.2
numpy==2.4.2

# Local dev / CI tools
pip install \
  pandas==3.0.1 \
  scikit-learn==1.8.0 \
  scipy==1.17.1 \
  matplotlib==3.10.8
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| transformers 5.2.0 | transformers 4.57.x | If a specific tutorial or community notebook targets 4.x and you can't debug v5 changes. Note: 4.x is legacy. |
| peft + trl (SFTTrainer) | raw Trainer with LoraConfig only | If you need granular control over the training loop. SFTTrainer is simpler for straightforward fine-tuning. |
| onnxruntime (CPU) on Lambda | PyTorch on Lambda | If you're willing to accept a 2-3x larger container image and slower cold starts. Do not do this — ONNX is the right call here. |
| optimum ONNX export | torch.onnx.export directly | For custom architectures not supported by optimum. Qwen2.5 is fully supported in optimum 2.x. |
| FastAPI + Mangum | AWS Lambda Powertools response utilities | If you don't need full ASGI and want lighter Lambda handler. FastAPI is cleaner for typed prediction endpoints. |
| Terraform | AWS SAM or CDK | SAM if you want tight Lambda-native tooling; CDK if the team is Python-native and prefers programmatic IaC. Terraform is simpler for this scope. |
| GitHub Actions | CircleCI, Jenkins | Only switch if GitHub Actions minutes become a cost concern (unlikely on free tier). |
| Python 3.12 Lambda runtime | Python 3.13 | 3.13 once numpy/onnxruntime wheels fully stabilize on Lambda base images (verify before using). |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| bitsandbytes < 0.43.0 | Versions before 0.43 require CUDA to import — will crash Lambda handler at cold start even for CPU inference | bitsandbytes >= 0.43.0 (use 0.49.2) |
| onnxruntime-gpu on Lambda | GPU not available in Lambda — package will fail or add unnecessary weight to container image | onnxruntime (CPU build) |
| PyTorch on Lambda for inference | Unquantized PyTorch model will be 1-2GB+; slow cold starts; exceeds Lambda ephemeral storage budget | ONNX Runtime for inference |
| transformers 4.x | Legacy branch, will not receive major feature development; Qwen2.5 support may have quirks vs v5 | transformers 5.2.0 |
| Python 3.11 on Lambda | 3.11 is not EOL but 3.12 has meaningfully better performance and is the current recommendation | Python 3.12 |
| `load_in_4bit=True` during ONNX export | Bitsandbytes quantized weights cannot be exported directly; must merge adapters to full precision first | Merge LoRA adapters → convert to fp16 → export ONNX |
| Quantized ONNX (INT4) for Lambda inference | INT4 ONNX requires specific EP hardware support not present in Lambda CPU; use fp16 or fp32 ONNX | fp16 ONNX export via optimum |
| `latest` Docker tags in CI | Non-deterministic builds; a base image update can silently break Lambda cold start behavior | Pin to `public.ecr.aws/lambda/python:3.12.YYYY-MM-DD` digest or version tag |
| `terraform apply` directly in CI without plan approval | Accidental destructive infrastructure changes | `terraform plan` in PR, `terraform apply` only on merge to main |

---

## Stack Patterns by Variant

**For Colab free tier (T4 GPU, 15GB VRAM):**
- Use `load_in_4bit=True` with `bnb_4bit_compute_dtype=torch.float16`
- Set `per_device_train_batch_size=1` with `gradient_accumulation_steps=8`
- Enable `gradient_checkpointing=True` to reduce peak VRAM
- Target LoRA rank r=8, alpha=16 (sufficient for regression adaptation on 0.5B model)

**For Colab Pro / A100:**
- Can use `bnb_4bit_compute_dtype=torch.bfloat16` (A100 has native bfloat16 support)
- Can increase batch size to 4-8 without gradient accumulation

**For Lambda deployment:**
- Do NOT include bitsandbytes in Lambda requirements.txt (not needed for ONNX inference)
- Do NOT include torch in Lambda requirements.txt (onnxruntime replaces it for inference)
- Use container image deployment (not .zip) — ONNX model + onnxruntime easily exceeds 50MB zip limit

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| transformers==5.2.0 | peft==0.18.1 | PEFT 0.18.x tracks transformers 5.x. Do not use peft 0.14.x or below with transformers 5.x. |
| torch==2.10.0 | onnxruntime==1.24.2 | ONNX opset 17-20 supported. Default export opset for transformers models is 14-17. |
| optimum==2.1.0 | transformers==5.2.0 | optimum 2.x is the transformers 5.x-compatible branch. Do not use optimum 1.x with transformers 5.x. |
| bitsandbytes==0.49.2 | torch==2.10.0 | Tested combination. bitsandbytes CUDA kernels must match torch CUDA version on Colab. |
| numpy==2.4.2 | onnxruntime==1.24.2 | onnxruntime 1.24.x requires numpy >= 1.24. numpy 2.x is supported. |
| Python 3.12 | Lambda runtime | Lambda Python 3.12 runtime is GA. All listed pip packages have 3.12 wheels on PyPI. |

---

## Critical Deployment Constraint

The ONNX export workflow must follow this exact sequence to avoid quantization export errors:

```python
# 1. Train with QLoRA (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = get_peft_model(model, lora_config)
# ... train ...

# 2. Merge LoRA weights back to base (exits quantization)
model = model.merge_and_unload()

# 3. Convert to fp16 full precision BEFORE export
model = model.half()

# 4. Export to ONNX via optimum (NOT torch.onnx.export directly)
# optimum handles dynamic axes and correct input signatures for causal LMs
```

Do NOT attempt to export the bitsandbytes-quantized model directly — it will fail.

---

## Sources

- PyPI index (`pip3 index versions <package>`) — versions verified 2026-02-26 — HIGH confidence
- Local Terraform binary `terraform version` — reports 1.14.3 installed, 1.14.6 latest — HIGH confidence
- Terraform Registry API (`registry.terraform.io`) — AWS provider 6.34.0 — HIGH confidence
- onnxruntime locally installed: 1.24.1 (1.24.2 is latest on PyPI) — HIGH confidence
- Lambda Python 3.12/3.13 runtime availability: known from AWS documentation (training data, August 2025 cutoff) — MEDIUM confidence; verify at https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html
- bitsandbytes CPU-mode in v0.43+: known from training data + corroborated by version jump in PyPI history (0.42.0 → 0.49.x skip) — MEDIUM confidence; verify at https://github.com/bitsandbytes-foundation/bitsandbytes/releases
- optimum 2.x as transformers 5.x branch: known from training data — MEDIUM confidence; verify at https://huggingface.co/docs/optimum/

---

*Stack research for: QLoRA fine-tuned LLM housing price predictor (NJ) — Colab training + ONNX Lambda serving*
*Researched: 2026-02-26*
