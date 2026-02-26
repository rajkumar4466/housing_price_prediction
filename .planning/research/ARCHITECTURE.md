# Architecture Research

**Domain:** QLoRA ML Pipeline with ONNX Lambda Serving
**Researched:** 2026-02-26
**Confidence:** HIGH (official HuggingFace docs + AWS docs verified)

## Standard Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRAINING ENVIRONMENT (Google Colab)              │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  01_data_prep.   │  │  02_train.ipynb  │  │  03_evaluate.ipynb   │  │
│  │  ipynb           │  │  QLoRA fine-tune │  │  MAE/RMSE/R²/MAPE    │  │
│  │  NJ housing data │  │  Qwen2.5-0.5B    │  │  plots + metrics     │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────────────┘  │
│           │                     │                                        │
│           ▼                     ▼                                        │
│  ┌────────────────┐    ┌────────────────────────────────────────────┐   │
│  │  data/         │    │  04_export.ipynb                           │   │
│  │  train.jsonl   │    │  merge_and_unload() → ONNX export          │   │
│  │  val.jsonl     │    │  (optimum-cli or ORTModelForCausalLM)      │   │
│  │  test.jsonl    │    └────────────────┬───────────────────────────┘   │
│  └────────────────┘                     │                                │
└─────────────────────────────────────────│───────────────────────────────┘
                                          │ model.onnx + tokenizer files
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARTIFACT STORAGE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Amazon ECR — Container Image (model.onnx bundled inside)       │    │
│  │  OR  Google Drive / GitHub Releases (ONNX artifact, < 1GB)      │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
└─────────────────────────────────│───────────────────────────────────────┘
                                  │ docker pull / image URI
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION SERVING (AWS)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  API Gateway (REST)                                           │       │
│  │  POST /predict                                                │       │
│  └──────────────────────────┬─────────────────────────────────────┘      │
│                             │ HTTP invoke                                │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  AWS Lambda (container image, Python 3.12)                   │       │
│  │  - Loads model.onnx + tokenizer at cold start                │       │
│  │  - Formats input as text prompt                               │       │
│  │  - Runs onnxruntime.InferenceSession                          │       │
│  │  - Returns predicted price as JSON                            │       │
│  └──────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE / CI-CD                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐        ┌─────────────────────────────────┐    │
│  │  Terraform (IaC)     │        │  GitHub Actions (CI/CD)         │    │
│  │  - Lambda function   │        │  - On push: lint + test         │    │
│  │  - API Gateway       │        │  - On release tag: build image  │    │
│  │  - IAM roles         │        │    → push ECR → terraform apply │    │
│  │  - ECR repository    │        └─────────────────────────────────┘    │
│  └──────────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `01_data_prep.ipynb` | Load NJ housing data, clean, generate synthetic records, format as text prompts, split train/val/test, save as JSONL | pandas, datasets, Google Drive mount |
| `02_train.ipynb` | Load Qwen2.5-0.5B with 4-bit quantization, apply QLoRA config, train with SFTTrainer or custom loop, save LoRA adapter | transformers, PEFT, bitsandbytes, trl |
| `03_evaluate.ipynb` | Load trained adapter, run inference on test set, compute MAE/RMSE/R²/MAPE, plot results | sklearn metrics, matplotlib |
| `04_export.ipynb` | Call `merge_and_unload()` on PeftModel to get plain model, run `optimum-cli export onnx`, validate ONNX output | PEFT, optimum, onnxruntime |
| `lambda/handler.py` | Parse API Gateway event, format property features as text prompt, tokenize, run ONNX inference session, return price prediction as JSON | onnxruntime, transformers tokenizer |
| `Dockerfile` | Bundle model.onnx + tokenizer + handler into Python Lambda container image | AWS base image `public.ecr.aws/lambda/python:3.12` |
| `terraform/` | Declare Lambda function (image URI), API Gateway REST API, IAM execution role, ECR repository | Terraform AWS provider |
| `.github/workflows/` | Run tests on PR; on release tag build + push Docker image to ECR, then run `terraform apply` to update Lambda | `docker/build-push-action`, AWS credentials action |

## Recommended Project Structure

```
housing_price_predictor/
├── notebooks/
│   ├── 01_data_prep.ipynb       # data loading, cleaning, prompt formatting
│   ├── 02_train.ipynb           # QLoRA fine-tuning
│   ├── 03_evaluate.ipynb        # metrics and plots
│   └── 04_export.ipynb          # merge + ONNX export + validation
├── lambda/
│   ├── handler.py               # Lambda function entry point
│   ├── requirements.txt         # onnxruntime, transformers (tokenizer only)
│   └── Dockerfile               # container image definition
├── terraform/
│   ├── main.tf                  # Lambda + API Gateway + ECR
│   ├── variables.tf             # image_uri, region, memory_size, timeout
│   ├── outputs.tf               # API Gateway invoke URL
│   └── versions.tf              # provider version pins
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint + test on PR
│       └── deploy.yml           # build image → ECR → terraform apply on tag
├── data/                        # gitignored — large files
│   ├── raw/                     # source CSV from data.gov / Kaggle
│   └── processed/               # train.jsonl, val.jsonl, test.jsonl
├── models/                      # gitignored — large artifacts
│   ├── lora_adapter/            # saved PEFT adapter weights
│   └── onnx/                    # model.onnx + tokenizer files
└── tests/
    └── test_handler.py          # unit tests for Lambda handler logic
```

### Structure Rationale

- **notebooks/:** All training work happens here; numbered prefix enforces execution order
- **lambda/:** Isolated deployment unit — only inference dependencies, no training libs
- **terraform/:** Kept flat (no modules) for a simple single-service deployment
- **data/ and models/:** Gitignored because large files don't belong in Git; transferred via Google Drive mount in Colab or manual upload

## Architectural Patterns

### Pattern 1: Merge-Before-Export

**What:** Call `peft_model.merge_and_unload()` before ONNX export to fold LoRA adapter weights (A and B matrices) into the base model weights. The result is a standard `PreTrainedModel` with no PEFT wrappers.

**When to use:** Always, before any ONNX export. ONNX exporters do not understand PEFT's dynamic adapter injection — they need a plain PyTorch model.

**Trade-offs:** Produces a larger model file (full weights instead of small LoRA delta), but eliminates the adapter dependency at inference time, which is correct for Lambda deployment.

**Example:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
peft_model = PeftModel.from_pretrained(base_model, "./models/lora_adapter/")

# Merge LoRA weights into base model — eliminates PEFT dependency
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./models/merged/")
```

### Pattern 2: ONNX Export with text-generation-with-past Task

**What:** Export the merged model using `optimum-cli export onnx` with the `text-generation-with-past` task. This produces two ONNX files: `model.onnx` (first token) and `model_with_past.onnx` (subsequent tokens), enabling KV-cache reuse during autoregressive generation.

**When to use:** For any text generation task including regression-as-generation. The `with-past` suffix is the correct default for causal LMs.

**Trade-offs:** Two ONNX files vs one, but significantly faster multi-token generation. For price prediction with a single output token, this still applies but `--monolith` flag can collapse to one file.

**Example:**
```bash
# Export merged local model to ONNX
optimum-cli export onnx \
  --model ./models/merged/ \
  --task text-generation-with-past \
  --device cpu \
  --dtype fp32 \
  ./models/onnx/
```

### Pattern 3: Lambda Cold Start Optimization via Global Model Load

**What:** Load the ONNX InferenceSession and tokenizer at module level (outside the handler function), not inside it. Lambda re-uses warm execution environments, so subsequent invocations skip model loading.

**When to use:** Always for ML models on Lambda — the model load is the expensive operation.

**Trade-offs:** Higher memory reservation needed (set Lambda memory to accommodate model in RAM), but eliminates model reload penalty on warm invocations.

**Example:**
```python
import onnxruntime as ort
from transformers import AutoTokenizer
import os

# Loaded once per Lambda execution environment (cold start only)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"]
)

def handler(event, context):
    # tokenizer and session already loaded
    body = json.loads(event["body"])
    prompt = format_prompt(body)
    inputs = tokenizer(prompt, return_tensors="np")
    outputs = session.run(None, dict(inputs))
    price = decode_price(outputs, tokenizer)
    return {"statusCode": 200, "body": json.dumps({"predicted_price": price})}
```

### Pattern 4: Prompt-Formatted Tabular Input

**What:** Convert the 7 property features into a structured natural language string before tokenization. This is the interface contract between the training data and the inference service — both must use the exact same format.

**When to use:** Required for any LoRA fine-tuning approach on tabular data. The prompt template is defined once in a shared utility and used in both the data prep notebook and the Lambda handler.

**Example:**
```python
def format_prompt(features: dict) -> str:
    return (
        f"Predict the price of a property in New Jersey. "
        f"Type: {features['property_type']}. "
        f"Location: zip code {features['zip_code']}. "
        f"Size: {features['sqft']} sqft, lot {features['lot_size']} sqft. "
        f"Bedrooms: {features['bedrooms']}, Bathrooms: {features['bathrooms']}. "
        f"Year built: {features['year_built']}. "
        f"Predicted price: $"
    )
```

## Data Flow

### Training Pipeline (Colab, one-time)

```
Raw NJ housing CSV (data.gov / Kaggle)
    ↓ [01_data_prep.ipynb]
Cleaned + synthetic records
    ↓ format_prompt() applied to each row
JSONL files: {"text": "<prompt>$<price>"}
    ↓ [02_train.ipynb]
Qwen2.5-0.5B (4-bit quantized) + LoRA adapter
    ↓ save_pretrained()
./models/lora_adapter/  (adapter_model.safetensors + adapter_config.json)
    ↓ [03_evaluate.ipynb] — validates on test.jsonl
Metrics report (MAE, RMSE, R², MAPE) + plots
    ↓ [04_export.ipynb]
merge_and_unload() → merged model
    ↓ optimum-cli export onnx
./models/onnx/  (model.onnx + tokenizer files)
    ↓ manual upload / GitHub Release artifact
Docker image build (model.onnx bundled)
    ↓ push to Amazon ECR
```

### Production Inference Request Flow

```
HTTP POST /predict
  {"bedrooms": 3, "bathrooms": 2, "sqft": 1800, ...}
    ↓ [API Gateway]
AWS Lambda invocation (event dict)
    ↓ [handler.py] parse event body
format_prompt(features) → text string
    ↓ tokenizer.encode()
input_ids tensor (numpy)
    ↓ onnxruntime.InferenceSession.run()
logits → argmax on next token
    ↓ tokenizer.decode() + parse float
{"predicted_price": 425000.0}
    ↓ [API Gateway]
HTTP 200 JSON response
```

### CI/CD Deployment Flow

```
git push --tag v1.0.0
    ↓ [GitHub Actions: deploy.yml triggered on tag]
docker build --platform linux/amd64 --provenance=false
    ↓ docker push → Amazon ECR
aws ecr describe-images (verify push)
    ↓ terraform init + terraform plan
terraform apply -var="image_uri=<ecr_uri>:<tag>"
    ↓ aws_lambda_function.image_uri updated
Lambda function redeployed with new container
    ↓ smoke test: curl API Gateway endpoint
Deploy complete
```

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-100 req/day | Default Lambda config sufficient; free tier covers cost; single container image |
| 100-10K req/day | Provision concurrency to eliminate cold starts (~3-5 warm instances); consider Lambda memory bump to 2GB for faster inference |
| 10K+ req/day | Lambda scales automatically but API Gateway throttling becomes relevant; consider response caching at API Gateway level; model size limits Lambda concurrency scaling rate |

### Scaling Priorities

1. **First bottleneck — cold start latency:** ONNX model load takes 5-15 seconds at cold start. Fix with provisioned concurrency (maintains warm instances). Cost increases but latency becomes consistent.
2. **Second bottleneck — Lambda timeout under load:** At high concurrency, Lambda may hit the 15-minute timeout if inference is slow. Fix by profiling ONNX session and enabling ONNX Runtime graph optimizations (`--optimize O2`).

## Anti-Patterns

### Anti-Pattern 1: Exporting the PeftModel Directly to ONNX

**What people do:** Call `optimum-cli export onnx` on the PEFT-wrapped model directory (containing `adapter_config.json` and `adapter_model.safetensors`) instead of the merged model.

**Why it's wrong:** ONNX exporters trace the model's computational graph. PEFT adapters are applied as dynamic weight injections that the ONNX tracer cannot follow correctly. The export either fails or produces an ONNX model that doesn't include the fine-tuned behavior.

**Do this instead:** Always call `merge_and_unload()` first, save the merged model to a new directory, then export that directory to ONNX.

### Anti-Pattern 2: Installing Training Dependencies in the Lambda Image

**What people do:** Reuse the Colab `requirements.txt` (PyTorch, PEFT, bitsandbytes, trl, datasets, etc.) for the Lambda Dockerfile.

**Why it's wrong:** PyTorch alone is ~2GB. The Lambda container would approach or exceed the 10GB image size limit, cold start times become multi-minute, and the training libraries are never used at inference.

**Do this instead:** Lambda only needs `onnxruntime` (CPU build, ~200MB) and `transformers` for the tokenizer. Keep the Dockerfile minimal — install only inference-time requirements.

### Anti-Pattern 3: Hardcoding the Prompt Template in Two Places

**What people do:** Write the `format_prompt()` function separately in the data prep notebook and again in the Lambda handler, with slight differences.

**Why it's wrong:** A mismatch between training prompt format and inference prompt format causes the model to see a different distribution at inference time, degrading prediction accuracy. This is silent — no error is thrown.

**Do this instead:** Define a single `format_prompt()` function in a shared file (e.g., `lambda/prompt_utils.py`). Import it in the notebooks by adding the project root to the path, and include it in the Lambda package.

### Anti-Pattern 4: Committing Model Artifacts to Git

**What people do:** Add `model.onnx` or the LoRA adapter checkpoint to the Git repository for convenience.

**Why it's wrong:** ONNX models for even 0.5B parameter models are 300MB+, pushing the repo into LFS territory or failing entirely. GitHub has a 100MB file size hard limit.

**Do this instead:** Store model artifacts in Google Drive (accessible from Colab) or upload as GitHub Release assets. Reference the artifact URI in Terraform variables or Docker build args, not in Git.

### Anti-Pattern 5: Running ONNX Inference Inside a Docker Build

**What people do:** Bundle model loading and a validation inference call in the `RUN` step of the Dockerfile to "verify" the model works.

**Why it's wrong:** Docker build runs on the CI runner, not Lambda. Inference inside a build step adds 30-60 seconds to every CI run, increases build memory requirements, and provides no guarantee the model works on Lambda's actual execution environment.

**Do this instead:** Validate ONNX output in the Colab export notebook (04_export.ipynb) before containerization. Run a post-deploy smoke test against the live Lambda endpoint after `terraform apply`.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Amazon ECR | Docker push via GitHub Actions; `aws ecr get-login-password` then `docker push` | Requires IAM credentials in GitHub Secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) |
| API Gateway | Terraform-managed REST API; Lambda proxy integration — full event/context passed to handler | 6MB response payload limit; plan for < 1KB prediction response |
| Google Colab | Manual notebook execution; Google Drive mounted for data and model artifact persistence | No CI integration — training is intentionally manual and one-time |
| GitHub Releases | Optional ONNX artifact hosting; alternative to bundling model in Docker image | Suitable if model < 2GB; keeps container image small |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Notebooks ↔ Lambda handler | Shared `format_prompt()` function via `lambda/prompt_utils.py` | Critical boundary — prompt format mismatch = silent accuracy degradation |
| Training ↔ Export | File system: `./models/lora_adapter/` directory written by 02, read by 04 | Notebooks must be run in order; no formal dependency enforcement |
| Lambda handler ↔ ONNX model | ONNX Runtime `InferenceSession.run()` with numpy arrays | Input/output names must match the exported graph; validate in 04_export.ipynb |
| Terraform ↔ ECR | `image_uri` variable passed at apply time; Lambda function updated in-place | `terraform apply` only needed when image URI changes (new tag pushed) |
| GitHub Actions ↔ Terraform | Actions runs `terraform apply` with AWS credentials from Secrets | Terraform state must be in S3 backend or Terraform Cloud for CI to work; local state breaks team workflows |

## Build Order (Phase Dependencies)

Based on component dependencies, the mandatory build sequence is:

```
Phase 1: Data Preparation
  → 01_data_prep.ipynb produces train/val/test JSONL
  → Nothing else can proceed without this data

Phase 2: Model Training
  → 02_train.ipynb requires Phase 1 data
  → Produces LoRA adapter weights

Phase 3: Evaluation
  → 03_evaluate.ipynb requires Phase 2 adapter + Phase 1 test data
  → Can run in parallel with Phase 4 if metrics are not a gate

Phase 4: ONNX Export
  → 04_export.ipynb requires Phase 2 adapter (merge_and_unload + export)
  → Produces model.onnx — required for Lambda

Phase 5: Lambda Service
  → lambda/handler.py + Dockerfile requires Phase 4 ONNX artifact
  → Container image built and pushed to ECR

Phase 6: Infrastructure
  → terraform/ requires ECR image URI from Phase 5
  → Deploys Lambda + API Gateway

Phase 7: CI/CD Automation
  → .github/workflows/ automates Phases 5-6
  → Requires Terraform state backend (S3) to be set up first
```

## Sources

- HuggingFace Optimum ONNX Export documentation: https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model (verified 2026-02-26) — HIGH confidence
- HuggingFace Transformers serialization guide: https://huggingface.co/docs/transformers/main/en/serialization (verified 2026-02-26) — HIGH confidence
- PEFT model merging documentation: https://huggingface.co/docs/peft/main/en/developer_guides/model_merging (verified 2026-02-26) — HIGH confidence
- PEFT LoRA API reference (merge_and_unload via PeftModel): https://huggingface.co/docs/peft/main/en/package_reference/lora (verified 2026-02-26) — HIGH confidence
- AWS Lambda quotas (10GB image limit, 900s timeout, 10GB memory): https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html (verified 2026-02-26) — HIGH confidence
- AWS Lambda Python container image deployment: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html (verified 2026-02-26) — HIGH confidence

---
*Architecture research for: NJ Housing Price Predictor (QLoRA + ONNX + Lambda)*
*Researched: 2026-02-26*
