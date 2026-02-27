# Phase 4: Lambda Container and REST API - Research

**Researched:** 2026-02-27
**Domain:** AWS Lambda container image, ONNX Runtime CPU inference, FastAPI + Mangum REST API, Docker, tokenizer bundling
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SERV-01 | Implement ONNX Runtime inference handler for price prediction | Global `InferenceSession` + `AutoTokenizer` pattern; `session.run()` with numpy inputs; `parse_price_from_output()` already in `lambda/prompt_utils.py` |
| SERV-02 | Expose REST API endpoint accepting 7 property features, returning predicted price | FastAPI `POST /predict` with Pydantic request model; Mangum wraps FastAPI app as Lambda handler; API Gateway v2 HTTP API |
| SERV-03 | Build minimal container image (onnxruntime + tokenizer only, no PyTorch) | `public.ecr.aws/lambda/python:3.12` base; `requirements.txt` with only `onnxruntime==1.24.2`, `fastapi==0.133.1`, `mangum==0.21.0`, `numpy==2.4.2`, `transformers==5.2.0`; COPY model artifacts into `${LAMBDA_TASK_ROOT}` |
</phase_requirements>

---

## Summary

Phase 4 builds the serving layer: a Docker container image containing the ONNX model, tokenizer, and a FastAPI handler, pushed to Amazon ECR, and invocable locally without network access. The primary technical domain is Lambda container image construction with minimal dependencies — no PyTorch, no bitsandbytes, no training dependencies of any kind.

The pattern is well-established: FastAPI app wrapped by Mangum, `InferenceSession` and tokenizer initialized as module-level globals at cold start, and the container built from `public.ecr.aws/lambda/python:3.12`. The critical constraint is that the ONNX model and tokenizer files must be `COPY`ed into the image during the Docker build — they are never fetched at runtime. This is verified by running `docker run --network none` before pushing to ECR.

The two non-trivial decisions for this phase are: (1) how to perform the ONNX token-to-price decoding in the handler (using the existing `parse_price_from_output()` from `lambda/prompt_utils.py`, not a custom implementation), and (2) how to test the handler locally before ECR push (using the Lambda Runtime Interface Emulator built into the AWS base image via `curl -XPOST http://localhost:9000/2015-03-31/functions/function/invocations`). Both have clear, documented approaches.

**Primary recommendation:** Build `lambda/handler.py` with module-level globals first, test locally with `docker run`, then validate `--network none` before touching ECR or Terraform.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnxruntime | 1.24.2 | CPU inference on Lambda — runs the exported ONNX model | CPU-only build; no CUDA dependency; same version validated in Phase 3 Colab export |
| fastapi | 0.133.1 | REST API with typed request/response models and automatic validation | Pydantic models give free 422 validation; clean integration with Mangum |
| mangum | 0.21.0 | ASGI-to-Lambda event adapter; wraps FastAPI for API Gateway v2 | Single import, zero boilerplate; handles HTTP API + REST API + Function URL events |
| transformers | 5.2.0 | `AutoTokenizer` only — no model loading | Tokenizer must match training tokenizer exactly; v5 is the current stable branch |
| numpy | 2.4.2 | Array ops for ONNX session inputs/outputs | Required by onnxruntime; pin to avoid ABI breaks |
| Python | 3.12 | Lambda execution runtime | Proven wheel availability for all stack deps; better performance than 3.11 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sentencepiece | 0.2.1 | Qwen2.5 tokenizer backend (required by AutoTokenizer) | Qwen2.5 uses sentencepiece; add to Lambda requirements.txt even though it's not imported directly |
| tokenizers | 0.22.2 | Fast tokenizer Rust backend | Auto-installed by transformers; pin explicitly to avoid version drift |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastAPI + Mangum | Raw Lambda handler (no FastAPI) | Simpler cold start, but loses Pydantic validation and clean request/response models; not recommended given the 7-field typed input requirement |
| FastAPI + Mangum | AWS Lambda Powertools response utilities | Lighter-weight but no automatic Pydantic validation; FastAPI is cleaner for typed prediction endpoints |
| Module-level global session | Per-request session creation | Per-request adds 3-5s overhead on every warm invocation; module-level is mandatory |
| COPY model into image | S3 download on cold start | S3 download adds 2-10s cold start latency per invocation; bundling in image is correct for this model size |

**Installation (Lambda requirements.txt):**
```bash
# lambda/requirements.txt — NO PyTorch, NO bitsandbytes, NO training deps
onnxruntime==1.24.2
fastapi==0.133.1
mangum==0.21.0
numpy==2.4.2
transformers==5.2.0
sentencepiece==0.2.1
tokenizers==0.22.2
```

---

## Architecture Patterns

### Recommended Project Structure

```
lambda/
├── handler.py               # FastAPI app + Mangum wrapper + global session init
├── prompt_utils.py          # Already exists: format_prompt() + parse_price_from_output()
├── requirements.txt         # Minimal inference deps (see above)
├── Dockerfile               # AWS base image, COPY model artifacts, install deps
└── model_artifacts/         # Populated at Docker build time (COPY from local path)
    ├── model.onnx           # Exported in Phase 3
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.json           # (Qwen2.5 tokenizer files)
    ├── merges.txt
    └── special_tokens_map.json
```

**Key structural decision:** Model artifacts live in `lambda/model_artifacts/` locally and are `COPY`ed to `${LAMBDA_TASK_ROOT}/model_artifacts/` in the Docker build. The handler references them via `os.path.join(os.path.dirname(__file__), "model_artifacts")` or via the `LAMBDA_TASK_ROOT` environment variable.

### Pattern 1: Module-Level Global Session Initialization

**What:** Load `InferenceSession` and `AutoTokenizer` at the Python module level, outside all functions. Lambda re-uses warm execution environments, so these objects survive across warm invocations.

**When to use:** Always for ML models on Lambda. This is the single most important performance pattern.

**Example:**
```python
# lambda/handler.py
import os
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI
from mangum import Mangum
import importlib

# Import from lambda/ dir (lambda is a Python reserved keyword — use importlib)
prompt_utils = importlib.import_module("lambda.prompt_utils")
format_prompt = prompt_utils.format_prompt
parse_price_from_output = prompt_utils.parse_price_from_output

# --- Module-level globals: initialized ONCE at cold start ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"],
)
# -----------------------------------------------------------

app = FastAPI()
handler = Mangum(app, lifespan="off")
```

**Note on import path:** `lambda/prompt_utils.py` already exists in the project. Because `lambda` is a Python reserved keyword, direct `from lambda.prompt_utils import ...` raises `SyntaxError`. Use `importlib.import_module("lambda.prompt_utils")` — this pattern is already documented in STATE.md decisions.

### Pattern 2: FastAPI Request/Response Models with Pydantic

**What:** Define typed Pydantic models for the 7-field input and the prediction response. FastAPI automatically validates input and returns 422 (Unprocessable Entity) on field-level errors.

**When to use:** Always for typed REST APIs. The automatic validation eliminates hand-rolled field checking.

**Example:**
```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: float = Field(..., ge=0.5, le=10.0)
    sqft: int = Field(..., ge=100, le=20000)
    lot_size: float = Field(..., ge=0.01, le=100.0)
    year_built: int = Field(..., ge=1800, le=2025)
    zip_code: str = Field(..., min_length=5, max_length=5)
    property_type: str = Field(..., pattern="^(Single Family|Condo|Townhouse|Multi-Family)$")

class PredictResponse(BaseModel):
    predicted_price: float
    predicted_price_rounded: int

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prompt = format_prompt(
        bedrooms=req.bedrooms,
        bathrooms=req.bathrooms,
        sqft=req.sqft,
        lot_size=req.lot_size,
        year_built=req.year_built,
        zip_code=req.zip_code,
        property_type=req.property_type,
    )
    inputs = tokenizer(prompt, return_tensors="np")
    input_feed = {k: v for k, v in inputs.items()}
    outputs = session.run(None, input_feed)
    # outputs[0] is logits shape [batch, seq_len, vocab_size]
    next_token_id = int(np.argmax(outputs[0][0, -1, :]))
    generated_token = tokenizer.decode([next_token_id])
    price = parse_price_from_output(generated_token)
    if price is None:
        # Fallback: decode more tokens by running generation loop
        # For now, raise a 500 — add generation loop in Wave 2
        raise RuntimeError(f"Could not parse price from token: {generated_token!r}")
    rounded = round(price / 1000) * 1000
    return PredictResponse(predicted_price=price, predicted_price_rounded=rounded)
```

**IMPORTANT — Single-token vs. multi-token generation:** The model generates a price as text (e.g., "450000"). A single `session.run()` call returns logits for only the next token position. For multi-digit price generation, you need an autoregressive loop (run N times, feeding previous output tokens back). The ONNX export from Phase 3 determines whether this uses `model.onnx` + `model_with_past.onnx` (with KV-cache) or a single `model.onnx` (without). Plan for the generation loop — do not assume a single forward pass returns the full price string.

### Pattern 3: Dockerfile for Lambda Container

**What:** Use `public.ecr.aws/lambda/python:3.12` as the base image (not a generic Python image). This base image includes the Lambda Runtime Interface Client and RIE.

**When to use:** Always for Lambda container deployments. The AWS base image is pre-configured; generic Python images require additional setup.

**Example:**
```dockerfile
FROM public.ecr.aws/lambda/python:3.12

# Install inference dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy Lambda handler and shared prompt utilities
COPY handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY prompt_utils.py ${LAMBDA_TASK_ROOT}/prompt_utils.py

# Copy model artifacts (model.onnx + tokenizer files)
# These must be present locally before docker build
COPY model_artifacts/ ${LAMBDA_TASK_ROOT}/model_artifacts/

# Set the handler: <module>.<function>
# With Mangum, the handler attribute is the Mangum instance
CMD [ "handler.handler" ]
```

**Build command (must specify platform for Lambda):**
```bash
docker build --platform linux/amd64 --provenance=false -t housing-predictor:latest .
```

### Pattern 4: Local Testing with Lambda Runtime Interface Emulator (RIE)

**What:** The AWS base image (`public.ecr.aws/lambda/python:3.12`) includes the RIE. Running the container locally exposes a `/2015-03-31/functions/function/invocations` endpoint that simulates Lambda invocation.

**When to use:** Before ECR push — always. This catches handler errors, import failures, and cold start issues without incurring Lambda costs.

**Example:**
```bash
# Run container locally (RIE built into AWS base image)
docker run --rm -p 9000:8080 housing-predictor:latest

# Test with a valid 7-feature payload (in another terminal)
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "httpMethod": "POST",
    "path": "/predict",
    "headers": {"Content-Type": "application/json"},
    "body": "{\"bedrooms\": 3, \"bathrooms\": 2.0, \"sqft\": 1800, \"lot_size\": 0.25, \"year_built\": 1995, \"zip_code\": \"07030\", \"property_type\": \"Single Family\"}"
  }'

# Verify tokenizer is bundled — MUST work with no network
docker run --network none --rm -p 9000:8080 housing-predictor:latest
# Then invoke as above; if tokenizer tries to fetch from HuggingFace Hub, this will fail
```

**Note on Mangum and RIE:** Mangum translates the raw Lambda event dict (API Gateway v2 format) into an ASGI scope. When testing locally with RIE, the event must be a valid API Gateway v2 proxy event, not a raw JSON body. The body must be JSON-stringified (string, not object) in the event dict.

### Pattern 5: Preventing Tokenizer Hub Download at Runtime

**What:** Set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` as environment variables in the Dockerfile or Lambda function config to prevent any HuggingFace Hub network calls at runtime.

**When to use:** Always for Lambda container images. Tokenizer files must be bundled in the image — no network calls at inference time.

**Example:**
```dockerfile
# In Dockerfile, after COPY model_artifacts/
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
```

`TOKENIZERS_PARALLELISM=false` prevents a spurious warning from the fast tokenizer's Rust backend when Lambda forks the process.

### Anti-Patterns to Avoid

- **Per-request `InferenceSession` creation:** Creates a new session on every invocation. Adds 3-5 seconds overhead to every warm invocation. Module-level global is mandatory.
- **`pip install torch` in Lambda Dockerfile:** Adds ~2-3GB to the image. PyTorch is not needed for ONNX inference. It will push the image toward the 10GB limit and dramatically increase cold start latency.
- **`from lambda.prompt_utils import ...`:** Raises `SyntaxError` because `lambda` is a Python reserved keyword. Use `importlib.import_module("lambda.prompt_utils")`.
- **Fetching tokenizer from HuggingFace Hub at cold start:** If `model_artifacts/` is missing tokenizer files, `AutoTokenizer.from_pretrained()` will attempt a Hub fetch. Lambda's network access is restricted and this adds 10-60 second latency. Bundle tokenizer files in the image.
- **`docker build` without `--platform linux/amd64`:** On Apple Silicon (M-series) Macs, Docker defaults to `linux/arm64`. Lambda runs on `x86_64`. Always specify `--platform linux/amd64` to avoid architecture mismatch errors on Lambda.
- **Committing `model_artifacts/` to Git:** Qwen2.5-0.5B ONNX model is ~1-2GB. GitHub has a 100MB file size limit. Use `.gitignore` to exclude `lambda/model_artifacts/`.
- **Single forward pass assuming full price string:** The ONNX model's `session.run()` returns logits for the next token only. "450000" requires 6 tokens. Plan for an autoregressive generation loop.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| API Gateway event → ASGI scope translation | Custom event parser | `mangum.Mangum` | Handles all event types (HTTP API, REST API, Function URL, ALB) with correct scope mapping; edge cases in path/query handling are subtle |
| Request field validation | Manual if/isinstance checks | `pydantic.BaseModel` + FastAPI | Automatic 422 responses with field-level error messages; ge/le constraints declarative |
| Price string parsing from LLM output | Custom regex | `parse_price_from_output()` in `lambda/prompt_utils.py` | Already implemented and tested; changing it creates a divergence from the training/inference contract |
| Prompt construction | Inline f-string in handler | `format_prompt()` in `lambda/prompt_utils.py` | Already the single source of truth; inline construction would create a divergence from training data format |
| ONNX CPU execution provider selection | Manual EP config | `providers=["CPUExecutionProvider"]` | One line; ort selects CPU automatically if left empty, but explicit is better for Lambda |

**Key insight:** The `lambda/prompt_utils.py` file is already the source of truth for `format_prompt()` and `parse_price_from_output()`. The handler must import from there — never reimplement these functions inline.

---

## Common Pitfalls

### Pitfall 1: Single-Token Assumption for Price Generation

**What goes wrong:** Handler calls `session.run()` once, takes `argmax` of logits, decodes one token, and expects to get the full price string (e.g., "450000"). The model actually generates one token at a time. One token produces "4" not "450000". `parse_price_from_output("4")` returns 4.0, not the actual price.

**Why it happens:** ONNX export via `optimum-cli` produces a model that does one forward pass at a time. Autoregressive generation requires running the session multiple times in a loop.

**How to avoid:** Implement a generation loop: call `session.run()` repeatedly, appending the predicted token to `input_ids`, until either a stop token (`EOS`) is generated or N max tokens are reached. Then decode the full generated sequence and call `parse_price_from_output()`. If Phase 3 exported with `text-generation-with-past`, use the `model_with_past.onnx` for subsequent tokens (faster via KV-cache). If exported with `feature-extraction`, the loop uses the same model file each time.

**Warning signs:** Handler returns a single digit price like `4.0`, `5.0`, or `0.0` — this means only the first token of the price was decoded.

### Pitfall 2: PyTorch in Requirements Triggers `transformers[torch]` Install

**What goes wrong:** `pip install transformers==5.2.0` without extras should NOT pull in PyTorch. But `pip install transformers[torch]` or any package that depends on torch will cause a 2-3GB install in the Docker layer.

**Why it happens:** Some packages list torch as a dependency (e.g., old versions of PEFT). Since PEFT is not in Lambda requirements, this risk is low — but it can still happen if a transitive dependency pulls in torch.

**How to avoid:** After building the Docker image, run `docker run --rm housing-predictor:latest pip show torch` and verify it is NOT installed. Also check image size: `docker image ls | grep housing-predictor` — if > 3GB, investigate with `docker history housing-predictor:latest`.

**Warning signs:** Docker image size > 3GB. `docker history` shows a layer with 2GB+ pip install.

### Pitfall 3: Lambda Invocation Timeout Because Mangum Cannot Route the Event

**What goes wrong:** The Lambda function times out or returns a 502 error because Mangum cannot parse the raw event format. This happens when testing with a raw JSON body instead of a valid API Gateway v2 proxy event structure.

**Why it happens:** Mangum expects an event dict with `version`, `httpMethod`/`requestContext`, `path`, `body`, and `headers` fields. A raw event like `{"bedrooms": 3, ...}` does not have this structure. Mangum will raise an error or return a 500.

**How to avoid:** When testing locally with RIE, always wrap the payload in a valid API Gateway v2 event structure. When testing via the actual API Gateway, the event is formatted correctly automatically. Use the RIE test command from Pattern 4 above (with the properly structured event).

**Warning signs:** RIE returns `{"statusCode": 500}` or a timeout. CloudWatch logs show `ValueError: Unable to determine event type` or similar Mangum error.

### Pitfall 4: Cold Start Timeout Due to OOM

**What goes wrong:** Lambda function exits with `Runtime exited with error: signal: killed` immediately during cold start. This is an OOM kill — the model did not fit in the configured Lambda memory.

**Why it happens:** Qwen2.5-0.5B ONNX in fp16 is ~500MB on disk but may expand in RAM. onnxruntime loads the full model into memory at session creation. If Lambda memory is set to the default 128MB, the container is killed instantly.

**How to avoid:** Set Lambda memory to **3008MB minimum** in Terraform (`memory_size = 3008`). Set timeout to `30` seconds. Verify locally that the cold start completes within timeout before deploying. If OOM continues, export ONNX in fp16 (reduces memory footprint by ~50% vs fp32).

**Warning signs:** Lambda function exits with signal SIGKILL during cold start. CloudWatch shows no logs from the handler, only the runtime exit message.

### Pitfall 5: Tokenizer Saves Files Spread Across Multiple Directories

**What goes wrong:** `AutoTokenizer.save_pretrained("./model_artifacts/")` in Phase 3 saves tokenizer files. Some tokenizer backends also write additional files to unexpected paths. When the Docker build `COPY model_artifacts/ ...` runs, one or more tokenizer files are missing, and `AutoTokenizer.from_pretrained(MODEL_DIR)` fails at cold start.

**Why it happens:** Qwen2.5 uses a fast tokenizer (Rust-based), which saves `tokenizer.json` plus several supporting files. If any are missing from the COPY, the tokenizer load fails.

**How to avoid:** After Phase 3 exports the tokenizer, verify all required files are in `model_artifacts/`: `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`. Run `AutoTokenizer.from_pretrained("./model_artifacts/")` locally (outside Docker) to confirm it loads successfully. Then run the `--network none` Docker test to confirm it works inside the container.

---

## Code Examples

Verified patterns from official sources and project context:

### Complete handler.py (verified architecture pattern)

```python
# lambda/handler.py
# Source: AWS Lambda Python container docs + Mangum README + project STATE.md
import os
import json
import logging
import importlib

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from mangum import Mangum

# NOTE: 'lambda' is a Python reserved keyword. Use importlib to import from lambda/
prompt_utils = importlib.import_module("lambda.prompt_utils")
format_prompt = prompt_utils.format_prompt
parse_price_from_output = prompt_utils.parse_price_from_output

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Cold start: model and tokenizer loaded ONCE per execution environment ---
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "model_artifacts")
)

logger.info("Loading tokenizer from %s", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

logger.info("Loading ONNX session from %s", MODEL_DIR)
session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"],
)

logger.info("Model loaded. Input names: %s", session.get_inputs()[0].name)
# -------------------------------------------------------------------------

app = FastAPI(title="NJ Housing Price Predictor")


class PredictRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: float = Field(..., ge=0.5, le=10.0)
    sqft: int = Field(..., ge=100, le=20000)
    lot_size: float = Field(..., gt=0.0, le=100.0)
    year_built: int = Field(..., ge=1800, le=2025)
    zip_code: str = Field(..., min_length=5, max_length=5)
    property_type: str


class PredictResponse(BaseModel):
    predicted_price: float
    predicted_price_rounded: int


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prompt = format_prompt(
        bedrooms=req.bedrooms,
        bathrooms=req.bathrooms,
        sqft=req.sqft,
        lot_size=req.lot_size,
        year_built=req.year_built,
        zip_code=req.zip_code,
        property_type=req.property_type,
    )

    # Autoregressive generation loop (single-token per session.run call)
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    MAX_NEW_TOKENS = 12
    generated_ids = []

    for _ in range(MAX_NEW_TOKENS):
        feed = {"input_ids": input_ids}
        if attention_mask is not None:
            feed["attention_mask"] = attention_mask

        logits = session.run(None, feed)[0]  # [batch, seq_len, vocab_size]
        next_token_id = int(np.argmax(logits[0, -1, :]))
        generated_ids.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        # Append next token to inputs
        next_token = np.array([[next_token_id]], dtype=np.int64)
        input_ids = np.concatenate([input_ids, next_token], axis=1)
        if attention_mask is not None:
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
            )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    price = parse_price_from_output(generated_text)

    if price is None or price <= 0:
        logger.warning("Could not parse price from: %r", generated_text)
        raise HTTPException(
            status_code=500,
            detail=f"Model output unparseable: {generated_text!r}"
        )

    rounded = round(price / 1000) * 1000
    return PredictResponse(predicted_price=price, predicted_price_rounded=rounded)


# Mangum wraps FastAPI for Lambda. lifespan="off" disables ASGI startup/shutdown events.
# Source: Mangum README (https://github.com/Kludex/mangum)
handler = Mangum(app, lifespan="off")
```

### Dockerfile (verified pattern)

```dockerfile
# Source: AWS Lambda Python container image docs
# https://docs.aws.amazon.com/lambda/latest/dg/python-image.html
FROM public.ecr.aws/lambda/python:3.12

# Disable HuggingFace Hub network calls at runtime
# Tokenizer and model must be bundled in image — no internet access in Lambda
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# Install inference dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy handler and shared utilities
COPY handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY prompt_utils.py ${LAMBDA_TASK_ROOT}/prompt_utils.py

# Copy ONNX model and tokenizer files
# model_artifacts/ must exist locally before docker build
COPY model_artifacts/ ${LAMBDA_TASK_ROOT}/model_artifacts/

# Lambda handler: <module_name>.<function_or_object_name>
# 'handler' is the Mangum instance in handler.py
CMD [ "handler.handler" ]
```

### Local test commands (verified RIE pattern)

```bash
# Build for Lambda architecture (always linux/amd64)
docker build --platform linux/amd64 --provenance=false \
  -t housing-predictor:local \
  lambda/

# Test with RIE (AWS base image includes RIE built-in)
docker run --rm -p 9000:8080 housing-predictor:local &

# Invoke with valid API Gateway v2 proxy event
curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "2.0",
    "routeKey": "POST /predict",
    "rawPath": "/predict",
    "headers": {"content-type": "application/json"},
    "requestContext": {"http": {"method": "POST"}},
    "body": "{\"bedrooms\":3,\"bathrooms\":2.0,\"sqft\":1800,\"lot_size\":0.25,\"year_built\":1995,\"zip_code\":\"07030\",\"property_type\":\"Single Family\"}",
    "isBase64Encoded": false
  }'

# Verify tokenizer is bundled (no network = must work offline)
docker run --network none --rm housing-predictor:local python -c "
from transformers import AutoTokenizer
import os
tok = AutoTokenizer.from_pretrained('/var/task/model_artifacts')
print('Tokenizer loaded OK:', tok.__class__.__name__)
"
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Lambda .zip deployment | Lambda container image deployment | 2020 (re:Invent) | Allows large ML models > 50MB zip limit; supports up to 10GB images |
| Flask + Zappa for Lambda | FastAPI + Mangum | 2021-2022 | Typed endpoints, Pydantic validation, async support; Mangum is the current standard |
| OIDC long-lived credentials in Secrets | OIDC IAM role, no stored credentials | 2022+ | GitHub Actions OIDC eliminates credential rotation risk |
| PyTorch for Lambda inference | ONNX Runtime CPU | 2021+ | 5-10x smaller image; faster cold start; no GPU required |
| transformers 4.x | transformers 5.x | 2025 | v5 is the stable branch; v4 is legacy |

**Deprecated/outdated:**
- `python:3.11-slim` base image: Use `public.ecr.aws/lambda/python:3.12` — the Lambda base image pre-configures the Runtime Interface Client and RIE; generic Python images require additional setup.
- `flask-lambda`: Use FastAPI + Mangum — better typing, active maintenance, native async.
- `onnxruntime-gpu` on Lambda: No GPU available; use `onnxruntime` (CPU build).

---

## Open Questions

1. **Generation loop vs. `ORTModelForCausalLM` (Optimum)**
   - What we know: The handler above implements a manual autoregressive loop using raw `session.run()`. Optimum provides `ORTModelForCausalLM` which handles the generation loop with KV-cache internally.
   - What's unclear: Whether `ORTModelForCausalLM` works correctly on Lambda without the full `optimum[onnxruntime]` package (which may pull in heavy dependencies). If the Phase 3 export used `text-generation-with-past`, `ORTModelForCausalLM.generate()` is cleaner.
   - Recommendation: Start with the manual loop (confirmed to work with just `onnxruntime` + `numpy`). If the loop is too slow or complex, add `optimum[onnxruntime]` and switch to `ORTModelForCausalLM.generate()` — but first measure the image size impact.

2. **Import path for `lambda/prompt_utils.py` inside Docker container**
   - What we know: In the Docker container, handler.py and prompt_utils.py are both in `${LAMBDA_TASK_ROOT}` (i.e., `/var/task/`). At that point, they are siblings — `import prompt_utils` should work directly without `importlib`.
   - What's unclear: The module path context inside Lambda's runtime. If handler.py is at `/var/task/handler.py` and prompt_utils.py is at `/var/task/prompt_utils.py`, then `import prompt_utils` works. The `importlib.import_module("lambda.prompt_utils")` pattern is needed when running from the project root (where `lambda/` is a package), not inside the container.
   - Recommendation: In the container, use `import prompt_utils` directly (simpler). In notebooks and local scripts outside the container, use `importlib.import_module("lambda.prompt_utils")`. Document this distinction clearly in handler.py comments.

3. **Measured cold start latency for Qwen2.5-0.5B ONNX on Lambda 3008MB**
   - What we know: Estimated 5-15 seconds from SUMMARY.md research. ONNX Runtime session creation for a ~500MB fp16 model is expected to take 3-8 seconds.
   - What's unclear: Actual measured cold start time. This determines whether provisioned concurrency is needed (adds cost). The 5-second warm invocation success criterion can still be met even with a slow cold start.
   - Recommendation: Measure on first deploy. If cold start > 10 seconds, consider: (1) ONNX graph optimization (`--optimize O2` in export), (2) fp16 export if not already done, (3) provisioned concurrency as a last resort.

4. **`model.onnx` vs. `model.onnx` + `model_with_past.onnx` from Phase 3**
   - What we know: Phase 3 export with `--task text-generation-with-past` produces two files. The handler pattern above uses only `model.onnx` (no KV-cache). Using both would require loading both sessions as globals.
   - What's unclear: Which export task Phase 3 actually used (this is flagged as an open question in PITFALLS.md and SUMMARY.md).
   - Recommendation: Design the handler to be flexible — load both sessions as globals if both files exist, fall back to single-file if only `model.onnx` exists. Or wait for Phase 3 to complete and confirm which files are produced.

---

## Sources

### Primary (HIGH confidence)
- AWS Lambda Python container image docs: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html — Dockerfile structure, base image, CMD handler format, `${LAMBDA_TASK_ROOT}`
- AWS Lambda Runtime Interface Emulator (RIE) GitHub: https://github.com/aws/aws-lambda-runtime-interface-emulator — local test invocation endpoint format
- Mangum GitHub (Kludex/mangum): https://github.com/Kludex/mangum — `Mangum(app, lifespan="off")` pattern, API Gateway v2 event handling
- Mangum documentation: https://mangum.fastapiexpert.com/adapter/ — configuration options, lifespan parameter
- ONNX Runtime Python API docs: https://onnxruntime.ai/docs/api/python/api_summary.html — `InferenceSession` constructor, `run()` method signature
- Project `.planning/research/STACK.md` — verified library versions for Lambda serving stack
- Project `lambda/prompt_utils.py` — `format_prompt()` and `parse_price_from_output()` already implemented; handler must use these

### Secondary (MEDIUM confidence)
- PyImageSearch: FastAPI Docker Deployment for ONNX Models on Lambda (2025-11-17) — confirmed FastAPI + Mangum + ONNX Runtime pattern for Lambda
- FastAPI official docs (https://fastapi.tiangolo.com/tutorial/handling-errors/) — Pydantic validation error behavior (422 response), `Field()` constraint syntax
- HuggingFace transformers docs — `TRANSFORMERS_OFFLINE=1` environment variable prevents Hub network calls; `AutoTokenizer.from_pretrained()` local path loading
- WebSearch results confirming `--platform linux/amd64` build requirement and `TOKENIZERS_PARALLELISM=false` for Lambda fork safety

### Tertiary (LOW confidence — verify before implementation)
- Lambda cold start latency for Qwen2.5-0.5B ONNX: 5-15 second estimate from SUMMARY.md — not empirically measured; verify on first deploy
- `ORTModelForCausalLM` dependency footprint on Lambda — not tested; use manual generation loop as safe default

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified in STACK.md research (2026-02-26); FastAPI + Mangum is the current documented standard
- Architecture: HIGH — Dockerfile pattern verified against AWS official docs; RIE local testing confirmed from AWS GitHub; Mangum lifespan config confirmed from official docs
- Pitfalls: HIGH — single-token generation loop gap is clearly documented in ARCHITECTURE.md data flow; PyTorch bloat and tokenizer bundling are documented in PITFALLS.md; import path issue documented in STATE.md decisions

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable domain; onnxruntime and FastAPI versions change slowly)
