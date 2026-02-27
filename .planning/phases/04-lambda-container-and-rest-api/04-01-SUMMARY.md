---
phase: 04-lambda-container-and-rest-api
plan: 01
subsystem: api
tags: [onnxruntime, fastapi, mangum, lambda, docker, transformers, qwen2.5]

# Dependency graph
requires:
  - phase: 01-data-foundation
    provides: prompt_utils.py with format_prompt and parse_price_from_output contracts
  - phase: 03-evaluation-and-onnx-export
    provides: model_artifacts/ directory with model.onnx and tokenizer files
provides:
  - lambda/handler.py: FastAPI + Mangum Lambda handler with autoregressive ONNX inference
  - lambda/requirements.txt: Minimal inference-only dependency set
  - lambda/Dockerfile: Lambda container image definition with TRANSFORMERS_OFFLINE=1
affects: [04-lambda-container-and-rest-api/04-02, 05-infrastructure-and-cicd]

# Tech tracking
tech-stack:
  added: [onnxruntime==1.24.2, fastapi==0.133.1, mangum==0.21.0, numpy==2.4.2, transformers==5.2.0, sentencepiece==0.2.1, tokenizers==0.22.2]
  patterns: [module-level cold-start globals, autoregressive generation loop with MAX_NEW_TOKENS, Mangum ASGI-to-Lambda adapter, TRANSFORMERS_OFFLINE for air-gapped container]

key-files:
  created:
    - lambda/handler.py
    - lambda/requirements.txt
    - lambda/Dockerfile
  modified:
    - .gitignore

key-decisions:
  - "Inside the Lambda container, prompt_utils.py is a sibling at /var/task/ — use 'import prompt_utils' directly, NOT importlib.import_module('lambda.prompt_utils') which is only needed at project-root level in notebooks"
  - "MAX_NEW_TOKENS=12 for autoregressive loop — prices are at most 7 digits, buffer added for safety"
  - "tokenizer and session initialized at module level (not inside endpoint) — cold start once per Lambda execution environment"
  - "lambda/model_artifacts/ added to .gitignore — Qwen2.5-0.5B ONNX is 500MB+, exceeds GitHub 100MB limit"
  - "TOKENIZERS_PARALLELISM=false in Dockerfile to suppress Rust tokenizer fork warning in Lambda"

patterns-established:
  - "Module-level globals pattern: tokenizer and ort.InferenceSession loaded once at cold start, reused across warm invocations"
  - "Autoregressive loop: argmax over logits[0, -1, :], append token, concatenate to input_ids, break on EOS or MAX_NEW_TOKENS"
  - "Mangum adapter: handler = Mangum(app, lifespan='off') is the Lambda entry point"

requirements-completed: [SERV-01, SERV-02, SERV-03]

# Metrics
duration: 8min
completed: 2026-02-27
---

# Phase 4 Plan 01: Lambda Container and REST API Summary

**FastAPI + Mangum Lambda handler with autoregressive ONNX inference, Lambda container Dockerfile using public.ecr.aws/lambda/python:3.12 with TRANSFORMERS_OFFLINE=1, and inference-only requirements.txt (7 packages, no PyTorch)**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-27T16:59:52Z
- **Completed:** 2026-02-27T17:08:01Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created lambda/handler.py with FastAPI POST /predict endpoint, module-level tokenizer and ONNX session (cold-start once), and autoregressive generation loop (MAX_NEW_TOKENS=12)
- Created lambda/Dockerfile using public.ecr.aws/lambda/python:3.12 with TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1, and TOKENIZERS_PARALLELISM=false
- Created lambda/requirements.txt with 7 inference-only packages (no PyTorch, no bitsandbytes, no training dependencies)
- Updated .gitignore to exclude lambda/model_artifacts/ and models/ (500MB+ ONNX artifacts)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Lambda handler with FastAPI, Mangum, and ONNX inference** - `1cbbe91` (feat)
2. **Task 2: Create Dockerfile and update .gitignore for model artifacts** - `effd3c1` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `lambda/handler.py` - FastAPI + Mangum Lambda handler; module-level tokenizer/session globals; POST /predict with autoregressive generation; imports format_prompt/parse_price_from_output from sibling prompt_utils
- `lambda/requirements.txt` - Inference-only deps: onnxruntime, fastapi, mangum, numpy, transformers, sentencepiece, tokenizers
- `lambda/Dockerfile` - Lambda container image definition; public.ecr.aws/lambda/python:3.12 base; TRANSFORMERS_OFFLINE=1; CMD handler.handler
- `.gitignore` - Added lambda/model_artifacts/ and models/ exclusions

## Decisions Made

- Inside the Lambda container `/var/task/`, `prompt_utils.py` is a sibling file — `import prompt_utils` works directly. The `importlib.import_module('lambda.prompt_utils')` pattern is only needed in notebooks/scripts at the project root where `lambda` is a Python reserved keyword.
- `MAX_NEW_TOKENS=12` chosen because housing prices are at most 7 digits (e.g., "1250000"), with buffer for safety.
- `tokenizer` and `session` must be module-level globals to benefit from Lambda warm starts — if placed inside the endpoint function they would reload on every invocation.
- `lambda/model_artifacts/` gitignored because the Qwen2.5-0.5B ONNX export is 500MB+, far exceeding GitHub's 100MB per-file limit.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Model artifacts will be populated in Phase 3 ONNX export.

## Next Phase Readiness

- lambda/handler.py, lambda/Dockerfile, and lambda/requirements.txt are ready for docker build once Phase 3 ONNX export populates lambda/model_artifacts/
- Plan 04-02 can proceed to build, tag, push to ECR, and configure Lambda function
- Docker build command: `docker build --platform linux/amd64 --provenance=false -t housing-predictor:latest lambda/`

---
*Phase: 04-lambda-container-and-rest-api*
*Completed: 2026-02-27*
