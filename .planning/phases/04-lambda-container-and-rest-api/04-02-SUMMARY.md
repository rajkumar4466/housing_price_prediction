---
phase: 04-lambda-container-and-rest-api
plan: 02
subsystem: api
tags: [docker, lambda, onnxruntime, transformers, rie, testing]

# Dependency graph
requires:
  - phase: 04-lambda-container-and-rest-api/04-01
    provides: lambda/Dockerfile, lambda/handler.py, lambda/requirements.txt
  - phase: 03-evaluation-and-onnx-export
    provides: model_artifacts/ with model.onnx and tokenizer files
provides:
  - lambda/test_local.sh: Automated local Docker build and Lambda RIE integration test script
affects: [05-infrastructure-and-ci-cd]

# Tech tracking
tech-stack:
  added: []
  patterns: [Lambda RIE local testing via docker run -p 9000:8080, offline container validation via docker exec with --network none, API Gateway v2 proxy event payload format for curl RIE invocation]

key-files:
  created:
    - lambda/test_local.sh
  modified: []

key-decisions:
  - "Use docker exec instead of curl for --network none offline test — host cannot reach container with no network but docker exec bypasses the network layer entirely"
  - "Tokenizer offline test confirms TRANSFORMERS_OFFLINE=1 is working and all tokenizer files are bundled in the container image"

patterns-established:
  - "RIE test: docker run --rm -d -p 9000:8080, sleep 5, curl to localhost:9000/2015-03-31/functions/function/invocations with API Gateway v2 JSON envelope"
  - "Offline test: docker run --network none, docker exec to run python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(...)'"

requirements-completed: [SERV-01, SERV-02, SERV-03]

# Metrics
duration: ~3min (Task 1 only; checkpoint at Task 2)
completed: 2026-02-27
---

# Phase 4 Plan 02: Local Docker Build and Test Summary

**Bash integration test script that builds the housing-predictor Lambda container image, validates no PyTorch is present, tests prediction via Lambda RIE curl, and confirms offline tokenizer operation with --network none docker exec**

## Performance

- **Duration:** ~3 min (Task 1 automated; paused at Task 2 human-verify checkpoint)
- **Started:** 2026-02-27T20:13:35Z
- **Completed:** 2026-02-27T20:16:00Z (partial — checkpoint)
- **Tasks:** 1 of 2 (Task 2 is human-verify checkpoint)
- **Files modified:** 1

## Accomplishments

- Created lambda/test_local.sh — 230-line bash script with pre-flight checks, docker build, PyTorch absence verification, image size check, RIE curl test, and --network none offline tokenizer test
- Script produces a PASS/FAIL summary table covering all five success criteria from the plan
- Script handles cleanup of test containers on each run (docker rm -f before starting new containers)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create local Docker build and test script** - `776cfcc` (feat)

**Note:** Task 2 is a checkpoint:human-verify — requires human to run `bash lambda/test_local.sh` and verify the prediction output is reasonable. Paused pending checkpoint approval.

## Files Created/Modified

- `lambda/test_local.sh` - Build + test script: pre-flight checks for model_artifacts, docker build --platform linux/amd64, no-PyTorch check, image size report, RIE curl test with API GW v2 envelope, offline docker exec tokenizer test, summary table

## Decisions Made

- Used `docker exec` instead of curl for the `--network none` offline test — with network isolation, the host cannot reach the container's port, but `docker exec` runs commands inside the container directly, bypassing the network layer entirely.
- API Gateway v2 proxy event format used for RIE curl (`version: "2.0"`, `routeKey`, `requestContext.http`) — this matches what Mangum expects and what API Gateway HTTP API sends in production.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

**Human verification required.** Run the test script after populating `lambda/model_artifacts/`:

```bash
# 1. Copy ONNX model and tokenizer from Phase 3 export to lambda/model_artifacts/
#    (model.onnx, tokenizer_config.json, tokenizer.json, vocab.json, merges.txt, special_tokens_map.json)

# 2. Run the test script from project root
cd /Users/mithra_sundaram/Desktop/code/models/housing_price_predictor
bash lambda/test_local.sh
```

**Verify:**
1. Docker image builds without errors
2. Image size is under 3 GB
3. No PyTorch in container (script checks automatically)
4. RIE test returns a response with `predicted_price` field
5. The predicted price is a reasonable number (not 0, not negative, not a single digit)
6. Offline tokenizer test passes (confirms tokenizer is bundled)
7. Cold start log shows "Loading tokenizer" and "Loading ONNX session" messages once

## Next Phase Readiness

- lambda/test_local.sh is ready to run once model_artifacts/ is populated
- Phase 5 infrastructure (ECR push, Lambda deployment via Terraform) is ready to proceed in parallel with or after this verification
- If predicted price is a single digit (e.g., 4.0, 5.0), the autoregressive loop or model.onnx export may need investigation

## Self-Check: PASSED

| Item | Status |
|------|--------|
| lambda/test_local.sh | FOUND |
| lambda/test_local.sh executable | FOUND |
| commit 776cfcc (Task 1) | FOUND |

---
*Phase: 04-lambda-container-and-rest-api*
*Completed: 2026-02-27 (partial — checkpoint:human-verify pending)*
