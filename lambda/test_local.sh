#!/usr/bin/env bash
# test_local.sh — Build and test the housing-predictor Lambda container image locally.
#
# Usage:
#   cd /path/to/housing_price_predictor
#   bash lambda/test_local.sh
#
# Prerequisites:
#   - Docker is running
#   - lambda/model_artifacts/ contains:
#       model.onnx, tokenizer_config.json, tokenizer.json,
#       vocab.json, merges.txt, special_tokens_map.json
#   (Populate by running the Phase 3 ONNX export notebook on Colab first.)

set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

pass() { echo -e "${GREEN}PASS${NC}: $1"; }
fail() { echo -e "${RED}FAIL${NC}: $1"; }
warn() { echo -e "${YELLOW}WARN${NC}: $1"; }
info() { echo -e "INFO: $1"; }

# Track results for summary
BUILD_STATUS="FAIL"
TORCH_STATUS="FAIL"
IMAGE_SIZE="unknown"
RIE_STATUS="FAIL"
OFFLINE_STATUS="FAIL"
RIE_RESPONSE=""

# ---------------------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------------------
info "=== Pre-flight checks ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model_artifacts"

if [ ! -f "${MODEL_DIR}/model.onnx" ]; then
  echo ""
  fail "model_artifacts/model.onnx not found."
  echo "  model_artifacts/ not found or incomplete. Run Phase 3 ONNX export first,"
  echo "  then copy model.onnx and tokenizer files to lambda/model_artifacts/"
  echo ""
  echo "  Expected files:"
  echo "    lambda/model_artifacts/model.onnx"
  echo "    lambda/model_artifacts/tokenizer_config.json"
  echo "    lambda/model_artifacts/tokenizer.json"
  echo "    lambda/model_artifacts/vocab.json"
  echo "    lambda/model_artifacts/merges.txt"
  echo "    lambda/model_artifacts/special_tokens_map.json"
  exit 1
fi

if [ ! -f "${MODEL_DIR}/tokenizer_config.json" ]; then
  fail "lambda/model_artifacts/tokenizer_config.json not found. Tokenizer files are missing."
  echo "  Copy all tokenizer files from the Phase 3 ONNX export to lambda/model_artifacts/"
  exit 1
fi

info "model_artifacts/model.onnx found"
info "model_artifacts/tokenizer_config.json found"

if ! docker info > /dev/null 2>&1; then
  fail "Docker is not running. Start Docker Desktop (or the Docker daemon) and retry."
  exit 1
fi
info "Docker is running"

# ---------------------------------------------------------------------------
# 2. Build the image
# ---------------------------------------------------------------------------
info ""
info "=== Building Docker image ==="
info "Platform: linux/amd64 (required for AWS Lambda x86_64)"

if docker build --platform linux/amd64 --provenance=false \
    -t housing-predictor:local \
    "${SCRIPT_DIR}/"; then
  BUILD_STATUS="PASS"
  pass "Docker image built successfully"
else
  fail "Docker build failed — check output above"
  echo ""
  echo "Summary:"
  echo "  Build:            FAIL"
  exit 1
fi

# ---------------------------------------------------------------------------
# 3. Verify no PyTorch in image
# ---------------------------------------------------------------------------
info ""
info "=== Verifying no PyTorch in image ==="

if docker run --rm housing-predictor:local pip show torch 2>&1 | grep -qi "not found"; then
  TORCH_STATUS="PASS"
  pass "No PyTorch in container"
else
  TORCH_STATUS="FAIL"
  fail "PyTorch found in container — Dockerfile must not install torch"
fi

# ---------------------------------------------------------------------------
# 4. Check image size
# ---------------------------------------------------------------------------
info ""
info "=== Checking image size ==="

IMAGE_SIZE=$(docker image ls housing-predictor:local --format "{{.Size}}")
info "Image size: ${IMAGE_SIZE}"

# Warn if > 3GB (rough heuristic; docker ls Size is human-formatted)
if echo "${IMAGE_SIZE}" | grep -qE '^[3-9][0-9]*\.[0-9]+GB$|^[0-9]{2,}(\.[0-9]+)?GB$'; then
  warn "Image is ${IMAGE_SIZE} — larger than 3 GB. Lambda has a 10 GB container limit but cold starts may be slow."
else
  pass "Image size is ${IMAGE_SIZE}"
fi

# ---------------------------------------------------------------------------
# 5. Test with RIE (network enabled)
# ---------------------------------------------------------------------------
info ""
info "=== RIE test (network enabled) ==="

# Clean up any leftover container from a previous run
docker rm -f hp-test 2>/dev/null || true

info "Starting container with Lambda RIE on port 9000..."
docker run --rm -d -p 9000:8080 --name hp-test housing-predictor:local

info "Waiting 5 seconds for cold start..."
sleep 5

RIE_RESPONSE=$(curl -s --max-time 30 -XPOST \
  "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "2.0",
    "routeKey": "POST /predict",
    "rawPath": "/predict",
    "headers": {"content-type": "application/json"},
    "requestContext": {"http": {"method": "POST", "path": "/predict"}},
    "body": "{\"bedrooms\":3,\"bathrooms\":2.0,\"sqft\":1800,\"lot_size\":0.25,\"year_built\":1995,\"zip_code\":\"07030\",\"property_type\":\"Single Family\"}",
    "isBase64Encoded": false
  }' 2>&1 || true)

info "RIE response:"
echo "${RIE_RESPONSE}"

if echo "${RIE_RESPONSE}" | grep -q "predicted_price"; then
  RIE_STATUS="PASS"
  pass "RIE prediction test returned predicted_price"
else
  RIE_STATUS="FAIL"
  fail "RIE response does not contain 'predicted_price'"
fi

info "Stopping RIE container..."
docker stop hp-test 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6. Test with --network none (offline validation)
# ---------------------------------------------------------------------------
info ""
info "=== Offline test (--network none) ==="
info "Confirms tokenizer is bundled — not fetched from HuggingFace Hub at runtime"

# Clean up any leftover container
docker rm -f hp-test-offline 2>/dev/null || true

info "Starting container with --network none on port 9001..."
docker run --rm -d -p 9001:8080 --network none --name hp-test-offline housing-predictor:local

info "Waiting 5 seconds for cold start..."
sleep 5

# NOTE: --network none prevents host-to-container curl. Use docker exec instead.
OFFLINE_OUT=$(docker exec hp-test-offline python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/var/task/model_artifacts')
print('Tokenizer loaded offline OK:', tok.__class__.__name__)
" 2>&1 || true)

echo "${OFFLINE_OUT}"

if echo "${OFFLINE_OUT}" | grep -q "Tokenizer loaded offline OK"; then
  OFFLINE_STATUS="PASS"
  pass "Offline tokenizer test passed — tokenizer is bundled in the container"
else
  OFFLINE_STATUS="FAIL"
  fail "Offline tokenizer test failed — check if tokenizer files are in model_artifacts/"
fi

info "Stopping offline container..."
docker stop hp-test-offline 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " TEST SUMMARY"
echo "============================================================"
echo "  Build:                ${BUILD_STATUS}"
echo "  No PyTorch:           ${TORCH_STATUS}"
echo "  Image size:           ${IMAGE_SIZE}"
echo "  RIE prediction test:  ${RIE_STATUS}"
if [ -n "${RIE_RESPONSE}" ]; then
  SNIPPET=$(echo "${RIE_RESPONSE}" | head -c 200)
  echo "    Response snippet:   ${SNIPPET}"
fi
echo "  Offline tokenizer:    ${OFFLINE_STATUS}"
echo "============================================================"

# Exit with failure if any critical test failed
if [ "${BUILD_STATUS}" != "PASS" ] || [ "${RIE_STATUS}" != "PASS" ] || [ "${OFFLINE_STATUS}" != "PASS" ]; then
  echo ""
  fail "One or more critical tests FAILED — see above for details"
  exit 1
fi

pass "All critical tests PASSED"
