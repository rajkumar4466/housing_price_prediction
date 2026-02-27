FROM public.ecr.aws/lambda/python:3.12

# Prevent HuggingFace Hub network calls — tokenizer and model must be bundled
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# Install inference-only dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy handler and shared prompt utilities
COPY handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY prompt_utils.py ${LAMBDA_TASK_ROOT}/prompt_utils.py

# Copy ONNX model and tokenizer files (must exist locally before docker build)
# Populated by Phase 3 ONNX export:
#   model.onnx + tokenizer_config.json + tokenizer.json + vocab.json +
#   merges.txt + special_tokens_map.json
#
# NOTE: lambda/model_artifacts/ is gitignored (Qwen2.5-0.5B ONNX is 500MB+,
# exceeding GitHub's 100MB file size limit). Populate this directory by
# running the Phase 3 ONNX export notebook before building this image.
#
# Build command (from project root):
#   docker build --platform linux/amd64 --provenance=false -t housing-predictor:latest lambda/
COPY model_artifacts/ ${LAMBDA_TASK_ROOT}/model_artifacts/

# Mangum handler: handler.py exposes 'handler' as the Mangum(app) instance
CMD [ "handler.handler" ]
