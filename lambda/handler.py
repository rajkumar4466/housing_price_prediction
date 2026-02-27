"""
Lambda inference handler for NJ housing price prediction.

This module is the entry point for the AWS Lambda function. It exposes a FastAPI
app via Mangum so the same code works both as a Lambda handler and as a local
ASGI server.

Import note:
    Inside the Lambda container, all files in lambda/ are copied to /var/task/.
    So prompt_utils.py is a sibling module: `import prompt_utils` works directly.
    Do NOT use importlib.import_module("lambda.prompt_utils") here — that pattern
    is needed in notebooks and scripts at the project root where `lambda` is a
    Python reserved keyword and the directory must be accessed via importlib. Inside
    the container, there is no `lambda` package; the file is simply `prompt_utils.py`.
"""

import logging
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# Sibling import inside the container — prompt_utils.py is at /var/task/prompt_utils.py
import prompt_utils

format_prompt = prompt_utils.format_prompt
parse_price_from_output = prompt_utils.parse_price_from_output

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Module-level globals — initialized ONCE per Lambda execution environment
# (cold start). Do NOT move these inside any function.
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

logger.info("Loading tokenizer from %s", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
logger.info("Tokenizer loaded successfully")

logger.info("Loading ONNX session from %s", os.path.join(MODEL_DIR, "model.onnx"))
session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"],
)
logger.info("ONNX session loaded successfully")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NJ Housing Price Predictor",
    description="Predicts NJ housing prices from 7 property features using a QLoRA fine-tuned Qwen2.5-0.5B exported to ONNX.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 12  # Prices are at most ~7 digits (e.g. "1250000")


class PredictRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.5, le=10.0, description="Number of bathrooms (0.5 increments)")
    sqft: int = Field(..., ge=100, le=20000, description="Living area in square feet")
    lot_size: float = Field(..., gt=0.0, le=100.0, description="Lot size in acres")
    year_built: int = Field(..., ge=1800, le=2026, description="Year the property was built")
    zip_code: str = Field(..., min_length=5, max_length=5, description="5-digit NJ zip code (leading zero preserved)")
    property_type: str = Field(..., description="Property type: Single Family, Condo, Townhouse, Multi-Family")


class PredictResponse(BaseModel):
    predicted_price: float = Field(..., description="Raw predicted price from model")
    predicted_price_rounded: int = Field(..., description="Price rounded to nearest $1,000")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict NJ housing price from 7 property features.

    Runs autoregressive token generation through the ONNX-exported Qwen2.5-0.5B
    model, then parses the price from the generated text.
    """
    # Build the prompt using the shared utility (single source of truth)
    prompt = format_prompt(
        bedrooms=request.bedrooms,
        bathrooms=request.bathrooms,
        sqft=request.sqft,
        lot_size=request.lot_size,
        year_built=request.year_built,
        zip_code=request.zip_code,
        property_type=request.property_type,
    )
    logger.info("Prompt: %s", prompt)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids: np.ndarray = inputs["input_ids"]
    attention_mask: np.ndarray = inputs.get("attention_mask")

    # Discover KV-cache structure from ONNX model inputs
    num_kv_layers = sum(1 for inp in session.get_inputs() if inp.name.endswith(".key"))
    kv_head_dim = 64  # Qwen2.5-0.5B: 2 heads x 64 dim
    kv_num_heads = 2

    # Autoregressive generation loop with KV-cache.
    # First pass: full prompt. Subsequent passes: single token with cached KV.
    generated_ids: list[int] = []
    past_kv = None  # Will be populated after first pass

    for step in range(MAX_NEW_TOKENS):
        seq_len = input_ids.shape[1]

        feed: dict[str, np.ndarray] = {"input_ids": input_ids}
        if attention_mask is not None:
            feed["attention_mask"] = attention_mask

        # Position IDs
        if past_kv is not None:
            # Subsequent steps: position = total sequence length - 1
            past_seq_len = past_kv[0].shape[2]
            feed["position_ids"] = np.array([[past_seq_len]], dtype=np.int64)
        else:
            # First step: position IDs = 0, 1, 2, ..., seq_len-1
            feed["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # KV-cache inputs
        for i in range(num_kv_layers):
            if past_kv is not None:
                feed[f"past_key_values.{i}.key"] = past_kv[i * 2]
                feed[f"past_key_values.{i}.value"] = past_kv[i * 2 + 1]
            else:
                # First step: empty KV-cache
                feed[f"past_key_values.{i}.key"] = np.zeros(
                    (1, kv_num_heads, 0, kv_head_dim), dtype=np.float32
                )
                feed[f"past_key_values.{i}.value"] = np.zeros(
                    (1, kv_num_heads, 0, kv_head_dim), dtype=np.float32
                )

        outputs = session.run(None, feed)
        # outputs[0] = logits, outputs[1:] = updated KV-cache
        logits = outputs[0]
        next_token_id: int = int(np.argmax(logits[0, -1, :]))

        # Store updated KV-cache for next step
        past_kv = outputs[1:]

        # Stop on EOS token
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)

        # Next step: only the new token as input
        input_ids = np.array([[next_token_id]], dtype=np.int64)
        if attention_mask is not None:
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=attention_mask.dtype)], axis=1
            )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logger.info("Generated text: %r", generated_text)

    price = parse_price_from_output(generated_text)

    if price is None or price <= 0:
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to generate a valid price. Raw output: {generated_text!r}",
        )

    rounded = round(price / 1000) * 1000
    logger.info("Predicted price: %.2f, rounded: %d", price, rounded)

    return PredictResponse(predicted_price=price, predicted_price_rounded=rounded)


# ---------------------------------------------------------------------------
# Mangum handler — this is the AWS Lambda entry point
# ---------------------------------------------------------------------------

# Mangum wraps the FastAPI ASGI app so Lambda can invoke it via API Gateway.
# lifespan="off" disables ASGI lifespan events (not supported in Lambda).
handler = Mangum(app, lifespan="off")
