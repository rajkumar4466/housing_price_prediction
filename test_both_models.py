"""Quick local test: predict the same property with both XGBoost and QLoRA ONNX."""

import re

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import onnxruntime as ort
from transformers import AutoTokenizer

# ============================================================
# Test property
# ============================================================
test = {
    "bedrooms": 4,
    "bathrooms": 2.5,
    "sqft": 2200,
    "lot_size": 0.30,
    "year_built": 1985,
    "zip_code": "07650",
    "property_type": "Single Family",
}

print("=" * 60)
print("TEST PROPERTY")
print("=" * 60)
for k, v in test.items():
    print(f"  {k}: {v}")
print()

# ============================================================
# 1. XGBoost prediction
# ============================================================
model = XGBRegressor()
model.load_model("models/xgboost_baseline.json")

row = {
    "bathrooms": test["bathrooms"],
    "bedrooms": test["bedrooms"],
    "lot_size": test["lot_size"],
    "pt_Condo": 0,
    "pt_Multi-Family": 0,
    "pt_Single Family": 1 if test["property_type"] == "Single Family" else 0,
    "pt_Townhouse": 0,
    "sqft": test["sqft"],
    "year_built": test["year_built"],
    "zip_code": int(test["zip_code"]),
}
X = pd.DataFrame([row])
xgb_price = model.predict(X)[0]

print(f"XGBoost predicted price:    ${xgb_price:,.0f}")

# ============================================================
# 2. QLoRA ONNX prediction
# ============================================================
ONNX_DIR = "lambda/model_artifacts"
tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
session = ort.InferenceSession(f"{ONNX_DIR}/model.onnx")

prompt = (
    f"Property: {test['property_type']} in zip {test['zip_code']}. "
    f"{test['bedrooms']} bedrooms, {test['bathrooms']} bathrooms, "
    f"{test['sqft']} sqft living area, {test['lot_size']:.2f} acre lot, "
    f"built in {test['year_built']}. Predicted price: $"
)

inputs = tokenizer(prompt, return_tensors="np")
input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)

# Model was exported with KV cache: 24 layers, 2 heads, 64 dims per head
NUM_LAYERS = 24
NUM_HEADS = 2
HEAD_DIM = 64

# Build initial empty KV cache (past_sequence_length = 0)
def make_empty_kv():
    return np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)

all_token_ids = input_ids.copy()

# Autoregressive generation (greedy, 20 tokens max)
for step in range(20):
    seq_len = input_ids.shape[1]
    if step == 0:
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    else:
        position_ids = np.array([[all_token_ids.shape[1] - 1]], dtype=np.int64)

    feed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # Add KV cache entries
    if step == 0:
        for i in range(NUM_LAYERS):
            feed[f"past_key_values.{i}.key"] = make_empty_kv()
            feed[f"past_key_values.{i}.value"] = make_empty_kv()
    else:
        for i in range(NUM_LAYERS):
            feed[f"past_key_values.{i}.key"] = kv_cache[i][0]
            feed[f"past_key_values.{i}.value"] = kv_cache[i][1]

    outputs = session.run(None, feed)
    logits = outputs[0]

    # Extract updated KV cache from outputs (after logits)
    kv_outputs = outputs[1:]
    kv_cache = []
    for i in range(NUM_LAYERS):
        kv_cache.append((kv_outputs[i * 2], kv_outputs[i * 2 + 1]))

    next_token = np.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
    if next_token[0, 0] == tokenizer.eos_token_id:
        break
    all_token_ids = np.concatenate([all_token_ids, next_token], axis=1)
    # Next step: only feed the new token
    input_ids = next_token
    attention_mask = np.ones((1, all_token_ids.shape[1]), dtype=np.int64)

generated = tokenizer.decode(all_token_ids[0], skip_special_tokens=True)
suffix = generated[len(prompt):]
cleaned = suffix.replace(",", "")
match = re.search(r"\d+(?:\.\d+)?", cleaned)
qlora_price = float(match.group()) if match else None

if qlora_price:
    print(f"QLoRA ONNX predicted price: ${qlora_price:,.0f}")
else:
    print(f"QLoRA ONNX: could not parse price from: {suffix!r}")

print()
print("=" * 60)
