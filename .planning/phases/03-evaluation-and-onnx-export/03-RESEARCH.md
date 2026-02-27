# Phase 3 Research: Evaluation and ONNX Export

**Phase:** 03-evaluation-and-onnx-export
**Researched:** 2026-02-27
**Confidence:** HIGH for evaluation metrics and merge pattern; MEDIUM-HIGH for ONNX task type
**Requirements addressed:** EVAL-01, EVAL-02, EVAL-03, ONNX-01, ONNX-02, ONNX-03

---

## Research Question

What do I need to know to plan Phase 3 well?

Phase 3 has two distinct sub-goals:
1. **Evaluation** (`03_evaluate.ipynb`): Load the LoRA adapter, run inference on the HF test split, compute 4 regression metrics, generate 2 plots.
2. **ONNX Export** (same or separate notebook): Merge LoRA weights via fp32 reload pattern, export to ONNX via optimum, validate numerically at `atol=1e-3`.

---

## Part 1: Evaluation

### What Phase 2 Produced (Inputs to Phase 3)

The training notebook (`02_train.ipynb`) produced:
- LoRA adapter weights at `/content/drive/MyDrive/housing_model/lora_adapter/` on Google Drive
- Required files: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files
- Training was done with `task_type=TaskType.CAUSAL_LM` — the model is a text generator, not a regression head
- The training text format is: `"Property: {type} in zip {zip}. ... Predicted price: $" + str(int(round(price)))`
- The model learns to generate the price digits immediately after `"Predicted price: $"`
- `parse_price_from_output()` in `lambda/prompt_utils.py` extracts the first numeric match from generated text

### Data Source for Evaluation

**Critical finding:** The training notebook (`02_train.ipynb` as implemented) loads data from the HuggingFace dataset `rajkumar4466/nj-housing-prices`. The evaluation notebook must also load from this dataset (test split), not from local JSONL files. The dataset has `prompt` and `price` columns. The `prompt` column already ends with `"Predicted price: $"`.

```python
from datasets import load_dataset
ds = load_dataset("rajkumar4466/nj-housing-prices")
test_data = ds["test"]  # has 'prompt' and 'price' columns
```

### Inference Pattern for Evaluation

The evaluation notebook must perform inference on each test record:
1. Load the base model with 4-bit quantization (same as training)
2. Load the LoRA adapter from Drive with `PeftModel.from_pretrained()`
3. For each test record: tokenize `prompt`, call `model.generate()` with `max_new_tokens=20, do_sample=False`
4. Decode output, strip the prompt prefix, call `parse_price_from_output()` to extract the float
5. Handle parse failures (None returns) — log them, skip from metric computation or substitute with a sentinel

**Parse failure handling strategy:** Records where `parse_price_from_output()` returns `None` should be counted separately. If parse failure rate exceeds 5%, log a warning. For metric computation, only use records with valid parsed prices. This is consistent with the PITFALLS.md guidance (< 1% parse failure rate is the target; > 5% is a warning sign).

**Inference speed consideration:** The test set has approximately 1,050 records (15% of 7,000). At ~1-3 seconds per inference on Colab T4, this is 17-52 minutes — potentially exceeding Colab session limits. Mitigation: batch inference is not straightforward for autoregressive generation, but `max_new_tokens=20` keeps each call short. Alternatively, evaluate on a 200-sample subset first to confirm metric quality, then run full evaluation.

**Better approach for eval speed:** Load the model in 4-bit for inference (same BitsAndBytesConfig as training). Do not use fp32 for evaluation inference — save fp32 reload for the ONNX export step. This keeps VRAM usage low and inference faster.

### Regression Metrics (EVAL-01)

All four metrics are required on the test set:

| Metric | Formula | Library | Notes |
|--------|---------|---------|-------|
| MAE | mean(|y_pred - y_actual|) | `sklearn.metrics.mean_absolute_error` | Dollars; e.g. "$35,000 average error" |
| RMSE | sqrt(mean((y_pred - y_actual)^2)) | `numpy.sqrt(sklearn.metrics.mean_squared_error)` | Penalizes large errors more than MAE |
| R² | 1 - SS_res/SS_tot | `sklearn.metrics.r2_score` | 1.0 = perfect; 0.0 = predicts mean; negative = worse than mean |
| MAPE | mean(|y_pred - y_actual| / |y_actual|) * 100 | Custom or `sklearn.metrics.mean_absolute_percentage_error` | Percentage; scale-independent |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
```

**Expected metric ranges for synthetic data (baseline):** With a QLoRA fine-tuned 0.5B model on synthetic NJ housing data, expect MAE in the $30,000–$80,000 range, MAPE 10–25%, R² 0.3–0.7. Lower quality (parse failures, poor convergence) will push metrics toward MAPE > 40%. These are not hard gates — EVAL-01 only requires the metrics are computed, not that they meet a threshold.

### Plots (EVAL-02, EVAL-03)

**EVAL-02: Predicted vs. actual scatter plot**
- X-axis: actual price, Y-axis: predicted price
- Include a `y=x` reference line (perfect prediction diagonal)
- Color-code by property type (optional, adds clarity)
- Save as PNG to Google Drive

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, alpha=0.4, s=10)
lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
ax.plot(lims, lims, 'r--', label='Perfect prediction', linewidth=1.5)
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title("Predicted vs. Actual Housing Prices\nQwen2.5-0.5B QLoRA Fine-tuned")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_PLOTS, "predicted_vs_actual.png"), dpi=150)
```

**EVAL-03: Training loss curve**
This is already generated by `02_train.ipynb` and saved to Google Drive at `housing_model/plots/training_loss_curve.png`. However, the success criterion says the loss curve is "generated as image files by the evaluation notebook." There are two interpretations:
- Option A: Re-load the trainer log history from a saved JSON and re-plot in the eval notebook
- Option B: The loss curve from `02_train.ipynb` already satisfies EVAL-03; the eval notebook just needs to load it from Drive

**Decision:** To strictly satisfy "generated as image files by the evaluation notebook," the eval notebook should re-generate the loss curve from trainer log history saved to a JSON file. The training notebook should save `trainer.state.log_history` to a JSON file on Drive, and the eval notebook loads and plots it. This is a clean separation: training saves raw data, evaluation generates the visualization.

Implementation in `02_train.ipynb` (add to training notebook):
```python
import json
log_history_path = os.path.join(DRIVE_BASE, "trainer_log_history.json")
with open(log_history_path, "w") as f:
    json.dump(trainer.state.log_history, f)
```

Implementation in `03_evaluate.ipynb`:
```python
import json
with open(os.path.join(DRIVE_BASE, "trainer_log_history.json")) as f:
    log_history = json.load(f)
# Plot training loss curve from log_history
```

**Alternative simpler interpretation:** The loss curve was already plotted in `02_train.ipynb` to a Drive file. The eval notebook can copy/display it. This avoids adding state-saving to the training notebook. However, this is less clean and may not satisfy the literal success criterion. Plan should generate the loss curve in the eval notebook by loading saved log data.

---

## Part 2: ONNX Export

### The Merge Pattern (ONNX-01)

The success criterion explicitly states: "The merge was performed via the two-step fp32 reload pattern (base reloaded without `load_in_4bit`, then `merge_and_unload()` called)."

This is the canonical pattern from PITFALLS.md and STACK.md:

```python
# Step 1: Load base model in fp32 on CPU (no quantization)
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,   # NOT load_in_4bit=True
    device_map="cpu",             # CPU merge is fine — no GPU needed
    trust_remote_code=True,
)

# Step 2: Load adapter on top of fp32 base
merged = PeftModel.from_pretrained(base, DRIVE_ADAPTER)

# Step 3: Merge adapter weights into base and remove adapter wrapper
merged = merged.merge_and_unload()

# Step 4: Save merged fp32 model
merged.save_pretrained(DRIVE_MERGED)
tokenizer.save_pretrained(DRIVE_MERGED)
```

**Why CPU and fp32:** The `merge_and_unload()` operation dequantizes the LoRA delta matrices and adds them to the base weights. This requires the base weights to be in a dequantizable state (fp32 or fp16, not 4-bit bitsandbytes format). CPU is used to avoid VRAM pressure from loading the full fp32 model (Qwen2.5-0.5B fp32 is ~2GB; Colab T4 has 15GB VRAM but it may be partially occupied).

**Expected merged model size:** Qwen2.5-0.5B in fp32 is approximately 2GB on disk. Saving to Google Drive is required — the merged model is too large for Colab's `/content` ephemeral storage alone.

### ONNX Export Task Type (ONNX-02)

**This was the primary open research question from SUMMARY.md.** Research findings:

**Confirmed:** Qwen2 (Qwen1.5) is listed as a supported architecture in Optimum's ONNX export (confirmed from the Optimum supported architectures page, 2026-02-27).

**Task type decision for this project:**

This project uses the LLM in a text-generation mode — the model autoregressively generates price digits after the prompt ends with `"Predicted price: $"`. This means the correct task type is `text-generation-with-past`, not `feature-extraction`.

Rationale:
- `feature-extraction` exports the model as a single forward pass returning hidden states (not logits for generation). It does NOT support `model.generate()` — it is for embedding extraction, not token generation.
- `text-generation-with-past` exports the model with KV-cache support (produces `model.onnx` + `model_with_past.onnx`), which is required for efficient autoregressive generation in ONNX Runtime. Optimum's `ORTModelForCausalLM` can then use these files with `.generate()`.
- `text-generation` (without `-with-past`) exports without KV-cache reuse — inference works but is slower (recomputes all attention for each token), which was flagged as a performance pitfall.

**Recommended export command:**
```bash
optimum-cli export onnx \
  --model ./merged_model/ \
  --task text-generation-with-past \
  --dtype fp16 \
  --trust-remote-code \
  ./onnx_output/
```

**Why `--dtype fp16`:** Reduces model.onnx from ~2GB (fp32) to ~1GB (fp16). STACK.md notes fp32 may cause OOM on Lambda at 3008MB memory. fp16 is the right tradeoff — small accuracy loss, significant size reduction.

**Note on `--dynamo` flag:** Optimum recommends `--dynamo` for opset >= 18. For Qwen2.5 with rotary position embeddings, the dynamo exporter handles the RoPE operations more cleanly than the TorchScript exporter. If export fails with the default exporter, add `--dynamo --opset 18`.

**Output files from `text-generation-with-past` export:**
- `model.onnx` — decoder without past key values (first token generation)
- `model_with_past.onnx` — decoder with past key values (subsequent tokens)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, etc.

Optimum by default merges these into a single file with `--no-post-process` disabled (default). To get a single merged ONNX file (simpler for Lambda), the default behavior applies (Optimum merges decoder + decoder-with-past into one file). Verify with `--monolith` flag if needed.

### ONNX Inference Pattern for Validation (ONNX-03)

The success criterion requires validation at `atol=1e-3` comparing ONNX Runtime output against PyTorch output.

**Approach 1 — Use Optimum's built-in validation:**
`optimum-cli export onnx` runs validation automatically and reports "All good" if atol passes. The default atol for causal LMs is typically 1e-4 or 1e-3. Check the output logs for the validation result.

**Approach 2 — Manual validation (explicit and verifiable):**
```python
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

# Load test prompt
tokenizer = AutoTokenizer.from_pretrained(DRIVE_MERGED, trust_remote_code=True)
test_prompt = ds["test"][0]["prompt"]
inputs = tokenizer(test_prompt, return_tensors="pt")

# PyTorch inference (logits for first token after prompt)
pt_model = AutoModelForCausalLM.from_pretrained(
    DRIVE_MERGED,
    torch_dtype=torch.float16,
    device_map="cpu",
)
with torch.no_grad():
    pt_outputs = pt_model(**inputs)
pt_logits = pt_outputs.logits[0, -1, :].numpy()  # last token logits

# ONNX Runtime inference
ort_model = ORTModelForCausalLM.from_pretrained(DRIVE_ONNX)
with torch.no_grad():
    ort_outputs = ort_model(**inputs)
ort_logits = ort_outputs.logits[0, -1, :].numpy()

# Validate
max_diff = np.max(np.abs(pt_logits - ort_logits))
print(f"Max absolute difference: {max_diff:.6f}")
assert max_diff < 1e-3, f"ONNX validation FAILED: max_diff={max_diff} > atol=1e-3"
print("ONNX-03 PASSED: Numerical validation at atol=1e-3")
```

**Alternative manual validation using raw ONNX Runtime session:**
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(os.path.join(DRIVE_ONNX, "model.onnx"))
inputs_ort = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy(),
}
ort_out = session.run(None, inputs_ort)
ort_logits = ort_out[0][0, -1, :]  # batch=0, last token, all vocab
max_diff = np.max(np.abs(pt_logits - ort_logits))
```

**Note on fp16 precision:** When exporting with `--dtype fp16`, the ONNX model operates in fp16. The PyTorch comparison model should also be in fp16 (`torch_dtype=torch.float16`) for a fair comparison. An fp16-to-fp32 precision mismatch will produce differences larger than 1e-3 even if the export is numerically correct.

### Notebook Architecture Decision: One or Two Notebooks?

The phase description says the evaluation notebook is `03_evaluate.ipynb` and implies a single notebook. But the SUMMARY.md describes them as potentially separate (notebooks 3 and 4). The success criteria language puts evaluation and ONNX export under one phase gate.

**Decision:** Use two notebooks to keep concerns separated and avoid Colab VRAM conflicts:
- `03_evaluate.ipynb` — loads adapter (4-bit quantized), runs inference on test set, computes metrics, generates plots. Does NOT do the merge (saves VRAM).
- `04_export.ipynb` — loads adapter (fp32, CPU), merges via `merge_and_unload()`, exports to ONNX via optimum, validates numerically.

This is consistent with SUMMARY.md's architectural description (notebooks 3 and 4 are separate) and avoids a common pitfall: loading the model in fp32 for ONNX export while still having the 4-bit quantized eval model in memory.

**However,** the phase description says "The model is evaluated against held-out test data...and a numerically validated ONNX artifact is ready for containerization" as the combined goal of Phase 3. Both notebooks are delivered as part of this phase. Phase 3 = 2 notebooks + all success criteria met.

---

## Part 3: Critical Implementation Details

### Google Drive Paths

Consistent with Phase 2 Drive organization:
```
/content/drive/MyDrive/housing_model/
├── lora_adapter/           # Input: LoRA adapter weights from Phase 2
├── checkpoints/            # Phase 2 training checkpoints
├── plots/
│   ├── training_loss_curve.png    # From 02_train.ipynb (or re-generated by 03_evaluate.ipynb)
│   ├── predicted_vs_actual.png    # New: generated by 03_evaluate.ipynb
├── trainer_log_history.json       # New: training notebook must save this
├── merged_model/           # New: fp32 merged model from 04_export.ipynb
└── onnx_model/             # New: ONNX export output from 04_export.ipynb
```

### The `trainer_log_history.json` Gap

**Current state of `02_train.ipynb`:** The training notebook already plots the loss curve and saves the PNG. But it does NOT save `trainer.state.log_history` to JSON. If Phase 3's eval notebook needs to re-generate the loss curve from raw data, the training notebook needs a small addition.

**Options:**
1. Add a cell to `02_train.ipynb` that saves `trainer.state.log_history` to JSON on Drive. (Requires modifying Phase 2 notebook — acceptable since Phase 2 Colab execution hasn't happened yet.)
2. Have the eval notebook re-plot from the existing PNG (just display it). This satisfies EVAL-03 if the criterion is interpreted as "the loss curve image file exists."
3. Have the eval notebook accept the already-saved PNG as satisfying EVAL-03.

**Recommended:** Add a single cell to `02_train.ipynb` to save `trainer.state.log_history` as JSON to Drive. This enables full reproducibility of the plot. Since Phase 2 Task 2 (Colab execution) is still pending (checkpoint:human-verify not yet approved), this addition can be made before the user runs the training notebook.

### Import Pattern for Eval Notebook

Same as training notebook — use `importlib.import_module` for the `lambda` module:
```python
import importlib
_prompt_mod = importlib.import_module("lambda.prompt_utils")
format_prompt = _prompt_mod.format_prompt
parse_price_from_output = _prompt_mod.parse_price_from_output
```

Dataset loading uses HuggingFace `datasets` (already installed in training environment):
```python
from datasets import load_dataset
ds = load_dataset("rajkumar4466/nj-housing-prices")
test_data = ds["test"]
```

### Dependencies for Phase 3 Notebooks

**`03_evaluate.ipynb`** (same as training environment — already installed):
```
transformers==5.2.0
peft==0.18.1
bitsandbytes==0.49.2
accelerate==1.12.0
datasets==4.6.0
sentencepiece==0.2.1
scikit-learn==1.8.0
matplotlib==3.10.8
numpy==2.4.2
```

**`04_export.ipynb`** (adds optimum and onnxruntime):
```
transformers==5.2.0
peft==0.18.1
accelerate==1.12.0
optimum[onnx]==2.1.0    # includes onnx==1.20.1
onnxruntime==1.24.2
sentencepiece==0.2.1
numpy==2.4.2
```

Note: `bitsandbytes` is NOT needed for `04_export.ipynb` — the merge step reloads in fp32 (no quantization). Do not install it unnecessarily (it has a CUDA dependency on Colab).

---

## Part 4: Risks and Mitigations

### Risk 1: ONNX Export Fails for Qwen2.5

**Probability:** LOW-MEDIUM. Qwen2 is confirmed supported in Optimum's architecture list. However, Qwen2.5-0.5B is a specific variant and rotary position embeddings (RoPE) have historically caused ONNX export issues with the TorchScript exporter.

**Mitigation:**
- Try default export first (TorchScript exporter)
- If it fails, add `--dynamo --opset 18` (dynamo exporter handles RoPE more cleanly)
- If dynamo fails, use `torch.onnx.export` directly with manual input/output specification
- The plan should include a fallback cell with the dynamo export command

### Risk 2: Test Set Inference Takes Too Long

**Probability:** MEDIUM. The full test set is ~1,050 records. At 1-3s each on T4 GPU, that's 17-52 minutes.

**Mitigation:**
- Evaluate on a 200-sample stratified subset first for quick metric check
- Run full evaluation in the background while working on export notebook
- Alternatively, run both notebooks back-to-back in a single Colab session

### Risk 3: Parse Failure Rate Too High

**Probability:** LOW-MEDIUM. The model was fine-tuned with strict `"Predicted price: $"` prompt ending, so it should generate digits. But if training loss didn't converge well, the model may generate narrative text instead.

**Mitigation:**
- Log parse failure count explicitly
- If parse failure rate > 10%, investigate top-5 failure examples for pattern
- `parse_price_from_output()` uses regex `\d+(?:\.\d+)?` which catches most numeric formats
- If failure rate is high, the metric computation should still proceed on valid predictions with a count warning

### Risk 4: ONNX atol=1e-3 Check Fails

**Probability:** LOW-MEDIUM for fp16 export, LOW for fp32 export.

**Mitigation:**
- If fp16 export fails atol check, re-export with `--dtype fp32`
- fp32 export will pass atol more easily but doubles model size (~2GB vs ~1GB)
- The plan should try fp16 first, with fp32 as explicit fallback

### Risk 5: Merged Model Too Large for Colab `/content`

**Probability:** LOW. Colab has ~12GB ephemeral disk. fp32 Qwen2.5-0.5B is ~2GB. ONNX export output is another ~1-2GB. Total ~4GB is within Colab disk limits.

**Mitigation:** Save merged model and ONNX output to Google Drive (not just `/content`).

---

## Part 5: Deliverables Checklist

| Requirement | Deliverable | Location |
|-------------|-------------|----------|
| EVAL-01 | MAE, RMSE, R², MAPE printed in `03_evaluate.ipynb` | Notebook cell output |
| EVAL-02 | Predicted vs. actual scatter plot PNG | `DRIVE_PLOTS/predicted_vs_actual.png` |
| EVAL-03 | Training loss curve PNG | `DRIVE_PLOTS/training_loss_curve.png` |
| ONNX-01 | Two-step fp32 merge: base reloaded without `load_in_4bit`, `merge_and_unload()` called | `04_export.ipynb` merge cell |
| ONNX-02 | `model.onnx` (and supporting files) exported via optimum | `DRIVE_ONNX/` |
| ONNX-03 | Numerical validation: max abs diff < 1e-3 vs. PyTorch logits | Validation cell output in `04_export.ipynb` |

---

## Part 6: Key Decisions to Lock In Before Planning

1. **Two notebooks (03_evaluate + 04_export) vs. one combined notebook.** Recommendation: two notebooks to keep VRAM usage separated and match the architectural description in SUMMARY.md.

2. **Add JSON log save to training notebook.** To support EVAL-03 from the eval notebook, `02_train.ipynb` needs a cell that saves `trainer.state.log_history` to `DRIVE_BASE/trainer_log_history.json`. Since the Colab execution hasn't happened yet, this can be added before the user runs the notebook.

3. **ONNX export task: `text-generation-with-past`.** This is correct for a causal LM being used for autoregressive token generation. `feature-extraction` is for embedding extraction only and will not support `model.generate()`.

4. **Export dtype: fp16 with fp32 fallback.** Attempt fp16 first (smaller artifact for Lambda). If atol=1e-3 fails with fp16, fallback to fp32 export.

5. **Evaluation data source: HF dataset test split.** Load from `rajkumar4466/nj-housing-prices` test split (consistent with how training notebook evolved from JSONL to HF dataset).

6. **Inference model state for eval: 4-bit quantized (same as training).** Do NOT use fp32 for the evaluation inference — keep memory usage low. fp32 reload is only for the ONNX export step.

---

## Sources

- HuggingFace Optimum ONNX Export Guide — confirmed `text-generation-with-past` as correct task for causal LM generation; confirmed Qwen2 is supported architecture — HIGH confidence, verified 2026-02-27
- HuggingFace Optimum Overview page — confirmed Qwen2(Qwen1.5) in supported architectures list — HIGH confidence, verified 2026-02-27
- HuggingFace PEFT docs — `merge_and_unload()` merge pattern, two-step reload requirement — HIGH confidence
- PITFALLS.md (project research) — quantized merge pitfall, ONNX task type pitfall, performance traps — HIGH confidence (previously vetted)
- STACK.md (project research) — pinned library versions, fp16 vs fp32 tradeoffs, Lambda size constraints — HIGH confidence (previously vetted)
- `02_train.ipynb` (actual notebook) — confirmed HF dataset loading pattern, prompt format, Drive path conventions — HIGH confidence (source of truth for Phase 2 output contract)
- `lambda/prompt_utils.py` — confirmed `parse_price_from_output()` regex pattern — HIGH confidence (actual code)
- scikit-learn docs — MAE, RMSE, R², MAPE API — HIGH confidence

---

*Research completed: 2026-02-27*
*Addresses open research flag from SUMMARY.md: "Correct `optimum-cli export onnx --task` flag for Qwen2.5 regression"*
*Conclusion: use `--task text-generation-with-past` (not `feature-extraction`)*
