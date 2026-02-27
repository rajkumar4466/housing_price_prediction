---
phase: 03-evaluation-and-onnx-export
plan: 02
subsystem: onnx-export
tags: [jupyter, colab, onnx, optimum, peft, transformers, qwen, lora, onnxruntime]

# Dependency graph
requires:
  - phase: 03-01-evaluation
    provides: "Trained LoRA adapter saved to Google Drive at housing_model/lora_adapter/"
  - phase: 02-qlora-training
    provides: "QLoRA fine-tuned Qwen2.5-0.5B adapter at housing_model/lora_adapter/"
provides:
  - "notebooks/04_export.ipynb: complete ONNX export and validation notebook for Colab"
  - "Two-step fp32 merge pipeline: reload base in float32 on CPU, then merge_and_unload()"
  - "ONNX model at housing_model/onnx_model/ on Google Drive (Colab-executed: model.onnx + model.onnx_data ~2.5 GB)"
  - "Numerical validation confirming ONNX logits match PyTorch at max_diff=0.000029 < atol=1e-3 (ONNX-03 PASSED)"
affects: [04-lambda-container-and-rest-api]

# Tech tracking
tech-stack:
  added: ["optimum[onnxruntime]==2.1.0", onnxruntime==1.24.2, sentencepiece==0.2.1]
  patterns: [fp32-two-step-merge, onnx-export-text-generation-with-past, logit-level-numerical-validation, dynamo-fallback-exporter]

key-files:
  created:
    - notebooks/04_export.ipynb
  modified: []

key-decisions:
  - "fp32 two-step merge: reload base model with torch_dtype=torch.float32 and device_map=cpu BEFORE loading adapter -- quantized merge silently corrupts weights"
  - "text-generation-with-past is the correct ONNX task for autoregressive causal LM generation (NOT feature-extraction which is for embeddings)"
  - "bitsandbytes NOT installed in this notebook -- fp32 merge requires no quantization at all"
  - "Dynamo fallback cell included with --dynamo --opset 18 flags to handle RoPE exporter compatibility issues with TorchScript default"
  - "Numerical validation compares last-token logits (the generation decision point) not all-token logits -- most meaningful for autoregressive use case"
  - "transformers pinned to >=4.45.0,<5.0.0 -- pad_token_id attribute error in 5.x breaks Qwen2 ONNX export in Colab"
  - "pad_token setup removed -- Qwen2 tokenizer handles it internally; explicit assignment caused warnings"
  - "Raw ORT validation requires position_ids + past_key_values inputs for text-generation-with-past graph; zero-fill applied"
  - "importlib.metadata.version('optimum') used instead of optimum.__version__ which raises AttributeError"

patterns-established:
  - "Two-step fp32 merge: always reload base in fp32 on CPU before PeftModel.from_pretrained() + merge_and_unload()"
  - "ONNX export verification: compare last-token logits between PyTorch and ONNX Runtime at atol=1e-3"
  - "Fallback cell pattern: check if primary output exists before running alternative approach"
  - "Pin transformers to <5.0.0 for Qwen2 ONNX export workflows until 5.x Qwen2 bugs are resolved"

requirements-completed: [ONNX-01, ONNX-02, ONNX-03]

# Metrics
duration: 20min (notebook creation + Colab execution 16.2 min)
completed: 2026-02-27
---

# Phase 3 Plan 02: ONNX Export Notebook Summary

**Qwen2.5-0.5B + LoRA ONNX export completed on Colab: fp32 merge via merge_and_unload(), optimum-cli text-generation-with-past export (~2.5 GB), ONNX-03 validated at max_diff=0.000029 < atol=1e-3**

## Performance

- **Duration:** ~20 min total (4 min notebook creation + 16.2 min Colab export runtime)
- **Started:** 2026-02-27T18:17:40Z
- **Completed:** 2026-02-27
- **Tasks:** 2 of 2 complete (Task 1 auto, Task 2 human-verify checkpoint APPROVED)
- **Files modified:** 1

## Accomplishments
- Created `notebooks/04_export.ipynb` with 14 cells covering the full ONNX export pipeline
- Implemented two-step fp32 merge pattern: reload base in float32 on CPU, load LoRA adapter, call merge_and_unload(), save merged model to Drive
- ONNX export via optimum-cli with `text-generation-with-past` task (confirmed working on Colab — ONNX-02 PASSED)
- Dynamo fallback cell (Cell 8b) with `--dynamo --opset 18` for RoPE exporter compatibility
- Numerical validation cell (Cell 9) comparing last-token logits between PyTorch and ONNX Runtime at atol=1e-3 (ONNX-03 PASSED: max_diff=0.000029, mean_diff=0.000004)
- Raw onnxruntime.InferenceSession fallback validation (Cell 9b) — fixed to include position_ids + past_key_values
- End-to-end generation test (Cell 10) verified ONNX model produces parseable price numbers
- Applied 4 Colab-discovered fixes and recommitted (transformers pin, pad_token removal, raw ORT inputs, importlib.metadata)

## Colab Execution Results

| Requirement | Status | Evidence |
|-------------|--------|----------|
| ONNX-01 (fp32 merge) | PASSED | Merged model saved to Drive at housing_model/merged_model/ |
| ONNX-02 (export) | PASSED | model.onnx (1.1 MB) + model.onnx_data (2520.7 MB) = ~2.5 GB |
| ONNX-03 (validation) | PASSED | max_diff=0.000029, mean_diff=0.000004 (threshold: atol=1e-3) |
| Generation test | PASSED | ONNX model generated parseable price in end-to-end test |
| Total export time | 16.2 min | CPU runtime on Colab |

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ONNX export notebook with fp32 merge, export, and numerical validation** - `48f24bc` (feat)
2. **Task 2: Notebook compatibility fixes from Colab testing** - `a080400` (fix)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `notebooks/04_export.ipynb` - Complete ONNX export and validation notebook for Google Colab (14 cells), updated with Colab compatibility fixes

## Decisions Made
- Two-step fp32 merge requires `torch_dtype=torch.float32` + `device_map="cpu"` -- NOT `load_in_4bit=True` which would silently corrupt merged weights
- `text-generation-with-past` ONNX task chosen for autoregressive causal LM (supports model.generate(); `feature-extraction` does not)
- bitsandbytes intentionally excluded from pip installs -- fp32 merge needs no quantization
- transformers pinned to `>=4.45.0,<5.0.0` — 5.x has pad_token_id AttributeError with Qwen2 models during ONNX export
- Numerical validation uses last-token logits (index [0, -1, :]) -- the actual generation decision point, most meaningful for autoregressive price prediction

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] transformers version pinned to <5.0.0 due to Qwen2 pad_token_id bug**
- **Found during:** Task 2 (Colab execution)
- **Issue:** Research phase specified `transformers==5.2.0` but 5.x introduced a breaking AttributeError on pad_token_id for Qwen2 models during ONNX export
- **Fix:** Changed `transformers==5.2.0` to `transformers>=4.45.0,<5.0.0` in the pip install cell
- **Files modified:** notebooks/04_export.ipynb
- **Verification:** Colab ONNX-02 PASSED with updated pin
- **Committed in:** a080400

**2. [Rule 1 - Bug] Removed unnecessary pad_token assignment that caused warnings**
- **Found during:** Task 2 (Colab execution)
- **Issue:** `tokenizer.pad_token = tokenizer.eos_token` raised a warning because Qwen2 tokenizer manages pad token internally
- **Fix:** Removed the pad_token setup block from Cell 5
- **Files modified:** notebooks/04_export.ipynb
- **Verification:** No pad_token warning in Colab output; export completed cleanly
- **Committed in:** a080400

**3. [Rule 1 - Bug] Fixed raw onnxruntime validation to pass required inputs**
- **Found during:** Task 2 (Colab execution)
- **Issue:** Cell 9b (raw ORT validation) only passed input_ids + attention_mask, but the text-generation-with-past ONNX graph also requires position_ids and past_key_values — session.run failed
- **Fix:** Added position_ids construction (np.arange) and zero-fill for all past_key_values inputs
- **Files modified:** notebooks/04_export.ipynb
- **Verification:** Raw ORT session completed; max_diff=0.000029 confirmed
- **Committed in:** a080400

**4. [Rule 1 - Bug] Switched optimum version check to importlib.metadata**
- **Found during:** Task 2 (Colab execution)
- **Issue:** `optimum.__version__` raises AttributeError — the package does not expose this attribute at module level
- **Fix:** Used `from importlib.metadata import version as pkg_version; pkg_version('optimum')`
- **Files modified:** notebooks/04_export.ipynb
- **Verification:** Version printed correctly in Colab cell output
- **Committed in:** a080400

---

**Total deviations:** 4 auto-fixed (all Rule 1 - bugs discovered during Colab execution)
**Impact on plan:** All fixes necessary for ONNX export to complete successfully. No scope creep.

## Issues Encountered

- transformers 5.x incompatibility with Qwen2 required version pin to 4.x branch — resolved cleanly; 4.x is stable for this export use case
- Raw ORT validation needed position_ids + past_key_values for text-generation-with-past task — fixed with zero-fill and arange construction

## User Setup Required

**Google Colab execution completed:**
- Uploaded notebooks/04_export.ipynb to Colab, ran all cells on CPU runtime
- Verified all three requirements (ONNX-01, ONNX-02, ONNX-03) PASSED
- ONNX model files saved to Google Drive at housing_model/onnx_model/

## Next Phase Readiness
- ONNX model files are on Google Drive at housing_model/onnx_model/ (~2.5 GB: model.onnx + model.onnx_data)
- Phase 4 (Lambda container) has already been implemented (04-01) and needs model files from Drive placed in lambda/model_artifacts/
- lambda/model_artifacts/ is gitignored (file size exceeds GitHub 100MB limit)
- Phase 3 is now complete — all 2 plans done, all requirements met

---
*Phase: 03-evaluation-and-onnx-export*
*Completed: 2026-02-27*
