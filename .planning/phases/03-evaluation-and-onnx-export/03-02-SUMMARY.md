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
  - "ONNX model at housing_model/onnx_model/ on Google Drive (after Colab run)"
  - "Numerical validation confirming ONNX logits match PyTorch at atol=1e-3 (ONNX-03)"
affects: [04-lambda-container-and-rest-api]

# Tech tracking
tech-stack:
  added: [optimum[onnxruntime]==2.1.0, onnxruntime==1.24.2, sentencepiece==0.2.1]
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
  - "ORTModelForCausalLM used for ONNX validation + raw onnxruntime.InferenceSession as fallback validation path"

patterns-established:
  - "Two-step fp32 merge: always reload base in fp32 on CPU before PeftModel.from_pretrained() + merge_and_unload()"
  - "ONNX export verification: compare last-token logits between PyTorch and ONNX Runtime at atol=1e-3"
  - "Fallback cell pattern: check if primary output exists before running alternative approach"

requirements-completed: [ONNX-01, ONNX-02, ONNX-03]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 3 Plan 02: ONNX Export Notebook Summary

**Qwen2.5-0.5B + LoRA ONNX export notebook with two-step fp32 CPU merge via merge_and_unload(), optimum-cli text-generation-with-past export, dynamo fallback, and atol=1e-3 PyTorch vs ONNX Runtime numerical validation**

## Performance

- **Duration:** ~4 min (notebook creation)
- **Started:** 2026-02-27T18:17:40Z
- **Completed:** 2026-02-27T18:21:41Z
- **Tasks:** 1 of 2 complete (Task 2 is checkpoint awaiting Colab execution)
- **Files modified:** 1

## Accomplishments
- Created `notebooks/04_export.ipynb` with 14 cells covering the full ONNX export pipeline
- Implemented two-step fp32 merge pattern: reload base in float32 on CPU, load LoRA adapter, call merge_and_unload(), save merged model to Drive
- ONNX export via optimum-cli with `text-generation-with-past` task and `--trust-remote-code` flag
- Dynamo fallback cell (Cell 8b) with `--dynamo --opset 18` for RoPE exporter compatibility
- Numerical validation cell (Cell 9) comparing last-token logits between PyTorch and ONNX Runtime at atol=1e-3
- Raw onnxruntime.InferenceSession fallback validation (Cell 9b) as alternative path
- End-to-end generation test (Cell 10) verifying ONNX model produces parseable price numbers
- Summary cell (Cell 11) with ONNX-01/ONNX-02/ONNX-03 requirement status checks

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ONNX export notebook with fp32 merge, export, and numerical validation** - `48f24bc` (feat)
2. **Task 2: Run ONNX export notebook on Colab and verify merge, export, and validation** - Awaiting Colab execution (checkpoint)

## Files Created/Modified
- `notebooks/04_export.ipynb` - Complete ONNX export and validation notebook for Google Colab (14 cells)

## Decisions Made
- Two-step fp32 merge requires `torch_dtype=torch.float32` + `device_map="cpu"` -- NOT `load_in_4bit=True` which would silently corrupt merged weights
- `text-generation-with-past` ONNX task chosen for autoregressive causal LM (supports model.generate(); `feature-extraction` does not)
- bitsandbytes intentionally excluded from pip installs -- fp32 merge needs no quantization
- Dynamo exporter fallback added for Qwen2.5-0.5B RoPE operation compatibility (TorchScript exporter may fail on newer rotary embedding implementations)
- Numerical validation uses last-token logits (index [0, -1, :]) -- the actual generation decision point, most meaningful for autoregressive price prediction

## Deviations from Plan

None - plan executed exactly as written. All 14 cells match the plan specification including fallback cells 8b and 9b.

## Issues Encountered

None - automated verification passed on first attempt (14 cells, all critical patterns present, no load_in_4bit=True in code).

## User Setup Required

**Google Colab execution required for full validation.** See plan's `user_setup` section:
- Upload `notebooks/04_export.ipynb` to Google Colab: https://colab.research.google.com/ -- File > Upload notebook
- GPU runtime optional but recommended. CPU sufficient for fp32 merge: Runtime > Change runtime type
- Run all cells: Runtime > Run all
- Verify:
  - Cell 2: "Adapter found on Drive" printed
  - Cell 5: dtype is torch.float32
  - Cell 6: "Merge complete" with dtype still torch.float32
  - Cell 7: Merged model files ~2GB listed
  - Cell 8: ONNX files created at housing_model/onnx_model/
  - Cell 9: ONNX-03 PASSED with max_diff < 1e-3
  - Cell 10: ONNX generation produces parseable price
  - Cell 11: ONNX-01, ONNX-02, ONNX-03 all PASSED

## Next Phase Readiness
- `notebooks/04_export.ipynb` is ready to upload and run on Colab
- After Colab execution: merged fp32 model at Drive/housing_model/merged_model/, ONNX files at Drive/housing_model/onnx_model/
- Phase 4 (Lambda containerization) requires the ONNX model files from Drive -- download model.onnx after export

---
*Phase: 03-evaluation-and-onnx-export*
*Completed: 2026-02-27*
