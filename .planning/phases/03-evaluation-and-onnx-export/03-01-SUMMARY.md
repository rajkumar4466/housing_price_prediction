---
phase: 03-evaluation-and-onnx-export
plan: 01
subsystem: evaluation
tags: [jupyter, colab, qlora, onnx, sklearn, matplotlib, peft, transformers, qwen]

# Dependency graph
requires:
  - phase: 02-qlora-training
    provides: "Trained LoRA adapter saved to Google Drive at housing_model/lora_adapter/"
  - phase: 01-data-foundation
    provides: "HuggingFace dataset rajkumar4466/nj-housing-prices with test split"
provides:
  - "notebooks/03_evaluate.ipynb: full evaluation pipeline for Colab (model load, inference, metrics, plots)"
  - "notebooks/02_train.ipynb updated with trainer_log_history.json save (EVAL-03 prerequisite)"
  - "Evaluation metrics: MAE, RMSE, R2, MAPE on held-out test set (EVAL-01)"
  - "predicted_vs_actual.png scatter plot with y=x reference line (EVAL-02)"
  - "training_loss_curve.png regenerated from JSON log history (EVAL-03)"
affects: [04-lambda-container-and-rest-api, 03-02-onnx-export]

# Tech tracking
tech-stack:
  added: [scikit-learn==1.8.0, matplotlib==3.10.8, datasets==4.6.0, peft==0.18.1, bitsandbytes==0.49.2]
  patterns: [colab-notebook, 4bit-inference, lora-adapter-loading, regression-metrics, parse-price-inline-copy]

key-files:
  created:
    - notebooks/03_evaluate.ipynb
  modified:
    - notebooks/02_train.ipynb

key-decisions:
  - "parse_price_from_output defined inline in Cell 7 of 03_evaluate.ipynb (verbatim copy of lambda/prompt_utils.py) for Colab portability -- avoids requiring full repo clone just for one function"
  - "Load model in 4-bit for eval inference (same config as training) -- fp32 reload only needed for ONNX export notebook"
  - "200-sample quick eval runs first for fast feedback before full test set (~1,050 records) which can take 17-52 min on T4"
  - "do_sample=False ensures deterministic generation for reproducible metrics"
  - "Parse failure rate tracked with 5% warning threshold"

patterns-established:
  - "Quick eval first (200 samples) then full eval -- gives early feedback signal before long-running full evaluation"
  - "Inline function copies for Colab portability -- functions from lambda/ mirrored inline with comment noting source"
  - "Metrics saved as JSON to Drive (eval_metrics.json) alongside PNG plots for programmatic access"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03]

# Metrics
duration: 9min
completed: 2026-02-27
---

# Phase 3 Plan 01: Evaluation Notebook Summary

**QLoRA Qwen2.5-0.5B evaluation notebook (03_evaluate.ipynb) with 4-bit inference, MAE/RMSE/R2/MAPE metrics, predicted-vs-actual scatter plot, and training loss curve from saved JSON log history**

## Performance

- **Duration:** ~9 min (notebook creation)
- **Started:** 2026-02-27T17:40:16Z
- **Completed:** 2026-02-27T17:49:00Z
- **Tasks:** 1 of 2 completed locally (Task 2 is Colab human-verify checkpoint)
- **Files modified:** 2

## Accomplishments
- Created `notebooks/03_evaluate.ipynb` with 15 cells covering the full evaluation pipeline
- Updated `notebooks/02_train.ipynb` with cell-13b-save-log-history that saves trainer.state.log_history as JSON to Google Drive (enables EVAL-03 loss curve regeneration)
- Evaluation notebook covers: Drive mount, pip installs, 4-bit model + LoRA adapter load, parse_price_from_output (inlined), inference function, 200-sample quick eval, full test set eval, all 4 metrics (EVAL-01), scatter plot with y=x line (EVAL-02), loss curve from JSON (EVAL-03), and summary with requirement status checks

## Task Commits

Each task was committed atomically:

1. **Task 1: Create evaluation notebook and update training notebook with log history save** - `56bff62` (feat)
2. **Task 2: Run evaluation notebook on Colab and verify metrics and plots** - Awaiting human checkpoint (Colab verification)

**Plan metadata:** (pending after checkpoint approval)

## Files Created/Modified
- `notebooks/03_evaluate.ipynb` - Full evaluation pipeline notebook for Google Colab (15 cells)
- `notebooks/02_train.ipynb` - Added cell-13b-save-log-history to save trainer.state.log_history as JSON to Drive

## Decisions Made
- parse_price_from_output inlined in Cell 7 as verbatim copy of lambda/prompt_utils.py -- Colab portability (no repo clone required). Function body is identical to source of truth.
- 4-bit quantization used for inference (same bnb_config as training) -- fp32 reload only for ONNX export phase
- 200-sample quick eval added before full eval to provide fast early feedback (~50-200 seconds vs 17-52 min for full set)
- do_sample=False for deterministic/reproducible inference metrics
- Metrics saved to eval_metrics.json on Drive alongside PNG plots for programmatic consumption

## Deviations from Plan

None - plan executed exactly as written. The inline parse_price_from_output approach was explicitly called out in the plan's IMPORTANT NOTE as an acceptable design choice for Colab portability.

## Issues Encountered

None - all automated verification checks passed on first attempt.

## User Setup Required

**Google Colab execution required for full validation.** See plan's `user_setup` section:
- Upload `notebooks/03_evaluate.ipynb` to Google Colab
- Ensure GPU runtime: Runtime > Change runtime type > T4 GPU
- If re-running training: re-upload updated `notebooks/02_train.ipynb` first to save trainer_log_history.json
- Run all cells: Runtime > Run all
- Verify: all 4 metrics printed, both PNG plots saved to Drive, EVAL-01/02/03 show PASSED in summary

## Next Phase Readiness
- Evaluation notebook ready for Colab execution
- After Colab verification (checkpoint approval), phase 3 plan 1 is complete
- Phase 3 plan 2 (ONNX export) can proceed once metrics are confirmed
- eval_metrics.json on Drive will contain quantitative evidence of model quality before ONNX export

---
*Phase: 03-evaluation-and-onnx-export*
*Completed: 2026-02-27*
