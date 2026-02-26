---
phase: 01-data-foundation
plan: 02
subsystem: data
tags: [python, jupyter, pandas, numpy, scikit-learn, scipy, matplotlib, qlora, qwen2.5, nj-housing]

# Dependency graph
requires:
  - "lambda/prompt_utils.py: format_prompt() and parse_price_from_output()"
  - "data/splits/: output directory created in plan 01"
  - "requirements.txt: scipy, matplotlib, scikit-learn, pandas, numpy"
provides:
  - "notebooks/01_data_prep.ipynb: Complete data generation and splitting pipeline"
  - "data/splits/train.jsonl: Training split (pending SR1A verification)"
  - "data/splits/val.jsonl: Validation split (pending SR1A verification)"
  - "data/splits/test.jsonl: Test split (pending SR1A verification)"
  - "data/price_distribution_check.png: Distribution validation plot"
affects:
  - "02-training (training loop reads train.jsonl/val.jsonl)"
  - "03-evaluate (eval notebook reads test.jsonl)"

# Tech tracking
tech-stack:
  added:
    - "jupyter nbformat 4 notebook structure"
    - "numpy.random.default_rng for reproducible generation"
    - "log-normal county price distributions calibrated to NJ 2024-2025 medians"
    - "sklearn.model_selection.train_test_split with price quartile stratification"
  patterns:
    - "importlib.import_module('lambda.prompt_utils') instead of from-import (reserved keyword)"
    - "Price-first synthetic generation: price generated from log-normal, features derived from price"
    - "Stratified 70/15/15 split on price quartiles for even price distribution across splits"
    - "5-char zip code: 3-char prefix + 2-char suffix (no leading zero loss)"

key-files:
  created:
    - "notebooks/01_data_prep.ipynb"
  modified: []

key-decisions:
  - "importlib.import_module used for lambda.prompt_utils import (not from-import syntax) — lambda is Python reserved keyword"
  - "Price-first generation strategy: generate price from county log-normal, derive all other features from price for realistic correlations"
  - "Zip code generation: 3-char county prefix + 2-char random suffix = 5-char string (not 3+3=6)"
  - "SR1A column names are PLACEHOLDERS in Cell 6 — user must update to match actual SR1A file headers after download"
  - "DATA-04 30% real records requirement has a soft-fail path: notebook warns but continues with synthetic-only if SR1A is not present"

# Metrics
duration: 5min
completed: 2026-02-26
---

# Phase 1 Plan 02: Data Preparation Notebook Summary

**Jupyter notebook 01_data_prep.ipynb built with 8 cells covering county log-normal synthetic generation, SR1A real data loading, 70/15/15 stratified JSONL export, and price distribution validation**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-26T22:59:53Z
- **Completed:** 2026-02-26T23:05:00Z (Task 1 complete; Task 2 at checkpoint)
- **Tasks:** 1 of 2 complete (Task 2 awaiting human verification)
- **Files modified:** 1

## Status

**CHECKPOINT REACHED** — Task 2 is `type="checkpoint:human-verify"`. The notebook is fully built (all 8 cells including SR1A loading, split, and JSONL export). User must download SR1A data, run the notebook, and verify outputs before this plan is complete.

## Accomplishments

- Created `notebooks/01_data_prep.ipynb` as a valid nbformat 4 Jupyter notebook with 8 cells
- Cell 1: Markdown description including import note about lambda reserved keyword
- Cell 2: Setup, imports, repo root detection, importlib-based lambda.prompt_utils import (verified working)
- Cell 3: 21 NJ counties defined in `COUNTY_PRICE_PARAMS` with log-normal mu/sigma and zip_prefix lists
- Cell 4: `generate_synthetic_record()` (price-first strategy) and `validate_price_distribution()` functions
- Cell 5: Generate 7,000 synthetic records with 5-char zip code validation
- Cell 6: SR1A 2024 real data loader (graceful fallback if file not present, placeholder column constants)
- Cell 7: Combine datasets, validate price distribution, stratified 70/15/15 split with price quartile stratification
- Cell 8: `export_to_jsonl()` writing `{prompt, price}` JSONL records, spot-check validation

## Task Commits

1. **Task 1: Create notebook cells 1-8** - `9a4a785` (feat) — includes all cells including SR1A loading, split, and export

## Files Created/Modified

- `notebooks/01_data_prep.ipynb` - Complete data generation pipeline (8 cells)

## Decisions Made

- **importlib for lambda import:** `from lambda.prompt_utils import format_prompt` raises SyntaxError because `lambda` is a Python reserved keyword. Used `importlib.import_module('lambda.prompt_utils')` instead.
- **Price-first generation:** Generate price from county log-normal distribution first, then derive sqft, bedrooms, bathrooms from price. Produces realistic feature-price correlations.
- **5-char zip code fix:** Plan code had a bug — first computed `prefix + 3-digit suffix = 6 chars`, then immediately recomputed with `prefix + 2-digit suffix = 5 chars`. Kept only the correct 5-char version.
- **SR1A placeholders:** Column names in Cell 6 (`SALE_PRICE_COL`, `PROPERTY_CLASS_COL`, etc.) are PLACEHOLDERS that user must update after inspecting the actual SR1A file.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed import syntax for lambda reserved keyword**
- **Found during:** Task 1 (Cell 2 implementation)
- **Issue:** Plan specified `from lambda.prompt_utils import format_prompt` which raises `SyntaxError: invalid syntax` because `lambda` is a Python reserved keyword
- **Fix:** Used `importlib.import_module('lambda.prompt_utils')` pattern as documented in 01-01-SUMMARY.md
- **Files modified:** notebooks/01_data_prep.ipynb (Cell 2)
- **Commit:** 9a4a785

**2. [Rule 1 - Bug] Fixed zip code generation (6-char bug in plan)**
- **Found during:** Task 1 (Cell 4 implementation)
- **Issue:** Plan's `generate_synthetic_record` code first computed `prefix(3) + suffix(3) = 6 chars`, then redundantly recomputed with `prefix(3) + suffix(2) = 5 chars` — the 6-char computation was dead code with the wrong length
- **Fix:** Removed the erroneous 6-char computation, kept only the correct `prefix(3) + suffix(2) = 5` version
- **Files modified:** notebooks/01_data_prep.ipynb (Cell 4)
- **Commit:** 9a4a785

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes were necessary for correctness. No scope change.

## Issues Encountered

- SR1A 2024 file not yet downloaded by user — Cell 6 gracefully handles missing file with informational warnings
- Column names in Cell 6 are PLACEHOLDERS — user must consult SR1A_FileLayout_Description.pdf and update constants

## User Setup Required

- Download SR1A 2024 Sales File from: https://www.nj.gov/treasury/taxation/lpt/statdata.shtml
- Extract and place at: `data/raw/SR1A_2024.txt`
- Update column name constants in Cell 6 to match actual SR1A column headers
- Run notebook end-to-end in Jupyter

## Next Phase Readiness

- Notebook is complete and structurally valid (verified with json.load + assertion checks)
- importlib import pattern verified working in Python 3
- Awaiting human SR1A data download and notebook execution for final data split outputs

---
*Phase: 01-data-foundation*
*Completed: 2026-02-26 (Task 1 only — Task 2 at checkpoint)*

## Self-Check: PARTIAL

- notebooks/01_data_prep.ipynb: FOUND
- Task 1 commit 9a4a785: verified present
- data/splits/train.jsonl: NOT YET (pending notebook execution — expected after checkpoint)
- data/splits/val.jsonl: NOT YET (pending notebook execution — expected after checkpoint)
- data/splits/test.jsonl: NOT YET (pending notebook execution — expected after checkpoint)
