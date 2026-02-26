---
phase: 01-data-foundation
plan: 01
subsystem: data
tags: [python, prompt-engineering, qwen2.5, qlora, pandas, numpy, scikit-learn]

# Dependency graph
requires: []
provides:
  - "lambda/prompt_utils.py: format_prompt() and parse_price_from_output() — the training/inference prompt contract"
  - "lambda/__init__.py: Python package init for lambda/ directory"
  - "data/SCHEMA.md: complete 10-field dataset schema documentation"
  - "requirements.txt: pinned Phase 1 Python dependencies"
  - "Project directory scaffold: data/raw/, data/processed/, data/splits/, notebooks/, scripts/"
affects:
  - "02-data-foundation (data generation notebook must import from lambda.prompt_utils)"
  - "03-training (training loop must use same prompt format)"
  - "04-inference (Lambda handler uses format_prompt for inference)"

# Tech tracking
tech-stack:
  added:
    - "pandas==3.0.1"
    - "numpy==2.4.2"
    - "scikit-learn==1.8.0"
    - "scipy==1.17.1"
    - "matplotlib==3.10.8"
  patterns:
    - "Prompt-as-contract: prompt format is frozen at training time; all modules import from single source"
    - "lambda/ as Python package: importable via importlib.import_module('lambda.prompt_utils') since lambda is a reserved keyword"

key-files:
  created:
    - "lambda/__init__.py"
    - "lambda/prompt_utils.py"
    - "data/SCHEMA.md"
    - "requirements.txt"
    - ".gitignore"
    - "data/raw/.gitkeep"
    - "data/processed/.gitkeep"
    - "data/splits/.gitkeep"
    - "notebooks/.gitkeep"
    - "scripts/.gitkeep"
  modified: []

key-decisions:
  - "lambda/ named after AWS Lambda deployment target even though it clashes with Python reserved keyword — import must use importlib.import_module('lambda.prompt_utils') not from-import syntax"
  - "Prompt template is FINAL training/inference contract — any change requires regenerating all data and retraining from scratch"
  - "zip_code typed as str (not int) to preserve leading zeros for NJ north-jersey zips (e.g., '07650')"
  - "lot_size uses :.2f formatting for consistent decimal representation in prompts"

patterns-established:
  - "Prompt format: 'Property: {type} in zip {zip}. {N} bedrooms, {N} bathrooms, {N} sqft living area, {N:.2f} acre lot, built in {year}. Predicted price: $'"
  - "Parser: strip commas, regex first numeric sequence, return float or None"

requirements-completed: [DATA-03, DATA-04]

# Metrics
duration: 2min
completed: 2026-02-26
---

# Phase 1 Plan 01: Project Scaffold and Prompt Contract Summary

**format_prompt() and parse_price_from_output() established as the frozen training/inference contract in lambda/prompt_utils.py, with full project scaffold and 10-field dataset schema**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T22:55:49Z
- **Completed:** 2026-02-26T22:57:40Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Created `lambda/prompt_utils.py` with `format_prompt()` returning prompts ending with `"Predicted price: $"` and `parse_price_from_output()` handling comma-formatted prices
- Created `data/SCHEMA.md` documenting all 10 dataset fields (7 features + price + source + county) with type, source, range, and notes
- Created project directory scaffold with pinned Phase 1 dependencies and comprehensive `.gitignore`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create lambda/prompt_utils.py** - `e016182` (feat)
2. **Task 2: Create project scaffold** - `20cf63a` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `lambda/__init__.py` - Empty package init making lambda/ importable as Python package
- `lambda/prompt_utils.py` - format_prompt() and parse_price_from_output() — the prompt contract
- `data/SCHEMA.md` - Documents all 10 dataset fields with type, source, range, notes
- `requirements.txt` - Pinned deps: pandas 3.0.1, numpy 2.4.2, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8
- `.gitignore` - Excludes data/raw/, data/processed/, data/splits/, model artifacts, .ipynb_checkpoints
- `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/splits/.gitkeep` - Track directories with ignored contents
- `notebooks/.gitkeep`, `scripts/.gitkeep` - Track empty directories

## Decisions Made
- **lambda reserved keyword:** The `lambda/` directory name matches the AWS Lambda deployment target but collides with Python's `lambda` reserved keyword. Direct `from lambda.prompt_utils import ...` syntax fails with SyntaxError. Notebooks and scripts must use `importlib.import_module('lambda.prompt_utils')` to load the module. This is an established constraint to document for all downstream work.
- **Prompt template frozen:** The prompt format string is the training/inference contract. Any change after data generation requires full regeneration + retraining.
- **zip_code as str:** Preserves leading zeros for NJ north-jersey zip codes (e.g., "07650") that would be lost if stored as int.
- **Force-added .gitkeep files:** data/raw/, data/processed/, data/splits/ are in .gitignore (to prevent committing large data files), so .gitkeep files required `git add -f` to be tracked.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] .gitkeep files in ignored directories required force-add**
- **Found during:** Task 2 (project scaffold)
- **Issue:** .gitignore correctly excludes data/raw/, data/processed/, data/splits/ but this also prevents `git add` of .gitkeep files without -f flag
- **Fix:** Used `git add -f` for the .gitkeep files in ignored directories — this is the standard pattern for tracking empty directories
- **Files modified:** data/raw/.gitkeep, data/processed/.gitkeep, data/splits/.gitkeep
- **Verification:** All files appear in git history as tracked
- **Committed in:** 20cf63a (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Force-add for .gitkeep files is the standard git pattern for this use case. No scope change.

**Important note for downstream work:** `lambda` is a Python reserved keyword. Notebooks importing from `lambda/prompt_utils.py` must use:
```python
import importlib, sys
sys.path.insert(0, '/path/to/repo')
pu = importlib.import_module('lambda.prompt_utils')
format_prompt = pu.format_prompt
```
Not: `from lambda.prompt_utils import format_prompt` (SyntaxError)

## Issues Encountered
- Python reserved keyword `lambda` prevents standard import syntax for `lambda/prompt_utils.py`. Verified that `importlib.import_module('lambda.prompt_utils')` works correctly after `sys.path.insert(0, '.')`.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Prompt contract established and verified: `format_prompt()` produces correct output, `parse_price_from_output()` handles comma-formatted numbers
- Project scaffold ready for data generation notebook (Plan 02)
- All downstream notebooks must use `importlib.import_module('lambda.prompt_utils')` for imports — document this pattern in notebook cells
- Data directories created and tracked; data generation can populate data/raw/ and data/splits/

---
*Phase: 01-data-foundation*
*Completed: 2026-02-26*

## Self-Check: PASSED
- All 10 created files verified present on disk
- Task commits verified: e016182 (Task 1), 20cf63a (Task 2)
