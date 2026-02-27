---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-02-27T18:22:48.628Z"
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 9
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Accurately predict NJ housing prices from 7 property features using a QLoRA fine-tuned Qwen2.5-0.5B, demonstrating the full ML pipeline from training to production inference.
**Current focus:** Phase 3 Plan 2 - ONNX export notebook created; awaiting Colab execution checkpoint

## Current Position

Phase: 3 of 5 (Evaluation and ONNX Export) — In Progress
Plan: 2 of 2 in phase 03 — Checkpoint reached (Task 2 awaiting Colab execution)
Status: 04_export.ipynb committed (48f24bc); checkpoint awaiting human Colab execution to verify fp32 merge, ONNX export, and atol=1e-3 numerical validation
Last activity: 2026-02-27 — Completed 03-02 Task 1 (ONNX export notebook 14 cells); checkpoint at Task 2 (Colab verification)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: ~11 min
- Total execution time: ~22 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-foundation | 2 | ~22 min | ~11 min |
| 02-qlora-training | 1 | ~140 min (Colab) | ~140 min |
| 04-lambda-container-and-rest-api | 1 | ~8 min | ~8 min |
| 05-infrastructure-and-ci-cd | 2 | ~6 min | ~3 min |

**Recent Trend:**
- Last 5 plans: ~2 min, ~20 min, ~5 min, ~8 min, ~4 min
- Trend: Stable

| 03-evaluation-and-onnx-export | 2 | ~13 min | ~6.5 min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Setup]: Qwen2.5-0.5B chosen — modern (2024), fits Colab + Lambda, Apache 2.0
- [Setup]: ONNX for inference — smaller artifact, faster Lambda cold start, no PyTorch in container
- [Setup]: Synthetic + public data — avoids TOS scraping issues; must use county-level price priors
- [01-01]: lambda/ named after AWS Lambda but clashes with Python reserved keyword — import must use importlib.import_module('lambda.prompt_utils'), not from-import syntax
- [01-01]: Prompt template in format_prompt() is FINAL training/inference contract — any change requires full data regeneration + retraining
- [01-01]: zip_code typed as str to preserve leading zeros for NJ zip codes (e.g., '07650')
- [01-02]: importlib.import_module used for lambda.prompt_utils import (lambda is Python reserved keyword — from-import raises SyntaxError)
- [01-02]: Price-first synthetic generation: county log-normal price generated first, features derived from price for realistic correlations
- [01-02]: Zip code is 3-char county prefix + 2-char random suffix = 5-char string (not 3+3=6)
- [01-02]: SR1A column name constants in Cell 6 are PLACEHOLDERS — user must update to match actual SR1A file headers after download
- [01-02]: Cell 9 added post-checkpoint to push dataset to HuggingFace Hub (rajkumar4466/nj-housing-prices) with graceful auth fallback
- [02-01]: prepare_model_for_kbit_training must appear before get_peft_model in peft imports AND in call order — verification script checks raw string position
- [02-01]: fp16 caused _amp_foreach_non_finite_check_and_unscale_cuda on T4 — bf16=True resolved it despite T4 not officially supporting bf16; use bf16 in practice
- [02-01]: packing=False in SFTTrainer — housing records must not be concatenated; each is one training sample
- [02-01]: optim=paged_adamw_8bit required for QLoRA memory efficiency on Colab free tier
- [02-01]: TRAIN-02 20-min budget not met (133.7 min actual on 4,900 records / T4 free tier) — budget assumes much smaller dataset; document as known limitation
- [02-01]: Data loaded from HuggingFace Hub (rajkumar4466/nj-housing-prices) in Colab — avoids manual file upload; works because Phase 1 pushed dataset to Hub
- [04-01]: Inside Lambda container /var/task/, use 'import prompt_utils' directly (sibling); importlib.import_module('lambda.prompt_utils') only needed at project-root level in notebooks
- [04-01]: MAX_NEW_TOKENS=12 for autoregressive loop — prices are at most 7 digits, buffer added
- [04-01]: tokenizer and session must be module-level globals for Lambda warm-start reuse
- [04-01]: lambda/model_artifacts/ gitignored — Qwen2.5-0.5B ONNX is 500MB+, exceeds GitHub 100MB limit
- [Phase 05-infrastructure-and-ci-cd]: use_lockfile=true for S3 backend locking (DynamoDB locking deprecated in Terraform 1.10+)
- [Phase 05-infrastructure-and-ci-cd]: thumbprint_list=[] for GitHub OIDC provider (AWS trusts GitHub CA natively since December 2024)
- [Phase 05-infrastructure-and-ci-cd]: payload_format_version=2.0 on API Gateway integration (required for Mangum event parsing)
- [Phase 05-infrastructure-and-ci-cd]: Lambda function omits handler and runtime fields (invalid for package_type=Image)
- [05-02]: CI workflow has no AWS credentials — unit tests mock ONNX/model; no cloud access needed for lint+test
- [05-02]: Deploy workflow tags Docker image with github.ref_name (git tag) AND latest — git tag makes image_uri change on every release so Terraform always triggers Lambda UpdateFunctionCode
- [05-02]: linux/amd64 platform flag mandatory — Lambda runs x86_64; ARM runner without this flag produces incompatible images
- [05-02]: GH_ACTIONS_ROLE_ARN stored as GitHub Actions Variable (not Secret) — role ARNs are not sensitive credentials
- [Phase 03-evaluation-and-onnx-export]: parse_price_from_output inlined in 03_evaluate.ipynb Cell 7 (verbatim copy of lambda/prompt_utils.py) for Colab portability -- no repo clone required
- [Phase 03-evaluation-and-onnx-export]: 200-sample quick eval runs before full test set in 03_evaluate.ipynb -- fast early feedback signal before 17-52 min full eval on T4
- [Phase 03-evaluation-and-onnx-export]: fp32 two-step merge: reload base model with torch_dtype=torch.float32 and device_map=cpu BEFORE loading adapter -- quantized merge silently corrupts weights
- [Phase 03-evaluation-and-onnx-export]: text-generation-with-past is the correct ONNX task for autoregressive causal LM (NOT feature-extraction)
- [Phase 03-evaluation-and-onnx-export]: Dynamo fallback cell with --dynamo --opset 18 included for RoPE exporter compatibility issues

### Pending Todos

- Download SR1A 2024 from NJ Treasury, update Cell 6 column constants, rerun notebook to meet DATA-04 (30% real records) before final production training run

### Blockers/Concerns

- [Phase 1 - Open]: DATA-04 30% real records not yet met — current splits are synthetic-only (7,000 records). SR1A download + column mapping required.
- [Phase 3]: Correct `optimum-cli export onnx --task` flag for Qwen2.5 regression needs empirical verification (may be `feature-extraction` or `text-generation-with-past`)
- [01-01]: Downstream notebooks must use importlib.import_module('lambda.prompt_utils') — document pattern clearly in notebook cell comments

## Session Continuity

Last session: 2026-02-27
Stopped at: Checkpoint in 03-02-PLAN.md Task 2 — 04_export.ipynb created (48f24bc), awaiting Colab execution to verify fp32 merge + ONNX export + atol=1e-3 validation (ONNX-01/02/03)
Resume file: None
