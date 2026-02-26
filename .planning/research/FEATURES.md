# Feature Research

**Domain:** ML Housing Price Prediction Pipeline (QLoRA fine-tuned LLM + ONNX + Lambda)
**Researched:** 2026-02-26
**Confidence:** MEDIUM — WebSearch and WebFetch unavailable; findings derived from training knowledge of PEFT, HuggingFace, ONNX Runtime, and AWS Lambda. Core pipeline patterns are mature and stable. Specific version behaviors flagged where uncertainty exists.

---

## Feature Landscape

### Table Stakes (Pipeline Breaks Without These)

Features that must exist for the pipeline to function end-to-end. Missing any one of these blocks downstream stages.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Dataset generation with train/val/test split | ML pipeline requires held-out evaluation sets; without splits, you cannot measure generalization | LOW | 80/10/10 split is standard. Must stratify by price range to avoid distribution skew in small NJ dataset. |
| Natural language prompt formatting | QLoRA fine-tunes a language model, not a tabular model; features must be text | LOW | Template: "Property in zip {zip}, {beds} beds, {baths} baths, {sqft} sqft, {lot} lot, built {year}, type {type}. Price:" |
| QLoRA 4-bit quantization setup | Lambda size constraint requires merged weights < 1GB; 4-bit quantization is the mechanism | MEDIUM | Requires bitsandbytes + PEFT. BitsAndBytesConfig for 4-bit. Must load_in_4bit=True before applying LoRA adapters. |
| LoRA adapter configuration | QLoRA fine-tuning attaches rank-decomposed matrices to transformer attention layers; this is the trainable parameter set | MEDIUM | LoraConfig with r=8 or r=16 typical for 0.5B model. Target modules: q_proj, v_proj at minimum. |
| Regression head or token-based regression | Qwen2.5 is a text generator; it must output a numeric prediction, not a classification label | HIGH | Two approaches: (1) generate price as text token, parse float; (2) add linear regression head on top of last hidden state. Token approach is simpler but fragile. Regression head requires custom model class. |
| Training loop on Colab GPU | Must complete in < 20 min; requires gradient checkpointing, correct batch size, bf16/fp16 | MEDIUM | Colab T4 GPU. Use HuggingFace Trainer or custom loop. Gradient accumulation if batch size constrained by VRAM. |
| MAE, RMSE, R², MAPE metrics | Standard regression evaluation; without these you cannot assess prediction quality | LOW | compute_metrics callback or post-training evaluation. sklearn.metrics covers all four. |
| LoRA weight merging | Before ONNX export, LoRA adapters must be merged back into base model weights | MEDIUM | model.merge_and_unload() in PEFT. Must happen before torch.onnx.export or optimum export. |
| ONNX export of merged model | Lambda cannot run PyTorch inference at 0.5B scale without unacceptable cold start times; ONNX Runtime is required | HIGH | Use HuggingFace Optimum or torch.onnx.export. Qwen2.5 uses rotary embeddings and grouped-query attention — verify ONNX opset compatibility (opset >= 17). Dynamic axes required for variable sequence length. |
| ONNX validation on Colab | Must verify ONNX outputs match PyTorch outputs before deployment | LOW | Compare logits/predictions within tolerance (rtol=1e-3). Run on same test samples used for PyTorch eval. |
| Lambda container image with ONNX Runtime | Lambda function must load and run ONNX model; Python-based container image with onnxruntime package | MEDIUM | Container image, not zip deploy — model won't fit in 250MB zip limit. ECR-hosted image. onnxruntime-gpu not needed for Lambda (CPU inference). |
| REST API endpoint via API Gateway | The product's output surface — without an API, nothing can consume predictions | LOW | Single POST endpoint: /predict. JSON body with property features. HTTP API Gateway (not REST API v1 — cheaper). |
| GitHub Actions CI/CD | Automates training artifact validation and Lambda deployment | MEDIUM | Workflow triggered on push to main. At minimum: lint, test, build container, push to ECR, update Lambda. |
| Terraform infrastructure as code | Lambda + API Gateway + ECR must be provisioned reproducibly | MEDIUM | Provider: aws. Resources: aws_lambda_function, aws_api_gateway_v2_api, aws_ecr_repository, aws_iam_role. |

---

### Differentiators (Learning Value / Competitive Advantage)

Features that demonstrate the QLoRA workflow clearly, make the project educational, or add production robustness. Not required to make the pipeline function, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Training loss and validation loss curves (matplotlib) | Makes training dynamics visible; essential for diagnosing overfitting, underfitting, or learning rate issues | LOW | Log loss per step/epoch from Trainer callbacks. Plot with matplotlib after training completes. |
| Prediction vs actual scatter plot | Visually confirms model quality; strong R² should show tight diagonal cluster | LOW | Save as PNG artifact. Useful for README and understanding where model fails (outlier properties). |
| Residual distribution histogram | Shows error shape — whether bias exists, fat tails, or heteroscedasticity | LOW | Residuals = predicted - actual. Normal distribution centered at 0 = good regression. |
| Per-zip-code error breakdown | NJ housing market varies heavily by region; aggregate RMSE hides zip-level failures | MEDIUM | Group test predictions by zip code, compute per-group MAE. Highlights geographic generalization gaps. |
| Synthetic data generation with realistic NJ statistics | Public NJ datasets are sparse; synthetic augmentation controls distribution and prevents overfitting to small dataset | MEDIUM | Use median home prices by NJ county/zip from public sources (NJ Treasury, Zillow Research data downloads). Gaussian noise around real statistics. Must document generation process clearly. |
| Prompt template versioning | If prompt format changes, model must be retrained; versioning prevents silent mismatches at inference | LOW | Store prompt template as a constant in a config file. Include template version in model metadata. |
| Cold start latency measurement | Lambda cold starts with a 0.5B ONNX model can exceed 10 seconds; knowing actual latency informs whether provisioned concurrency is needed | LOW | Time first invocation separately from warm invocations. Log in Lambda response. |
| Inference confidence via prediction interval | Housing price point predictions are hard to trust alone; providing a ±range increases usability | HIGH | Requires either: (1) ensemble of LoRA adapters with different seeds, (2) MC Dropout, or (3) empirical calibration from validation set errors. Option 3 is simplest: report median absolute error as a heuristic band. |
| Model card / artifact metadata | Documents base model, training data, hyperparameters, metrics; standard practice for reproducibility | LOW | JSON file alongside ONNX model: base_model, qlora_rank, training_samples, val_mae, val_rmse, export_date. |
| Input validation and error messages in API | Lambda handler should return clear 400 errors for missing/invalid fields, not cryptic 500s | LOW | Validate all 7 required fields before running inference. Return field-specific error messages. |

---

### Anti-Features (Deliberately Out of Scope)

Features that look useful but create disproportionate complexity, risk, or scope creep for this project.

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| Real-time Zillow/Realtor scraping | Fresh data = better predictions | TOS violations. Scraping these sites is legally gray and frequently broken by anti-bot measures. Data pipeline becomes unreliable and fragile. | Use static public datasets (NJ Treasury property records, data.gov, Kaggle NJ housing) + synthetic augmentation. Reproducible and legally clean. |
| Web UI / frontend | Makes the project feel more complete | Doubles scope. Frontend is a separate domain. API-first design is the right call for an ML pipeline project. | Document API with curl examples and Postman collection. |
| Multi-state generalization (beyond NJ) | Broader utility | Distribution shift — model trained on NJ data will underperform in CA or TX without retraining. Creates false confidence. Managing multi-state data multiplies complexity. | NJ-only v1. If multi-state is needed later, treat each state as a separate fine-tuned adapter. |
| Real-time model retraining on new data | Model freshness | Requires streaming data pipeline, online learning infrastructure, and continuous validation. None of this is Lambda-compatible. Budget and timeline blown. | Batch retrain on a schedule (monthly) when new data is accumulated. Git-triggered retraining via GitHub Actions. |
| Large base model (7B+ params) | Better accuracy | Won't fit in Lambda 10GB container with dependencies. Colab training will exceed 20-minute budget. Diminishing returns for tabular-style regression. | Qwen2.5-0.5B is the right choice. If accuracy is genuinely insufficient, try 1.5B next — don't jump to 7B. |
| GPU inference on Lambda | Faster inference | Lambda GPU (currently in preview/limited availability) costs dramatically more than free tier. Inference for a 0.5B ONNX model on CPU is adequate (< 5s target). | CPU Lambda + ONNX Runtime. Benchmark first — CPU is likely fast enough. |
| Streaming token output | Feels more LLM-like | This is a regression task, not a chat interface. Streaming a price prediction token-by-token adds complexity with zero user value. | Return JSON with `predicted_price` field. Single synchronous response. |
| Custom ONNX operator implementations | Edge case precision | If the model uses ops not in standard ONNX opset, the temptation is to write custom ops. This path is a maintenance nightmare. | If ONNX export fails due to unsupported ops, use optimum-cli or switch to opset 17+ which covers most modern transformer ops. |
| MLflow / MLOps platform integration | "Production ML" credibility | Adds infrastructure complexity (MLflow server, experiment tracking DB) that exceeds the project's training budget. GitHub Actions logs are sufficient. | Log metrics as JSON artifacts in GitHub Actions. Simple, searchable, no additional infrastructure. |

---

## Feature Dependencies

```
[Dataset Generation + Splitting]
    └──requires──> [Prompt Formatting]
                       └──requires──> [QLoRA Training Setup]
                                          └──requires──> [Training Loop on Colab GPU]
                                                             └──requires──> [MAE/RMSE/R²/MAPE Evaluation]
                                                                                └──requires──> [LoRA Weight Merging]
                                                                                                   └──requires──> [ONNX Export]
                                                                                                                      └──requires──> [ONNX Validation on Colab]
                                                                                                                                         └──requires──> [Lambda Container Image]
                                                                                                                                                            └──requires──> [REST API Endpoint]

[Terraform Infrastructure]
    └──requires──> [Lambda Container Image] (image must exist in ECR before Lambda can reference it)

[GitHub Actions CI/CD]
    └──requires──> [Terraform Infrastructure] (pipeline deploys to existing infra)
    └──requires──> [ONNX Validation on Colab] (CI validates artifacts before deploying)

[Training Loss + Val Loss Curves] ──enhances──> [Training Loop on Colab GPU]
[Scatter Plot + Residuals] ──enhances──> [MAE/RMSE/R²/MAPE Evaluation]
[Per-Zip-Code Error Breakdown] ──enhances──> [MAE/RMSE/R²/MAPE Evaluation]
[Prompt Template Versioning] ──enhances──> [Prompt Formatting]
[Cold Start Latency Measurement] ──enhances──> [Lambda Container Image]
[Input Validation in API] ──enhances──> [REST API Endpoint]
[Model Card / Artifact Metadata] ──enhances──> [ONNX Export]

[Synthetic Data Generation] ──enhances──> [Dataset Generation + Splitting]

[Regression Head vs Token Regression] ──conflicts──> each other
    Note: Choose ONE approach at training setup time. Token-based regression is simpler
    but requires output parsing at inference. Regression head is cleaner but requires
    custom model class and complicates ONNX export. Token approach recommended for v1.
```

### Dependency Notes

- **ONNX Export requires LoRA Weight Merging:** You cannot export a model with detached LoRA adapters to ONNX. merge_and_unload() must be called first, producing a standard transformer checkpoint.
- **Lambda Container Image requires ONNX Validation:** Deploy only a validated model. Pushing a broken ONNX artifact to Lambda wastes deploy cycles and makes debugging harder.
- **Terraform Infrastructure requires Container Image in ECR:** The aws_lambda_function resource references a specific ECR image URI. The image must be pushed before `terraform apply` can succeed (or use a placeholder image for initial provisioning).
- **GitHub Actions CI/CD requires all of the above:** CI is the last thing wired up. It orchestrates the full pipeline but depends on all prior stages existing and working.
- **Synthetic Data enhances Dataset Generation:** NJ public housing datasets are likely sparse (hundreds to low thousands of records). Synthetic data generation multiplies training samples while preserving realistic distribution. Dependency is soft — pipeline works without it, but model quality may suffer.
- **Token regression conflicts with Regression head:** These are two incompatible training paradigms. Token regression treats the price as a generated text string. Regression head adds a linear layer on top of the LLM's final hidden state. Pick token regression for v1 (less ONNX complexity).

---

## MVP Definition

### Launch With (v1) — Core Pipeline End-to-End

The v1 goal is a working end-to-end pipeline that demonstrates QLoRA fine-tuning through to Lambda inference. Every item below is a hard dependency.

- [ ] **Dataset generation with train/val/test split** — pipeline has nothing to train on without data
- [ ] **Prompt formatting (text template for 7 features)** — language model input format; must be finalized before training
- [ ] **QLoRA 4-bit training on Colab GPU (< 20 min)** — the core learning objective of the project
- [ ] **MAE, RMSE, R², MAPE evaluation metrics** — cannot assess model quality without these
- [ ] **Training loss and validation loss curves** — essential for diagnosing training; LOW complexity, HIGH signal
- [ ] **LoRA weight merging** — prerequisite for ONNX export
- [ ] **ONNX export and Colab validation** — validates the model is correctly exported before deployment
- [ ] **Lambda container image with ONNX Runtime** — production inference environment
- [ ] **REST API endpoint via API Gateway** — the externally consumable product
- [ ] **Terraform infrastructure as code** — reproducible deployment
- [ ] **GitHub Actions CI/CD** — automation of the deploy pipeline

### Add After Validation (v1.x)

Features to add once the pipeline is confirmed working end-to-end.

- [ ] **Prediction vs actual scatter plot** — add when base evaluation is working; improves interpretability
- [ ] **Residual distribution histogram** — add alongside scatter plot; < 1 hour to add
- [ ] **Synthetic data augmentation** — add if validation MAE is poor; helps if public dataset is too small
- [ ] **Per-zip-code error breakdown** — add when aggregate metrics are passing; surfaces geographic gaps
- [ ] **Input validation and error messages in API** — add when Lambda is deployed; hardens the API for real use
- [ ] **Cold start latency measurement** — add after first successful Lambda invocation; informs provisioned concurrency decision
- [ ] **Model card / artifact metadata JSON** — add alongside ONNX export; documents reproducibility

### Future Consideration (v2+)

Defer until v1 is stable and the learning objectives are confirmed met.

- [ ] **Prediction intervals / confidence bands** — defer: HIGH complexity, requires ensemble or calibration work
- [ ] **Prompt template versioning system** — defer: LOW risk in v1 since template won't change; add if iterating on prompt design
- [ ] **Monthly retrain pipeline** — defer: requires stable data pipeline and production monitoring first
- [ ] **Multi-state support** — defer: scope expansion; NJ-only is the explicit v1 boundary

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Dataset generation + splits | HIGH | LOW | P1 |
| Prompt formatting | HIGH | LOW | P1 |
| QLoRA 4-bit training setup | HIGH | MEDIUM | P1 |
| MAE/RMSE/R²/MAPE metrics | HIGH | LOW | P1 |
| LoRA weight merging | HIGH | LOW | P1 |
| ONNX export + validation | HIGH | HIGH | P1 |
| Lambda container + ONNX Runtime | HIGH | MEDIUM | P1 |
| REST API endpoint | HIGH | LOW | P1 |
| Terraform IaC | HIGH | MEDIUM | P1 |
| GitHub Actions CI/CD | HIGH | MEDIUM | P1 |
| Training loss/val loss curves | MEDIUM | LOW | P1 |
| Scatter plot + residuals | MEDIUM | LOW | P2 |
| Synthetic data augmentation | MEDIUM | MEDIUM | P2 |
| Per-zip-code error breakdown | MEDIUM | LOW | P2 |
| Input validation in API | MEDIUM | LOW | P2 |
| Cold start latency measurement | LOW | LOW | P2 |
| Model card / artifact metadata | LOW | LOW | P2 |
| Prediction intervals | MEDIUM | HIGH | P3 |
| Prompt template versioning | LOW | LOW | P3 |
| Monthly retrain pipeline | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch — pipeline broken without it
- P2: Should have — add during or immediately after core pipeline
- P3: Nice to have — future iteration

---

## Competitor Feature Analysis

This is a learning/portfolio project rather than a commercial product, so "competitors" here means comparable public ML pipeline projects and tutorials.

| Feature | Typical HF Tutorial | Typical Kaggle Notebook | This Project's Approach |
|---------|---------------------|-------------------------|-------------------------|
| Data source | Provided dataset | Provided dataset | Public NJ data + synthetic generation |
| Model | General LLM | scikit-learn / XGBoost | QLoRA fine-tuned Qwen2.5-0.5B (LLM for regression) |
| Training environment | Local or cloud | Kaggle GPU | Google Colab (free tier, < 20 min) |
| Evaluation | Basic accuracy/loss | MAE, RMSE | MAE, RMSE, R², MAPE + visual plots |
| Inference export | PyTorch checkpoint | Model pickle | ONNX (framework-agnostic, optimized) |
| Serving | None / local script | None | AWS Lambda + API Gateway (REST API) |
| Infrastructure | None | None | Terraform (IaC) |
| Automation | None | None | GitHub Actions CI/CD |
| Documentation | Notebook comments | Notebook comments | Model card JSON + reproducible pipeline |

**Key differentiator:** The combination of LLM fine-tuning (QLoRA) with ONNX export and Lambda serving is unusual. Most housing price tutorials use XGBoost or sklearn. Using a language model for tabular regression and getting it to production on Lambda is the learning-value differentiator.

---

## Sources

- Training knowledge of PEFT/HuggingFace QLoRA patterns (MEDIUM confidence — stable API since 2023, but verify current merge_and_unload behavior for Qwen2.5 architecture)
- Training knowledge of ONNX Runtime and torch.onnx.export patterns (MEDIUM confidence — Qwen2.5 rotary embedding ops should be in opset 17+, but verify opset compatibility before export)
- Training knowledge of AWS Lambda container image deployment limits (HIGH confidence — 10GB limit is documented AWS constraint, unlikely to change)
- Training knowledge of GitHub Actions + Terraform patterns (HIGH confidence — these are stable, well-documented workflows)
- PROJECT.md requirements and constraints (HIGH confidence — source of truth for this project)

**Flags for phase-specific verification:**
- Qwen2.5-0.5B ONNX export: verify opset version and dynamic axis configuration against current HuggingFace Optimum docs before implementing
- Token regression vs regression head: verify which approach produces a simpler ONNX graph for Qwen2.5 architecture specifically
- Lambda cold start with ONNX 0.5B: benchmark actual cold start times before deciding on provisioned concurrency

---
*Feature research for: NJ Housing Price Predictor (QLoRA + ONNX + Lambda ML Pipeline)*
*Researched: 2026-02-26*
