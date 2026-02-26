# Requirements: NJ Housing Price Predictor

**Defined:** 2026-02-26
**Core Value:** Accurately predict NJ housing prices from 7 property features using a QLoRA fine-tuned Qwen2.5-0.5B, demonstrating the full ML pipeline from training to production inference.

## v1 Requirements

### Data Pipeline

- [ ] **DATA-01**: Generate NJ housing dataset with 7 features (bedrooms, bathrooms, sqft, lot size, year built, zip code, property type)
- [ ] **DATA-02**: Create train/validation/test splits (70/15/15)
- [ ] **DATA-03**: Implement shared `format_prompt()` function for text-formatting tabular features
- [ ] **DATA-04**: Generate synthetic data with county-level NJ price distributions + source public datasets

### Model Training

- [ ] **TRAIN-01**: Fine-tune Qwen2.5-0.5B with QLoRA (4-bit quantization) on Google Colab GPU
- [ ] **TRAIN-02**: Complete training within 20 minutes on Colab free tier

### Evaluation

- [ ] **EVAL-01**: Compute regression metrics on test set: MAE, RMSE, R², MAPE
- [ ] **EVAL-02**: Generate predicted vs actual scatter plot with matplotlib
- [ ] **EVAL-03**: Generate training loss curve with matplotlib

### ONNX Export

- [ ] **ONNX-01**: Merge LoRA weights into base model (fp32 reload → merge_and_unload)
- [ ] **ONNX-02**: Export merged model to ONNX format via optimum
- [ ] **ONNX-03**: Validate ONNX numerical accuracy against PyTorch output on Colab

### Lambda Serving

- [ ] **SERV-01**: Implement ONNX Runtime inference handler for price prediction
- [ ] **SERV-02**: Expose REST API endpoint accepting 7 property features, returning predicted price
- [ ] **SERV-03**: Build minimal container image (onnxruntime + tokenizer only, no PyTorch)

### Infrastructure

- [ ] **INFRA-01**: Terraform configuration for Lambda + API Gateway + ECR
- [ ] **INFRA-02**: S3 backend for Terraform state
- [ ] **INFRA-03**: GitHub Actions workflow for lint/test on PR and Terraform plan/apply on merge

## v2 Requirements

### Evaluation Enhancements

- **EVAL-04**: Residual distribution plot
- **EVAL-05**: Feature importance analysis
- **EVAL-06**: Before/after fine-tuning comparison (base vs fine-tuned)

### Training Enhancements

- **TRAIN-03**: Google Drive checkpointing for Colab disconnect protection
- **TRAIN-04**: Learning rate scheduling for better convergence
- **TRAIN-05**: Hyperparameter experimentation framework

### Serving Enhancements

- **SERV-04**: Input validation and error handling on API endpoint
- **SERV-05**: Health check endpoint

### Monitoring

- **INFRA-04**: CloudWatch logging and monitoring for Lambda
- **INFRA-05**: Model artifact versioning

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-state predictions | NJ only for v1 — keeps dataset focused and manageable |
| Real-time model retraining | Batch training only — Lambda is for inference, not training |
| Web UI / frontend | API-only for v1 — UI adds complexity without core value |
| Zillow/Realtor.com scraping | TOS concerns — use public datasets + synthetic instead |
| Large LLM (7B+ params) | Lambda size constraints + Colab training time budget |
| Automated Colab retraining | CI/CD triggers for retraining are complex — defer to v2+ |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| ONNX-01 | Phase 3 | Pending |
| ONNX-02 | Phase 3 | Pending |
| ONNX-03 | Phase 3 | Pending |
| SERV-01 | Phase 4 | Pending |
| SERV-02 | Phase 4 | Pending |
| SERV-03 | Phase 4 | Pending |
| INFRA-01 | Phase 5 | Pending |
| INFRA-02 | Phase 5 | Pending |
| INFRA-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 (complete)

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after roadmap creation*
