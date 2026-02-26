# NJ Housing Price Predictor

## What This Is

A machine learning system that predicts New Jersey housing prices using a QLoRA fine-tuned Qwen2.5-0.5B model. Property features are formatted as natural language prompts, the model is trained on Google Colab with GPU, exported to ONNX for inference, and served via a REST API on AWS Lambda. This is both a learning exercise for the LoRA/QLoRA workflow and a functional housing price prediction tool.

## Core Value

Accurately predict NJ housing prices from property features (bedrooms, bathrooms, sqft, lot size, year built, zip code, property type) using a fine-tuned LLM with LoRA, demonstrating the full ML pipeline from training to production inference.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Generate and curate NJ housing dataset (real public data + synthetic) with train/val/test splits
- [ ] Fine-tune Qwen2.5-0.5B with QLoRA (4-bit quantization) on Google Colab GPU
- [ ] Format 7 property features as natural language prompts for model input
- [ ] Visualize training progress and model performance with matplotlib
- [ ] Track regression metrics: MAE, RMSE, R², MAPE
- [ ] Export fine-tuned model to ONNX format
- [ ] Run ONNX inference on Google Colab to validate
- [ ] Serve ONNX model via REST API on AWS Lambda (free tier)
- [ ] Deploy infrastructure with Terraform (Lambda + API Gateway)
- [ ] Automate CI/CD with GitHub Actions

### Out of Scope

- Multi-state predictions — NJ only for v1
- Real-time model retraining — batch training only
- Web UI / frontend — API-only for v1
- Zillow/Realtor.com scraping — TOS concerns, use public datasets + synthetic instead
- Large LLM (7B+ params) — Lambda size constraints, training time budget

## Context

- **Base Model**: Qwen2.5-0.5B (Alibaba, late 2024, Apache 2.0 license)
- **Fine-tuning**: QLoRA with 4-bit quantization via PEFT/bitsandbytes
- **Training Environment**: Google Colab free tier GPU, target < 20 min training time
- **Inference**: ONNX Runtime on AWS Lambda (merged LoRA weights exported to ONNX)
- **Data Features**: bedrooms, bathrooms, sqft, lot size, year built, zip code, property type
- **Data Sources**: Public NJ housing datasets (data.gov, Kaggle) combined with realistic synthetic data based on NJ market statistics
- **Framework**: PyTorch + HuggingFace transformers/PEFT
- **Approach**: Format tabular features as text prompts → fine-tune LLM for regression → merge LoRA weights → export ONNX → deploy Lambda

## Constraints

- **Training Time**: < 20 minutes on Colab free tier GPU — drives model size choice (0.5B params)
- **Lambda Size**: 10GB container image limit — ONNX model must fit comfortably (target < 1GB)
- **Lambda Timeout**: 15 min max — inference must be fast (target < 5 seconds)
- **AWS Budget**: Free tier only — Lambda + API Gateway free tier limits
- **Colab First**: All training and validation must run on Google Colab before any production deployment
- **License**: Base model must be free and open-source (Apache 2.0 / MIT)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Qwen2.5-0.5B as base model | Modern (2024), small enough for Colab + Lambda, Apache 2.0, great HF support | — Pending |
| Text-formatted features (not tabular) | Authentic LoRA/QLoRA workflow, transferable skill, demonstrates LLM fine-tuning | — Pending |
| ONNX for inference (not PyTorch) | Smaller artifact, faster inference, framework-agnostic, fits Lambda well | — Pending |
| GitHub Actions over Argo | Simpler, generous free tier, code already on GitHub, sufficient for this pipeline | — Pending |
| Synthetic + public data (no scraping) | Avoids TOS issues with Zillow/Realtor, reproducible, controllable distribution | — Pending |

---
*Last updated: 2026-02-26 after initialization*
