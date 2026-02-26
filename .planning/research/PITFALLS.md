# Pitfalls Research

**Domain:** QLoRA fine-tuning + ONNX export + Lambda serving (NJ Housing Price Predictor)
**Researched:** 2026-02-26
**Confidence:** MEDIUM-HIGH (official HuggingFace PEFT docs + AWS Lambda docs verified; some Lambda/ONNX specifics from official sources; Colab gotchas from training knowledge)

---

## Critical Pitfalls

### Pitfall 1: Merging QLoRA Weights Fails Because Base Model is Still 4-bit Quantized

**What goes wrong:**
You train with QLoRA (4-bit bitsandbytes), then try to call `model.merge_and_unload()` on the still-quantized base model. The merge either errors out or silently produces garbage weights because you cannot dequantize bitsandbytes 4-bit tensors back to full precision in-place. The resulting merged model has corrupted weights and ONNX export produces a broken model.

**Why it happens:**
Developers conflate "the adapter is trained" with "the model is ready to export." The QLoRA workflow requires a **two-step save process**: save the adapter separately, then reload the base model in full precision (fp32 or bf16), load the adapter on top, and *then* merge. Skipping the reload step is the single most common QLoRA export failure.

**How to avoid:**
Use the correct two-step merge pattern:
```python
# Step 1: Save only the adapter after training
model.save_pretrained("./lora_adapter/")
tokenizer.save_pretrained("./lora_adapter/")

# Step 2: Reload base model WITHOUT quantization, then merge
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,  # NOT load_in_4bit
    device_map="cpu",            # CPU is fine for merge
)
merged = PeftModel.from_pretrained(base, "./lora_adapter/")
merged = merged.merge_and_unload()
merged.save_pretrained("./merged_model/")
```
Do this merge step on Colab before ONNX export, not in the Lambda container.

**Warning signs:**
- `merge_and_unload()` raises a CUDA or dtype error
- Merged model produces identical output for all inputs (sign of zero weights)
- ONNX validation step shows large numerical difference (atol check fails)

**Phase to address:** Data + Training phase (Phase 2) — build merge step into training notebook from day one.

---

### Pitfall 2: ONNX Export of a Causal LM Without Past Key/Values Makes Lambda Inference Too Slow

**What goes wrong:**
You export the merged Qwen2.5-0.5B using `optimum-cli export onnx --task text-generation` (without `--task text-generation-with-past`). Inference on Lambda works but runs token-by-token recomputing all previous attention states from scratch, making a 200-token generation take 30+ seconds — well over the 5-second target and approaching Lambda's 15-minute timeout for cold starts under load.

**Why it happens:**
The `text-generation` task exports without KV-cache reuse. The `text-generation-with-past` task exports two ONNX files (decoder + decoder-with-past) that support caching. Developers use the shorter task name or don't know the `-with-past` suffix exists.

**How to avoid:**
Always export with past key/values for generation tasks:
```bash
optimum-cli export onnx \
  --model ./merged_model/ \
  --task text-generation-with-past \
  ./onnx_model/
```
This produces `model.onnx` and `model_with_past.onnx`. Use `ORTModelForCausalLM` which handles the split automatically. For a regression task (single forward pass to get logits, not autoregressive generation), use `--task feature-extraction` or a custom config — this avoids the KV-cache complexity entirely and is the correct approach for this project.

**Warning signs:**
- Single inference call takes >2 seconds on Colab CPU
- Lambda execution log shows timeout on warm invocations
- ONNX model file is unexpectedly small (missing the `-with-past` model)

**Phase to address:** ONNX Export phase (Phase 3) — validate inference latency on Colab CPU before Lambda deployment.

---

### Pitfall 3: LLM Used for Regression Without a Numeric Output Head — Loss Never Converges

**What goes wrong:**
You fine-tune the LLM with `task_type="CAUSAL_LM"` and train it to generate a text string like `"450000"`. At inference, you parse the generated number from the output token sequence. This is fragile: the model sometimes generates `"$450,000"`, `"450k"`, `"approximately 450000"`, or refuses to answer entirely. The output is unparseable 15-30% of the time on held-out test sets.

**Why it happens:**
Using generation for regression is indirect. The model is optimizing cross-entropy over tokens, not mean squared error over a numeric value. With small datasets (< 10k samples), the model often hasn't learned that it must generate exactly one parseable number.

**How to avoid:**
Use one of two approaches (pick at training time, not after):
1. **Token parsing with strict prompt format**: Train with a prompt template that ends with `"The price is: $"` and parse exactly the next N tokens. Add a post-processing fallback that extracts the first sequence of digits. Validate this on 100 training examples before committing to training.
2. **Sequence classification head**: Use `task_type="SEQ_CLS"` with a regression head (single output neuron). This changes the ONNX export task to `text-classification`. Simpler, more reliable, but slightly different from the "true LLM generation" workflow.

For this project, approach 1 with strict prompt engineering is recommended since the goal is to demonstrate the full LoRA text-generation workflow.

**Warning signs:**
- Validation MAPE > 40% despite low training loss
- Model generates multiple numbers or non-numeric text during eval
- `int(output_text)` raises `ValueError` more than 5% of the time

**Phase to address:** Data + Training phase (Phase 2) — define the output format and parsing logic *before* writing training code.

---

### Pitfall 4: Colab Free Tier Session Disconnect Kills Training Mid-Run

**What goes wrong:**
Google Colab free tier disconnects after 90 minutes of inactivity or 12 hours of total runtime. If training is in-progress without checkpointing, you lose all progress and must restart from scratch. With a 20-minute training target, this is usually fine — but loading the model, tokenizing the dataset, and running training can push past 20 minutes if not optimized.

**Why it happens:**
Colab free tier has hard session limits. The T4 GPU is not guaranteed — you may be assigned a slower GPU or get queue-delayed. Model downloads from HuggingFace Hub count toward session time.

**How to avoid:**
- Mount Google Drive and save checkpoints every epoch: `trainer.save_model("/content/drive/MyDrive/checkpoints/")`
- Cache the tokenized dataset to Drive: `dataset.save_to_disk("/content/drive/MyDrive/tokenized_dataset")`
- Download the base model to Drive on first run, load from Drive on subsequent runs
- Use `TrainingArguments(save_steps=50, save_total_limit=2, ...)`
- Test with 1 epoch before running full training

**Warning signs:**
- Training takes > 15 minutes to reach the first checkpoint
- No `save_steps` configured in `TrainingArguments`
- Model not downloaded to Drive before starting a long run

**Phase to address:** Data + Training phase (Phase 2) — set up Drive mounting and checkpointing as the first cell in the training notebook.

---

### Pitfall 5: ONNX Model File Exceeds Lambda 10GB Container Limit

**What goes wrong:**
The merged Qwen2.5-0.5B model in fp32 is approximately 2GB on disk. The ONNX file is similar. The Lambda container image includes the ONNX model, ONNX Runtime (~200MB), Python packages (~500MB), and the Lambda runtime. Total easily reaches 3-4GB, which is under the 10GB limit — but if you accidentally export in fp32 and include redundant packages, it can bloat. More critically, Lambda only supports containers up to 10GB *uncompressed*, and ECR image layers count separately.

**Why it happens:**
Developers include full PyTorch in the Lambda container (for "just in case" use), which adds 2-3GB to the image. ONNX Runtime alone is sufficient for inference — PyTorch is not needed at serving time.

**How to avoid:**
- Lambda container should have: `onnxruntime-cpu`, `transformers` (tokenizer only), `numpy`, `boto3`, `mangum`. Nothing else.
- Export ONNX in fp16 if model quality is acceptable: `optimum-cli export onnx --dtype fp16 ...`
- Use multi-stage Docker build: build stage installs everything, final stage copies only what's needed
- Target image size: under 2GB uncompressed
- Validate image size locally with `docker image ls` before pushing to ECR

**Warning signs:**
- `pip install torch` or `pip install transformers[torch]` is in the Dockerfile
- Docker image size > 3GB
- ECR push timing out

**Phase to address:** Lambda Serving phase (Phase 4) — define the minimal dependency set in Dockerfile before writing any Lambda code.

---

### Pitfall 6: Synthetic Housing Data Has Unrealistic Price Distribution That Breaks Generalization

**What goes wrong:**
You generate synthetic NJ housing data using a simple formula (e.g., `price = sqft * 300 + bedrooms * 10000 + noise`). The model trains to near-zero loss on training data but achieves MAPE > 50% on real-world test data. The distribution of synthetic prices doesn't match actual NJ market statistics — median prices, zip code variance, lot size correlations, and seasonal patterns are all wrong.

**Why it happens:**
Synthetic data generation is treated as a simple data augmentation step rather than a market modeling exercise. NJ housing prices vary enormously by county (Hudson/Bergen county median ~$600k vs. Salem county ~$200k). A uniform distribution across zip codes destroys the model's ability to learn the zip-code price signal.

**How to avoid:**
- Use actual NJ county-level median price statistics as priors for synthetic data generation
- Generate price as a log-normal distribution (housing prices are log-normally distributed)
- Assign zip codes to counties and use county-level price multipliers
- Validate that synthetic data price distribution (mean, std, 25th/75th percentile) matches public NJ real estate statistics
- Use at least 30% real public data (data.gov NJ property sales) alongside synthetic data
- Plot price histograms of synthetic vs. real data before training

**Warning signs:**
- Synthetic data price mean differs by > 20% from known NJ median (~$450k)
- All zip codes have similar price distributions
- Model MAE on synthetic validation set << MAE on any real-world example you manually check

**Phase to address:** Data Generation phase (Phase 1) — validate data distribution against known statistics before creating train/val/test splits.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip `prepare_model_for_kbit_training()` before adding LoRA | One less line of code | Gradient checkpointing not enabled; higher VRAM usage; may silently train incorrectly on quantized model | Never — always call this |
| Export ONNX without validating numerical output | Saves 5 minutes | Lambda serves wrong predictions silently; no error raised | Never — always run atol validation |
| Use `text-generation` task instead of `feature-extraction` for regression | Simpler export | Token parsing failures at inference; non-deterministic outputs | Never for a regression API |
| Skip Drive checkpointing in Colab | Cleaner notebook | Lose all training progress on disconnect | Never on free tier |
| Include PyTorch in Lambda container | Can do ad-hoc testing | Image size bloat (+2-3GB); slower cold starts | Only during development (remove before production) |
| Generate 100% synthetic data (no real data) | No data sourcing work | Poor generalization; distribution mismatch with real NJ market | Only for initial smoke tests |
| Hard-code sequence length at export time | Simpler ONNX graph | Breaks at inference if prompt length differs from export length | Only if you fully control all inference input lengths |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HuggingFace PEFT + bitsandbytes | Call `get_peft_model()` before `prepare_model_for_kbit_training()` | Always call `prepare_model_for_kbit_training(model)` first, then `get_peft_model(model, config)` |
| Optimum ONNX export | Specify `--task text-generation` for a regression model | For regression (single forward pass for a numeric value), use `--task feature-extraction` or a custom `ORTModelForSequenceClassification` config |
| Qwen2.5 tokenizer on Lambda | Ship the model's tokenizer config inside the container | Tokenizer files (vocab, merges, special tokens) must be in the container; they are NOT loaded from HuggingFace at runtime |
| AWS Lambda + ONNX Runtime | Use `onnxruntime` (includes GPU deps) | Use `onnxruntime-cpu` — Lambda has no GPU; the GPU version is larger and errors on import |
| Terraform + ECR | Push image before applying Lambda Terraform | ECR repository must exist before Lambda function resource references the image URI; use `depends_on` in Terraform |
| GitHub Actions + Colab | Try to automate Colab training via CI | Colab is not automatable from GitHub Actions; training is manual; CI/CD should cover Lambda deployment only |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading tokenizer from HuggingFace Hub at Lambda cold start | First invocation takes 10-60s due to network download | Bundle tokenizer files in the container image | Every cold start |
| ONNX Runtime session created per request (not shared) | Each invocation re-loads the model (3-5s overhead) | Create `InferenceSession` as a module-level global in Lambda handler, initialized once on cold start | Every warm invocation without session reuse |
| Large ONNX model not cached in `/tmp` | Repeated S3 downloads (if model is stored in S3) | Use `InferenceSession` global (model already in memory); if using S3, download to `/tmp` on cold start only | With S3-backed model storage |
| Using fp32 ONNX on Lambda 3008MB memory | OOM errors during model load | Export in fp16 or use `onnxruntime` optimization level O2 | Lambda functions with < 3GB memory configured |
| Tokenizing full 2048-token context at inference | Inference latency 2-10x higher than needed | Cap input to the actual prompt length (typically < 128 tokens for 7-feature housing prompt) | Single-property inference should be < 128 tokens |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Hardcode AWS credentials in Dockerfile or Lambda handler | Credential exposure in ECR image or GitHub history | Use IAM roles for Lambda execution; never use AWS_ACCESS_KEY in container |
| Return raw model logits or internal errors in API response | Information leakage about model internals | Return only `{"predicted_price": 450000, "confidence": "medium"}` — never stack traces |
| No input validation on Lambda API | Prompt injection or resource exhaustion via very long inputs | Validate all 7 input fields (type, range) before building the text prompt; cap prompt length before tokenization |
| Expose raw Terraform state with AWS account IDs | Account enumeration | Use S3 remote state with encryption; never commit `terraform.tfstate` |

---

## UX Pitfalls

*(This project is API-only, so UX pitfalls relate to API developer experience)*

| Pitfall | Developer Impact | Better Approach |
|---------|-----------------|-----------------|
| Return price as a raw float with 10 decimal places | Confusing output; useless precision | Round to nearest $1000: `round(price / 1000) * 1000` |
| No confidence or quality signal in response | Callers can't distinguish reliable vs. unreliable predictions | Include a simple `"confidence": "low/medium/high"` based on how close input features are to training distribution |
| Lambda returns 500 on invalid input | No diagnostic information | Return 400 with `{"error": "invalid_input", "field": "bedrooms", "message": "must be between 1-10"}` |

---

## "Looks Done But Isn't" Checklist

- [ ] **QLoRA Training**: Checkpoint saved to Google Drive — verify `ls /content/drive/MyDrive/checkpoints/` has adapter files
- [ ] **LoRA Merge**: Merged model validated by running inference and checking output is not identical for all inputs
- [ ] **ONNX Export**: Numerical validation run with `atol=1e-3` check — verify `optimum-cli` shows `"All good"` message
- [ ] **ONNX Inference on Colab**: Test inference with actual tokenized housing prompt — verify output is a parseable number, not random tokens
- [ ] **Lambda Container**: Image tested locally with `docker run` before pushing to ECR — verify cold start < 10s and warm inference < 3s
- [ ] **Lambda Timeout**: Lambda function timeout set to 30s (not default 3s) — verify in Terraform config
- [ ] **Lambda Memory**: Memory set to at least 3008MB — verify ONNX model loads without OOM
- [ ] **API Gateway**: CORS configured if future frontend consumption is possible — verify with `curl`
- [ ] **Tokenizer in Container**: Tokenizer files included in Docker image — verify no network calls during inference with `--network none` test
- [ ] **Regression Metrics**: All four metrics computed (MAE, RMSE, R², MAPE) — verify on test set, not just training set

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Merge failed — adapter saved but merged model corrupted | LOW | Delete merged model, reload base in fp32 on CPU, re-run merge (15 min) |
| ONNX export produced wrong task type | LOW | Re-run `optimum-cli export onnx` with correct `--task` flag (10 min) |
| Colab session disconnected mid-training | MEDIUM | Reload from Drive checkpoint, resume training from last saved step |
| Lambda container too large for ECR | MEDIUM | Audit and remove unnecessary packages from Dockerfile; rebuild and re-push (1-2 hours) |
| Synthetic data distribution is wrong | HIGH | Rewrite data generation with county-level priors, regenerate dataset, retrain from scratch (full day) |
| ONNX model produces NaN on Lambda | HIGH | Re-export with fp32 (not fp16), check for division by zero in preprocessing, re-validate on Colab first |
| Lambda cold start > 15s (timeout risk) | MEDIUM | Enable Lambda Provisioned Concurrency (costs money) or reduce model size via ONNX quantization |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| QLoRA merge requires non-quantized base | Phase 2: Training — include two-step save/merge in notebook | Run inference on merged model, check outputs differ across inputs |
| ONNX task type wrong for regression | Phase 3: ONNX Export — define export task before writing export code | `optimum-cli` validation passes with `atol=1e-3` |
| LLM output unparseable as number | Phase 2: Training — define prompt format and parser before training | Parse 100 validation outputs, confirm < 1% parse failure rate |
| Colab disconnect kills training | Phase 2: Training — Drive checkpointing in first cell | Training resumes successfully after manual disconnect test |
| Lambda container too large | Phase 4: Lambda Serving — define minimal deps in Dockerfile | `docker image ls` shows < 3GB uncompressed |
| Synthetic data distribution mismatch | Phase 1: Data Generation — validate distribution vs. NJ stats | Price histogram matches NJ median/std within 20% |
| Tokenizer not in container | Phase 4: Lambda Serving — verify no network calls in container | Run container with `--network none`, confirm no errors |
| ONNX Runtime global session not initialized | Phase 4: Lambda Serving — module-level global in handler | Warm invocation < 1s (no re-load overhead) |

---

## Sources

- HuggingFace PEFT Official Docs — Quantization Guide: https://huggingface.co/docs/peft/main/en/developer_guides/quantization (HIGH confidence)
- HuggingFace PEFT Official Docs — Troubleshooting: https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting (HIGH confidence)
- HuggingFace PEFT Official Docs — LoRA Config Reference: https://huggingface.co/docs/peft/main/en/package_reference/lora (HIGH confidence)
- HuggingFace Optimum ONNX Export Guide: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model (HIGH confidence)
- HuggingFace Optimum ONNX Runtime Inference: https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models (HIGH confidence)
- AWS Lambda Container Image Docs: https://docs.aws.amazon.com/lambda/latest/dg/images-create.html (HIGH confidence)
- AWS Lambda Runtime Docs: https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html (HIGH confidence)
- Qwen2.5-0.5B-Instruct Model Card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct (MEDIUM confidence — inferred from model card)
- QLoRA merge workflow pattern: MEDIUM confidence — based on PEFT official docs + known community pattern; verify with a test merge before committing to the full pipeline

---

*Pitfalls research for: QLoRA fine-tuning + ONNX export + Lambda serving (NJ Housing Price Predictor)*
*Researched: 2026-02-26*
