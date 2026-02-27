# Phase 5: Infrastructure and CI/CD - Research

**Researched:** 2026-02-27
**Domain:** Terraform (Lambda + API Gateway v2 + ECR + IAM), S3 remote state, GitHub Actions OIDC, container-image Lambda deployment
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | Terraform configuration for Lambda + API Gateway + ECR | `aws_lambda_function` (package_type = "Image", image_uri), `aws_apigatewayv2_api` + integration + route + stage, `aws_ecr_repository`, `aws_iam_role` — all documented below |
| INFRA-02 | S3 backend for Terraform state | `terraform { backend "s3" { ... } }` block; S3 bucket must be created manually before first `terraform init`; DynamoDB locking deprecated in Terraform 1.x — use S3 native locking (`use_lockfile = true`) |
| INFRA-03 | GitHub Actions workflow for lint/test on PR and Terraform plan/apply on merge | Two workflow files: `ci.yml` (triggered on PR) runs flake8 + pytest; `deploy.yml` (triggered on git tag) runs docker build + ECR push + `terraform apply`; OIDC auth via `aws-actions/configure-aws-credentials@v4` |
</phase_requirements>

---

## Summary

Phase 5 provisions the production AWS infrastructure and wires up automated deployment. The Terraform configuration is a flat, single-module setup (no nested modules needed for a single-Lambda deployment) creating five resource types: ECR repository, Lambda function, API Gateway v2 HTTP API, IAM execution role, and Lambda permission granting API Gateway invocation rights. The S3 remote state backend is the prerequisite that must exist before any CI run — the bucket is created manually once (outside Terraform) and never destroyed.

The CI/CD split is clean: a `ci.yml` workflow runs on pull requests (lint + Lambda handler unit tests, no AWS credentials needed), and a `deploy.yml` workflow runs on a pushed git tag (build Docker image, push to ECR, run `terraform apply` with the new image URI). GitHub Actions authenticates to AWS via OIDC — no long-lived credentials stored in Secrets. The `aws-actions/configure-aws-credentials@v4` action handles the STS AssumeRoleWithWebIdentity exchange.

The one non-obvious complexity in this phase is the chicken-and-egg ordering between ECR and the Lambda function: the Lambda Terraform resource requires an `image_uri` that references an image in ECR, but the ECR image does not exist until the GitHub Actions deploy workflow pushes it. The resolution is to provision the ECR repository first (either in a separate Terraform run or via `terraform apply -target=aws_ecr_repository.predictor`), push an initial image manually (or via CI), and only then let the full `terraform apply` create the Lambda function. The `depends_on` meta-argument in the Lambda resource is insufficient here — a data source or `null_resource` sentinel pattern is the standard solution.

**Primary recommendation:** Write Terraform in two logical groups — (1) ECR repository + IAM role (no image dependency), (2) Lambda + API Gateway (depends on ECR image existing) — and apply group 1 first, push image, then apply group 2 via the CI deploy workflow.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Terraform | 1.14.6 | Declarative IaC for all AWS resources | Verified locally (see STACK.md); provider ecosystem is mature; no CDK/SAM complexity needed for single-Lambda scope |
| AWS Provider (Terraform) | ~> 6.0 (6.34.0 current) | AWS resource definitions | Constraint `~> 6.0` allows patch updates; AWS Provider 6.x is the current major; verified from Terraform Registry API |
| actions/checkout | v4 | Source checkout in GitHub Actions | Current major; supports sparse checkout |
| aws-actions/configure-aws-credentials | v4 | OIDC-based AWS auth in GitHub Actions | No long-lived credentials; handles STS AssumeRoleWithWebIdentity; v4 is current |
| aws-actions/amazon-ecr-login | v2 | ECR push authentication | Works with OIDC credentials automatically; v2 is current |
| hashicorp/setup-terraform | v3 | Terraform CLI in Actions | Caches Terraform binary; supports `terraform fmt` check gates |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| docker/build-push-action | v6 | Multi-platform Docker build + push in GitHub Actions | Handles `--platform linux/amd64` correctly; integrates with ECR login action |
| docker/setup-buildx-action | v3 | Docker Buildx setup (required by build-push-action) | Required for `--platform linux/amd64` cross-compile on GitHub runner |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flat terraform/ directory | terraform-aws-modules/lambda module | Module adds abstraction; overkill for a single Lambda; flat files are easier to audit |
| `aws_apigatewayv2_api` (HTTP API) | `aws_api_gateway_rest_api` (REST API) | HTTP API is cheaper, lower latency, sufficient for simple POST; REST API adds per-request overhead and more complex Terraform |
| OIDC IAM role (no stored credentials) | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` in Secrets | Long-lived credentials must be rotated; OIDC is the 2025 industry standard |
| S3 backend | Terraform Cloud | Terraform Cloud is free for small teams but adds external dependency; S3 is simpler and already in the AWS account |
| Git tag trigger for deploy | Push to main trigger | Tag trigger gives explicit control over when deploys happen; tag is tied to a Docker image version |

---

## Architecture Patterns

### Recommended Project Structure

```
terraform/
├── main.tf          # All resources: ECR, Lambda, API GW, IAM, Lambda permission
├── variables.tf     # image_uri, aws_region, lambda_memory_size, lambda_timeout
├── outputs.tf       # api_gateway_url, lambda_function_name, ecr_repository_url
└── versions.tf      # required_providers block, Terraform version constraint

.github/
└── workflows/
    ├── ci.yml       # PR: lint (flake8) + test (pytest tests/test_handler.py)
    └── deploy.yml   # Tag: docker build → ECR push → terraform apply → smoke test
```

### Pattern 1: Terraform Resources for Container Lambda + API Gateway v2

**What:** Minimal set of 6 Terraform resources to provision a container-image Lambda behind an HTTP API.

**When to use:** Always for this project. Flat file, no modules.

**Example:**
```hcl
# terraform/main.tf

# --- ECR Repository ---
resource "aws_ecr_repository" "predictor" {
  name                 = "housing-price-predictor"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- IAM Execution Role ---
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "housing-predictor-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- Lambda Function (Container Image) ---
resource "aws_lambda_function" "predictor" {
  function_name = "housing-price-predictor"
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = var.image_uri  # Set by CI: "<account>.dkr.ecr.<region>.amazonaws.com/housing-price-predictor:<git_sha>"

  memory_size = var.lambda_memory_size  # 3008 minimum (from Phase 4 research)
  timeout     = var.lambda_timeout      # 30 seconds

  # NOTE: handler and runtime are NOT set for container-image Lambda
  # (package_type = "Image" ignores both fields)
}

# --- API Gateway v2 (HTTP API) ---
resource "aws_apigatewayv2_api" "predictor" {
  name          = "housing-price-predictor-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.predictor.id
  integration_type   = "AWS_PROXY"
  integration_method = "POST"
  integration_uri    = aws_lambda_function.predictor.invoke_arn

  payload_format_version = "2.0"  # Required for HTTP API Lambda proxy
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.predictor.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.predictor.id
  name        = "$default"
  auto_deploy = true
}

# --- Lambda Permission (allows API Gateway to invoke Lambda) ---
resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predictor.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.predictor.execution_arn}/*/*"
}
```

**Critical notes:**
- `handler` and `runtime` fields MUST NOT be set for `package_type = "Image"` Lambda — Terraform will reject the config if both are present
- `payload_format_version = "2.0"` is required for HTTP API Lambda proxy integration; Mangum in the Lambda handler expects v2.0 event format
- `source_arn` uses `/*/*` wildcard to allow all methods and routes; scoping tighter is optional

### Pattern 2: Terraform Variables and Outputs

```hcl
# terraform/variables.tf
variable "image_uri" {
  description = "ECR image URI for Lambda container (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com/housing-price-predictor:abc1234)"
  type        = string
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "lambda_memory_size" {
  description = "Lambda memory in MB (minimum 3008 for ONNX model load)"
  type        = number
  default     = 3008
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 30
}

# terraform/outputs.tf
output "api_gateway_url" {
  description = "Live API Gateway invoke URL for smoke test"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.predictor.repository_url
}

# terraform/versions.tf
terraform {
  required_version = ">= 1.14.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    bucket  = "housing-predictor-tfstate"  # Must exist before terraform init
    key     = "housing-price-predictor/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    # use_lockfile = true  # Terraform 1.10+ native S3 locking (replaces DynamoDB)
  }
}

provider "aws" {
  region = var.aws_region
}
```

### Pattern 3: S3 Remote State Backend

**What:** Store Terraform state in an S3 bucket so GitHub Actions CI can run `terraform apply` without conflicting with local runs.

**Critical prerequisite:** The S3 bucket must be created manually BEFORE the first `terraform init`. Terraform cannot bootstrap its own state bucket.

**Manual one-time setup:**
```bash
# Run ONCE before any terraform init — not managed by Terraform
aws s3api create-bucket \
  --bucket housing-predictor-tfstate \
  --region us-east-1

aws s3api put-bucket-versioning \
  --bucket housing-predictor-tfstate \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket housing-predictor-tfstate \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-public-access-block \
  --bucket housing-predictor-tfstate \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**State locking note (Terraform 1.10+):** DynamoDB-based state locking is deprecated in recent Terraform versions. Native S3 locking via `use_lockfile = true` in the backend block replaces DynamoDB. Since the project uses Terraform 1.14.6, use `use_lockfile = true` and skip DynamoDB. (Verify this in the official Terraform S3 backend docs at the time of implementation — this changed in 1.10.)

### Pattern 4: GitHub Actions OIDC IAM Role (Terraform-managed)

**What:** Create an IAM OIDC identity provider and a role that GitHub Actions can assume via short-lived tokens. No long-lived credentials stored in GitHub Secrets.

**When to use:** Always. Storing `AWS_ACCESS_KEY_ID` in Secrets requires rotation and creates credential leak risk.

**Key 2025 update:** AWS added GitHub to its root certificate authorities. Thumbprints in the `aws_iam_openid_connect_provider` resource are no longer required. The Terraform AWS Provider 6.x made thumbprints optional. Use `thumbprint_list = []` or omit the field.

```hcl
# Add to terraform/main.tf (or a separate iam_oidc.tf)

# GitHub OIDC Provider — created once per AWS account, not per repo
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = []  # Thumbprints optional as of AWS Provider 6.x (AWS trusts GitHub CA)
}

# Trust policy: only THIS repo's workflows can assume this role
data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      # Replace with your actual repo: repo:<org>/<repo>:*
      values   = ["repo:YOUR_GITHUB_ORG/housing_price_predictor:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "housing-predictor-github-actions-role"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json
}

# Permissions needed by GitHub Actions deploy workflow
resource "aws_iam_role_policy" "github_actions_permissions" {
  name = "github-actions-deploy-policy"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # ECR push permissions
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "*"
      },
      {
        # Lambda update + Terraform plan/apply permissions
        Effect = "Allow"
        Action = [
          "lambda:GetFunction",
          "lambda:UpdateFunctionCode",
          "lambda:UpdateFunctionConfiguration",
          "lambda:CreateFunction",
          "lambda:DeleteFunction",
          "lambda:AddPermission",
          "lambda:RemovePermission",
          "lambda:GetFunctionConfiguration",
          "apigateway:*",
          "iam:GetRole",
          "iam:PassRole",
          "iam:CreateRole",
          "iam:AttachRolePolicy",
          "iam:PutRolePolicy",
          "iam:GetRolePolicy",
          "iam:CreateOpenIDConnectProvider",
          "iam:GetOpenIDConnectProvider",
          "iam:TagOpenIDConnectProvider"
        ]
        Resource = "*"
      },
      {
        # S3 backend state access
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::housing-predictor-tfstate",
          "arn:aws:s3:::housing-predictor-tfstate/*"
        ]
      }
    ]
  })
}

output "github_actions_role_arn" {
  description = "ARN of IAM role for GitHub Actions OIDC — add to repo variable GH_ACTIONS_ROLE_ARN"
  value       = aws_iam_role.github_actions.arn
}
```

**Important:** The OIDC provider (`aws_iam_openid_connect_provider`) is an account-level resource. If it already exists in the AWS account (from another project), import it instead of creating a new one. One provider per AWS account is the correct model.

### Pattern 5: GitHub Actions Workflows

**ci.yml (PR workflow — no AWS credentials):**
```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install test dependencies
        run: |
          pip install flake8 pytest
          # Install Lambda handler test deps (no onnxruntime needed for unit tests with mocks)
          pip install fastapi pydantic mangum

      - name: Lint with flake8
        run: |
          flake8 lambda/ tests/ --max-line-length=120

      - name: Run Lambda handler unit tests
        run: |
          pytest tests/test_handler.py -v

      - name: Terraform fmt check
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.14.6"
      - run: terraform -chdir=terraform fmt -check -recursive
```

**deploy.yml (tag-triggered deploy workflow):**
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    tags:
      - 'v*'  # Triggers on v1.0.0, v1.1.0, etc.

permissions:
  id-token: write   # Required for OIDC token request
  contents: read

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: housing-price-predictor

jobs:
  deploy:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.GH_ACTIONS_ROLE_ARN }}  # Set in repo variables
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push Docker image to ECR
        uses: docker/build-push-action@v6
        with:
          context: lambda/
          platforms: linux/amd64      # Lambda runs x86_64 — must specify even on ARM runners
          provenance: false           # Reduces image manifest complexity for ECR
          push: true
          tags: |
            ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.ref_name }}
            ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:latest

      - name: Set up Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.14.6"

      - name: Terraform Init
        run: terraform init
        working-directory: terraform/

      - name: Terraform Apply
        run: |
          terraform apply -auto-approve \
            -var="image_uri=${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.ref_name }}"
        working-directory: terraform/

      - name: Get API Gateway URL
        id: get-url
        run: |
          API_URL=$(terraform output -raw api_gateway_url)
          echo "api_url=$API_URL" >> $GITHUB_OUTPUT
        working-directory: terraform/

      - name: Smoke test
        run: |
          RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${{ steps.get-url.outputs.api_url }}/predict" \
            -H "Content-Type: application/json" \
            -d '{"bedrooms":3,"bathrooms":2.0,"sqft":1800,"lot_size":0.25,"year_built":1995,"zip_code":"07030","property_type":"Single Family"}')
          echo "HTTP status: $RESPONSE"
          if [ "$RESPONSE" != "200" ]; then
            echo "Smoke test FAILED: expected 200, got $RESPONSE"
            exit 1
          fi
          echo "Smoke test PASSED"
```

**Note on `permissions: id-token: write`:** This YAML block is required at the workflow or job level. Without it, GitHub Actions will not generate an OIDC token, and the `configure-aws-credentials@v4` action will silently fail or fall through to other credential methods.

### Pattern 6: ECR Chicken-and-Egg Resolution

**Problem:** `aws_lambda_function.image_uri` requires an ECR image to exist, but the ECR image is built and pushed by the deploy workflow — which runs AFTER `terraform apply`. This creates a circular dependency.

**Resolution — two-phase apply strategy:**

Phase A (run once, before first CI deploy):
```bash
# Apply only ECR repository and IAM roles (no Lambda yet)
terraform apply -target=aws_ecr_repository.predictor \
               -target=aws_iam_role.lambda_exec \
               -target=aws_iam_role_policy_attachment.lambda_basic \
               -target=aws_iam_openid_connect_provider.github \
               -target=aws_iam_role.github_actions \
               -target=aws_iam_role_policy.github_actions_permissions

# Push initial image manually (or trigger the deploy workflow)
docker build --platform linux/amd64 --provenance=false -t housing-predictor:v0.0.0 lambda/
docker push <ECR_URL>:v0.0.0
```

Phase B (first full apply — via deploy workflow):
```bash
terraform apply -var="image_uri=<ECR_URL>:v0.0.0"
# This now creates Lambda, API Gateway, routes, stage, and Lambda permission
```

Subsequent deploys:
```bash
# Deploy workflow handles this automatically on every git tag push
terraform apply -var="image_uri=<ECR_URL>:<new_tag>"
# Terraform detects image_uri changed → runs UpdateFunctionCode → Lambda redeployed
```

### Anti-Patterns to Avoid

- **`handler` and `runtime` fields in container Lambda:** Setting these alongside `package_type = "Image"` causes Terraform to reject the config. Container-image Lambda functions don't use these fields.
- **`terraform apply` without `-var="image_uri=..."` in CI:** Terraform will use a default or fail. Always pass `image_uri` explicitly at apply time.
- **Committing `terraform.tfstate` to the repo:** State file contains AWS account IDs, resource ARNs, and potentially sensitive values. The S3 backend ensures state is never local. Add `*.tfstate` and `*.tfstate.backup` to `.gitignore`.
- **Using `terraform apply -auto-approve` on PR:** Only run `terraform plan` on PRs. Run `terraform apply` only on merge to main or a git tag. Auto-approve without plan review can destroy and recreate resources unexpectedly.
- **`payload_format_version = "1.0"` with API Gateway v2:** Version 1.0 uses a different event format than v2.0. Mangum in the Lambda handler is configured to handle v2.0 format (the `version: "2.0"` field in the event). Use `payload_format_version = "2.0"` in the Terraform integration resource.
- **Missing `aws_lambda_permission` resource:** Without this, API Gateway cannot invoke Lambda. The symptom is HTTP 403 or 500 from API Gateway with no Lambda invocation logged.
- **Storing `GH_ACTIONS_ROLE_ARN` in GitHub Secrets instead of Variables:** The role ARN is not sensitive (it's a resource identifier, not a credential). Use GitHub Actions Variables (not Secrets) for non-sensitive config. Using Secrets for non-secret values masks them in logs unnecessarily.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| API Gateway → Lambda event translation | Custom event parser in handler | Mangum (already used in Phase 4) | Handles all event format versions; edge cases in path/query handling are subtle |
| OIDC token exchange | Custom JWT logic | `aws-actions/configure-aws-credentials@v4` | Handles STS AssumeRoleWithWebIdentity automatically; token expiry managed |
| Multi-platform Docker build | `docker build` with cross-compile flags | `docker/build-push-action@v6` + `docker/setup-buildx-action@v3` | Handles QEMU emulation, layer caching, and manifest correctly |
| Terraform state management | Local state file | S3 backend | Local state fails in CI (no persistent filesystem); S3 backend is the documented standard for multi-user/CI environments |
| Lambda invoke URL | Manual URL construction | `aws_apigatewayv2_stage.default.invoke_url` output | Terraform computes the correct URL; manual construction gets the stage suffix wrong |

**Key insight:** All heavy lifting in this phase is handled by existing Terraform resources and GitHub Actions actions. The planner's task is wiring them together correctly — there is almost nothing to build from scratch.

---

## Common Pitfalls

### Pitfall 1: Lambda Function Created Before ECR Image Exists

**What goes wrong:** `terraform apply` fails with `InvalidParameterValueException: Source image <ECR_URI> does not exist` when trying to create `aws_lambda_function` before an image has been pushed to ECR.

**Why it happens:** Terraform creates resources in dependency order. The Lambda function depends on the ECR repository (via `image_uri`), but not on a specific image being present in that repository. If the workflow tries to apply the full config before the first image push, Terraform will attempt to create the Lambda and fail.

**How to avoid:** Use the two-phase apply strategy described in Pattern 6. Apply ECR and IAM resources first (`-target`), push an initial image, then run the full `terraform apply`. After the first deploy, the problem disappears — subsequent applies always have a valid image URI.

**Warning signs:** Terraform error containing `Source image` and `does not exist`; Lambda function not appearing in AWS console after apply.

### Pitfall 2: Missing `permissions: id-token: write` in GitHub Actions Workflow

**What goes wrong:** `aws-actions/configure-aws-credentials@v4` silently fails to obtain an OIDC token, falls through to environment-variable credential lookup, finds nothing, and the job errors with `Error: Could not load credentials from any providers`.

**Why it happens:** GitHub Actions blocks OIDC token generation by default for security. The `id-token: write` permission must be explicitly granted at the workflow or job level.

**How to avoid:** Add to the `deploy.yml` workflow (top-level or per-job):
```yaml
permissions:
  id-token: write
  contents: read
```

**Warning signs:** CI fails with `Error: Could not load credentials from any providers` or `Error: Could not resolve host: sts.amazonaws.com` even though the role ARN is set.

### Pitfall 3: `payload_format_version` Mismatch Causes Mangum Parse Failures

**What goes wrong:** API Gateway returns HTTP 500 or passes a malformed event to Lambda. Mangum raises `ValueError: Unable to determine event type` or FastAPI returns 422 on every request.

**Why it happens:** API Gateway v2 HTTP API supports two payload formats: `1.0` (legacy, compatible with REST API) and `2.0` (native HTTP API format). Mangum's default behavior detects the version from the event. If Terraform sets `payload_format_version = "1.0"` but Mangum expects `2.0`, the event structure mismatch causes failures.

**How to avoid:** Set `payload_format_version = "2.0"` in `aws_apigatewayv2_integration`. This matches the event format tested locally in Phase 4 RIE tests (the test curl command uses `"version": "2.0"` in the event).

**Warning signs:** API Gateway returns 502 or 500; Lambda CloudWatch logs show Mangum errors rather than handler errors.

### Pitfall 4: Terraform State Drift When Image URI Doesn't Change

**What goes wrong:** A new Docker image is pushed to ECR with the same tag (e.g., `latest`). `terraform apply` sees no change to `image_uri` (the tag is the same) and does NOT update the Lambda function. The Lambda continues running the old image.

**Why it happens:** Terraform tracks state by value, not by content. If `image_uri` = `<ECR_URL>:latest` in state and the new plan also has `<ECR_URL>:latest`, Terraform considers no change needed.

**How to avoid:** Tag images by git SHA or version tag, never by `latest`. The `deploy.yml` workflow tags with `${{ github.ref_name }}` (the git tag, e.g., `v1.0.1`). Each deploy gets a unique tag, ensuring Terraform detects the change and updates the Lambda function.

**Warning signs:** `terraform apply` reports "No changes" even after a new image was pushed; Lambda is running an old version confirmed by checking the image digest in the AWS console.

### Pitfall 5: Smoke Test Runs Before Lambda Is Warm

**What goes wrong:** Smoke test `curl` request times out or returns 502/503 because the Lambda function is still in the `Pending` state while AWS optimizes the container image. First invocation after `UpdateFunctionCode` can take 10-30 seconds for the optimization to complete.

**Why it happens:** After `UpdateFunctionCode`, Lambda performs a one-time container image optimization (extracting layers, warming the execution environment). During this time, invocations may fail.

**How to avoid:** Add a brief wait or retry loop in the smoke test step:
```bash
# Retry smoke test up to 5 times with 15-second intervals
for i in 1 2 3 4 5; do
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"bedrooms":3,...}')
  if [ "$RESPONSE" = "200" ]; then
    echo "Smoke test PASSED on attempt $i"
    break
  fi
  echo "Attempt $i: HTTP $RESPONSE — waiting 15s"
  sleep 15
done
if [ "$RESPONSE" != "200" ]; then exit 1; fi
```

**Warning signs:** Smoke test fails with HTTP 502 or timeout immediately after `terraform apply`; retry resolves it within 1-2 minutes.

### Pitfall 6: OIDC Provider Conflict (Already Exists in AWS Account)

**What goes wrong:** `terraform apply` fails with `EntityAlreadyExists: Provider with url https://token.actions.githubusercontent.com already exists`.

**Why it happens:** The GitHub OIDC provider is an AWS account-level resource. If another project or a team member already created it, Terraform's create attempt fails.

**How to avoid:** Before adding the OIDC provider resource, check if it exists:
```bash
aws iam list-open-id-connect-providers | grep github
```
If it exists, import it into Terraform state instead of creating:
```bash
terraform import aws_iam_openid_connect_provider.github \
  arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com
```

**Warning signs:** `EntityAlreadyExists` error during `terraform apply` on the OIDC provider resource.

---

## Code Examples

Verified patterns from official sources:

### Complete `terraform apply` with image_uri variable

```bash
# From deploy.yml — terraform apply passing image URI
terraform apply -auto-approve \
  -var="image_uri=$(terraform output -raw ecr_repository_url):${GITHUB_REF_NAME}"
```

### Post-deploy smoke test with retry

```bash
# Retry-aware smoke test for post-UpdateFunctionCode Lambda warmup
API_URL=$(terraform output -raw api_gateway_url)
PAYLOAD='{"bedrooms":3,"bathrooms":2.0,"sqft":1800,"lot_size":0.25,"year_built":1995,"zip_code":"07030","property_type":"Single Family"}'

for i in 1 2 3 4 5; do
  HTTP_STATUS=$(curl -s -o response.json -w "%{http_code}" \
    -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")
  echo "Attempt $i: HTTP $HTTP_STATUS"
  cat response.json
  if [ "$HTTP_STATUS" = "200" ]; then
    echo "Smoke test PASSED"
    exit 0
  fi
  sleep 15
done
echo "Smoke test FAILED after 5 attempts"
exit 1
```

### Checking image_uri triggers Terraform update

```bash
# Verify Terraform will update Lambda when image changes
terraform plan -var="image_uri=<ECR_URL>:<new_tag>"
# Should show: "aws_lambda_function.predictor will be updated in-place"
# Lines: ~ image_uri = "<old_tag>" -> "<new_tag>"
```

### Local Terraform plan before CI (development verification)

```bash
# Run locally before pushing to verify no syntax errors
terraform init
terraform validate
terraform plan -var="image_uri=placeholder:latest"  # Validates resource graph even with placeholder URI
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Long-lived `AWS_ACCESS_KEY_ID` in GitHub Secrets | OIDC IAM role, no stored credentials | 2022 (GitHub OIDC GA) | Eliminates credential rotation, reduces breach surface |
| DynamoDB for Terraform state locking | Native S3 locking (`use_lockfile = true`) | Terraform 1.10 (2024) | No DynamoDB table needed; simpler backend setup |
| Required thumbprint list for GitHub OIDC provider | Thumbprints optional (AWS trusts GitHub CA) | December 2024 (AWS SDK update) | No thumbprint maintenance; `thumbprint_list = []` is valid |
| `aws_api_gateway_rest_api` (REST API) | `aws_apigatewayv2_api` (HTTP API) | 2019 (HTTP API GA) | Lower cost, lower latency, simpler Terraform; HTTP API is default for new single-Lambda deployments |
| Lambda ZIP deployment | Lambda container image deployment | 2020 (re:Invent) | Supports large ML models; no 50MB zip limit |
| `payload_format_version = "1.0"` (REST API compatibility) | `payload_format_version = "2.0"` (HTTP API native) | 2020 | Cleaner event structure; required for Mangum v2 event handling |

**Deprecated/outdated:**
- DynamoDB Terraform state locking: Deprecated in Terraform 1.x. Use `use_lockfile = true` in the S3 backend config.
- GitHub OIDC thumbprint pinning: No longer required as of December 2024. AWS directly trusts GitHub's CA.
- `terraform apply` directly without plan in PR: Industry anti-pattern — PRs should only run `terraform plan`; apply only on merge/tag.

---

## Open Questions

1. **OIDC provider already exists in the target AWS account**
   - What we know: `aws_iam_openid_connect_provider` for `token.actions.githubusercontent.com` is account-level, not project-level. Conflict errors if it already exists.
   - What's unclear: Whether the target AWS account already has this provider from another project.
   - Recommendation: Check with `aws iam list-open-id-connect-providers` before writing the resource. If it exists, use `terraform import`. Document this check as the first task in Phase 5.

2. **Terraform state locking: `use_lockfile` vs. DynamoDB**
   - What we know: `use_lockfile = true` is the Terraform 1.10+ replacement for DynamoDB locking. The project uses Terraform 1.14.6.
   - What's unclear: Whether `use_lockfile` is fully stable in 1.14.6 for single-writer CI scenarios (only one GitHub Actions job runs `terraform apply` at a time).
   - Recommendation: Use `use_lockfile = true`. For single-writer CI, even without locking, conflicts are unlikely. If in doubt, add locking after the initial setup.

3. **Lambda provisioned concurrency decision**
   - What we know: Cold start latency for Qwen2.5-0.5B ONNX is estimated 5-15 seconds (from Phase 4 open questions). Provisioned concurrency eliminates cold starts but adds ~$0.015/hour.
   - What's unclear: Actual measured cold start time on first deploy.
   - Recommendation: Do NOT provision concurrency in Phase 5. Measure cold start time from CloudWatch after first deploy. Add `aws_lambda_provisioned_concurrency_config` resource only if cold start exceeds 15 seconds and use case requires consistent latency.

4. **ECR image lifecycle policy**
   - What we know: Each git tag push creates a new ECR image. Without a lifecycle policy, ECR will accumulate images indefinitely (cost: ~$0.10/GB/month).
   - What's unclear: Whether this matters at v1 scale (likely < 10 images in Phase 5).
   - Recommendation: Add a simple lifecycle policy to retain only the last 5 tagged images. Low-priority — do it in Phase 5 if time allows, defer if not.

---

## Sources

### Primary (HIGH confidence)
- AWS Prescriptive Guidance — Build and push Docker images to Amazon ECR using GitHub Actions and Terraform: https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/build-and-push-docker-images-to-amazon-ecr-using-github-actions-and-terraform.html — OIDC + ECR + GitHub Actions integrated pattern
- Terraform Registry — `aws_apigatewayv2_route`: https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/apigatewayv2_route — route_key, target syntax
- Terraform Registry — `aws_apigatewayv2_integration`: https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/apigatewayv2_integration — payload_format_version, integration_type
- HashiCorp Developer — Deploy serverless applications with AWS Lambda and API Gateway: https://developer.hashicorp.com/terraform/tutorials/aws/lambda-api-gateway — official Terraform tutorial for Lambda + API GW
- HashiCorp Developer — S3 backend: https://developer.hashicorp.com/terraform/language/backend/s3 — use_lockfile, encryption, versioning
- GitHub Docs — Configuring OIDC in AWS: https://docs.github.com/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services — `permissions: id-token: write`, trust policy format
- aws-actions/configure-aws-credentials GitHub: https://github.com/aws-actions/configure-aws-credentials — v4, role-to-assume parameter
- Project .planning/research/STACK.md — Terraform 1.14.6 + AWS Provider 6.34.0 + GitHub Actions versions verified 2026-02-26

### Secondary (MEDIUM confidence)
- Colin Barker blog — GitHub Actions and OIDC Update for Terraform and AWS (2025-01-12): https://colinbarker.me.uk/blog/2025-01-12-github-actions-oidc-update/ — thumbprints now optional, December 2024 AWS change
- devdosvid.blog — Mastering AWS API Gateway V2 HTTP and AWS Lambda With Terraform (2024): https://devdosvid.blog/2024/01/09/mastering-aws-api-gateway-v2-http-and-aws-lambda-with-terraform/ — complete working API GW v2 + Lambda Terraform example
- WebSearch results confirming `package_type = "Image"` excludes `handler` and `runtime` for container Lambda; `payload_format_version = "2.0"` for HTTP API

### Tertiary (LOW confidence — verify before implementation)
- Lambda provisioned concurrency cost estimate: ~$0.015/hour — from training data; verify at https://aws.amazon.com/lambda/pricing/
- `use_lockfile = true` stability in Terraform 1.14.6 for CI: confirmed available in 1.10+ but production stability in CI at 1.14.6 not independently verified

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tool versions verified in STACK.md (2026-02-26); Terraform resource names verified from Registry docs; GitHub Actions action versions confirmed from official repos
- Architecture: HIGH — Terraform resource configuration verified against official HashiCorp tutorials and Terraform Registry docs; OIDC pattern verified from GitHub official docs and AWS blog
- Pitfalls: HIGH — ECR chicken-and-egg confirmed from Terraform GitHub issues; thumbprint change confirmed from AWS/GitHub official source (December 2024); `payload_format_version` mismatch is documented behavior; OIDC `id-token: write` requirement is from GitHub official docs

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (Terraform AWS provider and GitHub Actions are stable; OIDC patterns are mature industry standard)
