---
phase: 05-infrastructure-and-ci-cd
plan: 01
subsystem: infra
tags: [terraform, aws, ecr, lambda, api-gateway, iam, oidc, github-actions, s3]

# Dependency graph
requires:
  - phase: 04-lambda-container-and-rest-api
    provides: Lambda handler (handler.py), Dockerfile, and container image that terraform provisions ECR and Lambda for

provides:
  - Terraform IaC for ECR repository, Lambda function (container image), API Gateway v2 HTTP API
  - GitHub Actions OIDC IAM role with scoped trust policy and deploy permissions
  - S3 remote state backend configuration (housing-predictor-tfstate bucket)
  - All four Terraform files ready for terraform init/apply

affects:
  - 05-02 (CI/CD wiring uses these terraform outputs and role ARN)
  - deployment workflows that reference ECR URL, API Gateway URL, lambda function name

# Tech tracking
tech-stack:
  added: [terraform >= 1.14.0, aws-provider ~> 6.0, s3-backend-native-locking]
  patterns:
    - "S3 native locking (use_lockfile=true) instead of deprecated DynamoDB locking"
    - "Lambda container image deployment (package_type=Image, no handler/runtime)"
    - "OIDC-based GitHub Actions auth instead of long-lived AWS credentials"
    - "API Gateway v2 with payload_format_version=2.0 for Mangum compatibility"

key-files:
  created:
    - terraform/main.tf
    - terraform/variables.tf
    - terraform/outputs.tf
    - terraform/versions.tf
  modified:
    - .gitignore

key-decisions:
  - "use_lockfile=true for S3 backend locking (DynamoDB locking deprecated in Terraform 1.10+)"
  - "thumbprint_list=[] for GitHub OIDC provider (AWS trusts GitHub CA natively since December 2024)"
  - "payload_format_version=2.0 on API Gateway integration (required for Mangum event parsing)"
  - "Lambda function omits handler and runtime fields (invalid for package_type=Image)"
  - "lambda_memory_size defaults to 3008 MB (minimum for ONNX model cold start)"
  - "OIDC trust policy scoped to repo:rajkumar4466/housing_price_predictor:* (not wildcard org)"

patterns-established:
  - "All AWS resources in single main.tf with logical grouping by resource type"
  - "Required variables without defaults (image_uri) force explicit CI supply at apply time"
  - "Outputs expose all values needed for CI/CD: api_gateway_url, ecr_repository_url, lambda_function_name, github_actions_role_arn"

requirements-completed: [INFRA-01, INFRA-02]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 5 Plan 01: Terraform IaC for Lambda + API Gateway v2 + ECR + IAM + OIDC Summary

**Complete Terraform IaC with ECR, Lambda container image, API Gateway v2 (payload_format_version=2.0), GitHub Actions OIDC role, and S3 native-locking backend across 4 files**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-27T17:21:18Z
- **Completed:** 2026-02-27T17:25:42Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created all four Terraform configuration files (versions.tf, main.tf, variables.tf, outputs.tf) ready for terraform init/apply
- Configured GitHub Actions OIDC authentication with scoped trust policy eliminating need for long-lived AWS credentials in CI
- Set up S3 remote state backend with native locking (use_lockfile=true) replacing deprecated DynamoDB approach

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Terraform versions.tf with S3 backend and provider configuration** - `b5518c7` (chore)
2. **Task 2: Create Terraform main.tf with all AWS resources and OIDC configuration** - `47d6647` (feat)

## Files Created/Modified

- `terraform/versions.tf` - Terraform version constraint (>=1.14.0), AWS provider ~>6.0, S3 backend with housing-predictor-tfstate bucket and use_lockfile=true
- `terraform/main.tf` - All AWS resources: ECR repository, Lambda IAM exec role, Lambda function (package_type=Image), API Gateway v2 HTTP API with route/integration/stage, Lambda permission, GitHub OIDC provider, GitHub Actions IAM role with deploy policy
- `terraform/variables.tf` - Input variables: image_uri (required, no default), aws_region, lambda_memory_size (default 3008), lambda_timeout (default 30), github_repo
- `terraform/outputs.tf` - Exported values: api_gateway_url, ecr_repository_url, lambda_function_name, github_actions_role_arn
- `.gitignore` - Added Terraform entries: *.tfstate, *.tfstate.backup, .terraform/, .terraform.lock.hcl

## Decisions Made

- **S3 native locking over DynamoDB:** use_lockfile=true is the Terraform 1.10+ approach; DynamoDB locking is deprecated and would add unnecessary infrastructure complexity
- **Empty thumbprint_list for OIDC:** AWS has natively trusted GitHub's OIDC CA since December 2024; thumbprint_list=[] is correct and future-proof
- **payload_format_version=2.0 mandated:** Mangum (the ASGI/FastAPI adapter in handler.py) requires v2.0 event format; using 1.0 causes parse failures
- **No handler/runtime on Lambda:** package_type=Image Lambda functions are configured via CMD in Dockerfile; Terraform rejects configs that set both package_type=Image and handler/runtime
- **image_uri required with no default:** This variable must be supplied at terraform apply time by CI/CD with the specific ECR image tag being deployed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

Before running `terraform init`, the S3 state bucket must be created manually:

```bash
aws s3 mb s3://housing-predictor-tfstate --region us-east-1
aws s3api put-bucket-versioning \
  --bucket housing-predictor-tfstate \
  --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption \
  --bucket housing-predictor-tfstate \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

After `terraform apply`, set the GitHub Actions role ARN as a repository variable:
- Variable name: `GH_ACTIONS_ROLE_ARN`
- Value: output of `terraform output github_actions_role_arn`

## Next Phase Readiness

- All 4 Terraform files are committed and ready for terraform init/apply
- Plan 05-02 can now wire CI/CD GitHub Actions workflows using the OIDC role ARN and ECR/Lambda outputs
- S3 state bucket must be created by user before terraform init (documented above)
- Prerequisite: Docker image must be built and pushed to ECR before terraform apply (image_uri variable)

---
*Phase: 05-infrastructure-and-ci-cd*
*Completed: 2026-02-27*
