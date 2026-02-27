---
phase: 05-infrastructure-and-ci-cd
plan: 02
subsystem: infra
tags: [github-actions, ci-cd, docker, ecr, terraform, oidc, lambda, smoke-test]

# Dependency graph
requires:
  - phase: 05-infrastructure-and-ci-cd
    plan: 01
    provides: Terraform IaC (ECR repo housing-price-predictor, Lambda, API Gateway, OIDC IAM role, outputs for api_gateway_url/ecr_repository_url)
  - phase: 04-lambda-container-and-rest-api
    plan: 01
    provides: lambda/Dockerfile and Lambda handler for Docker build context

provides:
  - GitHub Actions CI workflow (.github/workflows/ci.yml) for PR lint + test + terraform fmt
  - GitHub Actions deploy workflow (.github/workflows/deploy.yml) for tag-triggered ECR push + Terraform apply + smoke test

affects:
  - All future PRs and releases (workflows trigger automatically on PR and v* tag push)

# Tech tracking
tech-stack:
  added: [github-actions, aws-actions/configure-aws-credentials@v4, aws-actions/amazon-ecr-login@v2, docker/build-push-action@v6, hashicorp/setup-terraform@v3]
  patterns:
    - "OIDC-based GitHub Actions AWS auth (id-token:write permission + role-to-assume) eliminates long-lived credentials"
    - "Tag-triggered deploys (v* tags) with git tag as Docker image tag ensures Terraform detects image_uri changes"
    - "Smoke test retry loop (5 attempts, 15s intervals) handles Lambda container image warmup latency"
    - "provenance:false on docker/build-push-action to reduce manifest complexity for ECR compatibility"

key-files:
  created:
    - .github/workflows/ci.yml
    - .github/workflows/deploy.yml
  modified: []

key-decisions:
  - "CI workflow has no AWS credentials - flake8 lint and pytest unit tests run without cloud access (tests mock ONNX/model)"
  - "Deploy workflow tags Docker image with github.ref_name (git tag) AND latest - git tag ensures Terraform image_uri variable changes between deploys, triggering Lambda UpdateFunctionCode"
  - "platforms: linux/amd64 mandatory on build-push-action - Lambda runs x86_64 regardless of GitHub runner architecture"
  - "GH_ACTIONS_ROLE_ARN stored as GitHub Actions Variable (not Secret) - role ARNs are not sensitive credentials"
  - "Smoke test hits /predict with real NJ housing payload (zip 07030 = Hoboken) and verifies HTTP 200"

patterns-established:
  - "Separate CI (no AWS, PR-triggered) and Deploy (OIDC, tag-triggered) workflows — principle of least privilege"
  - "terraform apply receives image_uri at apply time from freshly-built ECR image, not from stored state"
  - "API Gateway URL retrieved dynamically via terraform output -raw after each apply (not hardcoded)"

requirements-completed: [INFRA-03]

# Metrics
duration: 2min
completed: 2026-02-27
---

# Phase 5 Plan 02: GitHub Actions CI and Deploy Workflows Summary

**Tag-triggered deploy pipeline with OIDC auth, ECR push (linux/amd64), Terraform apply with image_uri variable, and 5-attempt smoke test retry loop against /predict endpoint**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-27T17:29:28Z
- **Completed:** 2026-02-27T17:31:44Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created CI workflow that runs flake8, pytest, and terraform fmt on every PR to main without requiring AWS credentials
- Created deploy workflow with OIDC authentication, Docker linux/amd64 build, ECR push, Terraform apply, and smoke test with warmup-aware retry loop
- Wired git tag name as Docker image tag to guarantee Terraform detects image_uri changes on every release

## Task Commits

Each task was committed atomically:

1. **Task 1: Create CI workflow for PR lint, test, and Terraform fmt check** - `5cdf67b` (feat)
2. **Task 2: Create deploy workflow for tag-triggered ECR push, Terraform apply, and smoke test** - `4ddad01` (feat)

## Files Created/Modified

- `.github/workflows/ci.yml` - PR workflow: flake8 on lambda/ and tests/ (max-line-length=120), pytest tests/test_handler.py, terraform fmt -check -recursive; Python 3.12, ubuntu-22.04, Terraform 1.14.6; no AWS credentials
- `.github/workflows/deploy.yml` - Tag-triggered (v*) deploy workflow: OIDC AWS auth via GH_ACTIONS_ROLE_ARN variable, ECR login, Docker build for linux/amd64 with git tag + latest labels, terraform init + apply with image_uri, smoke test 5-attempt loop with 15s intervals

## Decisions Made

- **No AWS in CI:** Unit tests use mocks for ONNX runtime and model artifacts; no credentials needed and keeping them out reduces attack surface
- **Git tag as Docker image tag:** `github.ref_name` (e.g., `v1.2.0`) used as the primary ECR tag; this makes `image_uri` change on every release so Terraform always updates the Lambda function
- **linux/amd64 platform flag:** Lambda container functions run x86_64; building without this flag on ARM runners produces images that fail to start
- **provenance: false:** Reduces OCI manifest complexity; ECR can handle attestation manifests but the simpler format avoids edge-case login failures
- **GH_ACTIONS_ROLE_ARN as Variable (not Secret):** Role ARNs are 12-digit account IDs + role name — not sensitive enough to warrant secret masking, and Variables are easier to inspect/debug

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

Before the deploy workflow can run, one manual step is required after `terraform apply`:

Set the GitHub Actions role ARN as a repository variable:
- Go to: Repository Settings > Secrets and Variables > Actions > Variables
- Variable name: `GH_ACTIONS_ROLE_ARN`
- Value: output of `terraform output github_actions_role_arn`

This was documented in 05-01-SUMMARY.md as well.

## Next Phase Readiness

- All 5 phases complete - the project is fully built end-to-end
- To deploy: push a `v*` tag (e.g., `git tag v1.0.0 && git push origin v1.0.0`)
- Prerequisites before first deploy:
  1. S3 state bucket created (see 05-01-SUMMARY.md)
  2. `terraform apply` run once manually to create AWS resources and get the GitHub Actions role ARN
  3. `GH_ACTIONS_ROLE_ARN` set as GitHub Actions repository variable

---
*Phase: 05-infrastructure-and-ci-cd*
*Completed: 2026-02-27*
