output "api_gateway_url" {
  description = "Live API Gateway invoke URL for smoke test"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.predictor.repository_url
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.predictor.function_name
}

output "github_actions_role_arn" {
  description = "ARN of IAM role for GitHub Actions OIDC -- set as repo variable GH_ACTIONS_ROLE_ARN"
  value       = aws_iam_role.github_actions.arn
}
