variable "image_uri" {
  description = "ECR image URI for Lambda container (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com/housing-price-predictor:v1.0.0)"
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

variable "github_repo" {
  description = "GitHub repository in format org/repo for OIDC trust policy"
  type        = string
  default     = "rajkumar4466/housing_price_predictor"
}
