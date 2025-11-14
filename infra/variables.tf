variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "golf-coach"
}

variable "environment" {
  description = "Environment (dev, prod)"
  type        = string
  default     = "dev"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_username" {
  description = "RDS master username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "subnet_ids" {
  description = "Subnet IDs for RDS and ECS"
  type        = list(string)
  default     = ["subnet-0e962104467343d37", "subnet-071779c541428cabc"]
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access RDS"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "aws_access_key_id_secret_arn" {
  description = "ARN of AWS Access Key ID secret in Secrets Manager (optional)"
  type        = string
  default     = ""
}

variable "aws_secret_access_key_secret_arn" {
  description = "ARN of AWS Secret Access Key secret in Secrets Manager (optional)"
  type        = string
  default     = ""
}

variable "openai_api_key_secret_arn" {
  description = "ARN of OpenAI API Key secret in Secrets Manager (optional)"
  type        = string
  default     = ""
}

