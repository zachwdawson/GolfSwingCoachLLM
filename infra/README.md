# Infrastructure

Terraform configuration for AWS resources.

## Resources

- S3 bucket for video storage
- RDS Postgres with pgvector extension
- ECR repositories for backend and frontend
- ECS Fargate cluster
- Application Load Balancer
- IAM roles and security groups

## Usage

1. Set up AWS credentials
2. Create `terraform.tfvars` with required variables
3. Run `terraform init`
4. Run `terraform plan`
5. Run `terraform apply`

## Variables

See `variables.tf` for required variables. Create `terraform.tfvars`:

```hcl
aws_region      = "us-east-1"
project_name    = "golf-coach"
environment     = "dev"
db_instance_class = "db.t3.micro"
db_username     = "postgres"
db_password     = "your-secure-password"
subnet_ids      = ["subnet-xxx", "subnet-yyy"]
allowed_cidr_blocks = ["10.0.0.0/8"]
```

