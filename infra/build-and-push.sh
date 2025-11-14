#!/bin/bash
# Script to build and push Docker images to ECR

set -e

PROJECT_NAME=${PROJECT_NAME:-golf-coach}
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}

echo "=== Building and Pushing Docker Images ==="
echo ""

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BACKEND_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-backend"
ECR_FRONTEND_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-frontend"

# Login to ECR
echo "1. Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo "   ✓ Logged in to ECR"
echo ""

# Build and push backend
echo "2. Building backend image..."
cd "$(dirname "$0")/../backend"
docker build -t ${PROJECT_NAME}-backend:latest .
docker tag ${PROJECT_NAME}-backend:latest ${ECR_BACKEND_URL}:latest
echo "   ✓ Backend image built"
echo ""

echo "3. Pushing backend image to ECR..."
docker push ${ECR_BACKEND_URL}:latest
echo "   ✓ Backend image pushed"
echo ""

# Build and push frontend
echo "4. Building frontend image..."
cd "$(dirname "$0")/../frontend"
docker build -f Dockerfile.prod -t ${PROJECT_NAME}-frontend:latest .
docker tag ${PROJECT_NAME}-frontend:latest ${ECR_FRONTEND_URL}:latest
echo "   ✓ Frontend image built"
echo ""

echo "5. Pushing frontend image to ECR..."
docker push ${ECR_FRONTEND_URL}:latest
echo "   ✓ Frontend image pushed"
echo ""

echo "=== Build and Push Complete ==="
echo ""
echo "Images pushed:"
echo "  Backend:  ${ECR_BACKEND_URL}:latest"
echo "  Frontend: ${ECR_FRONTEND_URL}:latest"
echo ""
echo "Next steps:"
echo "  1. Run 'terraform apply' to create/update ECS services"
echo "  2. Or manually update ECS services to use the new images"

