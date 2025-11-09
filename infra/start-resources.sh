#!/bin/bash
# Script to start AWS resources

set -e

PROJECT_NAME=${PROJECT_NAME:-golf-coach}
ENVIRONMENT=${ENVIRONMENT:-dev}
DB_IDENTIFIER="${PROJECT_NAME}-db-${ENVIRONMENT}"
CLUSTER_NAME="${PROJECT_NAME}-cluster-${ENVIRONMENT}"

echo "=== Starting AWS Resources ==="
echo ""

# Start RDS
echo "1. Starting RDS instance: $DB_IDENTIFIER"
if aws rds describe-db-instances --db-instance-identifier "$DB_IDENTIFIER" --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null | grep -q "stopped"; then
    aws rds start-db-instance --db-instance-identifier "$DB_IDENTIFIER"
    echo "   ✓ RDS start initiated"
    echo "   Waiting for RDS to be available (this may take 5-10 minutes)..."
    aws rds wait db-instance-available --db-instance-identifier "$DB_IDENTIFIER" || echo "   ⚠ RDS not found"
    echo "   ✓ RDS is available"
elif aws rds describe-db-instances --db-instance-identifier "$DB_IDENTIFIER" --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null | grep -q "available"; then
    echo "   ✓ RDS already running"
else
    echo "   ⚠ RDS doesn't exist - run 'terraform apply' first"
fi

# Recreate ALB if needed
echo ""
echo "2. Checking ALB..."
ALB_EXISTS=$(aws elbv2 describe-load-balancers --names "${PROJECT_NAME}-alb-${ENVIRONMENT}" --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || echo "None")
if [ "$ALB_EXISTS" == "None" ]; then
    echo "   ALB not found, recreating..."
    cd "$(dirname "$0")"
    terraform apply -target=aws_lb.main -target=aws_security_group.alb -auto-approve
    echo "   ✓ ALB recreated"
else
    echo "   ✓ ALB already exists"
fi

# Scale ECS services back up
echo ""
echo "3. Scaling ECS services..."
for service in backend frontend; do
    if aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$service" --query 'services[0].serviceName' --output text 2>/dev/null | grep -q "$service"; then
        aws ecs update-service --cluster "$CLUSTER_NAME" --service "$service" --desired-count 1 > /dev/null 2>&1
        echo "   ✓ $service scaled to 1"
    else
        echo "   ⚠ $service service not found (may need to deploy first)"
    fi
done

echo ""
echo "=== Resources Started ==="
echo ""
echo "Resources are now running. Costs:"
echo "  - RDS: ~\$20/month"
echo "  - ALB: ~\$21-26/month"
echo "  - ECS: ~\$7-10/month per task"
