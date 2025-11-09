#!/bin/bash
# Script to stop AWS resources to save costs

set -e

PROJECT_NAME=${PROJECT_NAME:-golf-coach}
ENVIRONMENT=${ENVIRONMENT:-dev}
DB_IDENTIFIER="${PROJECT_NAME}-db-${ENVIRONMENT}"
CLUSTER_NAME="${PROJECT_NAME}-cluster-${ENVIRONMENT}"

echo "=== Stopping AWS Resources ==="
echo ""

# Stop RDS
echo "1. Stopping RDS instance: $DB_IDENTIFIER"
if aws rds describe-db-instances --db-instance-identifier "$DB_IDENTIFIER" --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null | grep -q "available"; then
    aws rds stop-db-instance --db-instance-identifier "$DB_IDENTIFIER"
    echo "   ✓ RDS stop initiated (takes 5-10 minutes)"
else
    echo "   ⚠ RDS already stopped or doesn't exist"
fi

# Scale ECS services to zero
echo ""
echo "2. Scaling ECS services to zero..."
for service in backend frontend; do
    if aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$service" --query 'services[0].serviceName' --output text 2>/dev/null | grep -q "$service"; then
        aws ecs update-service --cluster "$CLUSTER_NAME" --service "$service" --desired-count 0 > /dev/null 2>&1
        echo "   ✓ $service scaled to 0"
    else
        echo "   ⚠ $service service not found"
    fi
done

# Optionally delete ALB
echo ""
read -p "3. Delete ALB? (saves ~$21-26/month) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Deleting ALB..."
    cd "$(dirname "$0")"
    terraform destroy -target=aws_lb.main -target=aws_security_group.alb -auto-approve 2>/dev/null || echo "   ⚠ ALB already deleted or doesn't exist"
    echo "   ✓ ALB deleted"
else
    echo "   ⚠ ALB kept running (~$21-26/month)"
fi

echo ""
echo "=== Resources Stopped ==="
echo ""
echo "Estimated monthly savings:"
echo "  - RDS: ~\$20/month"
echo "  - ECS: ~\$7-10/month per task"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  - ALB: ~\$21-26/month"
fi
echo ""
echo "To start resources, run: ./start-resources.sh"
