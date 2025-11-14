#!/bin/bash
# Script to configure Route 53 DNS record pointing to ALB

set -e

PROJECT_NAME=${PROJECT_NAME:-golf-coach}
ENVIRONMENT=${ENVIRONMENT:-dev}
DOMAIN_NAME=${DOMAIN_NAME:-""}  # Your domain, e.g., example.com
SUBDOMAIN=${SUBDOMAIN:-""}       # Optional subdomain, e.g., "app" for app.example.com

if [ -z "$DOMAIN_NAME" ]; then
    echo "Error: DOMAIN_NAME environment variable is required"
    echo "Usage: DOMAIN_NAME=example.com SUBDOMAIN=app ./setup-route53.sh"
    exit 1
fi

echo "=== Configuring Route 53 DNS ==="
echo ""

# Get ALB DNS name from Terraform output
cd "$(dirname "$0")"
ALB_DNS=$(terraform output -raw alb_dns_name 2>/dev/null || echo "")
ALB_ZONE_ID=$(aws elbv2 describe-load-balancers --names "${PROJECT_NAME}-alb-${ENVIRONMENT}" --query 'LoadBalancers[0].CanonicalHostedZoneId' --output text 2>/dev/null || echo "")

if [ -z "$ALB_DNS" ] || [ -z "$ALB_ZONE_ID" ]; then
    echo "Error: Could not find ALB. Make sure Terraform has been applied."
    exit 1
fi

echo "ALB DNS: $ALB_DNS"
echo "ALB Zone ID: $ALB_ZONE_ID"
echo ""

# Get hosted zone ID for the domain
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones --query "HostedZones[?Name=='${DOMAIN_NAME}.'].Id" --output text | cut -d'/' -f3)

if [ -z "$HOSTED_ZONE_ID" ]; then
    echo "Error: Could not find Route 53 hosted zone for ${DOMAIN_NAME}"
    echo "Make sure the domain is managed in Route 53"
    exit 1
fi

echo "Hosted Zone ID: $HOSTED_ZONE_ID"
echo ""

# Determine record name
if [ -n "$SUBDOMAIN" ]; then
    RECORD_NAME="${SUBDOMAIN}.${DOMAIN_NAME}"
else
    RECORD_NAME="${DOMAIN_NAME}"
fi

echo "Creating/updating DNS record: ${RECORD_NAME} -> ${ALB_DNS}"
echo ""

# Create or update the record
aws route53 change-resource-record-sets \
    --hosted-zone-id "$HOSTED_ZONE_ID" \
    --change-batch "{
        \"Changes\": [{
            \"Action\": \"UPSERT\",
            \"ResourceRecordSet\": {
                \"Name\": \"${RECORD_NAME}\",
                \"Type\": \"A\",
                \"AliasTarget\": {
                    \"DNSName\": \"${ALB_DNS}\",
                    \"EvaluateTargetHealth\": true,
                    \"HostedZoneId\": \"${ALB_ZONE_ID}\"
                }
            }
        }]
    }"

echo ""
echo "=== Route 53 Configuration Complete ==="
echo ""
echo "DNS record created: ${RECORD_NAME} -> ${ALB_DNS}"
echo "DNS propagation may take a few minutes"

