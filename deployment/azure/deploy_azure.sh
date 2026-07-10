#!/bin/bash
# ─────────────────────────────────────────────────────────────
# deploy_azure.sh
# Milestone 3 – Full Azure Deployment Script
#
# Prerequisites:
#   • Azure CLI installed & logged in (az login)
#   • Docker installed and running
#   • Run from project root directory
# ─────────────────────────────────────────────────────────────

set -e  # Exit on any error

# ── Config (edit these) ───────────────────────────────────────
RESOURCE_GROUP="demand-forecasting-rg"
LOCATION="eastus"
ENVIRONMENT="prod"
IMAGE_TAG="latest"
APP_NAME="demand-forecasting"

echo "╔══════════════════════════════════════════════╗"
echo "║   Demand Forecasting – Azure Deployment      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 1: Login check ───────────────────────────────────────
echo "Step 1: Verifying Azure login..."
az account show --query "user.name" -o tsv || { echo "❌ Please run 'az login' first"; exit 1; }
echo "✅ Logged in"

# ── Step 2: Create Resource Group ────────────────────────────
echo ""
echo "Step 2: Creating Resource Group..."
az group create \
  --name     "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output   table
echo "✅ Resource Group: $RESOURCE_GROUP"

# ── Step 3: Deploy Azure Infrastructure ──────────────────────
echo ""
echo "Step 3: Deploying Azure infrastructure (Bicep)..."
DEPLOYMENT_OUTPUT=$(az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --template-file  "deployment/azure/azure_deploy.bicep" \
  --parameters     environment="$ENVIRONMENT" imageTag="$IMAGE_TAG" \
  --query          "properties.outputs" \
  --output         json)

ACR_SERVER=$(echo $DEPLOYMENT_OUTPUT | python3 -c "import sys,json; print(json.load(sys.stdin)['acrLoginServer']['value'])")
API_URL=$(echo    $DEPLOYMENT_OUTPUT | python3 -c "import sys,json; print(json.load(sys.stdin)['apiUrl']['value'])")
echo "✅ Infrastructure deployed"
echo "   ACR: $ACR_SERVER"

# ── Step 4: Build Docker Image ────────────────────────────────
echo ""
echo "Step 4: Building Docker image..."
docker build \
  -t "${APP_NAME}:${IMAGE_TAG}" \
  -f deployment/docker/Dockerfile \
  .
echo "✅ Docker image built"

# ── Step 5: Push to Azure Container Registry ─────────────────
echo ""
echo "Step 5: Pushing image to ACR..."
az acr login --name "${ACR_SERVER%%.*}"

docker tag "${APP_NAME}:${IMAGE_TAG}" "${ACR_SERVER}/${APP_NAME}:${IMAGE_TAG}"
docker push "${ACR_SERVER}/${APP_NAME}:${IMAGE_TAG}"
echo "✅ Image pushed to ACR"

# ── Step 6: Verify Deployment ─────────────────────────────────
echo ""
echo "Step 6: Verifying deployment..."
sleep 15  # Wait for container to start

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health" || echo "000")
if [ "$HTTP_STATUS" == "200" ]; then
    echo "✅ API is live and healthy!"
else
    echo "⚠️  API returned status: $HTTP_STATUS (may still be starting)"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║              Deployment Complete!             ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  API URL    : $API_URL"
echo "║  Docs URL   : ${API_URL}/docs"
echo "║  Health URL : ${API_URL}/health"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Test your API:"
echo "  curl ${API_URL}/health"
echo "  curl ${API_URL}/metrics"
echo "  curl -X POST ${API_URL}/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"date\":\"2024-06-01\",\"store_id\":1,\"product_id\":5,\"category\":\"Electronics\",\"store_type\":\"Urban\",\"price\":149.99}'"
