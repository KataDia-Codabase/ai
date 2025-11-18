#!/bin/bash
# Auto-deploy script for VPS - will be triggered by GitHub webhook or cron

set -e

echo "=== KataDia ML Auto-Deploy ==="
echo "Timestamp: $(date)"

# Navigate to project directory
cd ~/katadia-ml

# Pull latest code from GitHub
echo "Pulling latest code from GitHub..."
git pull origin main

# Stop existing containers
echo "Stopping containers..."
docker-compose down || true

# Build new image locally
echo "Building Docker image locally..."
docker-compose build --no-cache

# Start containers
echo "Starting containers..."
docker-compose up -d

# Show status
echo "Checking container status..."
docker-compose ps

echo "=== Deployment Complete ==="
echo "API URL: http://52.163.118.71"
echo ""
echo "View logs: docker-compose logs -f"
