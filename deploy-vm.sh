#!/bin/bash
# Deploy script for Azure VM

set -e

echo "=== KataDia ML Service Deployment Script ==="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
echo "Cloning repository..."
cd ~
git clone https://github.com/KataDia-Codabase/ai.git katadia-ml
cd katadia-ml

# Create .env file
echo "Creating environment variables..."
cat > .env << EOF
MYSQL_URI=mysql+pymysql://adminkatadia:123Hshi!@katadia-mysql.mysql.database.azure.com:3306/katadia_ml?charset=utf8mb4
REDIS_URL=rediss://:xkIliDVZlX1ZNAWmDDu72S5mjLADYzqYrAzCaHyFJZs=@katadia-redis.redis.cache.windows.net:6380/0?ssl_cert_reqs=required
DEBUG=False
LOG_LEVEL=INFO
EOF

# Build and start containers
echo "Building Docker images..."
docker-compose build

echo "Starting containers..."
docker-compose up -d

echo "=== Deployment Complete! ==="
echo "API accessible at: http://52.163.118.71"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f    # View logs"
echo "  docker-compose ps         # Check status"
echo "  docker-compose restart    # Restart services"
