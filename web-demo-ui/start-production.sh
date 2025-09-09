#!/bin/bash

# Production Startup Script for TikTok TechJam Live Streaming Platform
# This script sets up SSL, configures environment variables, and starts all services

set -e  # Exit on any error

echo "🚀 Starting TikTok TechJam Live Streaming Platform (Production Mode)"
echo "=================================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as root (needed for port 80/443)
if [[ $EUID -eq 0 ]]; then
   echo -e "${YELLOW}⚠️  Running as root. This is required for ports 80/443 but not recommended.${NC}"
fi

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from .env.production template..."
    cp .env.production .env
    print_warning "⚠️  Please edit .env file with your production settings before continuing!"
    print_warning "   - Set your domain name"
    print_warning "   - Set your server's public IP"
    print_warning "   - Change the SECRET_KEY"
    print_warning "   - Configure SSL certificates"
    echo ""
    read -p "Press Enter when you've configured .env file..."
fi

# Load environment variables
source .env

# Validate required environment variables
print_status "Validating configuration..."

if [ "$DOMAIN" = "your-domain.com" ]; then
    print_error "Please set DOMAIN in your .env file to your actual domain!"
    exit 1
fi

if [ "$ANNOUNCED_IP" = "your-server-public-ip" ]; then
    print_error "Please set ANNOUNCED_IP in your .env file to your server's public IP!"
    exit 1
fi

if [ "$SECRET_KEY" = "your-very-secure-secret-key-change-this-random-string-123456789" ]; then
    print_error "Please change SECRET_KEY in your .env file to a secure random string!"
    exit 1
fi

print_success "Configuration validated"

# Check SSL certificates
print_status "Checking SSL certificates..."

if [ "$SSL_ENABLED" = "true" ]; then
    if [ ! -f "$SSL_CERT_PATH" ] || [ ! -f "$SSL_KEY_PATH" ]; then
        print_warning "SSL certificates not found. Running SSL setup..."
        ./setup-ssl.sh
    else
        print_success "SSL certificates found"
    fi
else
    print_warning "SSL is disabled. For production, it's recommended to enable SSL."
fi

# Check Docker and Docker Compose
print_status "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker is installed and ready"

# Stop any running containers
print_status "Stopping any existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Pull latest images (if using remote images)
print_status "Pulling latest images..."
docker-compose pull 2>/dev/null || true

# Build images
print_status "Building application images..."
docker-compose build

# Start services based on profile
PROFILE=""
if [ "$SSL_ENABLED" = "true" ]; then
    PROFILE="--profile production"
    print_status "Starting services with SSL and Nginx..."
else
    print_status "Starting services without Nginx (development mode)..."
fi

# Start the services
print_status "Starting all services..."
docker-compose up -d $PROFILE

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 10

# Check service health
print_status "Checking service health..."

SERVICES=("backend" "mediasoup" "frontend")
ALL_HEALTHY=true

for service in "${SERVICES[@]}"; do
    if docker-compose ps | grep -q "${service}.*Up"; then
        print_success "$service is running"
    else
        print_error "$service is not running"
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = true ]; then
    print_success "All services are running successfully!"
    
    echo ""
    echo "🎉 Platform is ready!"
    echo "===================="
    
    if [ "$SSL_ENABLED" = "true" ]; then
        echo "🌐 Web Interface: https://$DOMAIN"
        echo "📡 WebRTC Server: https://$DOMAIN/mediasoup/"
        echo "🎥 Video API: https://$DOMAIN/video-api/"
        echo "🎧 Audio API: https://$DOMAIN/audio-api/"
        echo "🔒 SSL: Enabled"
    else
        echo "🌐 Web Interface: http://localhost:$FRONTEND_PORT"
        echo "📡 WebRTC Server: http://localhost:$MEDIASOUP_PORT"
        echo "🎥 Video API: http://localhost:$VIDEO_SERVICE_PORT"
        echo "🎧 Audio API: http://localhost:$AUDIO_SERVICE_PORT"
        echo "🔒 SSL: Disabled (Development Mode)"
    fi
    
    echo ""
    echo "📊 Service Status:"
    echo "=================="
    docker-compose ps
    
    echo ""
    echo "📝 Usage:"
    echo "========="
    echo "1. Host: Go to the web interface and click 'Start Streaming'"
    echo "2. Share the room ID with viewers"
    echo "3. Viewers: Enter the room ID to join the stream"
    
    echo ""
    echo "🛠️  Management Commands:"
    echo "======================="
    echo "• View logs: docker-compose logs -f [service_name]"
    echo "• Stop services: docker-compose down"
    echo "• Restart service: docker-compose restart [service_name]"
    echo "• View status: docker-compose ps"
    
    if [ "$SSL_ENABLED" = "true" ]; then
        echo ""
        echo "🔧 SSL Certificate Renewal:"
        echo "=========================="
        echo "• For Let's Encrypt: Set up auto-renewal with cron"
        echo "• Manual renewal: ./setup-ssl.sh"
    fi
    
else
    print_error "Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Optional: Show live logs
echo ""
read -p "Show live logs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Showing live logs (Ctrl+C to exit)..."
    docker-compose logs -f
fi