#!/bin/bash

# PrivaStream.site Deployment Script
# Single-machine deployment with all services on localhost, only frontend publicly accessible

set -e

echo "🚀 Deploying PrivaStream.site"
echo "============================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get server public IP
print_status "Detecting server public IP..."
PUBLIC_IP=$(curl -s ifconfig.me || curl -s ipinfo.io/ip || echo "MANUAL_SET_REQUIRED")

if [ "$PUBLIC_IP" = "MANUAL_SET_REQUIRED" ]; then
    read -p "Could not detect public IP. Please enter your server's public IP: " PUBLIC_IP
fi

print_success "Server public IP: $PUBLIC_IP"

# Create environment file
print_status "Creating environment configuration..."
cat > .env << EOF
# PrivaStream.site Single Machine Configuration
DOMAIN=privastream.site
SERVER_PUBLIC_IP=$PUBLIC_IP

# SSL Configuration
SSL_ENABLED=true
SSL_CERT_PATH=./ssl/cert.pem
SSL_KEY_PATH=./ssl/key.pem

# Security (CHANGE THIS!)
SECRET_KEY=$(openssl rand -base64 32)

# Internal Service Communication
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
MEDIASOUP_SERVER_URL=http://127.0.0.1:3001
VIDEO_SERVICE_URL=http://127.0.0.1:5001
AUDIO_SERVICE_URL=http://127.0.0.1:5002
FACE_ENROLLMENT_API_URL=http://127.0.0.1:5003

# MediaSoup Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=3001
ANNOUNCED_IP=$PUBLIC_IP
LISTEN_IP=0.0.0.0
RTC_MIN_PORT=10000
RTC_MAX_PORT=10100

# Frontend Environment Variables (Public URLs)
NEXT_PUBLIC_API_URL=https://privastream.site/api
NEXT_PUBLIC_MEDIASOUP_URL=https://privastream.site/mediasoup
NEXT_PUBLIC_VIDEO_SERVICE_URL=https://privastream.site/video-api
NEXT_PUBLIC_FACE_ENROLLMENT_API_URL=https://privastream.site/face-api
NEXT_PUBLIC_REDACTION_SERVICE_URL=https://privastream.site/audio-api

# CORS Configuration
CORS_ALLOWED_ORIGINS=https://privastream.site,https://www.privastream.site
SOCKETIO_CORS_ALLOWED_ORIGINS=https://privastream.site,https://www.privastream.site

# Processing Configuration
PROCESSING_DELAY_MS=8000
FRAME_RATE=30
PROCESS_EVERY_NTH_FRAME=15
BUFFER_DURATION_MS=3000
EOF

print_success "Environment configuration created"

# SSL Setup
print_status "Setting up SSL certificates..."
if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
    print_warning "SSL certificates not found. Setting up..."
    
    echo "Choose SSL setup:"
    echo "1) Let's Encrypt (recommended for production)"
    echo "2) Self-signed certificates (development/testing)"
    
    read -p "Enter choice (1-2): " ssl_choice
    
    case $ssl_choice in
        1)
            # Let's Encrypt setup
            if command -v certbot &> /dev/null; then
                print_status "Using Let's Encrypt for SSL certificates..."
                sudo certbot certonly --standalone -d privastream.site -d www.privastream.site --non-interactive --agree-tos --email admin@privastream.site
                
                mkdir -p ssl
                sudo cp /etc/letsencrypt/live/privastream.site/fullchain.pem ssl/cert.pem
                sudo cp /etc/letsencrypt/live/privastream.site/privkey.pem ssl/key.pem
                sudo chown $USER:$USER ssl/cert.pem ssl/key.pem
                chmod 644 ssl/cert.pem
                chmod 600 ssl/key.pem
                
                print_success "Let's Encrypt certificates installed"
            else
                print_error "certbot not found. Install with: apt-get install certbot"
                exit 1
            fi
            ;;
        2)
            # Self-signed certificates
            mkdir -p ssl
            openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=PrivaStream/CN=privastream.site"
            chmod 600 ssl/key.pem
            chmod 644 ssl/cert.pem
            print_warning "Self-signed certificates created (browsers will show security warning)"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
else
    print_success "SSL certificates found"
fi

# Check Docker
print_status "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "Docker Compose not found. Install Docker Compose first."
    exit 1
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.single-machine.yml down --remove-orphans 2>/dev/null || true

# Build and start services
print_status "Building and starting services..."
docker-compose -f docker-compose.single-machine.yml up -d --build

# Wait for services
print_status "Waiting for services to start..."
sleep 15

# Check health
print_status "Checking service health..."
HEALTHY=true

services=("backend" "mediasoup" "frontend" "nginx" "video-service" "audio-service")
for service in "${services[@]}"; do
    if docker-compose -f docker-compose.single-machine.yml ps | grep -q "${service}.*Up"; then
        print_success "$service is running"
    else
        print_error "$service is not running"
        HEALTHY=false
    fi
done

if [ "$HEALTHY" = true ]; then
    print_success "🎉 PrivaStream.site is live!"
    
    echo ""
    echo "📱 Access your platform:"
    echo "======================="
    echo "🌐 Website: https://privastream.site"
    echo "🔒 SSL: Enabled"
    echo "📡 WebRTC Ports: 10000-10100/udp"
    
    echo ""
    echo "🔧 Internal Services (localhost only):"
    echo "======================================"
    echo "• Backend: http://127.0.0.1:5000"
    echo "• MediaSoup: http://127.0.0.1:3001"
    echo "• Video Service: http://127.0.0.1:5001"
    echo "• Audio Service: http://127.0.0.1:5002"
    
    echo ""
    echo "📊 Service Status:"
    docker-compose -f docker-compose.single-machine.yml ps
    
    echo ""
    echo "🛠️  Management:"
    echo "=============="
    echo "• View logs: docker-compose -f docker-compose.single-machine.yml logs -f [service]"
    echo "• Stop: docker-compose -f docker-compose.single-machine.yml down"
    echo "• Restart: docker-compose -f docker-compose.single-machine.yml restart [service]"
    
    echo ""
    echo "🚪 Firewall Requirements:"
    echo "========================"
    echo "• Port 80 (HTTP redirect to HTTPS)"
    echo "• Port 443 (HTTPS)"
    echo "• Ports 10000-10100/UDP (WebRTC)"
    
    if [ "$ssl_choice" = "1" ]; then
        echo ""
        echo "🔄 SSL Certificate Auto-Renewal:"
        echo "==============================="
        echo "Add to crontab: 0 12 * * * /usr/bin/certbot renew --quiet"
    fi
    
else
    print_error "Some services failed to start. Check logs:"
    echo "docker-compose -f docker-compose.single-machine.yml logs"
    exit 1
fi