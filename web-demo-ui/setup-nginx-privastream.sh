#!/bin/bash

# Setup Nginx reverse proxy for PrivaStream.site production

set -e

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

echo "🌐 Setting up Nginx for PrivaStream.site"
echo "========================================"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (for port 80/443)"
   echo "Run with: sudo $0"
   exit 1
fi

# Install Nginx if not present
if ! command -v nginx &> /dev/null; then
    print_status "Installing Nginx..."
    apt update
    apt install -y nginx
    print_success "Nginx installed"
else
    print_success "Nginx is already installed"
fi

# Get server public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "MANUAL_SET_REQUIRED")

if [ "$PUBLIC_IP" = "MANUAL_SET_REQUIRED" ]; then
    read -p "Could not detect public IP. Please enter your server's public IP: " PUBLIC_IP
fi

print_success "Server public IP: $PUBLIC_IP"

# Setup SSL certificates
print_status "Setting up SSL certificates..."
mkdir -p /etc/nginx/ssl

if [ ! -f "/etc/nginx/ssl/privastream.site.crt" ] || [ ! -f "/etc/nginx/ssl/privastream.site.key" ]; then
    echo "Choose SSL setup:"
    echo "1) Let's Encrypt (recommended for production)"
    echo "2) Self-signed certificates (development/testing)"
    
    read -p "Enter choice (1-2): " ssl_choice
    
    case $ssl_choice in
        1)
            # Install certbot
            if ! command -v certbot &> /dev/null; then
                print_status "Installing certbot..."
                apt install -y certbot python3-certbot-nginx
            fi
            
            # Get Let's Encrypt certificate
            print_status "Getting Let's Encrypt certificate for privastream.site..."
            certbot certonly --standalone -d privastream.site -d www.privastream.site \
                --non-interactive --agree-tos --email admin@privastream.site
            
            # Copy certificates to nginx location
            cp /etc/letsencrypt/live/privastream.site/fullchain.pem /etc/nginx/ssl/privastream.site.crt
            cp /etc/letsencrypt/live/privastream.site/privkey.pem /etc/nginx/ssl/privastream.site.key
            
            # Set up auto-renewal
            echo "0 12 * * * /usr/bin/certbot renew --quiet && systemctl reload nginx" | crontab -
            print_success "SSL certificates installed with auto-renewal"
            ;;
        2)
            # Generate self-signed certificate
            openssl req -x509 -newkey rsa:4096 -keyout /etc/nginx/ssl/privastream.site.key \
                -out /etc/nginx/ssl/privastream.site.crt -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=PrivaStream/CN=privastream.site"
            print_warning "Self-signed certificates created (browsers will show security warning)"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
else
    print_success "SSL certificates already exist"
fi

# Set proper permissions
chmod 600 /etc/nginx/ssl/privastream.site.key
chmod 644 /etc/nginx/ssl/privastream.site.crt

# Create Nginx configuration
print_status "Creating Nginx configuration..."

cat > /etc/nginx/sites-available/privastream.site << 'EOF'
# Rate limiting
limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=websocket:10m rate=100r/s;

# Upstream servers (local services)
upstream frontend {
    server 127.0.0.1:3000;
}

upstream backend {
    server 127.0.0.1:5000;
}

upstream mediasoup {
    server 127.0.0.1:3001;
}

upstream video_service {
    server 127.0.0.1:5001;
}

upstream audio_service {
    server 127.0.0.1:5002;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name privastream.site www.privastream.site;
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://$host$request_uri;
    }
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name privastream.site www.privastream.site;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/privastream.site.crt;
    ssl_certificate_key /etc/nginx/ssl/privastream.site.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # CORS headers for WebRTC
    add_header Access-Control-Allow-Origin "https://privastream.site" always;
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization" always;

    # Handle preflight requests
    if ($request_method = 'OPTIONS') {
        add_header Access-Control-Allow-Origin "https://privastream.site";
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization";
        return 204;
    }

    # Frontend (Next.js) - Main application
    location / {
        limit_req zone=general burst=20 nodelay;
        
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Backend API (Flask + Socket.IO)
    location /api/ {
        limit_req zone=general burst=50 nodelay;
        
        proxy_pass http://backend/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Socket.IO WebSocket connections
    location /socket.io/ {
        limit_req zone=websocket burst=200 nodelay;
        
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
    }

    # MediaSoup WebRTC Server
    location /mediasoup/ {
        limit_req zone=websocket burst=200 nodelay;
        
        proxy_pass http://mediasoup/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Video Processing Service
    location /video-api/ {
        limit_req zone=general burst=100 nodelay;
        
        proxy_pass http://video_service/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        client_max_body_size 50M;
    }

    # Audio Processing Service
    location /audio-api/ {
        limit_req zone=general burst=100 nodelay;
        
        proxy_pass http://audio_service/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        client_max_body_size 50M;
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    # Static files with caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        proxy_pass http://frontend;
        expires 1M;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable the site
ln -sf /etc/nginx/sites-available/privastream.site /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
print_status "Testing Nginx configuration..."
if nginx -t; then
    print_success "Nginx configuration is valid"
else
    print_error "Nginx configuration has errors"
    exit 1
fi

# Start/restart Nginx
print_status "Starting Nginx..."
systemctl enable nginx
systemctl restart nginx

if systemctl is-active --quiet nginx; then
    print_success "Nginx is running"
else
    print_error "Failed to start Nginx"
    exit 1
fi

# Configure firewall (if ufw is installed)
if command -v ufw &> /dev/null; then
    print_status "Configuring firewall..."
    ufw allow 22/tcp      # SSH
    ufw allow 80/tcp      # HTTP
    ufw allow 443/tcp     # HTTPS
    ufw allow 10000:10100/udp  # WebRTC
    print_success "Firewall configured"
fi

print_success "🎉 Nginx setup complete!"

echo ""
echo "📱 PrivaStream.site is now accessible at:"
echo "========================================"
echo "🌐 https://privastream.site"
echo "🔒 SSL: Enabled"
echo "🔥 HTTP redirects to HTTPS"

echo ""
echo "🛠️  Management Commands:"
echo "======================"
echo "• Check Nginx status: systemctl status nginx"
echo "• Reload Nginx: systemctl reload nginx"
echo "• View Nginx logs: tail -f /var/log/nginx/error.log"
echo "• Test config: nginx -t"

echo ""
echo "🔧 Next Steps:"
echo "============="
echo "1. Make sure DNS points privastream.site to $PUBLIC_IP"
echo "2. Start your services with: ./start-native.sh"
echo "3. Test: https://privastream.site"

if [ "$ssl_choice" = "1" ]; then
    echo ""
    echo "🔄 SSL Auto-Renewal:"
    echo "==================="
    echo "• Certificates will auto-renew via cron"
    echo "• Manual renewal: certbot renew"
fi