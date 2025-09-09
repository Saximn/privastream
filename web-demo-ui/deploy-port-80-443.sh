#!/bin/bash

# PrivaStream.site Deployment on Ports 80/443 Only
# All services run internally, Nginx handles external access

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

echo "🚀 Deploying PrivaStream.site on Ports 80/443"
echo "=============================================="

# Check if running as root (needed for port 80/443)
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root to bind to ports 80/443"
   echo "Run with: sudo $0"
   exit 1
fi

# Get server public IP
print_status "Detecting server public IP..."
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "MANUAL_SET_REQUIRED")

if [ "$PUBLIC_IP" = "MANUAL_SET_REQUIRED" ]; then
    read -p "Could not detect public IP. Please enter your server's public IP: " PUBLIC_IP
fi

print_success "Server public IP: $PUBLIC_IP"

# Install Nginx if not present
if ! command -v nginx &> /dev/null; then
    print_status "Installing Nginx..."
    apt update
    apt install -y nginx curl lsof
    print_success "Nginx installed"
else
    print_success "Nginx is already installed"
fi

# Stop any existing services
print_status "Stopping existing services..."
systemctl stop nginx 2>/dev/null || true
pkill -f "python.*stream_backend.py" 2>/dev/null || true
pkill -f "python.*video_filter_api.py" 2>/dev/null || true
pkill -f "python.*audio_server_used.py" 2>/dev/null || true
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "npm.*start" 2>/dev/null || true

# Kill any processes using our internal ports
for port in 8000 8001 8002 8003 8004; do
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        kill -9 $pid 2>/dev/null && print_status "Freed port $port"
    fi
done

sleep 2

# Create environment files for internal ports
print_status "Setting up environment configuration..."

# Backend .env (port 8000)
cat > backend/.env << EOF
FLASK_HOST=127.0.0.1
FLASK_PORT=8000
FLASK_ENV=production
SECRET_KEY=$(openssl rand -base64 32)
SSL_ENABLED=false
MEDIASOUP_SERVER_URL=http://127.0.0.1:8003
VIDEO_SERVICE_URL=http://127.0.0.1:8001
AUDIO_SERVICE_URL=http://127.0.0.1:8002
FACE_ENROLLMENT_API_URL=http://127.0.0.1:8004
CORS_ALLOWED_ORIGINS=https://privastream.site,https://www.privastream.site
SOCKETIO_CORS_ALLOWED_ORIGINS=https://privastream.site,https://www.privastream.site
SOCKETIO_LOGGER=false
SOCKETIO_ENGINEIO_LOGGER=false
EOF

# MediaSoup .env (port 8003)
cat > mediasoup-server/.env << EOF
SERVER_HOST=127.0.0.1
SERVER_PORT=8003
NODE_ENV=production
SSL_ENABLED=false
ANNOUNCED_IP=$PUBLIC_IP
LISTEN_IP=0.0.0.0
RTC_MIN_PORT=10000
RTC_MAX_PORT=10100
REDACTION_SERVICE_URL=http://127.0.0.1:8002
VIDEO_SERVICE_URL=http://127.0.0.1:8001
PROCESSING_DELAY_MS=8000
FRAME_RATE=30
PROCESS_EVERY_NTH_FRAME=15
BUFFER_DURATION_MS=3000
CORS_ORIGINS=https://privastream.site,https://www.privastream.site
EOF

# Frontend .env (will be served by Nginx, no separate port needed)
cat > frontend/.env.local << EOF
NEXT_PUBLIC_API_URL=https://privastream.site/api
NEXT_PUBLIC_MEDIASOUP_URL=https://privastream.site/mediasoup
NEXT_PUBLIC_VIDEO_SERVICE_URL=https://privastream.site/video-api
NEXT_PUBLIC_FACE_ENROLLMENT_API_URL=https://privastream.site/face-api
NEXT_PUBLIC_REDACTION_SERVICE_URL=https://privastream.site/audio-api
EOF

print_success "Environment files created"

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

# Create logs directory
mkdir -p logs
chown -R $(logname):$(logname) logs 2>/dev/null || chown -R $SUDO_USER:$SUDO_USER logs 2>/dev/null || true

# Start services as non-root user if possible
REAL_USER=${SUDO_USER:-$(logname 2>/dev/null || echo "root")}

print_status "Starting services as user: $REAL_USER"

# Function to start service as specific user
start_service() {
    local name=$1
    local command=$2
    local logfile=$3
    local pidfile=$4
    local user=$5
    
    print_status "Starting $name..."
    
    if [ "$user" = "root" ]; then
        eval "$command" > "$logfile" 2>&1 &
    else
        su - "$user" -c "cd $(pwd) && $command" > "$logfile" 2>&1 &
    fi
    
    local pid=$!
    echo $pid > "$pidfile"
    sleep 2
    
    if kill -0 $pid 2>/dev/null; then
        print_success "$name started (PID: $pid)"
        return 0
    else
        print_error "$name failed to start. Check $logfile"
        return 1
    fi
}

# 1. Start Video Filter API (port 8001)
cd backend
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    su - "$REAL_USER" -c "cd $(pwd) && python3 -m venv venv"
fi
su - "$REAL_USER" -c "cd $(pwd) && source venv/bin/activate && pip install -q -r requirements.txt" || print_warning "Some pip packages may have failed"

FLASK_APP=video_filter_api.py FLASK_RUN_PORT=8001 su - "$REAL_USER" -c "cd $(pwd) && source venv/bin/activate && python -m flask run --host=127.0.0.1 --port=8001" > ../logs/video-service.log 2>&1 &
VIDEO_PID=$!
echo $VIDEO_PID > ../logs/video-service.pid
cd ..

# 2. Start Audio Service (port 8002)
cd audio-william
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment for audio..."
    su - "$REAL_USER" -c "cd $(pwd) && python3 -m venv venv"
fi
su - "$REAL_USER" -c "cd $(pwd) && source venv/bin/activate && pip install -q -r requirements.txt" || print_warning "Some pip packages may have failed"

# Modify audio_server_used.py to run on port 8002
sed -i 's/port=5002/port=8002/g' audio_server_used.py 2>/dev/null || true
sed -i 's/app.run(host="0.0.0.0", port=5002/app.run(host="127.0.0.1", port=8002/g' audio_server_used.py 2>/dev/null || true

su - "$REAL_USER" -c "cd $(pwd) && source venv/bin/activate && python audio_server_used.py" > ../logs/audio-service.log 2>&1 &
AUDIO_PID=$!
echo $AUDIO_PID > ../logs/audio-service.pid
cd ..

# 3. Start Stream Backend (port 8000)
cd backend
# Modify stream_backend.py to run on port 8000
sed -i 's/port=5000/port=8000/g' stream_backend.py 2>/dev/null || true
sed -i 's/app.run(host="0.0.0.0", port=5000/app.run(host="127.0.0.1", port=8000/g' stream_backend.py 2>/dev/null || true

su - "$REAL_USER" -c "cd $(pwd) && source venv/bin/activate && python stream_backend.py" > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
cd ..

# 4. Start MediaSoup Server (port 8003)
cd mediasoup-server
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    su - "$REAL_USER" -c "cd $(pwd) && npm install --production"
fi
su - "$REAL_USER" -c "cd $(pwd) && node server.js" > ../logs/mediasoup.log 2>&1 &
MEDIASOUP_PID=$!
echo $MEDIASOUP_PID > ../logs/mediasoup.pid
cd ..

# 5. Build Frontend (static files will be served by Nginx)
cd frontend
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies for frontend..."
    su - "$REAL_USER" -c "cd $(pwd) && npm install --production"
fi

print_status "Building frontend for production..."
su - "$REAL_USER" -c "cd $(pwd) && npm run build"
cd ..

# Wait for services to start
print_status "Waiting for services to initialize..."
sleep 10

# Create Nginx configuration
print_status "Creating Nginx configuration..."

cat > /etc/nginx/sites-available/privastream.site << EOF
# Rate limiting
limit_req_zone \$binary_remote_addr zone=general:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=websocket:10m rate=100r/s;

# Upstream servers (internal services)
upstream backend {
    server 127.0.0.1:8000;
}

upstream video_service {
    server 127.0.0.1:8001;
}

upstream audio_service {
    server 127.0.0.1:8002;
}

upstream mediasoup {
    server 127.0.0.1:8003;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name privastream.site www.privastream.site _;
    
    # Let's Encrypt challenge
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://\$host\$request_uri;
    }
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name privastream.site www.privastream.site _;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/privastream.site.crt;
    ssl_certificate_key /etc/nginx/ssl/privastream.site.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-XSS-Protection "1; mode=block" always;

    # CORS for WebRTC
    add_header Access-Control-Allow-Origin "https://privastream.site" always;
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization" always;

    # Frontend (Static Next.js build)
    root $(pwd)/frontend/out;
    index index.html;

    # Try static files first, then proxy to services
    location / {
        try_files \$uri \$uri/ @frontend;
    }

    # Fallback to serve Next.js files
    location @frontend {
        root $(pwd)/frontend/out;
        try_files \$uri \$uri/index.html /index.html;
    }

    # Backend API
    location /api/ {
        limit_req zone=general burst=50 nodelay;
        proxy_pass http://backend/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Socket.IO
    location /socket.io/ {
        limit_req zone=websocket burst=200 nodelay;
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
    }

    # MediaSoup WebRTC
    location /mediasoup/ {
        limit_req zone=websocket burst=200 nodelay;
        proxy_pass http://mediasoup/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Video Processing
    location /video-api/ {
        limit_req zone=general burst=100 nodelay;
        proxy_pass http://video_service/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        client_max_body_size 50M;
    }

    # Audio Processing
    location /audio-api/ {
        limit_req zone=general burst=100 nodelay;
        proxy_pass http://audio_service/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        client_max_body_size 50M;
    }

    # Health check
    location /health {
        access_log off;
        return 200 "PrivaStream.site healthy\\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Enable site and test config
ln -sf /etc/nginx/sites-available/privastream.site /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

if nginx -t; then
    print_success "Nginx configuration is valid"
else
    print_error "Nginx configuration has errors"
    exit 1
fi

# Start Nginx
systemctl enable nginx
systemctl restart nginx

print_success "🎉 PrivaStream.site is live on ports 80/443!"

echo ""
echo "🌐 Access Points:"
echo "================"
echo "🔗 Website: https://privastream.site"
echo "🔗 Alt: https://www.privastream.site" 
echo "🔒 SSL: Enabled"
echo "📡 WebRTC: Uses public IP $PUBLIC_IP:10000-10100"

echo ""
echo "🔧 Internal Services:"
echo "===================="
echo "• Stream Backend: 127.0.0.1:8000"
echo "• Video Filter API: 127.0.0.1:8001"
echo "• Audio Service: 127.0.0.1:8002"
echo "• MediaSoup Server: 127.0.0.1:8003"
echo "• Frontend: Served by Nginx (static)"

echo ""
echo "📊 Service Status:"
systemctl status nginx --no-pager -l
echo ""
ps aux | grep -E "(python.*(stream_backend|video_filter_api|audio_server_used)|node.*server.js)" | grep -v grep

echo ""
echo "🛠️ Management:"
echo "=============="
echo "• Stop all: sudo systemctl stop nginx && ./stop-privastream.sh"
echo "• View logs: tail -f logs/*.log"
echo "• Nginx logs: tail -f /var/log/nginx/error.log"
echo "• Reload Nginx: sudo systemctl reload nginx"

print_success "✅ Deployment complete!"
print_status "Visit https://privastream.site to test your platform"