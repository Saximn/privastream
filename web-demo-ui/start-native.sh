#!/bin/bash

# Native deployment script for PrivaStream.site
# Runs all services directly on the server without Docker

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

echo "🚀 Starting PrivaStream.site (Native Mode)"
echo "=========================================="

# Get server public IP
print_status "Detecting server public IP..."
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "127.0.0.1")
print_success "Server public IP: $PUBLIC_IP"

# Kill any existing processes
print_status "Stopping any existing services..."
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "python.*audio_redaction_server" 2>/dev/null || true
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "npm.*start" 2>/dev/null || true
pkill -f "next" 2>/dev/null || true
pkill -f "nginx" 2>/dev/null || true
sleep 2

# Create environment files
print_status "Setting up environment configuration..."

# Backend .env
cat > backend/.env << EOF
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_ENV=production
SECRET_KEY=$(openssl rand -base64 32)
SSL_ENABLED=false
MEDIASOUP_SERVER_URL=http://127.0.0.1:3001
VIDEO_SERVICE_URL=http://127.0.0.1:5001
AUDIO_SERVICE_URL=http://127.0.0.1:5002
FACE_ENROLLMENT_API_URL=http://127.0.0.1:5003
CORS_ALLOWED_ORIGINS=https://privastream.site,https://www.privastream.site,http://localhost:3000
SOCKETIO_CORS_ALLOWED_ORIGINS=https://privastream.site,https://www.privastream.site,http://localhost:3000
SOCKETIO_LOGGER=false
SOCKETIO_ENGINEIO_LOGGER=false
EOF

# MediaSoup .env
cat > mediasoup-server/.env << EOF
SERVER_HOST=127.0.0.1
SERVER_PORT=3001
NODE_ENV=production
SSL_ENABLED=false
ANNOUNCED_IP=$PUBLIC_IP
LISTEN_IP=0.0.0.0
RTC_MIN_PORT=10000
RTC_MAX_PORT=10100
REDACTION_SERVICE_URL=http://127.0.0.1:5002
VIDEO_SERVICE_URL=http://127.0.0.1:5001
PROCESSING_DELAY_MS=8000
FRAME_RATE=30
PROCESS_EVERY_NTH_FRAME=15
BUFFER_DURATION_MS=3000
CORS_ORIGINS=https://privastream.site,https://www.privastream.site,http://localhost:3000
EOF

# Frontend .env
cat > frontend/.env.local << EOF
NEXT_PUBLIC_API_URL=http://127.0.0.1:5000
NEXT_PUBLIC_MEDIASOUP_URL=http://127.0.0.1:3001
NEXT_PUBLIC_VIDEO_SERVICE_URL=http://127.0.0.1:5001
NEXT_PUBLIC_FACE_ENROLLMENT_API_URL=http://127.0.0.1:5003
NEXT_PUBLIC_REDACTION_SERVICE_URL=http://127.0.0.1:5002
EOF

print_success "Environment files created"

# Function to check if port is available
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        print_warning "Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to start service in background
start_service() {
    local name=$1
    local command=$2
    local logfile=$3
    local pidfile=$4
    
    print_status "Starting $name..."
    
    # Start the service
    eval "$command" > "$logfile" 2>&1 &
    local pid=$!
    echo $pid > "$pidfile"
    
    # Wait a moment and check if it's still running
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        print_success "$name started (PID: $pid)"
        return 0
    else
        print_error "$name failed to start. Check $logfile"
        return 1
    fi
}

# Check required ports
print_status "Checking port availability..."
for port in 5000 5001 5002 3001 3000; do
    if ! check_port $port; then
        print_error "Port $port is in use. Please stop the service using it."
        exit 1
    fi
done

# Create logs directory
mkdir -p logs

# 1. Start Video Service (Backend on port 5001)
print_status "Starting Video Service..."
cd backend
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment for video service..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || print_warning "Some pip packages may have failed"

# Start video service
FLASK_APP=video_app.py FLASK_RUN_PORT=5001 python -m flask run --host=127.0.0.1 --port=5001 > ../logs/video-service.log 2>&1 &
VIDEO_PID=$!
echo $VIDEO_PID > ../logs/video-service.pid
cd ..

# 2. Start Audio Service (audio-william)
print_status "Starting Audio Service..."
cd audio-william
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment for audio service..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || print_warning "Some pip packages may have failed"

# Start audio service
python audio_redaction_server_faster_whisper.py > ../logs/audio-service.log 2>&1 &
AUDIO_PID=$!
echo $AUDIO_PID > ../logs/audio-service.pid
cd ..

# 3. Start Backend (Flask + Socket.IO)
print_status "Starting Backend Service..."
cd backend
source venv/bin/activate
python app.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
cd ..

# 4. Start MediaSoup Server
print_status "Starting MediaSoup Server..."
cd mediasoup-server
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies for MediaSoup..."
    npm install
fi
node server.js > ../logs/mediasoup.log 2>&1 &
MEDIASOUP_PID=$!
echo $MEDIASOUP_PID > ../logs/mediasoup.pid
cd ..

# 5. Start Frontend
print_status "Starting Frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies for frontend..."
    npm install
fi

# Build for production
if [ ! -d ".next" ] || [ "$1" = "--build" ]; then
    print_status "Building frontend for production..."
    npm run build
fi

npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
cd ..

# Wait for services to start
print_status "Waiting for services to initialize..."
sleep 10

# Check if services are running
print_status "Checking service health..."
services_healthy=true

check_service() {
    local name=$1
    local pid_file=$2
    local url=$3
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            if curl -s "$url" > /dev/null 2>&1; then
                print_success "$name is healthy (PID: $pid)"
            else
                print_warning "$name is running but not responding at $url"
            fi
        else
            print_error "$name is not running"
            services_healthy=false
        fi
    else
        print_error "$name PID file not found"
        services_healthy=false
    fi
}

check_service "Video Service" "logs/video-service.pid" "http://127.0.0.1:5001/health"
check_service "Audio Service" "logs/audio-service.pid" "http://127.0.0.1:5002/health"
check_service "Backend" "logs/backend.pid" "http://127.0.0.1:5000/health"
check_service "MediaSoup" "logs/mediasoup.pid" "http://127.0.0.1:3001/health"
check_service "Frontend" "logs/frontend.pid" "http://127.0.0.1:3000"

if [ "$services_healthy" = true ]; then
    print_success "🎉 PrivaStream.site is running!"
    
    echo ""
    echo "📱 Access Points:"
    echo "================"
    echo "🌐 Frontend: http://127.0.0.1:3000"
    echo "🔌 Backend API: http://127.0.0.1:5000"
    echo "📡 MediaSoup: http://127.0.0.1:3001"
    echo "🎥 Video API: http://127.0.0.1:5001"
    echo "🎧 Audio API: http://127.0.0.1:5002"
    
    echo ""
    echo "🔥 Service Status:"
    echo "=================="
    ps aux | grep -E "(python.*app.py|python.*audio_redaction|node.*server.js|npm.*start)" | grep -v grep | awk '{printf "%-20s PID: %-8s CMD: %s\n", $1, $2, $11}'
    
    echo ""
    echo "📊 Real-time Logs:"
    echo "=================="
    echo "• All logs: tail -f logs/*.log"
    echo "• Backend: tail -f logs/backend.log"
    echo "• Video: tail -f logs/video-service.log"
    echo "• Audio: tail -f logs/audio-service.log"
    echo "• MediaSoup: tail -f logs/mediasoup.log"
    echo "• Frontend: tail -f logs/frontend.log"
    
    echo ""
    echo "🛑 Stop Services:"
    echo "================"
    echo "./stop-native.sh"
    
    # Save PIDs for easy stopping
    cat > logs/all-pids.txt << EOF
$VIDEO_PID
$AUDIO_PID
$BACKEND_PID
$MEDIASOUP_PID
$FRONTEND_PID
EOF
    
    echo ""
    echo "🔧 For Production (Public Access):"
    echo "=================================="
    echo "1. Set up reverse proxy (Nginx/Apache) for HTTPS"
    echo "2. Point privastream.site to this server"
    echo "3. Configure firewall for ports 80, 443, 10000-10100"
    
else
    print_error "Some services failed to start. Check logs in logs/ directory"
    exit 1
fi

# Option to show logs
echo ""
read -p "Show live logs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Showing live logs (Ctrl+C to exit)..."
    tail -f logs/*.log
fi