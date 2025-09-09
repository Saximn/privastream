#!/bin/bash

# PrivaStream.site Single Machine Deployment Script
# Uses the correct service files

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

echo "🚀 Starting PrivaStream.site (Single Machine)"
echo "============================================="

# Get server public IP
print_status "Detecting server public IP..."
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "127.0.0.1")
print_success "Server public IP: $PUBLIC_IP"

# Kill any existing processes
print_status "Stopping any existing services..."
pkill -f "python.*stream_backend.py" 2>/dev/null || true
pkill -f "python.*video_filter_api.py" 2>/dev/null || true
pkill -f "python.*audio_server_used.py" 2>/dev/null || true
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "npm.*start" 2>/dev/null || true
pkill -f "next" 2>/dev/null || true
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
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "Port $port is already in use"
        return 1
    fi
    return 0
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

# 1. Start Video Service (backend/video_filter_api.py on port 5001)
print_status "Starting Video Filter API Service..."
cd backend
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment for backend services..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || print_warning "Some pip packages may have failed"

# Set environment for video service to run on port 5001
FLASK_APP=video_filter_api.py FLASK_RUN_PORT=5001 python -m flask run --host=127.0.0.1 --port=5001 > ../logs/video-service.log 2>&1 &
VIDEO_PID=$!
echo $VIDEO_PID > ../logs/video-service.pid
print_success "Video Filter API started (PID: $VIDEO_PID)"
cd ..

# 2. Start Audio Service (audio-william/audio_server_used.py on port 5002)
print_status "Starting Audio Service..."
cd audio-william
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment for audio service..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || print_warning "Some pip packages may have failed"

# Start audio service
python audio_server_used.py > ../logs/audio-service.log 2>&1 &
AUDIO_PID=$!
echo $AUDIO_PID > ../logs/audio-service.pid
print_success "Audio Service started (PID: $AUDIO_PID)"
cd ..

# 3. Start Backend (backend/stream_backend.py on port 5000)
print_status "Starting Stream Backend Service..."
cd backend
source venv/bin/activate
python stream_backend.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
print_success "Stream Backend started (PID: $BACKEND_PID)"
cd ..

# 4. Start MediaSoup Server (mediasoup-server/server.js on port 3001)
print_status "Starting MediaSoup Server..."
cd mediasoup-server
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies for MediaSoup..."
    npm install --production
fi
node server.js > ../logs/mediasoup.log 2>&1 &
MEDIASOUP_PID=$!
echo $MEDIASOUP_PID > ../logs/mediasoup.pid
print_success "MediaSoup Server started (PID: $MEDIASOUP_PID)"
cd ..

# 5. Start Frontend (frontend/ on port 3000)
print_status "Starting Frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies for frontend..."
    npm install --production
fi

# Build for production if needed
if [ ! -d ".next" ] || [ "$1" = "--build" ]; then
    print_status "Building frontend for production..."
    npm run build
fi

npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
print_success "Frontend started (PID: $FRONTEND_PID)"
cd ..

# Wait for services to initialize
print_status "Waiting for services to initialize..."
sleep 10

# Check service health
print_status "Checking service health..."
services_healthy=true

check_service() {
    local name=$1
    local pid_file=$2
    local url=$3
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
                print_success "$name is healthy (PID: $pid)"
                return 0
            else
                print_warning "$name is running but not responding at $url (PID: $pid)"
                return 1
            fi
        else
            print_error "$name is not running"
            services_healthy=false
            return 1
        fi
    else
        print_error "$name PID file not found"
        services_healthy=false
        return 1
    fi
}

# Check all services
check_service "Video Filter API" "logs/video-service.pid" "http://127.0.0.1:5001/health"
check_service "Audio Service" "logs/audio-service.pid" "http://127.0.0.1:5002/health"
check_service "Stream Backend" "logs/backend.pid" "http://127.0.0.1:5000/health"
check_service "MediaSoup Server" "logs/mediasoup.pid" "http://127.0.0.1:3001/health"
check_service "Frontend" "logs/frontend.pid" "http://127.0.0.1:3000"

# Show results
if [ "$services_healthy" = true ]; then
    print_success "🎉 PrivaStream.site is running!"
    
    echo ""
    echo "📱 Local Access (Development):"
    echo "============================="
    echo "🌐 Frontend: http://127.0.0.1:3000"
    echo "🔌 Stream Backend: http://127.0.0.1:5000"
    echo "📡 MediaSoup Server: http://127.0.0.1:3001"
    echo "🎥 Video Filter API: http://127.0.0.1:5001"
    echo "🎧 Audio Service: http://127.0.0.1:5002"
    
    echo ""
    echo "🌍 For Public Access (Production):"
    echo "=================================="
    echo "1. Set up Nginx reverse proxy: sudo ./setup-nginx-privastream.sh"
    echo "2. Make sure DNS points privastream.site to $PUBLIC_IP"
    echo "3. Configure firewall: ports 80, 443, 10000-10100"
    echo "4. Access via: https://privastream.site"
    
    echo ""
    echo "🔥 Service Status:"
    echo "=================="
    echo "Process Status:"
    ps aux | grep -E "(python.*stream_backend|python.*video_filter_api|python.*audio_server_used|node.*server.js|npm.*start)" | grep -v grep | awk '{printf "%-15s PID: %-8s CPU: %-6s MEM: %-6s CMD: %s\n", $1, $2, $3, $4, $11}' || echo "No processes found in ps output"
    
    echo ""
    echo "Port Status:"
    for port in 3000 3001 5000 5001 5002; do
        if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
            echo "Port $port: IN USE ✅"
        else
            echo "Port $port: FREE ❌"
        fi
    done
    
    echo ""
    echo "📊 Real-time Logs:"
    echo "=================="
    echo "• All logs: tail -f logs/*.log"
    echo "• Stream Backend: tail -f logs/backend.log"
    echo "• Video Filter: tail -f logs/video-service.log"
    echo "• Audio Service: tail -f logs/audio-service.log"
    echo "• MediaSoup: tail -f logs/mediasoup.log"
    echo "• Frontend: tail -f logs/frontend.log"
    
    echo ""
    echo "🛑 Stop All Services:"
    echo "===================="
    echo "./stop-privastream.sh"
    
    # Save all PIDs for easy stopping
    cat > logs/all-pids.txt << EOF
$VIDEO_PID
$AUDIO_PID
$BACKEND_PID
$MEDIASOUP_PID
$FRONTEND_PID
EOF
    
else
    print_error "❌ Some services failed to start"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "=================="
    echo "• Check logs: cat logs/*.log"
    echo "• Check ports: netstat -tuln | grep -E ':(3000|3001|5000|5001|5002)'"
    echo "• Check processes: ps aux | grep -E '(python|node|npm)'"
    echo ""
    echo "📋 Service Files:"
    echo "================"
    echo "• Stream Backend: backend/stream_backend.py"
    echo "• Video Filter: backend/video_filter_api.py"  
    echo "• Audio Service: audio-william/audio_server_used.py"
    echo "• MediaSoup: mediasoup-server/server.js"
    echo "• Frontend: frontend/ (Next.js)"
    exit 1
fi

# Option to show live logs
echo ""
read -p "Show live logs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Showing live logs (Ctrl+C to exit)..."
    tail -f logs/*.log 2>/dev/null || echo "Could not tail logs (files may not exist yet)"
fi