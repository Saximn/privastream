#!/bin/bash

# Stop all PrivaStream.site services

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🛑 Stopping PrivaStream.site Services"
echo "====================================="

# Function to stop service by PID file
stop_service() {
    local name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
                print_success "$name stopped (force killed PID: $pid)"
            else
                print_success "$name stopped (PID: $pid)"
            fi
        else
            print_status "$name was not running"
        fi
        rm -f "$pid_file"
    else
        print_status "$name PID file not found"
    fi
}

# Stop services by PID files
if [ -d "logs" ]; then
    stop_service "Video Service" "logs/video-service.pid"
    stop_service "Audio Service" "logs/audio-service.pid"
    stop_service "Backend" "logs/backend.pid"
    stop_service "MediaSoup" "logs/mediasoup.pid"
    stop_service "Frontend" "logs/frontend.pid"
    
    # Clean up PID files
    rm -f logs/all-pids.txt
else
    print_status "No logs directory found"
fi

# Kill any remaining processes by name
print_status "Cleaning up any remaining processes..."

# Kill Python processes
pkill -f "python.*app.py" 2>/dev/null && print_success "Stopped remaining Python backend processes"
pkill -f "python.*audio_redaction_server" 2>/dev/null && print_success "Stopped remaining audio service processes"

# Kill Node.js processes
pkill -f "node.*server.js" 2>/dev/null && print_success "Stopped remaining MediaSoup processes"
pkill -f "npm.*start" 2>/dev/null && print_success "Stopped remaining npm processes"
pkill -f "next" 2>/dev/null && print_success "Stopped remaining Next.js processes"

# Kill any nginx processes if started manually
pkill -f "nginx.*privastream" 2>/dev/null || true

sleep 1

# Check if any services are still running
print_status "Checking for remaining services..."
remaining=$(ps aux | grep -E "(python.*app.py|python.*audio_redaction|node.*server.js|npm.*start)" | grep -v grep | wc -l)

if [ "$remaining" -eq 0 ]; then
    print_success "✅ All PrivaStream.site services stopped successfully"
else
    print_error "⚠️  Some processes may still be running:"
    ps aux | grep -E "(python.*app.py|python.*audio_redaction|node.*server.js|npm.*start)" | grep -v grep
    echo ""
    echo "To force kill all: pkill -f 'python|node|npm'"
fi

# Show port status
print_status "Port status:"
echo "============"
for port in 3000 3001 5000 5001 5002; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "Port $port: STILL IN USE"
    else
        echo "Port $port: Available"
    fi
done

echo ""
print_success "🏁 Shutdown complete"