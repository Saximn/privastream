#!/bin/bash

# Stop all PrivaStream.site services

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
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
            print_status "Stopping $name (PID: $pid)..."
            kill "$pid"
            
            # Wait up to 10 seconds for graceful shutdown
            local count=0
            while [ $count -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
                print_warning "$name force killed (PID: $pid)"
            else
                print_success "$name stopped gracefully (PID: $pid)"
            fi
        else
            print_status "$name was not running (PID: $pid)"
        fi
        rm -f "$pid_file"
    else
        print_status "$name PID file not found ($pid_file)"
    fi
}

# Stop services by PID files if logs directory exists
if [ -d "logs" ]; then
    stop_service "Video Filter API" "logs/video-service.pid"
    stop_service "Audio Service" "logs/audio-service.pid" 
    stop_service "Stream Backend" "logs/backend.pid"
    stop_service "MediaSoup Server" "logs/mediasoup.pid"
    stop_service "Frontend" "logs/frontend.pid"
    
    # Clean up PID files
    rm -f logs/all-pids.txt
else
    print_warning "No logs directory found"
fi

# Kill any remaining processes by name/pattern
print_status "Cleaning up any remaining processes..."

# Python services
if pkill -f "python.*stream_backend.py" 2>/dev/null; then
    print_success "Stopped remaining Stream Backend processes"
fi

if pkill -f "python.*video_filter_api.py" 2>/dev/null; then
    print_success "Stopped remaining Video Filter API processes"  
fi

if pkill -f "python.*audio_server_used.py" 2>/dev/null; then
    print_success "Stopped remaining Audio Service processes"
fi

# Node.js services
if pkill -f "node.*server.js" 2>/dev/null; then
    print_success "Stopped remaining MediaSoup Server processes"
fi

if pkill -f "npm.*start" 2>/dev/null; then
    print_success "Stopped remaining npm start processes"
fi

if pkill -f "next-server" 2>/dev/null; then
    print_success "Stopped remaining Next.js server processes"
fi

# Additional cleanup for any Flask processes on our ports
for port in 5000 5001 5002; do
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        kill -9 $pid 2>/dev/null && print_success "Killed process on port $port (PID: $pid)"
    fi
done

sleep 2

# Final check for remaining processes
print_status "Checking for remaining services..."
remaining_processes=$(ps aux | grep -E "(python.*(stream_backend|video_filter_api|audio_server_used)|node.*server.js|npm.*start)" | grep -v grep)

if [ -z "$remaining_processes" ]; then
    print_success "✅ All PrivaStream.site services stopped successfully"
else
    print_warning "⚠️  Some processes may still be running:"
    echo "$remaining_processes"
    echo ""
    print_status "To force kill all related processes:"
    echo "pkill -f 'python.*stream_backend'"
    echo "pkill -f 'python.*video_filter_api'" 
    echo "pkill -f 'python.*audio_server_used'"
    echo "pkill -f 'node.*server.js'"
    echo "pkill -f 'npm.*start'"
fi

# Show port status
print_status "Port status:"
echo "============"
for port in 3000 3001 5000 5001 5002; do
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "Port $port: STILL IN USE"
        # Show what's using the port
        local process=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$process" ]; then
            echo "  Process: $process ($(ps -p $process -o comm= 2>/dev/null))"
        fi
    else
        print_success "Port $port: Available"
    fi
done

echo ""
print_success "🏁 Shutdown complete"
echo ""
print_status "To start services again: ./start-privastream.sh"