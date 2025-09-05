#!/usr/bin/env python3
"""
Startup script for the complete audio redaction system.
Starts all required services in the correct order.
"""

import os
import sys
import subprocess
import time
import signal
import threading
import requests
from pathlib import Path

# Configuration
SERVICES = {
    'audio_redaction': {
        'command': [sys.executable, 'audio-processing/src/realtime_audio_redactor.py'],
        'port': 5002,
        'health_url': 'http://localhost:5002/health',
        'name': 'Audio Redaction Service'
    },
    'backend': {
        'command': [sys.executable, 'web-demo-ui/backend/app.py'],
        'port': 5000,
        'health_url': 'http://localhost:5000/health',
        'name': 'Backend Service'
    },
    'mediasoup': {
        'command': ['node', 'web-demo-ui/mediasoup-server/server.js'],
        'port': 3001,
        'health_url': 'http://localhost:3001/health',
        'name': 'Mediasoup SFU Server'
    }
}

class ServiceManager:
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def check_health(self, service_name, url, timeout=5):
        """Check if a service is healthy"""
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def wait_for_service(self, service_name, config, max_wait=60):
        """Wait for a service to become healthy"""
        print(f"⏳ Waiting for {config['name']} to start...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.check_health(service_name, config['health_url']):
                print(f"✅ {config['name']} is ready")
                return True
            time.sleep(2)
        
        print(f"❌ {config['name']} failed to start within {max_wait} seconds")
        return False
    
    def start_service(self, service_name, config):
        """Start a single service"""
        print(f"🚀 Starting {config['name']}...")
        
        try:
            # Change to the appropriate directory
            cwd = os.getcwd()
            
            # Start the process
            process = subprocess.Popen(
                config['command'],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes[service_name] = process
            
            # Start a thread to monitor the process output
            monitor_thread = threading.Thread(
                target=self.monitor_process,
                args=(service_name, process, config['name']),
                daemon=True
            )
            monitor_thread.start()
            
            return process
            
        except Exception as e:
            print(f"❌ Failed to start {config['name']}: {e}")
            return None
    
    def monitor_process(self, service_name, process, service_display_name):
        """Monitor process output"""
        while self.running:
            try:
                output = process.stdout.readline()
                if output:
                    print(f"[{service_display_name}] {output.strip()}")
                elif process.poll() is not None:
                    break
            except Exception as e:
                print(f"❌ Error monitoring {service_display_name}: {e}")
                break
    
    def stop_service(self, service_name):
        """Stop a single service"""
        if service_name in self.processes:
            process = self.processes[service_name]
            config = SERVICES[service_name]
            
            print(f"🛑 Stopping {config['name']}...")
            
            try:
                # Try graceful shutdown first
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f"✅ {config['name']} stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    print(f"⚠️ Force killing {config['name']}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {config['name']} force stopped")
                    
            except Exception as e:
                print(f"❌ Error stopping {config['name']}: {e}")
            
            del self.processes[service_name]
    
    def stop_all_services(self):
        """Stop all services"""
        print("\n🛑 Stopping all services...")
        self.running = False
        
        # Stop services in reverse order
        service_names = list(self.processes.keys())
        service_names.reverse()
        
        for service_name in service_names:
            self.stop_service(service_name)
        
        print("✅ All services stopped")
    
    def start_all_services(self):
        """Start all services in order"""
        print("🚀 Starting Audio PII Redaction System...")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Start services in order
        for service_name, config in SERVICES.items():
            # Start the service
            process = self.start_service(service_name, config)
            if not process:
                print(f"❌ Failed to start {config['name']}")
                return False
            
            # Wait for it to become healthy
            if not self.wait_for_service(service_name, config):
                print(f"❌ {config['name']} failed health check")
                return False
            
            print(f"✅ {config['name']} started successfully")
            print("-" * 30)
        
        print("\n🎉 All services started successfully!")
        print("\nService URLs:")
        for service_name, config in SERVICES.items():
            print(f"  • {config['name']}: http://localhost:{config['port']}")
        
        print(f"\n📱 Frontend available at: http://localhost:3000")
        print(f"🔧 Audio Redaction Panel: Access via frontend UI")
        print(f"\n💡 Press Ctrl+C to stop all services")
        
        return True
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        # Check Python
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print("✅ Python version OK")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, 'node')
            print(f"✅ Node.js version OK: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js not found or not working")
            return False
        
        # Check required directories
        required_dirs = [
            'audio-processing',
            'web-demo-ui/backend',
            'web-demo-ui/mediasoup-server',
            'models'
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                print(f"❌ Required directory not found: {dir_path}")
                return False
        print("✅ All required directories found")
        
        # Check key files
        key_files = [
            'audio-processing/src/realtime_audio_redactor.py',
            'web-demo-ui/backend/app.py',
            'web-demo-ui/mediasoup-server/server.js'
        ]
        
        for file_path in key_files:
            if not Path(file_path).exists():
                print(f"❌ Required file not found: {file_path}")
                return False
        print("✅ All key files found")
        
        return True
    
    def run(self):
        """Main run loop"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down...")
            self.stop_all_services()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start all services
        if not self.start_all_services():
            print("❌ Failed to start services")
            self.stop_all_services()
            sys.exit(1)
        
        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(1)
                
                # Check if any process has died
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        config = SERVICES[service_name]
                        print(f"⚠️ {config['name']} has stopped unexpectedly")
                        # Could implement auto-restart here
                        
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_services()


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'install':
            print("🔧 Installing dependencies...")
            
            # Install Python dependencies
            print("Installing Python dependencies...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 
                'audio-processing/requirements.txt'
            ], check=True)
            
            # Install Node.js dependencies
            print("Installing Node.js dependencies...")
            subprocess.run(['npm', 'install'], cwd='web-demo-ui/mediasoup-server', check=True)
            subprocess.run(['npm', 'install'], cwd='web-demo-ui/frontend', check=True)
            
            print("✅ Dependencies installed successfully!")
            return
        
        elif command == 'check':
            print("🔍 Checking system status...")
            manager = ServiceManager()
            
            for service_name, config in SERVICES.items():
                if manager.check_health(service_name, config['health_url']):
                    print(f"✅ {config['name']}: Running")
                else:
                    print(f"❌ {config['name']}: Not running")
            return
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: install, check")
            sys.exit(1)
    
    # Default behavior: start all services
    print("🎤 TikTok TechJam 2025 - Audio PII Redaction System")
    print("=" * 60)
    
    manager = ServiceManager()
    manager.run()


if __name__ == '__main__':
    main()