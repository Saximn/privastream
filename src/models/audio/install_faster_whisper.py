#!/usr/bin/env python3
"""
Installation script for Faster-Whisper Audio Redaction
Installs required dependencies and tests the setup
"""

import subprocess
import sys
import os
import torch

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error details:", e.stderr)
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n🎮 Checking CUDA availability...")
    
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️ CUDA not available - will use CPU mode")
        return False

def install_faster_whisper():
    """Install Faster-Whisper and dependencies"""
    print("\n🚀 Installing Faster-Whisper Audio Redaction Dependencies")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Error: Python 3.8+ required")
        return False
    
    # Check CUDA
    has_cuda = check_cuda()
    
    # Install PyTorch with appropriate CUDA support
    if has_cuda:
        print("\n📦 Installing PyTorch with CUDA support...")
        torch_cmd = "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("\n📦 Installing PyTorch (CPU only)...")
        torch_cmd = "pip install torch torchaudio"
    
    if not run_command(torch_cmd, "Installing PyTorch"):
        print("❌ Failed to install PyTorch")
        return False
    
    # Install Faster-Whisper
    if not run_command("pip install faster-whisper", "Installing Faster-Whisper"):
        print("❌ Failed to install Faster-Whisper")
        return False
    
    # Install other requirements
    commands = [
        ("pip install flask flask-cors", "Installing Flask web framework"),
        ("pip install librosa soundfile", "Installing audio processing libraries"),
        ("pip install transformers datasets", "Installing NLP libraries for PII detection"),
        ("pip install numpy python-dateutil", "Installing utility libraries")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Failed: {description}")
            return False
    
    return True

def test_installation():
    """Test the Faster-Whisper installation"""
    print("\n🧪 Testing Faster-Whisper installation...")
    
    try:
        # Test imports
        print("Testing imports...")
        from faster_whisper import WhisperModel
        import torch
        import librosa
        import soundfile as sf
        import numpy as np
        from flask import Flask
        from transformers import pipeline
        
        print("✅ All imports successful!")
        
        # Test model loading
        print("\nTesting Faster-Whisper model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading tiny model on {device} with {compute_type}...")
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        
        print("✅ Faster-Whisper model loaded successfully!")
        
        # Test with a dummy audio array
        print("\nTesting transcription...")
        # Create 1 second of silence at 16kHz
        dummy_audio = np.zeros(16000, dtype=np.float32)
        
        segments, info = model.transcribe(dummy_audio, word_timestamps=True)
        segments_list = list(segments)  # Convert generator to list
        
        print(f"✅ Transcription test successful!")
        print(f"   Language detected: {info.language}")
        print(f"   Segments: {len(segments_list)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🎤 Faster-Whisper Audio Redaction Setup")
    print("=" * 50)
    
    # Change to the audio-william directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📂 Working directory: {script_dir}")
    
    # Install dependencies
    if not install_faster_whisper():
        print("\n❌ Installation failed!")
        return False
    
    # Test installation
    if not test_installation():
        print("\n❌ Testing failed!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Faster-Whisper Audio Redaction is ready!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Start the audio redaction server:")
    print("   python audio_redaction_server_faster_whisper.py")
    print("\n2. Update your mediasoup server configuration to use port 5002")
    print("\n3. Test with the provided test script:")
    print("   python test_faster_whisper_api.py")
    
    # Show performance comparison info
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"\n⚡ Performance info:")
    print(f"   Device: {device}")
    print(f"   Expected speed: ~2-5x faster than standard Whisper")
    print(f"   Accuracy: Same or better than OpenAI Whisper")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)