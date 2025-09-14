#!/usr/bin/env python3
"""
GPU Check Script for Audio Redaction Server
Verifies CUDA installation and GPU compatibility
"""

import torch
import sys
import subprocess
import platform

def check_cuda_availability():
    """Check if CUDA is available"""
    print("🔍 Checking CUDA Availability...")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_properties = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {gpu_properties.name}")
            print(f"      Memory: {gpu_properties.total_memory / 1e9:.1f} GB")
            print(f"      Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")
        
        return True
    else:
        print("   ❌ CUDA not available")
        return False

def check_nvidia_driver():
    """Check NVIDIA driver version"""
    print("\n🔍 Checking NVIDIA Driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   ✅ {line.strip()}")
                    return True
        else:
            print("   ❌ nvidia-smi command failed")
            return False
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False

def test_gpu_operations():
    """Test basic GPU operations"""
    print("\n🔍 Testing GPU Operations...")
    
    if not torch.cuda.is_available():
        print("   ⏭️ Skipping GPU tests (CUDA not available)")
        return False
    
    try:
        # Create tensors on GPU
        print("   Testing tensor operations...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Perform computation
        z = torch.mm(x, y)
        print("   ✅ Matrix multiplication on GPU successful")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / 1e6
        print(f"   Memory allocated: {memory_allocated:.1f} MB")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        print("   ✅ GPU memory cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU operations failed: {e}")
        return False

def check_transformers_gpu():
    """Check if transformers can use GPU"""
    print("\n🔍 Testing Transformers GPU Support...")
    
    if not torch.cuda.is_available():
        print("   ⏭️ Skipping transformers GPU test (CUDA not available)")
        return False
    
    try:
        from transformers import pipeline
        
        # Test with a simple model
        print("   Loading test model on GPU...")
        classifier = pipeline("sentiment-analysis", device=0)
        
        print("   Testing inference...")
        result = classifier("This is a test sentence.")
        print(f"   ✅ Transformers GPU test successful: {result}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Transformers GPU test failed: {e}")
        return False

def provide_troubleshooting():
    """Provide troubleshooting steps"""
    print("\n🔧 Troubleshooting Steps:")
    print("1. Check NVIDIA GPU compatibility:")
    print("   - Ensure you have an NVIDIA GPU with CUDA support")
    print("   - Check: https://developer.nvidia.com/cuda-gpus")
    
    print("\n2. Install/Update NVIDIA Drivers:")
    print("   - Download latest drivers: https://www.nvidia.com/drivers")
    print("   - Restart after installation")
    
    print("\n3. Install CUDA Toolkit:")
    print("   - Download CUDA 12.8: https://developer.nvidia.com/cuda-downloads")
    print("   - Or use conda: conda install nvidia::cuda-toolkit")
    
    print("\n4. Reinstall PyTorch with CUDA:")
    print("   - pip uninstall torch")
    print("   - pip install torch --index-url https://download.pytorch.org/whl/cu128")
    
    print("\n5. Verify Installation:")
    print("   - Run: python -c \"import torch; print(torch.cuda.is_available())\"")
    
    print("\n6. Alternative: CPU-only mode:")
    print("   - The server will work on CPU (slower but functional)")
    print("   - ~6 seconds processing time vs ~1-3 seconds on GPU")

def main():
    print("🎮 GPU Compatibility Check for Audio Redaction Server")
    print("=" * 60)
    
    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    
    # Run checks
    results = []
    results.append(("CUDA Available", check_cuda_availability()))
    results.append(("NVIDIA Driver", check_nvidia_driver()))
    results.append(("GPU Operations", test_gpu_operations()))
    results.append(("Transformers GPU", check_transformers_gpu()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("🎉 GPU setup is working! Your audio server will use GPU acceleration.")
    elif passed == 0:
        print("⚠️ GPU not available. Server will run on CPU (slower but functional).")
        provide_troubleshooting()
    else:
        print("⚠️ Partial GPU support. Some features may not work optimally.")
        provide_troubleshooting()

if __name__ == "__main__":
    main()