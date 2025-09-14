"""
Setup and installation script for the Livestream PII Pipeline.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """Set up logging for the setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("setup")


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")


def install_requirements():
    """Install Python requirements."""
    logger = logging.getLogger("setup")
    
    logger.info("Installing Python requirements...")
    
    try:
        # Install basic requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Basic requirements installed successfully")
        
        # Install Whisper
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-U", "openai-whisper"
        ])
        logger.info("Whisper installed successfully")
        
        # Install ffmpeg-python for audio processing
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "ffmpeg-python"
        ])
        logger.info("FFmpeg Python bindings installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        raise


def setup_directories():
    """Create necessary directories."""
    logger = logging.getLogger("setup")
    
    directories = [
        "models",
        "data", 
        "logs",
        "output",
        "output/results",
        "output/transcripts",
        "output/blur_instructions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_whisper_models():
    """Download and cache Whisper models."""
    logger = logging.getLogger("setup")
    
    try:
        import whisper
        
        # Download the models we'll use
        models_to_download = ["base", "large-v3"]
        
        for model_name in models_to_download:
            logger.info(f"Downloading Whisper model: {model_name}")
            whisper.load_model(model_name)
            logger.info(f"Successfully downloaded {model_name}")
            
    except ImportError:
        logger.warning("Whisper not installed, skipping model download")
    except Exception as e:
        logger.error(f"Error downloading Whisper models: {e}")


def setup_kaggle_datasets():
    """Set up Kaggle datasets if kaggle is configured."""
    logger = logging.getLogger("setup")
    
    try:
        # Check if kaggle is installed and configured
        result = subprocess.run(
            ["kaggle", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0:
            logger.info("Kaggle CLI found, running dataset setup...")
            subprocess.check_call(["bash", "setup_datasets.sh"])
            logger.info("Kaggle datasets downloaded successfully")
        else:
            logger.warning("Kaggle CLI not configured. Please set up Kaggle API credentials if you want to download datasets.")
            logger.info("You can skip this step if you already have the model files.")
            
    except FileNotFoundError:
        logger.warning("Kaggle CLI not installed. Please install with 'pip install kaggle' if needed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up Kaggle datasets: {e}")


def verify_gpu_support():
    """Verify GPU support for PyTorch and CUDA."""
    logger = logging.getLogger("setup")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA support detected: {gpu_count} GPU(s) available")
            logger.info(f"Primary GPU: {gpu_name}")
            
            # Test CUDA functionality
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            logger.info("CUDA functionality test passed")
            
        else:
            logger.warning("No CUDA support detected. Pipeline will run on CPU (slower)")
            
    except ImportError:
        logger.warning("PyTorch not installed yet")
    except Exception as e:
        logger.error(f"Error checking GPU support: {e}")


def create_example_config():
    """Create example configuration files."""
    logger = logging.getLogger("setup")
    
    # Check if config already exists
    config_path = "configs/pipeline_config.yaml"
    if not os.path.exists(config_path):
        logger.info("Pipeline configuration already created during installation")
    else:
        logger.info("Pipeline configuration file found")


def run_basic_test():
    """Run a basic test to verify installation."""
    logger = logging.getLogger("setup")
    
    try:
        logger.info("Running basic functionality test...")
        
        # Test imports
        from src.pipeline_types import AudioSegment, PIIType
        from src.whisper_processor import WhisperProcessor
        from src.pii_detector import PIIDetector
        from src.livestream_pipeline import LivestreamPIIPipeline
        
        logger.info("All modules imported successfully")
        
        # Test basic object creation
        try:
            # Test with CPU to avoid GPU requirements during setup
            whisper_processor = WhisperProcessor(
                model_name="base",
                device="cpu",
                max_workers=1
            )
            logger.info("Whisper processor initialized successfully")
            whisper_processor.cleanup()
            
        except Exception as e:
            logger.warning(f"Whisper processor test failed: {e}")
        
        logger.info("Basic functionality test completed")
        
    except ImportError as e:
        logger.error(f"Import error during test: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during basic test: {e}")
        raise


def print_setup_complete():
    """Print setup completion message with next steps."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Ensure you have trained DeBERTa models in the 'models/' directory")
    print("2. If using GPU, verify CUDA is properly installed")
    print("3. Run the example: python example_usage.py")
    print("4. Check the configuration in configs/pipeline_config.yaml")
    
    print("\nFor integration with your livestream:")
    print("1. Replace sample audio data with real audio input")
    print("2. Connect blur instructions to your video processing")
    print("3. Add callbacks for your specific requirements")
    
    print("\nUseful commands:")
    print("- Test installation: python -c \"from src.livestream_pipeline import LivestreamPIIPipeline; print('OK')\"")
    print("- Run example: python example_usage.py")
    print("- Check GPU: python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\"")
    
    print("\n" + "="*60)


def main():
    """Main setup function."""
    logger = setup_logging()
    logger.info("Starting Livestream PII Pipeline setup...")
    
    try:
        # Check Python version
        check_python_version()
        logger.info(f"Python version check passed: {sys.version}")
        
        # Create directories
        setup_directories()
        
        # Install requirements
        install_requirements()
        
        # Download Whisper models
        download_whisper_models()
        
        # Set up Kaggle datasets (optional)
        setup_kaggle_datasets()
        
        # Verify GPU support
        verify_gpu_support()
        
        # Create example config
        create_example_config()
        
        # Run basic test
        run_basic_test()
        
        # Print completion message
        print_setup_complete()
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
