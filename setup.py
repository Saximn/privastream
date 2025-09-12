"""
Setup script for Privastream - AI-Powered Privacy Streaming Platform
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from file
def read_requirements():
    requirements = [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "ultralytics>=8.0.0",
        "opencv-python>=4.7.0",
        "numpy>=1.21.0",
        "flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "flask-cors>=4.0.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.3.0",
        "Pillow>=9.5.0",
        "requests>=2.31.0",
        "pyyaml>=6.0"
    ]
    return requirements

# Read long description from README
def read_long_description():
    readme_file = Path("README.md")
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="privastream",
    version="1.0.0",
    author="Privastream Team",
    author_email="team@privastream.ai",
    description="AI-Powered Privacy Streaming Platform for real-time PII detection and redaction",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/privastream/tiktok-techjam-2025",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "torchaudio>=0.12.0",
        ],
        "audio": [
            "faster-whisper>=0.9.0",
            "librosa>=0.9.0",
            "soundfile>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "privastream=privastream.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "privastream": [
            "configs/*.json",
            "configs/*.yaml",
            "data/*",
        ],
    },
    zip_safe=False,
)