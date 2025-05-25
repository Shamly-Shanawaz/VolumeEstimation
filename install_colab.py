#!/usr/bin/env python3
"""
Installation script for Google Colab environment.
This script installs Detectron2 and other dependencies properly in Colab.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def install_detectron2():
    """Install Detectron2 for Colab environment."""
    print("Installing Detectron2...")
    
    # Install dependencies first
    commands = [
        ("pip install pyyaml==5.4.1", "Installing PyYAML"),
        ("pip install torch torchvision torchaudio", "Installing PyTorch"),
        ("pip install opencv-python", "Installing OpenCV"),
        ("pip install 'git+https://github.com/facebookresearch/detectron2.git'", "Installing Detectron2")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"Failed to execute: {command}")
            return False
    
    return True

def install_other_requirements():
    """Install other project requirements."""
    requirements = [
        "numpy>=1.19.0",
        "Pillow>=8.0.0", 
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "timm>=0.4.12",
        "open3d>=0.13.0",
        "pycocotools>=2.0.0",
        "huggingface_hub>=0.19.0"
    ]
    
    for req in requirements:
        if not run_command(f"pip install '{req}'", f"Installing {req}"):
            print(f"Warning: Failed to install {req}")
    
    return True

def verify_installation():
    """Verify that Detectron2 is properly installed."""
    try:
        import detectron2
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.model_zoo import model_zoo
        print("✅ Detectron2 installation verified successfully!")
        print(f"Detectron2 version: {detectron2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Detectron2 verification failed: {e}")
        return False

def main():
    """Main installation function."""
    print("🚀 Starting VolumeEstimation installation for Google Colab...")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab environment detected")
    except ImportError:
        print("⚠️  Warning: Not running in Google Colab")
    
    # Install Detectron2
    if not install_detectron2():
        print("❌ Failed to install Detectron2")
        return False
    
    # Install other requirements
    if not install_other_requirements():
        print("⚠️  Some requirements failed to install")
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("1. Make sure your model files are in Google Drive at '/content/drive/My Drive/model_weights/'")
    print("2. Run the volume estimation code")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 