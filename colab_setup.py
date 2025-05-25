#!/usr/bin/env python3
"""
Complete setup script for Google Colab to run Volume Estimation with Detectron2.
This script handles all the installation and setup steps.
"""

import subprocess
import sys
import os
import torch

def run_cmd(command, description=""):
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    if description:
        print(f"🔧 {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def install_system_dependencies():
    """Install system-level dependencies."""
    print("📦 Installing system dependencies...")
    
    commands = [
        ("apt update", "Updating package lists"),
        ("apt install -y build-essential", "Installing build tools"),
        ("apt install -y ninja-build", "Installing Ninja build system"),
    ]
    
    for cmd, desc in commands:
        if not run_cmd(cmd, desc):
            print(f"⚠️ Warning: Failed to execute {cmd}")

def install_python_dependencies():
    """Install Python dependencies."""
    print("🐍 Installing Python dependencies...")
    
    # Core dependencies
    deps = [
        "torch==2.1.0",
        "torchvision==0.16.0", 
        "torchaudio==2.1.0",
        "opencv-python==4.8.1.78",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
        "timm>=0.4.12",
        "pycocotools>=2.0.0",
        "huggingface_hub>=0.19.0",
        "fvcore",
        "cloudpickle",
        "omegaconf>=2.1",
        "hydra-core>=1.1"
    ]
    
    for dep in deps:
        if not run_cmd(f"pip install '{dep}'", f"Installing {dep}"):
            print(f"⚠️ Warning: Failed to install {dep}")

def install_detectron2():
    """Install Detectron2 properly for Colab."""
    print("🤖 Installing Detectron2...")
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    
    # Install Detectron2 from pre-built wheel for CUDA 11.8 (most common in Colab)
    detectron2_cmd = "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    
    if not run_cmd(detectron2_cmd, "Installing Detectron2 from source"):
        print("❌ Failed to install Detectron2")
        return False
    
    return True

def verify_installation():
    """Verify that all components are installed correctly."""
    print("🔍 Verifying installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        # Test Detectron2
        import detectron2
        print(f"✅ Detectron2 {detectron2.__version__} installed")
        
        # Test Detectron2 components
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.model_zoo import model_zoo
        print("✅ Detectron2 components imported successfully")
        
        # Test other dependencies
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ All dependencies verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def setup_google_drive():
    """Mount Google Drive and check for model files."""
    print("📁 Setting up Google Drive...")
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        
        # Check for model directory
        model_dir = '/content/drive/My Drive/model_weights'
        if os.path.exists(model_dir):
            print(f"✅ Model directory found: {model_dir}")
            
            # List available models
            models = os.listdir(model_dir)
            print("📋 Available model files:")
            for model in models:
                if model.endswith(('.pkl', '.pt', '.bin')):
                    size = os.path.getsize(os.path.join(model_dir, model)) / (1024*1024)  # MB
                    print(f"  - {model} ({size:.1f} MB)")
        else:
            print(f"⚠️ Model directory not found: {model_dir}")
            print("Please upload your model files to this directory")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup Google Drive: {e}")
        return False

def create_test_script():
    """Create a simple test script."""
    test_code = '''
# Test script for Volume Estimation
import sys
sys.path.append('/content/VolumeEstimation/src')

try:
    from src.segmentation import InstanceSegmentation
    print("✅ Segmentation module imported successfully")
    
    # Test with a dummy model path (will fail but shows import works)
    # seg = InstanceSegmentation('/path/to/model.pkl')
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open('/content/test_volume_estimation.py', 'w') as f:
        f.write(test_code)
    
    print("✅ Test script created at /content/test_volume_estimation.py")

def main():
    """Main setup function."""
    print("🚀 Starting Volume Estimation setup for Google Colab...")
    print("This may take 5-10 minutes to complete.\n")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Google Colab environment detected")
    except ImportError:
        print("⚠️ Warning: Not running in Google Colab")
    
    # Step 1: Install system dependencies
    install_system_dependencies()
    
    # Step 2: Install Python dependencies
    install_python_dependencies()
    
    # Step 3: Install Detectron2
    if not install_detectron2():
        print("❌ Setup failed at Detectron2 installation")
        return False
    
    # Step 4: Verify installation
    if not verify_installation():
        print("❌ Setup failed at verification")
        return False
    
    # Step 5: Setup Google Drive
    setup_google_drive()
    
    # Step 6: Create test script
    create_test_script()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Make sure your model files are in '/content/drive/My Drive/model_weights/'")
    print("2. Run: python /content/test_volume_estimation.py")
    print("3. Use the volume estimation code")
    print("\n🔧 If you encounter issues:")
    print("- Restart runtime and run this script again")
    print("- Check that all model files are properly uploaded")
    print("- Verify model file formats (.pkl for segmentation)")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!") 