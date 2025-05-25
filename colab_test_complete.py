#!/usr/bin/env python3
"""
Complete Google Colab test script for Volume Estimation.
This script allows you to upload images and test the volume estimation system.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files
from google.colab import drive
import torch

# Add the project to Python path
sys.path.append('/content/VolumeEstimation/src')

def setup_directories():
    """Create necessary directories for testing."""
    os.makedirs('/content/test_images', exist_ok=True)
    os.makedirs('/content/results', exist_ok=True)
    print("✅ Test directories created")

def upload_test_images():
    """Upload test images for volume estimation."""
    print("📸 Upload your test images:")
    print("You can upload multiple images at once")
    
    uploaded = files.upload()
    
    image_paths = []
    for filename, data in uploaded.items():
        # Save uploaded file
        filepath = f'/content/test_images/{filename}'
        with open(filepath, 'wb') as f:
            f.write(data)
        image_paths.append(filepath)
        print(f"✅ Uploaded: {filename}")
    
    return image_paths

def test_segmentation_only(image_path, model_path):
    """Test just the segmentation component."""
    print(f"\n🔍 Testing segmentation on: {os.path.basename(image_path)}")
    
    try:
        from src.segmentation import InstanceSegmentation
        
        # Initialize segmentation model
        print("Initializing segmentation model...")
        seg_model = InstanceSegmentation(model_path)
        print("✅ Segmentation model loaded successfully!")
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        print("Running segmentation...")
        results = seg_model.segment(image_rgb)
        
        print(f"📊 Segmentation Results:")
        print(f"  - Objects detected: {len(results['class_names'])}")
        print(f"  - Classes found: {results['class_names']}")
        print(f"  - Confidence scores: {[f'{score:.3f}' for score in results['scores']]}")
        
        # Visualize results
        vis_image = seg_model.visualize_segmentation(image_rgb)
        
        return {
            'results': results,
            'visualization': vis_image,
            'original': image_rgb
        }
        
    except Exception as e:
        print(f"❌ Segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_volume_estimation(image_path):
    """Test the complete volume estimation pipeline."""
    print(f"\n🧮 Testing full volume estimation on: {os.path.basename(image_path)}")
    
    try:
        from src.download_models import get_volume_estimator
        
        # Initialize the full estimator
        print("Initializing volume estimator...")
        estimator = get_volume_estimator()
        print("✅ Volume estimator loaded successfully!")
        
        # Estimate volumes
        print("Estimating volumes...")
        volumes = estimator.estimate_volume(image_path)
        
        print(f"📊 Volume Estimation Results:")
        for obj_name, volume in volumes.items():
            print(f"  - {obj_name}: {volume:.4f} cubic units")
        
        # Create visualization
        vis_image = estimator.visualize_results(image_path, volumes)
        
        return {
            'volumes': volumes,
            'visualization': vis_image
        }
        
    except Exception as e:
        print(f"❌ Volume estimation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_results(results, image_name):
    """Display the results with matplotlib."""
    if results is None:
        print("❌ No results to display")
        return
    
    # Create figure
    if 'visualization' in results and 'original' in results:
        # Segmentation results
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(results['original'])
        axes[0].set_title(f'Original Image: {image_name}')
        axes[0].axis('off')
        
        # Segmentation visualization
        axes[1].imshow(results['visualization'])
        axes[1].set_title('Segmentation Results')
        axes[1].axis('off')
        
    elif 'visualization' in results:
        # Volume estimation results
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(results['visualization'])
        ax.set_title(f'Volume Estimation Results: {image_name}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    result_path = f'/content/results/result_{image_name}.png'
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"💾 Results saved to: {result_path}")

def run_comprehensive_test():
    """Run comprehensive tests on uploaded images."""
    print("🧪 COMPREHENSIVE VOLUME ESTIMATION TEST")
    print("="*60)
    
    # Check if models are available
    try:
        from src.download_models import check_models
        if not check_models():
            print("❌ Model files not found. Please ensure they're in Google Drive.")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False
    
    # Setup directories
    setup_directories()
    
    # Upload test images
    image_paths = upload_test_images()
    
    if not image_paths:
        print("❌ No images uploaded")
        return False
    
    # Get model path for segmentation-only test
    model_path = '/content/drive/My Drive/model_weights/pointtrend_rcnn_R_50_FPN_3x_coco.pkl'
    
    # Test each uploaded image
    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"🖼️  TESTING IMAGE {i+1}/{len(image_paths)}: {image_name}")
        print(f"{'='*60}")
        
        # Test 1: Segmentation only
        print("\n1️⃣ SEGMENTATION TEST")
        seg_results = test_segmentation_only(image_path, model_path)
        if seg_results:
            display_results(seg_results, f"seg_{image_name}")
        
        # Test 2: Full volume estimation
        print("\n2️⃣ VOLUME ESTIMATION TEST")
        vol_results = test_full_volume_estimation(image_path)
        if vol_results:
            display_results(vol_results, f"vol_{image_name}")
        
        print(f"\n✅ Completed testing: {image_name}")
    
    print(f"\n🎉 ALL TESTS COMPLETED!")
    print(f"📁 Results saved in: /content/results/")
    
    return True

def quick_single_test():
    """Quick test with a single image upload."""
    print("🚀 QUICK SINGLE IMAGE TEST")
    print("="*40)
    
    # Upload single image
    print("📸 Upload ONE test image:")
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No image uploaded")
        return
    
    # Get the uploaded image
    filename = list(uploaded.keys())[0]
    image_path = f'/content/test_images/{filename}'
    
    # Save the file
    with open(image_path, 'wb') as f:
        f.write(uploaded[filename])
    
    print(f"✅ Image saved: {filename}")
    
    # Test segmentation
    model_path = '/content/drive/My Drive/model_weights/pointtrend_rcnn_R_50_FPN_3x_coco.pkl'
    results = test_segmentation_only(image_path, model_path)
    
    if results:
        display_results(results, filename)
        print("🎉 Quick test completed successfully!")
    else:
        print("❌ Quick test failed")

def main_menu():
    """Main menu for testing options."""
    print("\n" + "="*60)
    print("🎯 VOLUME ESTIMATION TESTING MENU")
    print("="*60)
    print("Choose your testing option:")
    print("1️⃣  Quick Single Image Test (Segmentation only)")
    print("2️⃣  Comprehensive Multi-Image Test (Full pipeline)")
    print("3️⃣  Check System Status")
    print("4️⃣  Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        quick_single_test()
    elif choice == "2":
        run_comprehensive_test()
    elif choice == "3":
        check_system_status()
    elif choice == "4":
        print("👋 Goodbye!")
        return False
    else:
        print("❌ Invalid choice. Please try again.")
    
    return True

def check_system_status():
    """Check if everything is working properly."""
    print("\n🔍 SYSTEM STATUS CHECK")
    print("="*40)
    
    try:
        # Check imports
        from src.segmentation import InstanceSegmentation
        from src.download_models import check_models
        print("✅ All modules imported successfully")
        
        # Check models
        if check_models():
            print("✅ All model files found and valid")
        else:
            print("❌ Some model files missing or invalid")
        
        # Check CUDA
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        # Check directories
        if os.path.exists('/content/drive/My Drive/model_weights'):
            print("✅ Model directory accessible")
        else:
            print("❌ Model directory not found")
        
        print("\n📊 System is ready for testing!")
        
    except Exception as e:
        print(f"❌ System check failed: {e}")

if __name__ == "__main__":
    print("🚀 VOLUME ESTIMATION TESTING SYSTEM")
    print("="*60)
    print("This script will help you test the volume estimation system")
    print("with your own images!")
    
    # Mount Google Drive if not already mounted
    try:
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
    except:
        pass
    
    # Run main menu
    while True:
        if not main_menu():
            break 