#!/usr/bin/env python3
"""
Test script for the Detectron2-based volume estimation.
This script tests the new implementation in Google Colab.
"""

import sys
import os
sys.path.append('/content/VolumeEstimation/src')

from src.download_models import check_models, get_volume_estimator
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_segmentation_only():
    """Test just the segmentation component."""
    print("Testing segmentation component...")
    
    try:
        from src.segmentation import InstanceSegmentation
        
        # Path to your model
        model_path = '/content/drive/My Drive/model_weights/pointtrend_rcnn_R_50_FPN_3x_coco.pkl'
        
        print("Initializing segmentation model...")
        seg_model = InstanceSegmentation(model_path)
        print("✅ Segmentation model initialized successfully!")
        
        # Create a test image (you can replace this with a real image path)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("Running segmentation...")
        results = seg_model.segment(test_image)
        
        print(f"Segmentation results:")
        print(f"- Number of objects detected: {len(results['class_names'])}")
        print(f"- Object classes: {results['class_names']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the full volume estimation pipeline."""
    print("\nTesting full pipeline...")
    
    try:
        # Check if models are available
        if not check_models():
            print("❌ Model files not found")
            return False
        
        # Initialize the estimator
        estimator = get_volume_estimator()
        print("✅ Full pipeline initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Detectron2-based Volume Estimation")
    print("=" * 50)
    
    # Test 1: Segmentation only
    seg_success = test_segmentation_only()
    
    # Test 2: Full pipeline
    if seg_success:
        full_success = test_full_pipeline()
    else:
        print("Skipping full pipeline test due to segmentation failure")
        full_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Segmentation test: {'✅ PASSED' if seg_success else '❌ FAILED'}")
    print(f"Full pipeline test: {'✅ PASSED' if full_success else '❌ FAILED'}")
    
    if seg_success and full_success:
        print("\n🎉 All tests passed! The Detectron2 implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return seg_success and full_success

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 