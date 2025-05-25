import os
import torch
import numpy as np
import pickle
from google.colab import drive
from src.volume_estimation import VolumeEstimator
from src.segmentation import InstanceSegmentation
from src.depth_estimation import DepthEstimator

# Mount Google Drive
drive.mount('/content/drive')

# Define the base path for models in Google Drive
DRIVE_PATH = '/content/drive/My Drive/model_weights/'

# Define the model paths
MODEL_PATHS = {
    'pointrend': os.path.join(DRIVE_PATH, 'pointtrend_rcnn_R_50_FPN_3x_coco.pkl'),
    'dpt_hybrid': os.path.join(DRIVE_PATH, 'dpt_hybrid_pytorch_model.bin'),
    'dpt_large': os.path.join(DRIVE_PATH, 'dpt_large_pytorch_model.bin'),
    'midas_large': os.path.join(DRIVE_PATH, 'midas_v21_large_384.pt'),
    'midas_small': os.path.join(DRIVE_PATH, 'midas_v21_small_256.pt')
}

def load_model_file(file_path):
    """Load a model file with proper error handling."""
    try:
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith(('.pt', '.bin')):
            # Try different loading methods
            try:
                # First try with weights_only=False
                return torch.load(file_path, map_location='cpu', weights_only=False)
            except Exception as e1:
                print(f"First loading attempt failed: {str(e1)}")
                try:
                    # Try with weights_only=True
                    return torch.load(file_path, map_location='cpu', weights_only=True)
                except Exception as e2:
                    print(f"Second loading attempt failed: {str(e2)}")
                    # Try loading as state dict
                    return torch.load(file_path, map_location='cpu', pickle_module=pickle)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        raise

def check_models():
    """Check if all required model files exist in Google Drive."""
    print(f"Checking for required model files in Google Drive: {DRIVE_PATH}\n")
    all_found = True
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"[FOUND] {name}: {path}")
            # Verify file is readable
            try:
                model = load_model_file(path)
                print(f"[VERIFIED] {name} file is valid")
                # Print model type for debugging
                print(f"[INFO] {name} model type: {type(model)}")
            except Exception as e:
                print(f"[ERROR] {name} file is corrupted or invalid: {str(e)}")
                all_found = False
        else:
            print(f"[MISSING] {name}: {path}")
            all_found = False
    
    if all_found:
        print("\nAll required model files are present and valid in your Google Drive!")
    else:
        print("\nSome model files are missing or invalid. Please check the files in your Google Drive.")
    
    return all_found

def get_volume_estimator():
    """Initialize and return the VolumeEstimator with the correct model paths."""
    try:
        # Try loading the models first to catch any errors
        print("Loading segmentation model...")
        try:
            segmentation_model = load_model_file(MODEL_PATHS['pointrend'])
            print("Segmentation model loaded successfully")
        except Exception as e:
            print(f"Error loading segmentation model: {str(e)}")
            raise
            
        print("Loading depth model...")
        try:
            depth_model = load_model_file(MODEL_PATHS['dpt_hybrid'])
            print("Depth model loaded successfully")
        except Exception as e:
            print(f"Error loading depth model: {str(e)}")
            raise
        
        print("Initializing VolumeEstimator...")
        try:
            # Try initializing with each model separately to identify the problematic one
            print("Testing segmentation model initialization...")
            test_seg = InstanceSegmentation(MODEL_PATHS['pointrend'])
            print("Segmentation model initialized successfully")
            
            print("Testing depth model initialization...")
            test_depth = DepthEstimator(MODEL_PATHS['dpt_hybrid'], model_type="dpt_hybrid")
            print("Depth model initialized successfully")
            
            # If both tests pass, initialize the full estimator
            estimator = VolumeEstimator(
                segmentation_model_path=MODEL_PATHS['pointrend'],
                depth_model_path=MODEL_PATHS['dpt_hybrid'],
                depth_model_type="dpt_hybrid"
            )
            print("Models initialized successfully!")
            return estimator
        except Exception as e:
            print(f"Error during model initialization: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            raise
            
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure all model files are properly downloaded and not corrupted")
        print("2. Check if the model files are in the correct format")
        print("3. Verify that you have enough memory available")
        print("4. Try downloading the model files again")
        raise

if __name__ == "__main__":
    if check_models():
        try:
            estimator = get_volume_estimator()
        except Exception as e:
            print(f"Failed to initialize estimator: {str(e)}") 