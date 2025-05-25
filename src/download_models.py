import os
from google.colab import drive

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

def check_models():
    """Check if all required model files exist in Google Drive."""
    print(f"Checking for required model files in Google Drive: {DRIVE_PATH}\n")
    all_found = True
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"[FOUND] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            all_found = False
    
    if all_found:
        print("\nAll required model files are present in your Google Drive!")
    else:
        print("\nSome model files are missing. Please upload them to your Google Drive in the 'model_weights' folder.")
    
    return all_found

def get_volume_estimator():
    """Initialize and return the VolumeEstimator with the correct model paths."""
    from src.volume_estimation import VolumeEstimator
    
    estimator = VolumeEstimator(
        segmentation_model_path=MODEL_PATHS['pointrend'],
        depth_model_path=MODEL_PATHS['dpt_hybrid'],
        depth_model_type="dpt_hybrid"
    )
    print("Models initialized successfully!")
    return estimator

if __name__ == "__main__":
    if check_models():
        estimator = get_volume_estimator() 