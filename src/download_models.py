import os

# For Google Colab: mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = '/content/drive/My Drive/model_weights/'
except ImportError:
    print("Not running in Google Colab. Please run this script in a Colab environment.")
    DRIVE_PATH = None

REQUIRED_MODELS = {
    'DPT Hybrid': 'dpt_hybrid_pytorch_model.bin',
    'DPT Large': 'dpt_large_pytorch_model.bin',
    'MiDaS v21 Large': 'midas_v21_large_384.pt',
    'MiDaS v21 Small': 'midas_v21_small_256.pt',
    'PointRend COCO': 'pointtrend_rcnn_R_50_FPN_3x_coco.pkl',
}

def main():
    if DRIVE_PATH is None:
        print("Google Drive not mounted. Exiting.")
        return
    print(f"Checking for required model files in Google Drive: {DRIVE_PATH}\n")
    all_found = True
    for name, filename in REQUIRED_MODELS.items():
        path = os.path.join(DRIVE_PATH, filename)
        if os.path.exists(path):
            print(f"[FOUND] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            all_found = False
    if all_found:
        print("\nAll required model files are present in your Google Drive!")
    else:
        print("\nSome model files are missing. Please upload them to your Google Drive in the 'model_weights' folder.")

if __name__ == "__main__":
    main() 