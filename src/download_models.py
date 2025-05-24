import os
import requests
from tqdm import tqdm
import pixellib
from huggingface_hub import hf_hub_download

def download_file_with_progress(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/midas_depth/weights", exist_ok=True)
    
    print("Downloading PointRend model...")
    # Download PointRend model using pixellib
    pixellib.download_pointrend()
    
    print("\nDownloading MiDaS models...")
    # MiDaS models from Hugging Face
    midas_models = {
        "midas_v21": {
            "repo_id": "isl-org/MiDaS",
            "filename": "weights/midas_v21-f6b98070.pt"
        },
        "midas_v21_small": {
            "repo_id": "isl-org/MiDaS",
            "filename": "weights/midas_v21_small-70d6b9c8.pt"
        }
    }
    
    for name, model_info in midas_models.items():
        print(f"Downloading {name}...")
        try:
            # Download from Hugging Face
            model_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                repo_type="model"
            )
            # Copy to our models directory
            target_path = f"models/midas_depth/weights/{name}-{os.path.basename(model_info['filename'])}"
            import shutil
            shutil.copy2(model_path, target_path)
            print(f"Saved to {target_path}")
        except Exception as e:
            print(f"Error downloading {name}: {str(e)}")
    
    print("\nDownloading DPT models...")
    # DPT models from Hugging Face
    dpt_models = {
        "dpt_large": {
            "repo_id": "isl-org/DPT",
            "filename": "weights/dpt_large-midas-2f21e586.pt"
        },
        "dpt_hybrid": {
            "repo_id": "isl-org/DPT",
            "filename": "weights/dpt_hybrid-midas-501f0c75.pt"
        }
    }
    
    for name, model_info in dpt_models.items():
        print(f"Downloading {name}...")
        try:
            # Download from Hugging Face
            model_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                repo_type="model"
            )
            # Copy to our models directory
            target_path = f"models/midas_depth/weights/{name}-{os.path.basename(model_info['filename'])}"
            import shutil
            shutil.copy2(model_path, target_path)
            print(f"Saved to {target_path}")
        except Exception as e:
            print(f"Error downloading {name}: {str(e)}")
    
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    main() 