# Volume Estimation from Images

This project estimates the volume of objects in images using deep learning models for depth estimation and instance segmentation.

## Features

- Instance segmentation using PointRend
- Depth estimation using MiDaS/DPT models
- Volume calculation for detected objects
- Support for multiple depth estimation models
- Easy-to-use API for volume estimation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VolumeEstimation.git
cd VolumeEstimation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required models:
```bash
python src/download_models.py
```

## Usage

Basic usage:
```python
from volume_estimation import VolumeEstimator

# Initialize the estimator
estimator = VolumeEstimator(
    segmentation_model_path="models/pointrend_resnet50.pkl",
    depth_model_path="models/midas_depth/weights/dpt_hybrid-midas-501f0c75.pt",
    depth_model_type="dpt_hybrid"
)

# Estimate volume from an image
volume = estimator.estimate_volume("path/to/image.jpg")
print(f"Estimated volume: {volume} cubic units")
```

## Project Structure

```
VolumeEstimation/
├── src/                    # Source code
│   ├── __init__.py
│   ├── volume_estimation.py
│   ├── depth_estimation.py
│   ├── segmentation.py
│   └── utils.py
├── models/                 # Model weights
│   ├── pointrend_resnet50.pkl
│   └── midas_depth/
├── data/                   # Data directory
│   ├── input/             # Input images
│   └── output/            # Output results
├── tests/                 # Test files
├── examples/              # Example scripts
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PointRend model from Facebook AI Research
- MiDaS and DPT models from Intel ISL
- PixelLib library for instance segmentation 