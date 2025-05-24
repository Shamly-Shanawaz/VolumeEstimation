import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.volume_estimation import VolumeEstimator
import matplotlib.pyplot as plt

def main():
    # Initialize the volume estimator
    estimator = VolumeEstimator(
        segmentation_model_path="models/pointrend_resnet50.pkl",
        depth_model_path="models/midas_depth/weights/dpt_hybrid-midas-501f0c75.pt",
        depth_model_type="dpt_hybrid"
    )
    
    # Example with a reference object
    image_path = "data/input/example.jpg"
    reference_object = "bottle"  # Name of the reference object
    reference_depth = 20.0  # Known depth in cm
    
    # Estimate volumes
    volumes = estimator.estimate_volume(
        image_path,
        reference_object=reference_object,
        reference_depth=reference_depth
    )
    
    # Print results
    print("\nEstimated volumes:")
    for obj_name, volume in volumes.items():
        print(f"{obj_name}: {volume:.2f} cubic cm")
    
    # Visualize results
    result_image = estimator.visualize_results(image_path, volumes)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.axis('off')
    plt.title('Volume Estimation Results')
    plt.show()

if __name__ == "__main__":
    main() 