import os
import torch
import numpy as np
from PIL import Image
import cv2
from .depth_estimation import DepthEstimator
from .segmentation import InstanceSegmentation

class VolumeEstimator:
    def __init__(self, segmentation_model_path, depth_model_path, depth_model_type="dpt_hybrid"):
        """
        Initialize the Volume Estimator.
        
        Args:
            segmentation_model_path (str): Path to the PointRend model
            depth_model_path (str): Path to the depth estimation model
            depth_model_type (str): Type of depth model ('dpt_hybrid', 'dpt_large', 'midas_v21', 'midas_v21_small')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize segmentation model
        self.segmentation = InstanceSegmentation(segmentation_model_path)
        
        # Initialize depth estimation model
        self.depth_estimator = DepthEstimator(
            model_path=depth_model_path,
            model_type=depth_model_type
        )
        
    def preprocess_image(self, image_path):
        """Preprocess the input image."""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        return image
    
    def estimate_volume(self, image_path, reference_object=None, reference_depth=None):
        """
        Estimate the volume of objects in the image.
        
        Args:
            image_path (str): Path to the input image
            reference_object (str, optional): Name of the reference object for calibration
            reference_depth (float, optional): Known depth of the reference object
            
        Returns:
            dict: Dictionary containing volume estimates for each detected object
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Get segmentation results
        segmentation_results = self.segmentation.segment(image)
        
        # Get depth map
        depth_map = self.depth_estimator.estimate_depth(image)
        
        # If reference object and depth are provided, calibrate the depth map
        if reference_object and reference_depth:
            depth_map = self._calibrate_depth(depth_map, segmentation_results, reference_object, reference_depth)
        
        # Calculate volumes for each detected object
        volumes = {}
        for obj_name, mask in segmentation_results['masks'].items():
            volume = self._calculate_volume(depth_map, mask)
            volumes[obj_name] = volume
            
        return volumes
    
    def _calibrate_depth(self, depth_map, segmentation_results, reference_object, reference_depth):
        """Calibrate the depth map using a reference object."""
        if reference_object not in segmentation_results['masks']:
            raise ValueError(f"Reference object '{reference_object}' not found in the image")
            
        ref_mask = segmentation_results['masks'][reference_object]
        ref_depth = np.mean(depth_map[ref_mask > 0])
        scale_factor = reference_depth / ref_depth
        
        return depth_map * scale_factor
    
    def _calculate_volume(self, depth_map, mask):
        """Calculate the volume of an object using its depth map and mask."""
        # Get the object's depth values
        obj_depth = depth_map[mask > 0]
        
        # Calculate the volume using the depth values
        # This is a simplified calculation - you might want to use more sophisticated methods
        volume = np.sum(obj_depth) * (1.0 / depth_map.shape[0]) * (1.0 / depth_map.shape[1])
        
        return volume
    
    def visualize_results(self, image_path, volumes):
        """Visualize the results with bounding boxes and volume estimates."""
        image = self.preprocess_image(image_path)
        
        # Get segmentation results for visualization
        segmentation_results = self.segmentation.segment(image)
        
        # Draw bounding boxes and volume estimates
        for obj_name, volume in volumes.items():
            if obj_name in segmentation_results['boxes']:
                box = segmentation_results['boxes'][obj_name]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{obj_name}: {volume:.2f}", 
                          (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image 