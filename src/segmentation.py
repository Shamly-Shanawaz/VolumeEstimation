import torch
import numpy as np
import cv2
from pixellib.instance import instance_segmentation

class InstanceSegmentation:
    def __init__(self, model_path):
        """
        Initialize the Instance Segmentation model.
        
        Args:
            model_path (str): Path to the PointRend model weights
        """
        self.model = instance_segmentation()
        self.model.load_model(model_path)
        
    def segment(self, image):
        """
        Perform instance segmentation on the image.
        
        Args:
            image: Input image (path or numpy array)
            
        Returns:
            dict: Dictionary containing segmentation results
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Perform segmentation
        results = self.model.segmentFrame(image, show_bboxes=True)
        
        # Process results
        processed_results = {
            'masks': {},
            'boxes': {},
            'class_names': []
        }
        
        for i, (box, class_name, score) in enumerate(zip(
            results['boxes'], results['class_names'], results['scores'])):
            
            if score > 0.5:  # Confidence threshold
                processed_results['masks'][class_name] = results['masks'][:, :, i]
                processed_results['boxes'][class_name] = box
                processed_results['class_names'].append(class_name)
                
        return processed_results
    
    def visualize_segmentation(self, image, results):
        """
        Visualize segmentation results.
        
        Args:
            image: Input image
            results: Segmentation results from segment()
            
        Returns:
            numpy.ndarray: Image with visualization
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Create visualization
        vis_image = image.copy()
        
        # Draw masks
        for class_name, mask in results['masks'].items():
            color = np.random.randint(0, 255, 3).tolist()
            vis_image[mask > 0] = vis_image[mask > 0] * 0.5 + np.array(color) * 0.5
            
        # Draw bounding boxes
        for class_name, box in results['boxes'].items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, class_name, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return vis_image 