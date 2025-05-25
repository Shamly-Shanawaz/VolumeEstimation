import torch
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import os

class InstanceSegmentation:
    def __init__(self, model_path):
        """
        Initialize the Instance Segmentation model using Detectron2.
        
        Args:
            model_path (str): Path to the model weights (.pkl file)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup Detectron2 configuration
        self.cfg = get_cfg()
        
        # Auto-detect model type and use appropriate configuration
        model_name = os.path.basename(model_path).lower()
        
        if "pointrend" in model_name or "pointtrend" in model_name:
            # For PointRend models, use Mask R-CNN as base since PointRend isn't in main model zoo
            print("PointRend model detected, using Mask R-CNN configuration as base...")
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        else:
            # For other models, use standard Mask R-CNN
            print("Using Mask R-CNN configuration...")
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Set model weights path
        self.cfg.MODEL.WEIGHTS = model_path
        
        # Set device
        self.cfg.MODEL.DEVICE = str(self.device)
        
        # Set confidence threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        # Get metadata for class names
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        
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
        
        # Ensure image is in the right format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR for Detectron2
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            
        # Perform inference
        outputs = self.predictor(image_bgr)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        
        # Process results
        processed_results = {
            'masks': {},
            'boxes': {},
            'class_names': [],
            'scores': []
        }
        
        if len(instances) > 0:
            # Get class names
            class_ids = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()
            masks = instances.pred_masks.numpy()
            
            # Get class names from metadata
            class_names = [self.metadata.thing_classes[class_id] for class_id in class_ids]
            
            # Store results with unique identifiers for multiple objects of same class
            class_counts = {}
            for i, (class_name, score, box, mask) in enumerate(zip(class_names, scores, boxes, masks)):
                # Create unique identifier for objects of same class
                if class_name in class_counts:
                    class_counts[class_name] += 1
                    unique_name = f"{class_name}_{class_counts[class_name]}"
                else:
                    class_counts[class_name] = 1
                    unique_name = f"{class_name}_1"
                
                processed_results['masks'][unique_name] = mask
                processed_results['boxes'][unique_name] = box
                processed_results['class_names'].append(unique_name)
                processed_results['scores'].append(score)
                
        return processed_results
    
    def visualize_segmentation(self, image, results=None):
        """
        Visualize segmentation results using Detectron2's visualizer.
        
        Args:
            image: Input image (path or numpy array)
            results: Optional pre-computed segmentation results
            
        Returns:
            numpy.ndarray: Image with visualization
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to BGR for Detectron2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get predictions if not provided
        if results is None:
            outputs = self.predictor(image_bgr)
            # Create visualizer
            v = Visualizer(image_bgr[:, :, ::-1], self.metadata, scale=1.2)
            # Draw predictions
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            return out.get_image()[:, :, ::-1]  # Convert back to RGB
        else:
            # For custom visualization with pre-computed results
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
    
    def get_object_masks(self, image, class_filter=None):
        """
        Get masks for specific object classes.
        
        Args:
            image: Input image
            class_filter: List of class names to filter (None for all)
            
        Returns:
            dict: Dictionary of masks for filtered classes
        """
        results = self.segment(image)
        
        if class_filter is None:
            return results['masks']
        
        filtered_masks = {}
        for obj_name, mask in results['masks'].items():
            # Extract base class name (remove the _1, _2 suffix)
            base_class = obj_name.rsplit('_', 1)[0]
            if base_class in class_filter:
                filtered_masks[obj_name] = mask
                
        return filtered_masks 