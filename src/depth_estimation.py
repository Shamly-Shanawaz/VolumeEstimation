import torch
import numpy as np
import cv2
from PIL import Image
import timm
import matplotlib.pyplot as plt

class DepthEstimator:
    def __init__(self, model_path, model_type="dpt_hybrid"):
        """
        Initialize the Depth Estimator.
        
        Args:
            model_path (str): Path to the model weights
            model_type (str): Type of model ('dpt_hybrid', 'dpt_large', 'midas_v21', 'midas_v21_small')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Load the appropriate model
        if model_type.startswith("dpt"):
            self.model = self._load_dpt_model(model_path, model_type)
        else:
            self.model = self._load_midas_model(model_path, model_type)
            
        self.model.eval()
        self.model.to(self.device)
        
    def _load_dpt_model(self, model_path, model_type):
        """Load DPT model."""
        if model_type == "dpt_large":
            model = timm.create_model(
                'vit_large_patch16_384',
                pretrained=True,
                num_classes=1
            )
        else:  # dpt_hybrid
            model = timm.create_model(
                'vit_base_resnet50_384',
                pretrained=True,
                num_classes=1
            )
            
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model
    
    def _load_midas_model(self, model_path, model_type):
        """Load MiDaS model."""
        if model_type == "midas_v21_small":
            model = timm.create_model(
                'tf_efficientnet_lite3',
                pretrained=True,
                num_classes=1
            )
        else:  # midas_v21
            model = timm.create_model(
                'tf_efficientnet_b5',
                pretrained=True,
                num_classes=1
            )
            
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model
    
    def preprocess_image(self, image):
        """Preprocess the input image for the model."""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize image
        if self.model_type.startswith("dpt"):
            target_size = (384, 384)
        else:
            target_size = (256, 256)
            
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image.
        
        Args:
            image: Input image (path or numpy array)
            
        Returns:
            numpy.ndarray: Depth map
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Get depth prediction
        with torch.no_grad():
            depth = self.model(input_tensor)
            
        # Post-process depth map
        depth = depth.squeeze().cpu().numpy()
        
        # Resize to original image size if needed
        if isinstance(image, str):
            original_size = cv2.imread(image).shape[:2]
        else:
            original_size = image.shape[:2]
            
        depth = cv2.resize(depth, (original_size[1], original_size[0]))
        
        return depth
    
    def visualize_depth(self, depth_map):
        """Convert depth map to visualization format."""
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        depth_colored = (plt.cm.plasma(depth_norm) * 255).astype(np.uint8)
        return depth_colored 