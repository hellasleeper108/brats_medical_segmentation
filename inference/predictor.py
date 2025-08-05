import os
import time
import torch
import numpy as np
import nibabel as nib
from typing import Dict, Tuple, Optional, List
import yaml

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UNet3D, VisionTransformer3D
from data.transforms import get_inference_transforms, get_post_transforms
from utils.device import get_device
from utils.checkpoint import load_checkpoint


class MedicalImagePredictor:
    """Medical image segmentation predictor."""
    
    def __init__(self, model_path: str, model_type: str = 'unet', 
                 config: Optional[Dict] = None, device: str = 'cuda'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('unet', 'vit')
            config: Model configuration dictionary
            device: Device to run inference on
        """
        self.model_path = model_path
        self.model_type = model_type
        self.config = config or self._load_default_config()
        self.device = get_device(device)
        
        # Load model
        self.model = self._load_model()
        
        # Setup transforms
        self.transforms = get_inference_transforms(
            roi_size=tuple(self.config['model']['input_size']),
            spacing=(1.0, 1.0, 1.0),
            intensity_range=(-1000, 1000)
        )
        
        self.post_transforms = get_post_transforms()
        
        print(f"Predictor initialized with {model_type} model on {self.device}")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration if none provided."""
        return {
            'model': {
                'name': self.model_type,
                'input_size': [128, 128, 128],
                'num_classes': 4,
                'dropout': 0.0
            }
        }
    
    def _load_model(self):
        """Load the trained model."""
        # Initialize model architecture
        model_config = self.config['model']
        
        if self.model_type == 'unet':
            model = UNet3D(
                in_channels=1,
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout']
            )
        elif self.model_type == 'vit':
            model = VisionTransformer3D(
                img_size=tuple(model_config['input_size']),
                in_channels=1,
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout']
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load trained weights
        if os.path.exists(self.model_path):
            checkpoint = load_checkpoint(self.model_path, self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {self.model_path}")
            if 'best_dice' in checkpoint:
                print(f"Model best Dice score: {checkpoint['best_dice']:.4f}")
        else:
            print(f"Warning: Model file not found at {self.model_path}")
            print("Using randomly initialized weights")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_path: str) -> Tuple[np.ndarray, Dict, float]:
        """
        Perform segmentation prediction on a medical image.
        
        Args:
            image_path: Path to input medical image
        
        Returns:
            Tuple of (prediction array, metrics dict, processing time)
        """
        start_time = time.time()
        
        # Load and preprocess image
        data_dict = {'image': image_path}
        data_dict = self.transforms(data_dict)
        
        # Move to device
        image = data_dict['image'].unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            with torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad():
                logits = self.model(image)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy
        prediction_np = prediction.cpu().numpy().squeeze()
        probabilities_np = probabilities.cpu().numpy().squeeze()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_inference_metrics(
            prediction_np, probabilities_np, processing_time
        )
        
        return prediction_np, metrics, processing_time
    
    def predict_with_tta(self, image_path: str, 
                        tta_transforms: List[str] = ['flip_x', 'flip_y', 'flip_z']) -> Tuple[np.ndarray, Dict, float]:
        """
        Perform prediction with Test Time Augmentation (TTA).
        
        Args:
            image_path: Path to input medical image
            tta_transforms: List of TTA transforms to apply
        
        Returns:
            Tuple of (prediction array, metrics dict, processing time)
        """
        start_time = time.time()
        
        # Load and preprocess image
        data_dict = {'image': image_path}
        data_dict = self.transforms(data_dict)
        original_image = data_dict['image']
        
        predictions = []
        
        # Original prediction
        image = original_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image)
            probabilities = torch.softmax(logits, dim=1)
            predictions.append(probabilities.cpu())
        
        # TTA predictions
        for transform in tta_transforms:
            augmented_image = self._apply_tta_transform(original_image, transform)
            augmented_image = augmented_image.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(augmented_image)
                probabilities = torch.softmax(logits, dim=1)
                # Reverse the transformation on predictions
                probabilities = self._reverse_tta_transform(probabilities, transform)
                predictions.append(probabilities.cpu())
        
        # Average predictions
        avg_probabilities = torch.stack(predictions).mean(dim=0)
        prediction = torch.argmax(avg_probabilities, dim=1)
        
        # Convert to numpy
        prediction_np = prediction.numpy().squeeze()
        probabilities_np = avg_probabilities.numpy().squeeze()
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_inference_metrics(
            prediction_np, probabilities_np, processing_time
        )
        metrics['tta_used'] = True
        metrics['tta_transforms'] = tta_transforms
        
        return prediction_np, metrics, processing_time
    
    def _apply_tta_transform(self, image: torch.Tensor, transform: str) -> torch.Tensor:
        """Apply test time augmentation transform."""
        if transform == 'flip_x':
            return torch.flip(image, dims=[1])
        elif transform == 'flip_y':
            return torch.flip(image, dims=[2])
        elif transform == 'flip_z':
            return torch.flip(image, dims=[3])
        else:
            return image
    
    def _reverse_tta_transform(self, probabilities: torch.Tensor, transform: str) -> torch.Tensor:
        """Reverse test time augmentation transform on predictions."""
        if transform == 'flip_x':
            return torch.flip(probabilities, dims=[2])
        elif transform == 'flip_y':
            return torch.flip(probabilities, dims=[3])
        elif transform == 'flip_z':
            return torch.flip(probabilities, dims=[4])
        else:
            return probabilities
    
    def _calculate_inference_metrics(self, prediction: np.ndarray, 
                                   probabilities: np.ndarray, 
                                   processing_time: float) -> Dict:
        """Calculate inference metrics."""
        metrics = {
            'processing_time': processing_time,
            'prediction_shape': prediction.shape,
            'num_classes_predicted': len(np.unique(prediction)),
            'class_distribution': {}
        }
        
        # Calculate class distribution
        unique_classes, counts = np.unique(prediction, return_counts=True)
        total_voxels = prediction.size
        
        for class_id, count in zip(unique_classes, counts):
            percentage = (count / total_voxels) * 100
            metrics['class_distribution'][f'class_{int(class_id)}'] = {
                'voxels': int(count),
                'percentage': float(percentage)
            }
        
        # Calculate confidence statistics
        max_probs = np.max(probabilities, axis=0)
        metrics['confidence'] = {
            'mean': float(np.mean(max_probs)),
            'std': float(np.std(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs))
        }
        
        # Memory usage (if CUDA)
        if self.device.type == 'cuda':
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return metrics
    
    def batch_predict(self, image_paths: List[str], 
                     output_dir: str = 'predictions') -> List[Dict]:
        """
        Perform batch prediction on multiple images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save predictions
        
        Returns:
            List of result dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Perform prediction
                prediction, metrics, processing_time = self.predict(image_path)
                
                # Save prediction
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_prediction.nii.gz")
                
                # Load original image to get header info
                original_img = nib.load(image_path)
                pred_img = nib.Nifti1Image(prediction, original_img.affine, original_img.header)
                nib.save(pred_img, output_path)
                
                result = {
                    'input_path': image_path,
                    'output_path': output_path,
                    'metrics': metrics,
                    'processing_time': processing_time,
                    'success': True
                }
                
                print(f"  Completed in {processing_time:.2f}s")
                
            except Exception as e:
                result = {
                    'input_path': image_path,
                    'error': str(e),
                    'success': False
                }
                print(f"  Failed: {str(e)}")
            
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': self.model.get_model_size(),
            'input_size': self.config['model']['input_size'],
            'num_classes': self.config['model']['num_classes']
        }
        
        return info