import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from typing import Tuple, List, Optional, Union


class DiceScore(nn.Module):
    """Dice Score metric for segmentation evaluation."""
    
    def __init__(self, include_background: bool = False, 
                 reduction: str = 'mean', smooth: float = 1e-5):
        super(DiceScore, self).__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.smooth = smooth
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice score.
        
        Args:
            y_pred: Predicted segmentation [B, C, H, W, D]
            y_true: Ground truth segmentation [B, C, H, W, D] or [B, H, W, D]
        
        Returns:
            Dice score tensor
        """
        if y_pred.dim() == 4 and y_true.dim() == 4:
            # Convert to one-hot if needed
            num_classes = y_pred.shape[1]
            y_true = torch.nn.functional.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3)
        
        # Apply softmax to predictions if needed
        if y_pred.dtype == torch.float32:
            y_pred = torch.softmax(y_pred, dim=1)
        
        # Start from class 1 if background should be excluded
        start_idx = 1 if not self.include_background else 0
        
        dice_scores = []
        for i in range(start_idx, y_pred.shape[1]):
            pred_i = y_pred[:, i]
            true_i = y_true[:, i]
            
            intersection = (pred_i * true_i).sum(dim=(1, 2, 3))
            union = pred_i.sum(dim=(1, 2, 3)) + true_i.sum(dim=(1, 2, 3))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)
        
        if self.reduction == 'mean':
            return dice_scores.mean()
        elif self.reduction == 'none':
            return dice_scores
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class HausdorffDistance:
    """Hausdorff Distance metric for segmentation evaluation."""
    
    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
        
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Hausdorff distance.
        
        Args:
            y_pred: Predicted binary segmentation
            y_true: Ground truth binary segmentation
        
        Returns:
            Hausdorff distance
        """
        # Get surface points
        pred_points = self._get_surface_points(y_pred)
        true_points = self._get_surface_points(y_true)
        
        if len(pred_points) == 0 or len(true_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        hd1 = directed_hausdorff(pred_points, true_points)[0]
        hd2 = directed_hausdorff(true_points, pred_points)[0]
        
        # Return maximum (standard Hausdorff) or percentile
        if self.percentile == 100.0:
            return max(hd1, hd2)
        else:
            distances = np.concatenate([
                np.min(np.linalg.norm(pred_points[:, None] - true_points, axis=2), axis=1),
                np.min(np.linalg.norm(true_points[:, None] - pred_points, axis=2), axis=1)
            ])
            return np.percentile(distances, self.percentile)
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface points from binary mask."""
        from scipy import ndimage
        
        # Get boundary using morphological operations
        eroded = ndimage.binary_erosion(mask)
        boundary = mask ^ eroded
        
        # Get coordinates of boundary points
        coords = np.where(boundary)
        return np.column_stack(coords)


class SurfaceDistance:
    """Average Surface Distance metric."""
    
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate average surface distance.
        
        Args:
            y_pred: Predicted binary segmentation
            y_true: Ground truth binary segmentation
        
        Returns:
            Average surface distance
        """
        pred_points = self._get_surface_points(y_pred)
        true_points = self._get_surface_points(y_true)
        
        if len(pred_points) == 0 or len(true_points) == 0:
            return float('inf')
        
        # Calculate distances from pred to true
        distances_pt = np.min(np.linalg.norm(
            pred_points[:, None] - true_points, axis=2), axis=1)
        avg_dist_pt = np.mean(distances_pt)
        
        if not self.symmetric:
            return avg_dist_pt
        
        # Calculate distances from true to pred
        distances_tp = np.min(np.linalg.norm(
            true_points[:, None] - pred_points, axis=2), axis=1)
        avg_dist_tp = np.mean(distances_tp)
        
        return (avg_dist_pt + avg_dist_tp) / 2.0
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface points from binary mask."""
        from scipy import ndimage
        
        eroded = ndimage.binary_erosion(mask)
        boundary = mask ^ eroded
        coords = np.where(boundary)
        return np.column_stack(coords)


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculator."""
    
    def __init__(self, num_classes: int = 4, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        self.dice_metric = DiceScore(include_background=include_background)
        self.hausdorff_metric = HausdorffDistance()
        self.surface_metric = SurfaceDistance()
    
    def calculate_all_metrics(self, y_pred: torch.Tensor, 
                            y_true: torch.Tensor) -> dict:
        """
        Calculate all segmentation metrics.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Dice score
        dice = self.dice_metric(y_pred, y_true)
        metrics['dice'] = dice.item() if isinstance(dice, torch.Tensor) else dice
        
        # Convert to numpy for other metrics
        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
        else:
            y_pred_np = y_pred
            y_true_np = y_true
        
        # Per-class metrics
        start_idx = 1 if not self.include_background else 0
        class_metrics = {}
        
        for i in range(start_idx, self.num_classes):
            pred_class = (y_pred_np == i).astype(np.uint8)
            true_class = (y_true_np == i).astype(np.uint8)
            
            if pred_class.sum() > 0 and true_class.sum() > 0:
                hd = self.hausdorff_metric(pred_class, true_class)
                asd = self.surface_metric(pred_class, true_class)
                
                class_metrics[f'class_{i}'] = {
                    'hausdorff_distance': hd,
                    'average_surface_distance': asd
                }
        
        metrics['per_class'] = class_metrics
        
        # Sensitivity and Specificity
        sensitivity, specificity = self._calculate_sensitivity_specificity(
            y_pred_np, y_true_np)
        metrics['sensitivity'] = sensitivity
        metrics['specificity'] = specificity
        
        return metrics
    
    def _calculate_sensitivity_specificity(self, y_pred: np.ndarray, 
                                         y_true: np.ndarray) -> Tuple[float, float]:
        """Calculate sensitivity and specificity."""
        # Flatten arrays
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, 
                            labels=list(range(self.num_classes)))
        
        # Calculate sensitivity (recall) and specificity for each class
        sensitivities = []
        specificities = []
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        return np.mean(sensitivities), np.mean(specificities)