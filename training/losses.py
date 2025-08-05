import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, include_background: bool = False, 
                 smooth: float = 1e-5, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            input: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D] or [B, C, H, W, D]
        
        Returns:
            Dice loss
        """
        # Apply softmax to get probabilities
        input_soft = F.softmax(input, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:  # [B, H, W, D]
            target_one_hot = F.one_hot(target.long(), input.shape[1]).permute(0, 4, 1, 2, 3)
        else:  # Already one-hot [B, C, H, W, D]
            target_one_hot = target
        
        # Start from class 1 if background should be excluded
        start_idx = 1 if not self.include_background else 0
        
        dice_losses = []
        for i in range(start_idx, input.shape[1]):
            input_i = input_soft[:, i]
            target_i = target_one_hot[:, i].float()
            
            intersection = (input_i * target_i).sum(dim=(1, 2, 3))
            union = input_i.sum(dim=(1, 2, 3)) + target_i.sum(dim=(1, 2, 3))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice
            dice_losses.append(dice_loss)
        
        dice_losses = torch.stack(dice_losses, dim=1)
        
        if self.reduction == 'mean':
            return dice_losses.mean()
        elif self.reduction == 'sum':
            return dice_losses.sum()
        else:
            return dice_losses


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.
        
        Args:
            input: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D]
        
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            focal_loss = at.view(target.shape) * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice loss with class weighting."""
    
    def __init__(self, include_background: bool = False, 
                 smooth: float = 1e-5, reduction: str = 'mean'):
        super(GeneralizedDiceLoss, self).__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Generalized Dice loss.
        
        Args:
            input: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D] or [B, C, H, W, D]
        
        Returns:
            Generalized Dice loss
        """
        # Apply softmax to get probabilities
        input_soft = F.softmax(input, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:
            target_one_hot = F.one_hot(target.long(), input.shape[1]).permute(0, 4, 1, 2, 3)
        else:
            target_one_hot = target
        
        # Start from class 1 if background should be excluded
        start_idx = 1 if not self.include_background else 0
        
        # Calculate class weights (inverse of class frequency)
        w = []
        for i in range(start_idx, input.shape[1]):
            class_sum = target_one_hot[:, i].sum()
            w.append(1.0 / (class_sum + self.smooth))
        
        w = torch.tensor(w, device=input.device)
        w = w / w.sum()  # Normalize weights
        
        # Calculate weighted dice coefficients
        numerator = 0
        denominator = 0
        
        for i, class_idx in enumerate(range(start_idx, input.shape[1])):
            input_i = input_soft[:, class_idx]
            target_i = target_one_hot[:, class_idx].float()
            
            intersection = (input_i * target_i).sum()
            union = input_i.sum() + target_i.sum()
            
            numerator += w[i] * intersection
            denominator += w[i] * union
        
        gd_loss = 1.0 - (2.0 * numerator + self.smooth) / (denominator + self.smooth)
        
        return gd_loss


class TverskyLoss(nn.Module):
    """Tversky loss - generalization of Dice loss."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 include_background: bool = False, smooth: float = 1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.include_background = include_background
        self.smooth = smooth
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Tversky loss.
        
        Args:
            input: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D] or [B, C, H, W, D]
        
        Returns:
            Tversky loss
        """
        # Apply softmax to get probabilities
        input_soft = F.softmax(input, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:
            target_one_hot = F.one_hot(target.long(), input.shape[1]).permute(0, 4, 1, 2, 3)
        else:
            target_one_hot = target
        
        # Start from class 1 if background should be excluded
        start_idx = 1 if not self.include_background else 0
        
        tversky_losses = []
        for i in range(start_idx, input.shape[1]):
            input_i = input_soft[:, i]
            target_i = target_one_hot[:, i].float()
            
            tp = (input_i * target_i).sum(dim=(1, 2, 3))
            fp = (input_i * (1 - target_i)).sum(dim=(1, 2, 3))
            fn = ((1 - input_i) * target_i).sum(dim=(1, 2, 3))
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_loss = 1.0 - tversky
            tversky_losses.append(tversky_loss)
        
        tversky_losses = torch.stack(tversky_losses, dim=1)
        return tversky_losses.mean()


class CombinedLoss(nn.Module):
    """Combined loss function for better segmentation performance."""
    
    def __init__(self, num_classes: int = 4, include_background: bool = False,
                 dice_weight: float = 0.5, ce_weight: float = 0.3, 
                 focal_weight: float = 0.2, focal_gamma: float = 2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(include_background=include_background)
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            input: Predicted logits [B, C, H, W, D]
            target: Ground truth labels [B, H, W, D]
        
        Returns:
            Combined loss
        """
        # Squeeze target if needed
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        dice = self.dice_loss(input, target)
        ce = self.ce_loss(input, target.long())
        focal = self.focal_loss(input, target.long())
        
        total_loss = (self.dice_weight * dice + 
                     self.ce_weight * ce + 
                     self.focal_weight * focal)
        
        return total_loss