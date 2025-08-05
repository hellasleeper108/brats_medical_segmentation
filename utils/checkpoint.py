import os
import torch
from typing import Dict, Any, Optional


def save_checkpoint(state: Dict[str, Any], is_best: bool, 
                   checkpoint_dir: str, filename: Optional[str] = None) -> None:
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state and metadata
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Custom filename (optional)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = 'checkpoint.pth'
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    
    Returns:
        Dictionary containing model state and metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint