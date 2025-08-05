#!/usr/bin/env python3
"""
Quick training test to verify the system is ready.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        import monai
        import nibabel as nib
        from models import UNet3D
        from data.dataset import BraTSDataset
        from data.transforms import get_transforms
        from training.losses import DiceLoss
        from utils.metrics import DiceScore
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"X Import error: {e}")
        return False

def test_data_loading():
    """Test if data can be loaded."""
    print("Testing data loading...")
    
    try:
        # Check if sample data exists
        train_dir = Path("data/samples/train")
        val_dir = Path("data/samples/val")
        
        if not train_dir.exists() or not val_dir.exists():
            print("X Sample data directories not found")
            return False
        
        train_files = list(train_dir.glob("*.nii.gz"))
        val_files = list(val_dir.glob("*.nii.gz"))
        
        if len(train_files) == 0:
            print("X No training files found")
            return False
        
        print(f"✓ Found {len(train_files)//2} training samples, {len(val_files)//2} validation samples")
        return True
        
    except Exception as e:
        print(f"X Data loading error: {e}")
        return False

def test_model_creation():
    """Test if models can be created."""
    print("Testing model creation...")
    
    try:
        from models import UNet3D
        
        model = UNet3D(in_channels=1, num_classes=3, dropout=0.1)
        
        # Test forward pass
        test_input = torch.randn(1, 1, 64, 64, 32)  # Small test size
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ U-Net created successfully. Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"X Model creation error: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ CUDA available: {device} ({memory:.1f} GB)")
        return True
    else:
        print("! CUDA not available - will use CPU")
        return False

def create_minimal_config():
    """Create a minimal config for quick testing."""
    config = {
        'model': {
            'name': 'unet',
            'input_size': [64, 64, 32],  # Small size for quick testing
            'num_classes': 3,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 1,
            'learning_rate': 0.001,
            'epochs': 2,  # Just 2 epochs for testing
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'weight_decay': 0.01,
            'gradient_clipping': 1.0,
            'mixed_precision': torch.cuda.is_available()
        },
        'data': {
            'dataset': 'samples',
            'data_root': './data/samples',
            'cache_rate': 0.0,
            'augmentation': False,  # Disable for quick test
            'normalize': True
        },
        'hardware': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 2,
            'pin_memory': True,
            'persistent_workers': False
        },
        'logging': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_dir': './logs',
            'save_interval': 1
        },
        'paths': {
            'checkpoints': './checkpoints',
            'logs': './logs',
            'predictions': './predictions',
            'visualizations': './visualizations'
        }
    }
    
    # Save config
    os.makedirs('configs', exist_ok=True)
    with open('configs/quick_test_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✓ Created minimal test configuration")
    return config

def main():
    print("AI Medical Image Segmentation - Training Readiness Check")
    print("=" * 60)
    
    # Run tests
    tests_passed = []
    
    tests_passed.append(test_imports())
    tests_passed.append(test_data_loading())
    tests_passed.append(test_model_creation())
    cuda_available = test_cuda()
    
    # Create test config
    config = create_minimal_config()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    if all(tests_passed):
        print("✓ System is ready for training!")
        print("\nTo start training:")
        print("1. Quick test: python training/train.py --config configs/quick_test_config.yaml")
        print("2. Full training: python training/train.py --config configs/config.yaml")
        
        if cuda_available:
            print(f"\nGPU acceleration available - training will be fast!")
        else:
            print(f"\nCPU-only training - will be slower but still works")
        
        return True
    else:
        print("System not ready. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)