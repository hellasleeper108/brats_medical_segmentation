#!/usr/bin/env python3
"""
Simple training test without unicode characters.
"""

import os
import sys
import torch
import yaml

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("AI Medical Image Segmentation - Training Readiness Check")
    print("=" * 60)
    
    # Test imports
    try:
        import torch
        import monai
        import nibabel as nib
        from models import UNet3D
        print("PASS: All imports successful")
        imports_ok = True
    except Exception as e:
        print(f"FAIL: Import error: {e}")
        imports_ok = False
    
    # Test CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"PASS: CUDA available - {device_name} ({memory_gb:.1f} GB)")
        cuda_ok = True
    else:
        print("INFO: CUDA not available - will use CPU")
        cuda_ok = False
    
    # Check sample data
    train_dir = "data/samples/train"
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.nii.gz')]
        print(f"PASS: Found {len(train_files)//2} training samples")
        data_ok = True
    else:
        print("FAIL: No training data found")
        data_ok = False
    
    # Test model creation
    try:
        model = UNet3D(in_channels=1, num_classes=3)
        test_input = torch.randn(1, 1, 128, 128, 64)  # Larger test size
        with torch.no_grad():
            output = model(test_input)
        print(f"PASS: Model created - Output shape: {output.shape}")
        model_ok = True
    except Exception as e:
        print(f"FAIL: Model error: {e}")
        model_ok = False
    
    # Create test config
    try:
        config = {
            'model': {
                'name': 'unet',
                'input_size': [64, 64, 32],
                'num_classes': 3,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 1,
                'learning_rate': 0.001,
                'epochs': 2,
                'optimizer': 'adamw',
                'mixed_precision': cuda_ok
            },
            'data': {
                'dataset': 'samples',
                'data_root': './data/samples',
                'cache_rate': 0.0
            },
            'hardware': {
                'device': 'cuda' if cuda_ok else 'cpu',
                'num_workers': 0,  # Avoid multiprocessing issues
                'pin_memory': False
            },
            'paths': {
                'checkpoints': './checkpoints',
                'logs': './logs'
            }
        }
        
        os.makedirs('configs', exist_ok=True)
        with open('configs/test_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("PASS: Test configuration created")
        config_ok = True
    except Exception as e:
        print(f"FAIL: Config error: {e}")
        config_ok = False
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    all_tests = [imports_ok, data_ok, model_ok, config_ok]
    
    if all(all_tests):
        print("SYSTEM READY FOR TRAINING!")
        print("\nNext steps:")
        print("1. Quick test: python training/train.py --config configs/test_config.yaml")
        print("2. Full training: python training/train.py --config configs/config.yaml")
        
        if cuda_ok:
            print("\nGPU acceleration is available!")
        else:
            print("\nUsing CPU (slower but functional)")
        
        return True
    else:
        print("SYSTEM NOT READY - Please fix the issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)