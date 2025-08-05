#!/usr/bin/env python3
"""
Test script to verify BraTS 2021 dataset loading works correctly.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_brats_dataset_structure():
    """Test if the BraTS dataset structure is accessible."""
    
    # Check paths from config
    train_dir = Path("../PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet")
    val_dir = Path("../PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_ValidationSet")
    
    print("Testing BraTS 2021 Dataset Access")
    print("=" * 50)
    
    # Test training directory
    if train_dir.exists():
        print(f"[OK] Training directory found: {train_dir}")
        
        # Count training subjects
        train_subjects = []
        for subdir in train_dir.iterdir():
            if subdir.is_dir():
                # Check for different cancer center collections
                if subdir.name in ['ACRIN-FMISO-Brain', 'CPTAC-GBM', 'IvyGAP', 'TCGA-GBM', 'TCGA-LGG', 'UCSF-PDGM', 'UPENN-GBM', 'new-not-previously-in-TCIA']:
                    # Count subjects in this collection
                    for subject_dir in subdir.iterdir():
                        if subject_dir.is_dir() and subject_dir.name.startswith("BraTS2021_"):
                            train_subjects.append(subject_dir)
        
        print(f"Found {len(train_subjects)} training subjects")
        
        # Test first few subjects for complete data
        print("\nChecking first 5 subjects for complete modalities:")
        modalities = ['t1', 't1ce', 't2', 'flair', 'seg']
        
        for i, subject_dir in enumerate(train_subjects[:5]):
            print(f"\n  Subject: {subject_dir.name}")
            complete = True
            for modality in modalities:
                file_path = subject_dir / f"{subject_dir.name}_{modality}.nii.gz"
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024*1024)  # MB
                    print(f"    [OK] {modality}: {file_size:.1f}MB")
                else:
                    print(f"    [MISSING] {modality}")
                    complete = False
            
            if complete:
                print(f"    [COMPLETE] Subject {subject_dir.name}")
            else:
                print(f"    [INCOMPLETE] Subject {subject_dir.name}")
                
    else:
        print(f"[ERROR] Training directory not found: {train_dir}")
        return False
    
    # Test validation directory
    if val_dir.exists():
        print(f"\n[OK] Validation directory found: {val_dir}")
        
        # Count validation subjects
        val_subjects = []
        for subdir in val_dir.iterdir():
            if subdir.is_dir():
                for subject_dir in subdir.iterdir():
                    if subject_dir.is_dir() and subject_dir.name.startswith("BraTS2021_"):
                        val_subjects.append(subject_dir)
        
        print(f"Found {len(val_subjects)} validation subjects")
        
    else:
        print(f"[ERROR] Validation directory not found: {val_dir}")
    
    print(f"\nDataset verification complete!")
    print(f"Ready to train with {len(train_subjects)} training subjects")
    
    return True


def test_pytorch_dataset_loading():
    """Test PyTorch dataset loading."""
    try:
        from train_brats2021 import BraTS2021Dataset
        
        print("\nTesting PyTorch Dataset Loading")
        print("=" * 40)
        
        train_dir = "../PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet"
        
        # Create dataset
        dataset = BraTS2021Dataset(train_dir, target_size=(128, 128, 64), augment=False)
        
        print(f"[OK] Dataset created successfully")
        print(f"Dataset size: {len(dataset)} subjects")
        
        if len(dataset) > 0:
            # Test loading first sample
            print(f"\nTesting first sample loading...")
            sample = dataset[0]
            
            if isinstance(sample, tuple) and len(sample) == 2:
                image, label = sample
                print(f"[OK] Sample loaded successfully")
                print(f"   Image shape: {image.shape}")
                print(f"   Label shape: {label.shape}")
                print(f"   Image dtype: {image.dtype}")
                print(f"   Label dtype: {label.dtype}")
                
                # Check for reasonable value ranges
                print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
                print(f"   Unique labels: {sorted(list(set(label.flatten().tolist())))}")
                
            else:
                print(f"[WARNING] Unexpected sample format: {type(sample)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error testing dataset loading: {e}")
        return False


if __name__ == "__main__":
    print("BraTS 2021 Dataset Verification")
    print("=" * 60)
    
    # Test 1: Dataset structure
    structure_ok = test_brats_dataset_structure()
    
    # Test 2: PyTorch loading (if structure is OK)
    if structure_ok:
        loading_ok = test_pytorch_dataset_loading()
        
        if loading_ok:
            print(f"\n[SUCCESS] All tests passed! Ready to start training.")
        else:
            print(f"\n[WARNING] Dataset structure OK, but loading failed. Check dependencies.")
    else:
        print(f"\n[ERROR] Dataset structure test failed. Check paths.")