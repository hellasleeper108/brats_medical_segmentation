#!/usr/bin/env python3
"""
Prepare sample data for training.
"""

import os
import shutil
import json
from pathlib import Path

def prepare_sample_data():
    """Organize sample data into train/val/test splits."""
    
    # Paths
    base_dir = Path("data/samples")
    raw_dir = base_dir / "raw"
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    
    # Create directories
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_files = list(raw_dir.glob("*_[0-9][0-9][0-9].nii.gz"))
    image_files = [f for f in image_files if not f.name.endswith("_seg.nii.gz")]
    
    print(f"Found {len(image_files)} sample images")
    
    # Split data (use first 3 for training, last 1 for validation)
    train_files = image_files[:3]
    val_files = image_files[3:]
    
    # Copy training files
    for img_file in train_files:
        seg_file = img_file.parent / img_file.name.replace(".nii.gz", "_seg.nii.gz")
        
        # Copy to train directory
        shutil.copy2(img_file, train_dir / img_file.name)
        shutil.copy2(seg_file, train_dir / seg_file.name)
        print(f"Copied {img_file.name} to training set")
    
    # Copy validation files
    for img_file in val_files:
        seg_file = img_file.parent / img_file.name.replace(".nii.gz", "_seg.nii.gz")
        
        # Copy to val directory
        shutil.copy2(img_file, val_dir / img_file.name)
        shutil.copy2(seg_file, val_dir / seg_file.name)
        print(f"Copied {img_file.name} to validation set")
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    return len(train_files), len(val_files)

if __name__ == "__main__":
    prepare_sample_data()