#!/usr/bin/env python3
"""
Quick diagnostic script to check BraTS label issues
"""

import numpy as np
import nibabel as nib
from pathlib import Path

def check_label_ranges():
    """Check what label values exist in BraTS dataset"""
    
    print("CHECKING BRATS LABEL RANGES")
    print("=" * 40)
    
    # Check a few training subjects
    train_dir = Path("../PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_TrainingSet")
    
    label_values = set()
    subjects_checked = 0
    
    for center_dir in train_dir.iterdir():
        if center_dir.is_dir() and subjects_checked < 10:  # Check first 10 subjects
            for subject_dir in center_dir.iterdir():
                if subject_dir.is_dir() and subject_dir.name.startswith("BraTS2021_"):
                    seg_file = subject_dir / f"{subject_dir.name}_seg.nii.gz"
                    if seg_file.exists():
                        try:
                            # Load segmentation
                            seg_img = nib.load(str(seg_file))
                            seg_data = seg_img.get_fdata()
                            
                            # Get unique values
                            unique_vals = np.unique(seg_data)
                            label_values.update(unique_vals)
                            
                            print(f"Subject {subject_dir.name}: {sorted(unique_vals)}")
                            subjects_checked += 1
                            
                            if subjects_checked >= 10:
                                break
                                
                        except Exception as e:
                            print(f"Error loading {subject_dir.name}: {e}")
            
            if subjects_checked >= 10:
                break
    
    print(f"\nALL UNIQUE LABEL VALUES FOUND: {sorted(label_values)}")
    
    # Check if values are in expected range (0-3)
    max_val = max(label_values) if label_values else 0
    min_val = min(label_values) if label_values else 0
    
    print(f"Label range: {min_val} to {max_val}")
    
    if max_val > 3:
        print("🚨 PROBLEM: Labels exceed 3 (max class index for 4 classes)")
        print("   BraTS original labels: 0, 1, 2, 4")
        print("   Need to remap: 4 -> 3")
    
    if min_val < 0:
        print("🚨 PROBLEM: Negative label values found")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    check_label_ranges()