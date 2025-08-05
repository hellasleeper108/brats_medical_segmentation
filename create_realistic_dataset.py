#!/usr/bin/env python3
"""
Create realistic synthetic medical dataset for training.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json


def generate_realistic_brain(subject_id, shape=(160, 192, 144)):
    """Generate realistic synthetic brain with tumors."""
    
    # Create coordinate grids
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing='ij'
    )
    
    # Brain mask (ellipsoid)
    brain_mask = ((x/0.7)**2 + (y/0.8)**2 + (z/0.6)**2) < 1
    
    # Initialize image and segmentation
    np.random.seed(subject_id)  # Reproducible per subject
    image = np.zeros(shape)
    segmentation = np.zeros(shape, dtype=np.uint8)
    
    # Background noise
    image += np.random.normal(0, 0.05, shape)
    
    # Brain tissue
    brain_intensity = 0.7 + np.random.normal(0, 0.1, shape)
    image[brain_mask] = brain_intensity[brain_mask]
    segmentation[brain_mask] = 1
    
    # Add anatomical structures
    # Ventricles (darker regions)
    ventricle_mask = ((x/0.15)**2 + (y/0.3)**2 + (z/0.4)**2) < 1
    ventricle_mask = ventricle_mask & brain_mask
    image[ventricle_mask] *= 0.3
    
    # White/Gray matter variation
    wm_variation = 0.2 * np.sin(3*x) * np.cos(3*y) * np.sin(2*z)
    image[brain_mask] += wm_variation[brain_mask] * 0.1
    
    # Add tumors (random number and locations)
    num_tumors = np.random.randint(1, 4)
    
    for tumor_id in range(num_tumors):
        # Random tumor center within brain
        center = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.6, 0.6),
            np.random.uniform(-0.4, 0.4)
        ])
        
        # Tumor size
        radius = np.random.uniform(0.05, 0.15)
        
        # Tumor mask
        tumor_dist = np.sqrt(
            ((x - center[0])/radius)**2 +
            ((y - center[1])/radius)**2 +
            ((z - center[2])/radius)**2
        )
        
        # Core tumor
        tumor_core = (tumor_dist < 1) & brain_mask
        # Edema (larger region)
        tumor_edema = (tumor_dist < 1.5) & brain_mask & (~tumor_core)
        
        # Update image intensities
        if tumor_core.sum() > 0:
            image[tumor_core] = 1.2 + np.random.normal(0, 0.1, tumor_core.sum())
        if tumor_edema.sum() > 0:
            image[tumor_edema] = 0.9 + np.random.normal(0, 0.05, tumor_edema.sum())
        
        # Update segmentation
        segmentation[tumor_core] = 2  # Tumor core
        segmentation[tumor_edema] = 3  # Tumor edema
    
    # Add noise and artifacts
    image += np.random.normal(0, 0.02, shape)
    
    # Ensure realistic intensity range
    image = np.clip(image, 0, 1.5)
    
    return image, segmentation


def create_enhanced_dataset(num_subjects=50):
    """Create enhanced synthetic dataset."""
    print("Creating Enhanced Synthetic Medical Dataset")
    print("=" * 50)
    
    base_dir = Path("data/enhanced_synthetic")
    base_dir.mkdir(exist_ok=True)
    
    # Create splits
    splits = {
        'train': int(num_subjects * 0.7),
        'val': int(num_subjects * 0.2),
        'test': int(num_subjects * 0.1)
    }
    
    # Ensure at least 1 sample per split
    if splits['test'] == 0:
        splits['test'] = 1
        splits['train'] -= 1
    if splits['val'] == 0:
        splits['val'] = 1
        splits['train'] -= 1
    
    print(f"Creating {num_subjects} subjects:")
    print(f"  Training: {splits['train']}")
    print(f"  Validation: {splits['val']}")
    print(f"  Test: {splits['test']}")
    print()
    
    subject_id = 1
    
    for split_name, count in splits.items():
        split_dir = base_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"Generating {split_name} data...")
        
        for i in tqdm(range(count), desc=f"Creating {split_name}"):
            # Generate brain
            image, segmentation = generate_realistic_brain(subject_id)
            
            # Subject name
            subject_name = f"subject_{subject_id:03d}"
            
            # Create affine matrix (1mm isotropic)
            affine = np.eye(4)
            affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0
            
            # Save image
            img_path = split_dir / f"{subject_name}.nii.gz"
            img_nifti = nib.Nifti1Image(image.astype(np.float32), affine)
            nib.save(img_nifti, img_path)
            
            # Save segmentation
            seg_path = split_dir / f"{subject_name}_seg.nii.gz"
            seg_nifti = nib.Nifti1Image(segmentation.astype(np.uint8), affine)
            nib.save(seg_nifti, seg_path)
            
            subject_id += 1
    
    # Create metadata
    metadata = {
        "dataset_name": "Enhanced Synthetic Brain Dataset",
        "description": "Realistic synthetic brain MRI with tumor segmentation",
        "num_subjects": num_subjects,
        "modality": "Synthetic MRI",
        "image_size": [160, 192, 144],
        "voxel_spacing": [1.0, 1.0, 1.0],
        "classes": {
            0: "background",
            1: "brain_tissue", 
            2: "tumor_core",
            3: "tumor_edema"
        },
        "splits": splits,
        "intensity_range": [0.0, 1.5],
        "created_by": "aimedis_synthetic_generator"
    }
    
    with open(base_dir / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("Dataset creation complete!")
    print(f"Location: {base_dir}")
    print(f"Total subjects: {num_subjects}")
    print("Classes: Background, Brain Tissue, Tumor Core, Tumor Edema")
    
    return base_dir


if __name__ == "__main__":
    print("Enhanced Synthetic Medical Dataset Creator")
    print("=" * 50)
    
    try:
        num_subjects = input("Number of subjects to create (default 50): ").strip()
        num_subjects = int(num_subjects) if num_subjects else 50
    except:
        num_subjects = 50
    
    dataset_path = create_enhanced_dataset(num_subjects)
    
    print()
    print("NEXT STEPS:")
    print("1. Train with enhanced dataset:")
    print("   python train_enhanced.py")
    print("2. Or modify train_samples.py to use enhanced_synthetic data")
    print()
    print("The enhanced dataset has:")
    print("- More realistic brain anatomy")
    print("- Variable tumor sizes and locations") 
    print("- 4 segmentation classes")
    print("- Larger image dimensions (160x192x144)")
    print("- Better intensity distributions")