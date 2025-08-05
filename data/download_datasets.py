#!/usr/bin/env python3
"""
Download and prepare medical imaging datasets for training.

This script provides functionality to download and prepare:
- BraTS (Brain Tumor Segmentation) dataset
- COVID-CT dataset
- Custom dataset organization

Note: Some datasets require registration and manual download due to licensing.
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
import json


def create_directory_structure(base_dir: str):
    """Create standard directory structure for medical datasets."""
    dirs_to_create = [
        'raw',
        'preprocessed',
        'train',
        'val', 
        'test',
        'metadata'
    ]
    
    for dir_name in dirs_to_create:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")


def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download file with progress indication."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
    
    print(f"\nDownloaded: {destination}")


def setup_brats_dataset(data_dir: str):
    """
    Setup BraTS dataset structure and provide download instructions.
    
    Note: BraTS dataset requires registration at:
    https://www.med.upenn.edu/cbica/brats2021/
    """
    brats_dir = os.path.join(data_dir, 'brats')
    create_directory_structure(brats_dir)
    
    # Create README with instructions
    readme_content = """# BraTS Dataset Setup

The BraTS (Brain Tumor Segmentation) dataset requires registration and manual download.

## Steps to download:

1. Visit: https://www.med.upenn.edu/cbica/brats2021/
2. Register for an account
3. Download the training data
4. Extract the dataset to this directory

## Expected structure after extraction:

```
brats/
├── raw/
│   ├── BraTS2021_Training_Data/
│   │   ├── BraTS2021_00000/
│   │   │   ├── BraTS2021_00000_flair.nii.gz
│   │   │   ├── BraTS2021_00000_t1.nii.gz
│   │   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   │   ├── BraTS2021_00000_t2.nii.gz
│   │   │   └── BraTS2021_00000_seg.nii.gz
│   │   └── ...
├── train/
├── val/
├── test/
└── metadata/
```

## After manual download:

Run the preprocessing script:
```bash
python preprocess_brats.py --input_dir brats/raw --output_dir brats/
```
"""
    
    with open(os.path.join(brats_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    # Create metadata file
    metadata = {
        'dataset_name': 'BraTS2021',
        'description': 'Brain Tumor Segmentation Challenge 2021',
        'modalities': ['flair', 't1', 't1ce', 't2'],
        'classes': {
            0: 'background',
            1: 'necrotic_tumor_core',
            2: 'peritumoral_edema', 
            3: 'enhancing_tumor'
        },
        'image_size': [240, 240, 155],
        'spacing': [1.0, 1.0, 1.0],
        'url': 'https://www.med.upenn.edu/cbica/brats2021/',
        'license': 'Custom - Registration Required'
    }
    
    with open(os.path.join(brats_dir, 'metadata', 'dataset_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"BraTS dataset structure created in {brats_dir}")
    print("Please follow the README.md instructions to download the actual data.")


def setup_covid_ct_dataset(data_dir: str):
    """
    Setup COVID-CT dataset structure and provide download instructions.
    
    Note: This is a placeholder for COVID-CT dataset setup.
    Actual datasets may require specific permissions.
    """
    covid_dir = os.path.join(data_dir, 'covid_ct')
    create_directory_structure(covid_dir)
    
    readme_content = """# COVID-CT Dataset Setup

This directory is set up for COVID-CT lung segmentation datasets.

## Supported datasets:

1. **COVID-19 CT Lung and Infection Segmentation Dataset**
   - Available at: https://zenodo.org/record/3757476
   - Contains CT scans with lung and infection segmentations

2. **MedSeg COVID-19 CT Dataset**
   - Available at: http://medicalsegmentation.com/covid19/
   - Contains CT scans with various annotations

## Expected structure:

```
covid_ct/
├── raw/
│   ├── images/
│   │   ├── patient_001.nii.gz
│   │   └── ...
│   ├── lung_masks/
│   │   ├── patient_001.nii.gz
│   │   └── ...
│   └── infection_masks/
│       ├── patient_001.nii.gz
│       └── ...
├── train/
├── val/
├── test/
└── metadata/
```

## Usage:

1. Download datasets from the above sources
2. Organize them according to the expected structure
3. Run preprocessing: `python preprocess_covid_ct.py`
"""
    
    with open(os.path.join(covid_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    metadata = {
        'dataset_name': 'COVID-CT',
        'description': 'COVID-19 CT lung and infection segmentation',
        'modalities': ['ct'],
        'classes': {
            0: 'background',
            1: 'lung',
            2: 'infection'
        },
        'image_size': [512, 512, 'variable'],
        'spacing': 'variable',
        'sources': [
            'https://zenodo.org/record/3757476',
            'http://medicalsegmentation.com/covid19/'
        ]
    }
    
    with open(os.path.join(covid_dir, 'metadata', 'dataset_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"COVID-CT dataset structure created in {covid_dir}")


def download_sample_data(data_dir: str):
    """
    Download sample medical images for testing.
    
    This downloads publicly available sample medical images that can be used
    for testing the pipeline without requiring dataset registration.
    """
    sample_dir = os.path.join(data_dir, 'samples')
    create_directory_structure(sample_dir)
    
    print("Downloading sample medical images...")
    
    # Sample URLs (these are placeholders - in practice, you'd use real public datasets)
    samples = [
        {
            'name': 'sample_brain_mri.nii.gz',
            'description': 'Sample brain MRI scan',
            'url': 'https://github.com/InsightSoftwareConsortium/ITKExamples/raw/master/src/Core/Common/ReadUnknownImageType/Input/BrainProtonDensitySlice.png'
        }
    ]
    
    # Create synthetic sample data instead of downloading
    create_synthetic_samples(sample_dir)
    
    print(f"Sample data created in {sample_dir}")


def create_synthetic_samples(sample_dir: str):
    """Create synthetic medical image samples for testing."""
    import numpy as np
    import nibabel as nib
    
    print("Creating synthetic medical image samples...")
    
    # Create synthetic brain-like image
    shape = (128, 128, 64)
    
    # Generate synthetic image with brain-like structure
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]), 
        np.linspace(-1, 1, shape[2])
    )
    
    # Create brain-like ellipsoid
    brain_mask = (x**2/0.8 + y**2/0.8 + z**2/0.4) < 1
    
    # Add some noise and structures
    np.random.seed(42)
    image = np.random.normal(0, 0.1, shape)
    image[brain_mask] += 1.0
    
    # Add some "lesions"
    for i in range(3):
        center = np.random.randint(20, shape[0]-20, 3)
        radius = np.random.randint(3, 8)
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    if (dx**2 + dy**2 + dz**2) < radius**2:
                        try:
                            image[center[0]+dx, center[1]+dy, center[2]+dz] += 0.5
                        except IndexError:
                            pass
    
    # Create corresponding segmentation
    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[brain_mask] = 1
    
    # Add lesion segmentations
    lesion_mask = image > 1.3
    segmentation[lesion_mask] = 2
    
    # Save as NIfTI
    affine = np.eye(4)
    
    # Save image
    img_nifti = nib.Nifti1Image(image.astype(np.float32), affine)
    nib.save(img_nifti, os.path.join(sample_dir, 'raw', 'sample_brain_001.nii.gz'))
    
    # Save segmentation
    seg_nifti = nib.Nifti1Image(segmentation, affine)
    nib.save(seg_nifti, os.path.join(sample_dir, 'raw', 'sample_brain_001_seg.nii.gz'))
    
    # Create a few more samples
    for i in range(2, 5):
        # Vary the synthetic data slightly
        noise_image = image + np.random.normal(0, 0.05, shape)
        noise_seg = segmentation.copy()
        
        img_nifti = nib.Nifti1Image(noise_image.astype(np.float32), affine)
        nib.save(img_nifti, os.path.join(sample_dir, 'raw', f'sample_brain_{i:03d}.nii.gz'))
        
        seg_nifti = nib.Nifti1Image(noise_seg, affine)
        nib.save(seg_nifti, os.path.join(sample_dir, 'raw', f'sample_brain_{i:03d}_seg.nii.gz'))
    
    # Create metadata
    metadata = {
        'dataset_name': 'Synthetic Samples',
        'description': 'Synthetic medical images for testing',
        'num_samples': 4,
        'classes': {
            0: 'background',
            1: 'brain_tissue',
            2: 'lesion'
        },
        'image_size': list(shape),
        'spacing': [1.0, 1.0, 1.0]
    }
    
    with open(os.path.join(sample_dir, 'metadata', 'dataset_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created {metadata['num_samples']} synthetic brain samples")


def organize_custom_dataset(input_dir: str, output_dir: str):
    """
    Organize custom dataset into standard structure.
    
    Args:
        input_dir: Directory containing custom medical images
        output_dir: Output directory for organized dataset
    """
    create_directory_structure(output_dir)
    
    print(f"Organizing custom dataset from {input_dir} to {output_dir}")
    
    # This is a template - actual implementation would depend on
    # the specific organization of the custom dataset
    
    readme_content = """# Custom Dataset Organization

This script helps organize custom medical imaging datasets into a standard structure.

## Usage:

1. Place your medical images in the input directory
2. Run: `python download_datasets.py --organize --input_dir <input> --output_dir <output>`
3. The script will organize files into train/val/test splits

## Expected input formats:

- NIfTI files (.nii, .nii.gz)
- DICOM series
- Other medical imaging formats supported by SimpleITK

## Output structure:

```
custom_dataset/
├── raw/           # Original files
├── train/         # Training set (80%)
├── val/          # Validation set (15%)
├── test/         # Test set (5%)
└── metadata/     # Dataset information
```
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print("Custom dataset organization template created")
    print("Please implement specific organization logic based on your dataset structure")


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare medical imaging datasets'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data',
        help='Base directory for datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['brats', 'covid_ct', 'samples', 'all'],
        default='all',
        help='Dataset to setup'
    )
    parser.add_argument(
        '--organize',
        action='store_true',
        help='Organize custom dataset'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Input directory for custom dataset organization'
    )
    
    args = parser.parse_args()
    
    # Create base data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.organize:
        if not args.input_dir:
            print("Error: --input_dir required when using --organize")
            sys.exit(1)
        organize_custom_dataset(args.input_dir, args.data_dir)
        return
    
    print(f"Setting up datasets in {args.data_dir}")
    
    if args.dataset in ['brats', 'all']:
        setup_brats_dataset(args.data_dir)
    
    if args.dataset in ['covid_ct', 'all']:
        setup_covid_ct_dataset(args.data_dir)
    
    if args.dataset in ['samples', 'all']:
        download_sample_data(args.data_dir)
    
    print("\nDataset setup complete!")
    print("\nNext steps:")
    print("1. Follow dataset-specific README files for manual downloads")
    print("2. Run preprocessing scripts to prepare data for training")
    print("3. Start training with: python training/train.py")


if __name__ == '__main__':
    main()