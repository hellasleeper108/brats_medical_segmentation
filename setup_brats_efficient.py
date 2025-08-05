#!/usr/bin/env python3
"""
Efficient setup script for large BraTS dataset.
Optimized for 1,479 subjects with 407,245 files.
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import time

class EfficientBraTSSetup:
    """Efficient setup for large BraTS dataset."""
    
    def __init__(self, input_dir, output_dir="data/brats_large"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected BraTS modalities
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        
    def find_subjects_efficiently(self):
        """Find subjects by scanning directory structure instead of all files."""
        print("Efficiently scanning BraTS dataset structure...")
        
        subjects = {}
        
        # Look for subject directories first
        subject_dirs = []
        
        # Scan top-level directories
        for item in self.input_dir.iterdir():
            if item.is_dir():
                # Check if this looks like a subject directory
                if any(keyword in item.name.lower() for keyword in ['brats', 'subject', 'patient']):
                    subject_dirs.append(item)
                else:
                    # Look one level deeper
                    for subitem in item.iterdir():
                        if subitem.is_dir() and any(keyword in subitem.name.lower() for keyword in ['brats', 'subject', 'patient']):
                            subject_dirs.append(subitem)
        
        print(f"Found {len(subject_dirs)} potential subject directories")
        
        # Process each subject directory
        for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
            subject_files = self._analyze_subject_directory(subject_dir)
            if subject_files:
                subject_id = subject_dir.name
                subjects[subject_id] = subject_files
        
        # Filter complete subjects
        complete_subjects = {}
        required_files = self.modalities + ['seg']
        
        for subject_id, files in subjects.items():
            missing = [mod for mod in required_files if mod not in files]
            if len(missing) <= 1:  # Allow up to 1 missing modality
                complete_subjects[subject_id] = files
        
        print(f"Complete subjects: {len(complete_subjects)}")
        return complete_subjects
    
    def _analyze_subject_directory(self, subject_dir):
        """Analyze a single subject directory for required files."""
        files = {}
        
        # Look for NIfTI files in this directory and subdirectories
        nifti_files = list(subject_dir.rglob("*.nii.gz")) + list(subject_dir.rglob("*.nii"))
        
        for file_path in nifti_files:
            filename = file_path.name.lower()
            
            # Identify modality
            if 'seg' in filename or 'label' in filename:
                files['seg'] = file_path
            elif 't1ce' in filename or 't1c' in filename:
                files['t1ce'] = file_path
            elif 't1' in filename and 't1ce' not in filename and 't1c' not in filename:
                files['t1'] = file_path
            elif 't2' in filename:
                files['t2'] = file_path
            elif 'flair' in filename:
                files['flair'] = file_path
        
        return files if len(files) >= 4 else None  # Need at least 4 files
    
    def create_splits(self, subjects, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create train/validation/test splits."""
        subject_ids = list(subjects.keys())
        np.random.seed(42)
        np.random.shuffle(subject_ids)
        
        n_total = len(subject_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': subject_ids[:n_train],
            'val': subject_ids[n_train:n_train + n_val],
            'test': subject_ids[n_train + n_val:]
        }
        
        print(f"\nDataset splits:")
        for split, ids in splits.items():
            print(f"  {split}: {len(ids)} subjects ({len(ids)/n_total*100:.1f}%)")
        
        return splits
    
    def organize_dataset(self, subjects, splits):
        """Organize dataset into train/val/test directories."""
        print("\nOrganizing dataset...")
        
        for split_name, subject_ids in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing {split_name} split ({len(subject_ids)} subjects)...")
            
            for subject_id in tqdm(subject_ids, desc=f"Copying {split_name}"):
                self._copy_subject_files(subject_id, subjects[subject_id], split_dir)
    
    def _copy_subject_files(self, subject_id, files, split_dir):
        """Copy files for a single subject."""
        subject_dir = split_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        for modality, src_path in files.items():
            if modality == 'seg':
                dst_filename = f"{subject_id}_seg.nii.gz"
            else:
                dst_filename = f"{subject_id}_{modality}.nii.gz"
            
            dst_path = subject_dir / dst_filename
            
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
    
    def validate_dataset(self):
        """Validate the organized dataset."""
        print("\nValidating organized dataset...")
        
        validation_results = {}
        total_subjects = 0
        total_size_gb = 0
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
            
            subjects = [d for d in split_dir.iterdir() if d.is_dir()]
            valid_subjects = 0
            split_size_gb = 0
            
            for subject_dir in subjects:
                required_files = [
                    f"{subject_dir.name}_t1.nii.gz",
                    f"{subject_dir.name}_t1ce.nii.gz",
                    f"{subject_dir.name}_t2.nii.gz", 
                    f"{subject_dir.name}_flair.nii.gz",
                    f"{subject_dir.name}_seg.nii.gz"
                ]
                
                existing_files = [f for f in required_files if (subject_dir / f).exists()]
                if len(existing_files) >= 4:  # At least 4 files
                    valid_subjects += 1
                    
                    # Calculate size
                    for f in existing_files:
                        file_path = subject_dir / f
                        split_size_gb += file_path.stat().st_size / (1024**3)
            
            validation_results[split] = {
                'total_subjects': len(subjects),
                'valid_subjects': valid_subjects,
                'size_gb': split_size_gb
            }
            
            total_subjects += valid_subjects
            total_size_gb += split_size_gb
            
            print(f"  {split}: {valid_subjects}/{len(subjects)} valid subjects ({split_size_gb:.1f} GB)")
        
        print(f"\nTotal: {total_subjects} subjects ({total_size_gb:.1f} GB)")
        return validation_results
    
    def create_dataset_info(self, validation_results):
        """Create dataset information file."""
        total_subjects = sum(r['valid_subjects'] for r in validation_results.values())
        total_size = sum(r['size_gb'] for r in validation_results.values())
        
        dataset_info = {
            "dataset_name": "BraTS 2021 (RSNA-ASNR-MICCAI)",
            "description": "Brain Tumor Segmentation Challenge 2021",
            "source": "RSNA-ASNR-MICCAI BraTS 2021",
            "num_subjects": total_subjects,
            "total_size_gb": round(total_size, 1),
            "modalities": self.modalities,
            "classes": {
                "0": "background",
                "1": "necrotic_tumor_core",
                "2": "peritumoral_edematous_invaded_tissue",
                "3": "enhancing_tumor"
            },
            "splits": validation_results,
            "preprocessing": "Skull-stripped, co-registered, resampled",
            "created_by": "Efficient BraTS Setup Script"
        }
        
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset info saved to {self.output_dir}/dataset_info.json")
    
    def setup(self):
        """Main setup process."""
        print("Efficient BraTS Dataset Setup")
        print("=" * 50)
        
        start_time = time.time()
        
        if not self.input_dir.exists():
            print(f"ERROR: Input directory not found: {self.input_dir}")
            return False
        
        # Find subjects efficiently
        subjects = self.find_subjects_efficiently()
        
        if len(subjects) == 0:
            print("ERROR: No valid subjects found!")
            return False
        
        # Create splits
        splits = self.create_splits(subjects)
        
        # Organize dataset
        self.organize_dataset(subjects, splits)
        
        # Validate
        validation_results = self.validate_dataset()
        
        # Create info file
        self.create_dataset_info(validation_results)
        
        processing_time = time.time() - start_time
        
        print(f"\nBraTS Dataset Setup Complete!")
        print("=" * 50)
        print(f"Dataset location: {self.output_dir}")
        print(f"Processing time: {processing_time/60:.1f} minutes")
        print(f"Total subjects: {sum(r['valid_subjects'] for r in validation_results.values())}")
        
        print(f"\nNext steps:")
        print(f"1. cd X:\\Projects\\brats_medical_segmentation")
        print(f"2. python train_large_brats.py")
        print(f"3. Expected Dice score: >0.90")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Efficient BraTS dataset setup')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing BraTS dataset')
    parser.add_argument('--output_dir', type=str, default='data/brats_large',
                       help='Output directory for organized dataset')
    
    args = parser.parse_args()
    
    setup = EfficientBraTSSetup(args.input_dir, args.output_dir)
    success = setup.setup()
    
    if success:
        print("SUCCESS: BraTS dataset setup complete!")
    else:
        print("FAILED: Dataset setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()