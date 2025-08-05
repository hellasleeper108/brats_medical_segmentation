#!/usr/bin/env python3
"""
Setup script for BraTS 2021 dataset downloaded from TCIA.
Automatically organizes TCIA BraTS data into training format.
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


class TCIABraTSSetup:
    """Setup TCIA BraTS 2021 dataset for training."""
    
    def __init__(self, input_dir, output_dir="data/brats2021"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected BraTS modalities
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        
    def find_brats_files(self):
        """Find all BraTS files in the input directory."""
        print("🔍 Scanning for BraTS files...")
        
        # Look for NIfTI files
        all_files = list(self.input_dir.rglob("*.nii.gz"))
        all_files.extend(list(self.input_dir.rglob("*.nii")))
        
        print(f"Found {len(all_files)} NIfTI files")
        
        # Group by subject
        subjects = {}
        
        for file_path in all_files:
            filename = file_path.name
            
            # Extract subject ID (various naming conventions)
            if "BraTS" in filename:
                # Standard BraTS naming
                parts = filename.split('_')
                if len(parts) >= 2:
                    subject_id = '_'.join(parts[:2])  # e.g., BraTS_001
                else:
                    subject_id = parts[0]
            else:
                # Fallback: use directory name or filename
                subject_id = file_path.parent.name
                if subject_id == self.input_dir.name:
                    subject_id = filename.split('.')[0]
            
            if subject_id not in subjects:
                subjects[subject_id] = {}
            
            # Identify modality and segmentation
            filename_lower = filename.lower()
            if 'seg' in filename_lower:
                subjects[subject_id]['seg'] = file_path
            elif 't1ce' in filename_lower or 't1c' in filename_lower:
                subjects[subject_id]['t1ce'] = file_path
            elif 't1' in filename_lower:
                subjects[subject_id]['t1'] = file_path
            elif 't2' in filename_lower:
                subjects[subject_id]['t2'] = file_path
            elif 'flair' in filename_lower:
                subjects[subject_id]['flair'] = file_path
            else:
                # Try to guess from filename patterns
                if any(mod in filename_lower for mod in self.modalities):
                    for mod in self.modalities:
                        if mod in filename_lower:
                            subjects[subject_id][mod] = file_path
                            break
        
        # Filter complete subjects (have all modalities + segmentation)
        complete_subjects = {}
        for subject_id, files in subjects.items():
            required_files = self.modalities + ['seg']
            if all(mod in files for mod in required_files):
                complete_subjects[subject_id] = files
            else:
                missing = [mod for mod in required_files if mod not in files]
                print(f"⚠️  {subject_id}: Missing {missing}")
        
        print(f"✅ Found {len(complete_subjects)} complete subjects")
        return complete_subjects
    
    def create_splits(self, subjects, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
        """Create train/val/test splits."""
        subject_ids = list(subjects.keys())
        np.random.seed(42)  # Reproducible splits
        np.random.shuffle(subject_ids)
        
        n_total = len(subject_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': subject_ids[:n_train],
            'val': subject_ids[n_train:n_train + n_val],
            'test': subject_ids[n_train + n_val:]
        }
        
        print(f"📊 Dataset splits:")
        for split, ids in splits.items():
            print(f"  {split}: {len(ids)} subjects")
        
        return splits
    
    def copy_files_to_splits(self, subjects, splits):
        """Copy files to train/val/test directories."""
        print("📁 Organizing files into splits...")
        
        for split_name, subject_ids in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            print(f"\n📂 Processing {split_name} split...")
            
            for subject_id in tqdm(subject_ids, desc=f"Copying {split_name}"):
                files = subjects[subject_id]
                
                # Create subject directory
                subject_dir = split_dir / subject_id
                subject_dir.mkdir(exist_ok=True)
                
                # Copy all modalities and segmentation
                for modality, src_path in files.items():
                    # Standardized filename
                    if modality == 'seg':
                        dst_filename = f"{subject_id}_seg.nii.gz"
                    else:
                        dst_filename = f"{subject_id}_{modality}.nii.gz"
                    
                    dst_path = subject_dir / dst_filename
                    
                    # Copy file
                    shutil.copy2(src_path, dst_path)
        
        print("✅ File organization complete!")
    
    def validate_dataset(self):
        """Validate the organized dataset."""
        print("\n🔍 Validating organized dataset...")
        
        validation_results = {}
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            if not split_dir.exists():
                continue
            
            subjects = list(split_dir.iterdir())
            subjects = [s for s in subjects if s.is_dir()]
            
            valid_subjects = 0
            total_size_gb = 0
            
            for subject_dir in subjects:
                # Check for required files
                required_files = [
                    f"{subject_dir.name}_t1.nii.gz",
                    f"{subject_dir.name}_t1ce.nii.gz", 
                    f"{subject_dir.name}_t2.nii.gz",
                    f"{subject_dir.name}_flair.nii.gz",
                    f"{subject_dir.name}_seg.nii.gz"
                ]
                
                if all((subject_dir / f).exists() for f in required_files):
                    valid_subjects += 1
                    
                    # Calculate size
                    for f in required_files:
                        file_path = subject_dir / f
                        total_size_gb += file_path.stat().st_size / (1024**3)
            
            validation_results[split] = {
                'total_subjects': len(subjects),
                'valid_subjects': valid_subjects,
                'size_gb': total_size_gb
            }
            
            print(f"  {split}: {valid_subjects}/{len(subjects)} valid subjects ({total_size_gb:.2f} GB)")
        
        return validation_results
    
    def create_dataset_info(self, validation_results):
        """Create dataset information file."""
        dataset_info = {
            "dataset_name": "BraTS 2021 (TCIA)",
            "description": "Brain Tumor Segmentation Challenge 2021 from TCIA",
            "source": "The Cancer Imaging Archive (TCIA)",
            "url": "https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/",
            "modalities": ["t1", "t1ce", "t2", "flair"],
            "classes": {
                "0": "background",
                "1": "necrotic_tumor_core",
                "2": "peritumoral_edematous_invaded_tissue", 
                "3": "enhancing_tumor"
            },
            "image_size": "Variable (typically 240x240x155)",
            "voxel_spacing": "1.0x1.0x1.0 mm",
            "splits": validation_results,
            "preprocessing": "Skull-stripped, co-registered, resampled to 1mm isotropic",
            "license": "CC BY 4.0",
            "citation": "Baid et al. The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. arXiv:2107.02314",
            "created_by": "TCIA BraTS Setup Script"
        }
        
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"✅ Dataset info saved to {self.output_dir}/dataset_info.json")
    
    def setup(self):
        """Main setup process."""
        print("🧠 BraTS 2021 (TCIA) Dataset Setup")
        print("=" * 50)
        
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            print("Please download BraTS 2021 from TCIA first:")
            print("https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/")
            return False
        
        # Find BraTS files
        subjects = self.find_brats_files()
        if len(subjects) == 0:
            print("❌ No valid BraTS subjects found!")
            return False
        
        # Create splits
        splits = self.create_splits(subjects)
        
        # Copy files
        self.copy_files_to_splits(subjects, splits)
        
        # Validate
        validation_results = self.validate_dataset()
        
        # Create info file
        self.create_dataset_info(validation_results)
        
        print("\n🎉 BraTS 2021 Dataset Setup Complete!")
        print(f"📁 Dataset location: {self.output_dir}")
        print(f"📊 Total subjects: {sum(r['valid_subjects'] for r in validation_results.values())}")
        print(f"💾 Total size: {sum(r['size_gb'] for r in validation_results.values()):.2f} GB")
        
        print("\n🚀 Next steps:")
        print("1. python train_brats.py --dataset brats2021")
        print("2. Expected Dice score: >0.90")
        print("3. Training time: 4-6 hours on RTX 4080 Super")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Setup BraTS 2021 dataset from TCIA')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing downloaded TCIA BraTS 2021 data')
    parser.add_argument('--output_dir', type=str, default='data/brats2021',
                       help='Output directory for organized dataset')
    
    args = parser.parse_args()
    
    setup = TCIABraTSSetup(args.input_dir, args.output_dir)
    success = setup.setup()
    
    if success:
        print("✅ Setup successful!")
    else:
        print("❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()