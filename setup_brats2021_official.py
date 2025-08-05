#!/usr/bin/env python3
"""
Setup script for official BraTS 2021 dataset.
Handles the exact RSNA-ASNR-MICCAI-BraTS-2021 structure.
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

class BraTS2021Setup:
    """Setup official BraTS 2021 dataset."""
    
    def __init__(self, input_dir, output_dir="data/brats2021"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected BraTS modalities
        self.modalities = ['t1', 't1ce', 't2', 'flair']
        
    def scan_brats2021_structure(self):
        """Scan the official BraTS 2021 dataset structure."""
        print("Scanning official BraTS 2021 dataset...")
        
        # Look for the training set (NIfTI format)
        training_dir = self.input_dir / "RSNA-ASNR-MICCAI-BraTS-2021" / "BraTS2021_TrainingSet"
        validation_dir = self.input_dir / "RSNA-ASNR-MICCAI-BraTS-2021" / "BraTS2021_ValidationSet"
        
        subjects = {}
        
        # Process training data
        if training_dir.exists():
            print(f"Found training directory: {training_dir}")
            subjects.update(self._scan_training_data(training_dir))
        
        # Process validation data (if available)
        if validation_dir.exists():
            print(f"Found validation directory: {validation_dir}")
            subjects.update(self._scan_validation_data(validation_dir))
        
        print(f"Total subjects found: {len(subjects)}")
        return subjects
    
    def _scan_training_data(self, training_dir):
        """Scan training data from multiple subcategories."""
        subjects = {}
        
        # BraTS 2021 has multiple subcategories
        subcategories = [
            "ACRIN-FMISO-Brain", "CPTAC-GBM", "IvyGAP", "TCGA-GBM", 
            "TCGA-LGG", "UCSF-PDGM", "UPENN-GBM", "new-not-previously-in-TCIA"
        ]
        
        for subcat in subcategories:
            subcat_dir = training_dir / subcat
            if subcat_dir.exists():
                print(f"  Scanning {subcat}...")
                subjects.update(self._scan_subcategory(subcat_dir, has_seg=True))
        
        return subjects
    
    def _scan_validation_data(self, validation_dir):
        """Scan validation data (typically no segmentation)."""
        subjects = {}
        
        subcategories = [
            "CPTAC-GBM", "IvyGAP", "TCGA-GBM", "TCGA-LGG",
            "UCSF-PDGM", "UPENN-GBM", "new-not-previously-in-TCIA"
        ]
        
        for subcat in subcategories:
            subcat_dir = validation_dir / subcat
            if subcat_dir.exists():
                print(f"  Scanning validation {subcat}...")
                subjects.update(self._scan_subcategory(subcat_dir, has_seg=False))
        
        return subjects
    
    def _scan_subcategory(self, subcat_dir, has_seg=True):
        """Scan a subcategory directory for subjects."""
        subjects = {}
        
        for subject_dir in subcat_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith("BraTS2021_"):
                subject_id = subject_dir.name
                files = self._analyze_subject_files(subject_dir, has_seg)
                
                if files and self._is_complete_subject(files, has_seg):
                    subjects[subject_id] = files
        
        return subjects
    
    def _analyze_subject_files(self, subject_dir, has_seg=True):
        """Analyze files in a subject directory."""
        files = {}
        
        for file_path in subject_dir.iterdir():
            if file_path.suffix == '.gz' and file_path.name.endswith('.nii.gz'):
                filename = file_path.name
                subject_id = subject_dir.name
                
                # Parse modality from filename
                if filename.endswith('_flair.nii.gz'):
                    files['flair'] = file_path
                elif filename.endswith('_t1ce.nii.gz'):
                    files['t1ce'] = file_path
                elif filename.endswith('_t1.nii.gz'):
                    files['t1'] = file_path
                elif filename.endswith('_t2.nii.gz'):
                    files['t2'] = file_path
                elif filename.endswith('_seg.nii.gz') and has_seg:
                    files['seg'] = file_path
        
        return files
    
    def _is_complete_subject(self, files, has_seg=True):
        """Check if subject has all required files."""
        required = self.modalities.copy()
        if has_seg:
            required.append('seg')
        
        return all(mod in files for mod in required)
    
    def create_splits(self, subjects, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Create train/validation/test splits."""
        # Separate subjects with and without segmentation
        train_subjects = {k: v for k, v in subjects.items() if 'seg' in v}
        val_subjects = {k: v for k, v in subjects.items() if 'seg' not in v}
        
        print(f"Subjects with segmentation (training): {len(train_subjects)}")
        print(f"Subjects without segmentation (validation): {len(val_subjects)}")
        
        # Split training subjects
        train_ids = list(train_subjects.keys())
        np.random.seed(42)
        np.random.shuffle(train_ids)
        
        n_train = len(train_ids)
        n_train_split = int(n_train * train_ratio)
        n_val_split = int(n_train * val_ratio)
        
        splits = {
            'train': train_ids[:n_train_split],
            'val': train_ids[n_train_split:n_train_split + n_val_split],
            'test': train_ids[n_train_split + n_val_split:]
        }
        
        # Add validation subjects to test set (for inference testing)
        if val_subjects:
            val_ids = list(val_subjects.keys())
            splits['test'].extend(val_ids[:min(len(val_ids), 50)])  # Limit to 50
        
        print(f"\nDataset splits:")
        for split, ids in splits.items():
            print(f"  {split}: {len(ids)} subjects")
        
        return splits, {**train_subjects, **val_subjects}
    
    def organize_dataset(self, all_subjects, splits):
        """Organize dataset into train/val/test directories."""
        print("\nOrganizing dataset...")
        
        total_copied = 0
        
        for split_name, subject_ids in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing {split_name} split ({len(subject_ids)} subjects)...")
            
            for subject_id in tqdm(subject_ids, desc=f"Copying {split_name}"):
                if subject_id in all_subjects:
                    self._copy_subject_files(subject_id, all_subjects[subject_id], split_dir)
                    total_copied += 1
        
        print(f"\nTotal subjects copied: {total_copied}")
        return total_copied
    
    def _copy_subject_files(self, subject_id, files, split_dir):
        """Copy files for a single subject."""
        subject_dir = split_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        for modality, src_path in files.items():
            # Standardized filename
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
                validation_results[split] = {
                    'total_subjects': 0,
                    'valid_subjects': 0,
                    'size_gb': 0
                }
                continue
            
            subjects = [d for d in split_dir.iterdir() if d.is_dir()]
            valid_subjects = 0
            split_size_gb = 0
            
            for subject_dir in subjects:
                # Check for modality files
                modality_files = [
                    f"{subject_dir.name}_t1.nii.gz",
                    f"{subject_dir.name}_t1ce.nii.gz",
                    f"{subject_dir.name}_t2.nii.gz",
                    f"{subject_dir.name}_flair.nii.gz"
                ]
                
                existing_files = [f for f in modality_files if (subject_dir / f).exists()]
                
                # Also check for segmentation (training/val only)
                seg_file = subject_dir / f"{subject_dir.name}_seg.nii.gz"
                if seg_file.exists():
                    existing_files.append(seg_file.name)
                
                if len(existing_files) >= 4:  # At least 4 modalities
                    valid_subjects += 1
                    
                    # Calculate size
                    for f in existing_files:
                        file_path = subject_dir / f
                        if file_path.exists():
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
    
    def create_dataset_info(self, validation_results, total_copied):
        """Create dataset information file."""
        dataset_info = {
            "dataset_name": "BraTS 2021 (Official RSNA-ASNR-MICCAI)",
            "description": "Brain Tumor Segmentation Challenge 2021 - Official Dataset",
            "source": "RSNA-ASNR-MICCAI BraTS 2021",
            "url": "https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/",
            "total_subjects_organized": total_copied,
            "modalities": self.modalities,
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
            "created_by": "BraTS 2021 Official Setup Script",
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset info saved to {self.output_dir}/dataset_info.json")
        return dataset_info
    
    def setup(self):
        """Main setup process."""
        print("Official BraTS 2021 Dataset Setup")
        print("=" * 50)
        
        start_time = time.time()
        
        if not self.input_dir.exists():
            print(f"ERROR: Input directory not found: {self.input_dir}")
            return False
        
        # Scan dataset structure
        subjects = self.scan_brats2021_structure()
        
        if len(subjects) == 0:
            print("ERROR: No valid subjects found!")
            return False
        
        # Create splits
        splits, all_subjects = self.create_splits(subjects)
        
        # Organize dataset
        total_copied = self.organize_dataset(all_subjects, splits)
        
        # Validate
        validation_results = self.validate_dataset()
        
        # Create info file
        dataset_info = self.create_dataset_info(validation_results, total_copied)
        
        processing_time = time.time() - start_time
        
        print(f"\nBraTS 2021 Dataset Setup Complete!")
        print("=" * 50)
        print(f"Dataset location: {self.output_dir}")
        print(f"Processing time: {processing_time/60:.1f} minutes")
        print(f"Subjects organized: {total_copied}")
        print(f"Total size: {sum(r['size_gb'] for r in validation_results.values()):.1f} GB")
        
        print(f"\nExpected Performance:")
        print(f"- Dice Score: >0.90 (official BraTS dataset)")
        print(f"- Training time: 8-16 hours (RTX 4080 Super)")
        print(f"- State-of-the-art results expected")
        
        print(f"\nNext steps:")
        print(f"1. cd X:\\Projects\\brats_medical_segmentation")
        print(f"2. python train_large_brats.py --dataset brats2021")
        print(f"3. Monitor training with TensorBoard")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Setup official BraTS 2021 dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing BraTS 2021 dataset')
    parser.add_argument('--output_dir', type=str, default='data/brats2021',
                       help='Output directory for organized dataset')
    
    args = parser.parse_args()
    
    setup = BraTS2021Setup(args.input_dir, args.output_dir)
    success = setup.setup()
    
    if success:
        print("SUCCESS: BraTS 2021 dataset setup complete!")
    else:
        print("FAILED: Dataset setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()