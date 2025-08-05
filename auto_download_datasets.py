#!/usr/bin/env python3
"""
Automated Medical Dataset Downloader
Downloads publicly available medical imaging datasets automatically.
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import json
import nibabel as nib
import numpy as np


class AutoDatasetDownloader:
    """Automated downloader for public medical datasets."""
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def download_with_progress(self, url, destination):
        """Download file with progress bar."""
        try:
            print(f"📥 Downloading: {os.path.basename(destination)}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("✅ Download complete!")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_to):
        """Extract various archive formats."""
        print(f"📂 Extracting: {archive_path}")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"⚠️ Unknown archive format: {archive_path.suffix}")
                return False
            
            print("✅ Extraction complete!")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def download_sample_ct_data(self):
        """Download sample CT data from public sources."""
        print("\n🫁 Downloading Sample CT Data")
        print("=" * 50)
        
        ct_dir = self.base_dir / "sample_ct"
        ct_dir.mkdir(exist_ok=True)
        raw_dir = ct_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        # List of publicly available medical image samples
        sample_urls = [
            {
                "name": "sample_ct_chest.nii.gz",
                "url": "https://github.com/InsightSoftwareConsortium/ITKElastix/raw/master/examples/ITK_Example07_ParameterMaps/lung.nii.gz",
                "description": "Sample chest CT scan"
            }
        ]
        
        downloaded_files = []
        
        for sample in sample_urls:
            dest_path = raw_dir / sample["name"]
            
            if self.download_with_progress(sample["url"], dest_path):
                downloaded_files.append(dest_path)
                print(f"✅ Downloaded: {sample['description']}")
        
        # Create synthetic segmentations for the downloaded CT data
        if downloaded_files:
            self.create_synthetic_segmentations(downloaded_files, raw_dir)
        
        return len(downloaded_files) > 0
    
    def create_synthetic_segmentations(self, image_files, output_dir):
        """Create synthetic segmentation masks for real CT data."""
        print("\n🎨 Creating synthetic segmentation masks...")
        
        for img_file in image_files:
            try:
                # Load the real CT image
                img = nib.load(img_file)
                img_data = img.get_fdata()
                
                # Create synthetic segmentation based on intensity thresholds
                segmentation = np.zeros_like(img_data, dtype=np.uint8)
                
                # Background (air/low density)
                segmentation[img_data < -500] = 0
                
                # Soft tissue
                segmentation[(img_data >= -500) & (img_data < 100)] = 1
                
                # Dense tissue/bone
                segmentation[img_data >= 100] = 2
                
                # Save segmentation
                seg_filename = img_file.stem.replace('.nii', '_seg.nii') + '.gz'
                seg_path = output_dir / seg_filename
                
                seg_img = nib.Nifti1Image(segmentation, img.affine, img.header)
                nib.save(seg_img, seg_path)
                
                print(f"✅ Created segmentation: {seg_filename}")
                
            except Exception as e:
                print(f"❌ Failed to create segmentation for {img_file}: {e}")
        
        print("🎉 Synthetic segmentations created!")
    
    def download_medical_decathlon_task(self, task_id=1):
        """Download a specific Medical Decathlon task."""
        print(f"\n🏥 Downloading Medical Decathlon Task {task_id:02d}")
        print("=" * 50)
        
        # Medical Decathlon tasks
        tasks = {
            1: {"name": "BrainTumour", "modality": "MRI"},
            2: {"name": "Heart", "modality": "MRI"},
            3: {"name": "Liver", "modality": "CT"},
            4: {"name": "Hippocampus", "modality": "MRI"},
            5: {"name": "Prostate", "modality": "MRI"},
            6: {"name": "Lung", "modality": "CT"},
            7: {"name": "Pancreas", "modality": "CT"},
            8: {"name": "HepaticVessel", "modality": "CT"},
            9: {"name": "Spleen", "modality": "CT"},
            10: {"name": "Colon", "modality": "CT"}
        }
        
        if task_id not in tasks:
            print(f"❌ Invalid task ID: {task_id}")
            return False
        
        task_info = tasks[task_id]
        task_name = task_info["name"]
        
        decathlon_dir = self.base_dir / "medical_decathlon"
        decathlon_dir.mkdir(exist_ok=True)
        
        task_dir = decathlon_dir / f"Task{task_id:02d}_{task_name}"
        task_dir.mkdir(exist_ok=True)
        
        # Note: Medical Decathlon requires registration
        print(f"📋 Task: {task_name} ({task_info['modality']})")
        print("⚠️  Medical Decathlon requires registration at:")
        print("   https://medicaldecathlon.com/")
        print("\n📋 Steps to download:")
        print("1. Register at the Medical Decathlon website")
        print("2. Download the dataset manually")
        print("3. Extract to: data/medical_decathlon/")
        
        # Create metadata
        metadata = {
            "task_id": task_id,
            "task_name": task_name,
            "modality": task_info["modality"],
            "url": "https://medicaldecathlon.com/",
            "registration_required": True
        }
        
        with open(task_dir / "task_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def download_covid_chest_xray(self):
        """Download COVID chest X-ray dataset (publicly available)."""
        print("\n🫁 Downloading COVID-19 Chest X-ray Dataset")
        print("=" * 50)
        
        covid_xray_dir = self.base_dir / "covid_chest_xray"
        covid_xray_dir.mkdir(exist_ok=True)
        raw_dir = covid_xray_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        # GitHub COVID-19 Chest X-ray dataset
        dataset_url = "https://github.com/ieee8023/covid-chestxray-dataset/archive/master.zip"
        archive_path = raw_dir / "covid_chestxray_dataset.zip"
        
        if self.download_with_progress(dataset_url, archive_path):
            if self.extract_archive(archive_path, raw_dir):
                # Clean up archive
                os.remove(archive_path)
                
                # Create metadata
                metadata = {
                    "dataset_name": "COVID-19 Chest X-ray Dataset",
                    "description": "Chest X-rays of COVID-19 cases",
                    "modality": "X-ray",
                    "url": "https://github.com/ieee8023/covid-chestxray-dataset",
                    "license": "Apache 2.0"
                }
                
                with open(covid_xray_dir / "dataset_info.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print("🎉 COVID-19 Chest X-ray dataset downloaded!")
                return True
        
        return False
    
    def create_enhanced_synthetic_dataset(self, num_subjects=20):
        """Create a larger, more realistic synthetic dataset."""
        print(f"\n🧠 Creating Enhanced Synthetic Dataset ({num_subjects} subjects)")
        print("=" * 50)
        
        synthetic_dir = self.base_dir / "enhanced_synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            split_dir = synthetic_dir / split
            split_dir.mkdir(exist_ok=True)
        
        # Generate subjects
        subjects_per_split = {
            'train': int(num_subjects * 0.7),
            'val': int(num_subjects * 0.2),
            'test': int(num_subjects * 0.1)
        }
        
        subject_id = 1
        
        for split, count in subjects_per_split.items():
            split_dir = synthetic_dir / split
            
            print(f"📊 Generating {count} subjects for {split}...")
            
            for i in tqdm(range(count), desc=f"Creating {split} data"):
                # Generate more realistic synthetic brain
                image, segmentation = self._generate_realistic_brain(subject_id)
                
                # Save as NIfTI
                subject_name = f"subject_{subject_id:03d}"
                
                # Create affine matrix
                affine = np.eye(4)
                affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm isotropic
                
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
            "classes": {
                0: "background",
                1: "brain_tissue",
                2: "tumor_core",
                3: "tumor_edema"
            },
            "splits": subjects_per_split
        }
        
        with open(synthetic_dir / "dataset_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"🎉 Enhanced synthetic dataset created with {num_subjects} subjects!")
        return True
    
    def _generate_realistic_brain(self, subject_id):
        """Generate more realistic synthetic brain with tumors."""
        shape = (160, 192, 144)  # More realistic brain dimensions
        
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
            image[tumor_core] = 1.2 + np.random.normal(0, 0.1, tumor_core.sum())
            image[tumor_edema] = 0.9 + np.random.normal(0, 0.05, tumor_edema.sum())
            
            # Update segmentation
            segmentation[tumor_core] = 2  # Tumor core
            segmentation[tumor_edema] = 3  # Tumor edema
        
        # Add noise and artifacts
        image += np.random.normal(0, 0.02, shape)
        
        # Ensure realistic intensity range
        image = np.clip(image, 0, 1.5)
        
        return image, segmentation


def main():
    print("🤖 Automated Medical Dataset Downloader")
    print("=" * 60)
    
    downloader = AutoDatasetDownloader()
    
    while True:
        print("\n📋 Available Downloads:")
        print("1. Sample CT Data (public, automatic)")
        print("2. COVID-19 Chest X-ray Dataset (public, automatic)")
        print("3. Enhanced Synthetic Dataset (20 subjects)")
        print("4. Medical Decathlon Info (requires registration)")
        print("5. Download All Public Datasets")
        print("6. Exit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                downloader.download_sample_ct_data()
            elif choice == "2":
                downloader.download_covid_chest_xray()
            elif choice == "3":
                try:
                    num_subjects = int(input("Number of synthetic subjects (default 20): ") or "20")
                    downloader.create_enhanced_synthetic_dataset(num_subjects)
                except ValueError:
                    downloader.create_enhanced_synthetic_dataset(20)
            elif choice == "4":
                task_id = int(input("Medical Decathlon task ID (1-10): ") or "1")
                downloader.download_medical_decathlon_task(task_id)
            elif choice == "5":
                print("🚀 Downloading all available public datasets...")
                downloader.download_sample_ct_data()
                downloader.download_covid_chest_xray()
                downloader.create_enhanced_synthetic_dataset(30)
                print("🎉 All public datasets downloaded!")
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()