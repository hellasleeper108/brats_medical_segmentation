#!/usr/bin/env python3
"""
Medical Dataset Acquisition System
Helps acquire BraTS, COVID-CT, and other medical imaging datasets.
"""

import os
import sys
import requests
import zipfile
import tarfile
import json
from pathlib import Path
from urllib.parse import urlparse
import webbrowser
from tqdm import tqdm


class MedicalDatasetAcquisition:
    """Class to handle medical dataset acquisition."""
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_file_with_progress(self, url, destination):
        """Download file with progress bar."""
        try:
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
            
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def setup_brats_dataset(self):
        """Setup BraTS dataset - requires manual registration."""
        brats_dir = self.base_dir / "brats"
        brats_dir.mkdir(exist_ok=True)
        
        print("🧠 BraTS Dataset Setup")
        print("=" * 50)
        print("\nBraTS (Brain Tumor Segmentation) Challenge Dataset")
        print("This is the gold standard for brain tumor segmentation research.")
        
        print("\n📋 Registration Required:")
        print("1. BraTS requires registration due to patient privacy")
        print("2. Visit: https://www.synapse.org/#!Synapse:syn53708126")
        print("3. Create a Synapse account (free)")
        print("4. Accept the data use agreement")
        print("5. Download BraTS 2023 Training Data")
        
        print("\n📁 Expected Dataset Structure:")
        structure = """
brats/
├── raw/
│   └── BraTS2023_Training_Data/
│       ├── BraTS-GLI-00000-000/
│       │   ├── BraTS-GLI-00000-000-t1c.nii.gz
│       │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│       │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│       │   ├── BraTS-GLI-00000-000-t2w.nii.gz
│       │   └── BraTS-GLI-00000-000-seg.nii.gz
│       └── ... (more cases)
├── train/
├── val/
└── test/
        """
        print(structure)
        
        # Create metadata
        metadata = {
            "dataset_name": "BraTS2023",
            "description": "Brain Tumor Segmentation Challenge 2023",
            "url": "https://www.synapse.org/#!Synapse:syn53708126",
            "modalities": ["t1c", "t1n", "t2f", "t2w"],
            "classes": {
                0: "background",
                1: "necrotic_tumor_core",
                2: "peritumoral_edematous/invaded_tissue",
                3: "enhancing_tumor"
            },
            "num_subjects": 1251,
            "license": "Custom - Research Use Only",
            "acquisition_method": "manual_registration"
        }
        
        with open(brats_dir / "dataset_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create helper script
        helper_script = '''#!/usr/bin/env python3
"""
BraTS Dataset Preprocessor
Run this after downloading BraTS data manually.
"""

import os
import shutil
from pathlib import Path

def organize_brats_data():
    """Organize BraTS data into train/val/test splits."""
    raw_dir = Path("data/brats/raw/BraTS2023_Training_Data")
    
    if not raw_dir.exists():
        print("❌ BraTS raw data not found. Please download first.")
        print("Expected location: data/brats/raw/BraTS2023_Training_Data/")
        return False
    
    # Find all subject directories
    subjects = [d for d in raw_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subjects)} BraTS subjects")
    
    # Split data: 80% train, 15% val, 5% test
    n_train = int(len(subjects) * 0.8)
    n_val = int(len(subjects) * 0.15)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    # Create splits
    for split_name, split_subjects in [
        ("train", train_subjects),
        ("val", val_subjects), 
        ("test", test_subjects)
    ]:
        split_dir = Path(f"data/brats/{split_name}")
        split_dir.mkdir(exist_ok=True)
        
        for subject in split_subjects:
            dest_subject = split_dir / subject.name
            if not dest_subject.exists():
                shutil.copytree(subject, dest_subject)
        
        print(f"✅ {split_name}: {len(split_subjects)} subjects")
    
    print("🎉 BraTS data organization complete!")
    return True

if __name__ == "__main__":
    organize_brats_data()
'''
        
        with open(brats_dir / "preprocess_brats.py", 'w') as f:
            f.write(helper_script)
        
        print("\n🚀 Next Steps:")
        print("1. Open browser and register for BraTS dataset")
        print("2. Download the training data (several GB)")
        print("3. Extract to data/brats/raw/")
        print("4. Run: python data/brats/preprocess_brats.py")
        
        # Ask if user wants to open registration page
        try:
            open_browser = input("\n🌐 Open BraTS registration page in browser? (y/n): ").lower().strip()
            if open_browser == 'y':
                webbrowser.open("https://www.synapse.org/#!Synapse:syn53708126")
                print("✅ Opened BraTS registration page")
        except:
            pass
        
        return True
    
    def setup_covid_ct_dataset(self):
        """Setup COVID-CT dataset with automatic download options."""
        covid_dir = self.base_dir / "covid_ct"
        covid_dir.mkdir(exist_ok=True)
        
        print("\n🦠 COVID-CT Dataset Setup")
        print("=" * 50)
        
        # Available COVID-CT datasets
        datasets = {
            "1": {
                "name": "COVID-19 CT Lung and Infection Segmentation",
                "url": "https://zenodo.org/record/3757476",
                "download_url": "https://zenodo.org/record/3757476/files/Lung_and_Infection_Mask.zip",
                "size": "~2.5GB",
                "subjects": 100,
                "description": "CT scans with lung and infection masks"
            },
            "2": {
                "name": "MedSeg COVID-19 CT Dataset", 
                "url": "http://medicalsegmentation.com/covid19/",
                "download_url": None,  # Requires manual download
                "size": "~5GB",
                "subjects": 829,
                "description": "Large COVID-19 CT segmentation dataset"
            },
            "3": {
                "name": "COVID-19 CT Segmentation Dataset (Kaggle)",
                "url": "https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset",
                "download_url": None,  # Kaggle API required
                "size": "~1.2GB", 
                "subjects": 4000,
                "description": "CT slices with segmentation masks"
            }
        }
        
        print("Available COVID-CT Datasets:")
        for key, dataset in datasets.items():
            print(f"\n{key}. {dataset['name']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Subjects: {dataset['subjects']}")
            print(f"   URL: {dataset['url']}")
        
        try:
            choice = input("\nSelect dataset to setup (1-3): ").strip()
            
            if choice == "1":
                return self._setup_zenodo_covid_dataset(covid_dir, datasets["1"])
            elif choice == "2":
                return self._setup_medseg_covid_dataset(covid_dir, datasets["2"])
            elif choice == "3":
                return self._setup_kaggle_covid_dataset(covid_dir, datasets["3"])
            else:
                print("Invalid choice")
                return False
                
        except KeyboardInterrupt:
            print("\nSetup cancelled")
            return False
    
    def _setup_zenodo_covid_dataset(self, covid_dir, dataset_info):
        """Setup Zenodo COVID-19 dataset."""
        print(f"\n📥 Setting up {dataset_info['name']}")
        
        raw_dir = covid_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        download_path = raw_dir / "covid_lung_infection.zip"
        
        print("🌐 Downloading dataset...")
        if self.download_file_with_progress(dataset_info["download_url"], download_path):
            print("✅ Download complete!")
            
            # Extract
            print("📂 Extracting dataset...")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            
            # Clean up zip file
            os.remove(download_path)
            
            print("🎉 COVID-CT dataset setup complete!")
            return True
        else:
            print("❌ Download failed")
            return False
    
    def _setup_medseg_covid_dataset(self, covid_dir, dataset_info):
        """Setup MedSeg COVID-19 dataset (manual download)."""
        print(f"\n📋 Setting up {dataset_info['name']}")
        print("This dataset requires manual download due to registration.")
        
        print("\n📋 Steps:")
        print("1. Visit: http://medicalsegmentation.com/covid19/")
        print("2. Fill out the request form")
        print("3. Wait for download link via email")
        print("4. Download and extract to data/covid_ct/raw/")
        
        try:
            open_browser = input("\n🌐 Open MedSeg website? (y/n): ").lower().strip()
            if open_browser == 'y':
                webbrowser.open(dataset_info["url"])
        except:
            pass
        
        return True
    
    def _setup_kaggle_covid_dataset(self, covid_dir, dataset_info):
        """Setup Kaggle COVID-19 dataset."""
        print(f"\n📋 Setting up {dataset_info['name']}")
        print("This dataset requires Kaggle API setup.")
        
        print("\n📋 Steps:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Setup Kaggle credentials (kaggle.json)")
        print("3. Run: kaggle datasets download -d maedemaftouni/large-covid19-ct-slice-dataset")
        print("4. Extract to data/covid_ct/raw/")
        
        # Try to install kaggle API
        try:
            install_kaggle = input("\n📦 Install Kaggle API now? (y/n): ").lower().strip()
            if install_kaggle == 'y':
                os.system("pip install kaggle")
                print("✅ Kaggle API installed")
                print("⚠️  You still need to setup kaggle.json credentials")
        except:
            pass
        
        try:
            open_browser = input("\n🌐 Open Kaggle dataset page? (y/n): ").lower().strip()
            if open_browser == 'y':
                webbrowser.open(dataset_info["url"])
        except:
            pass
        
        return True
    
    def setup_public_datasets(self):
        """Setup publicly available medical datasets that don't require registration."""
        print("\n🏥 Public Medical Datasets")
        print("=" * 50)
        
        public_datasets = {
            "1": {
                "name": "BTCV (Multi-organ segmentation)",
                "url": "https://www.synapse.org/#!Synapse:syn3193805/wiki/89480",
                "description": "30 subjects with 13 organ segmentations",
                "modality": "CT"
            },
            "2": {
                "name": "Medical Segmentation Decathlon",
                "url": "http://medicaldecathlon.com/",
                "description": "10 medical segmentation tasks",
                "modality": "Various"
            },
            "3": {
                "name": "AMOS2022 (Abdominal organ segmentation)",
                "url": "https://amos22.grand-challenge.org/",
                "description": "600 CT and 100 MRI scans",
                "modality": "CT/MRI"
            }
        }
        
        print("Available Public Datasets:")
        for key, dataset in public_datasets.items():
            print(f"\n{key}. {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Modality: {dataset['modality']}")
            print(f"   URL: {dataset['url']}")
        
        try:
            choice = input("\nSelect dataset to learn more about (1-3): ").strip()
            if choice in public_datasets:
                webbrowser.open(public_datasets[choice]["url"])
                print(f"✅ Opened {public_datasets[choice]['name']} page")
        except:
            pass
        
        return True
    
    def check_dataset_status(self):
        """Check status of downloaded datasets."""
        print("\n📊 Dataset Status Check")
        print("=" * 50)
        
        datasets = ["brats", "covid_ct", "samples"]
        
        for dataset_name in datasets:
            dataset_dir = self.base_dir / dataset_name
            
            if not dataset_dir.exists():
                print(f"❌ {dataset_name}: Not found")
                continue
            
            # Check for data files
            raw_dir = dataset_dir / "raw"
            train_dir = dataset_dir / "train"
            val_dir = dataset_dir / "val"
            
            raw_files = len(list(raw_dir.glob("**/*.nii.gz"))) if raw_dir.exists() else 0
            train_files = len(list(train_dir.glob("**/*.nii.gz"))) if train_dir.exists() else 0
            val_files = len(list(val_dir.glob("**/*.nii.gz"))) if val_dir.exists() else 0
            
            status = "✅" if train_files > 0 else "⚠️"
            print(f"{status} {dataset_name}:")
            print(f"   Raw files: {raw_files}")
            print(f"   Training files: {train_files}")
            print(f"   Validation files: {val_files}")
        
        return True


def main():
    print("🏥 Medical Dataset Acquisition System")
    print("=" * 60)
    
    acquisition = MedicalDatasetAcquisition()
    
    while True:
        print("\n📋 Available Options:")
        print("1. Setup BraTS Dataset (Brain Tumor Segmentation)")
        print("2. Setup COVID-CT Dataset")
        print("3. Browse Public Medical Datasets")
        print("4. Check Dataset Status")
        print("5. Exit")
        
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                acquisition.setup_brats_dataset()
            elif choice == "2":
                acquisition.setup_covid_ct_dataset()
            elif choice == "3":
                acquisition.setup_public_datasets()
            elif choice == "4":
                acquisition.check_dataset_status()
            elif choice == "5":
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