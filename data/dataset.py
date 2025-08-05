import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from monai import transforms as T
from monai.data import Dataset as MonaiDataset
from monai.utils import first
import pandas as pd
from typing import List, Dict, Tuple, Optional


class MedicalDataset(Dataset):
    """Base class for medical imaging datasets."""
    
    def __init__(self, data_dir: str, transforms=None, cache_rate: float = 0.0):
        self.data_dir = data_dir
        self.transforms = transforms
        self.cache_rate = cache_rate
        self.data_list = []
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_dict = self.data_list[idx].copy()
        
        if self.transforms:
            data_dict = self.transforms(data_dict)
            
        return data_dict


class BraTSDataset(MedicalDataset):
    """BraTS (Brain Tumor Segmentation) dataset."""
    
    def __init__(self, data_dir: str, transforms=None, cache_rate: float = 0.0, 
                 modalities: List[str] = ['t1', 't1ce', 't2', 'flair']):
        super().__init__(data_dir, transforms, cache_rate)
        self.modalities = modalities
        self.data_list = self._prepare_data_list()
        
        if cache_rate > 0:
            self.dataset = MonaiDataset(
                data=self.data_list,
                transform=self.transforms,
                cache_rate=cache_rate
            )
    
    def _prepare_data_list(self) -> List[Dict]:
        """Prepare data list for BraTS dataset."""
        data_list = []
        
        # Scan for BraTS data structure
        for subject_dir in os.listdir(self.data_dir):
            subject_path = os.path.join(self.data_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
                
            # Find modality files
            image_files = []
            seg_file = None
            
            for file in os.listdir(subject_path):
                if file.endswith('.nii.gz'):
                    full_path = os.path.join(subject_path, file)
                    if 'seg' in file.lower():
                        seg_file = full_path
                    else:
                        for modality in self.modalities:
                            if modality in file.lower():
                                image_files.append(full_path)
                                break
            
            if len(image_files) == len(self.modalities) and seg_file:
                data_dict = {
                    'image': image_files,
                    'label': seg_file,
                    'subject_id': subject_dir
                }
                data_list.append(data_dict)
        
        return data_list
    
    def __getitem__(self, idx):
        if hasattr(self, 'dataset') and self.cache_rate > 0:
            return self.dataset[idx]
        else:
            return super().__getitem__(idx)


class COVIDCTDataset(MedicalDataset):
    """COVID-CT dataset for lung segmentation."""
    
    def __init__(self, data_dir: str, transforms=None, cache_rate: float = 0.0,
                 metadata_file: Optional[str] = None):
        super().__init__(data_dir, transforms, cache_rate)
        self.metadata_file = metadata_file
        self.data_list = self._prepare_data_list()
        
        if cache_rate > 0:
            self.dataset = MonaiDataset(
                data=self.data_list,
                transform=self.transforms,
                cache_rate=cache_rate
            )
    
    def _prepare_data_list(self) -> List[Dict]:
        """Prepare data list for COVID-CT dataset."""
        data_list = []
        
        # If metadata file exists, use it
        if self.metadata_file and os.path.exists(self.metadata_file):
            df = pd.read_csv(self.metadata_file)
            for _, row in df.iterrows():
                image_path = os.path.join(self.data_dir, row['image_path'])
                label_path = os.path.join(self.data_dir, row['label_path'])
                
                if os.path.exists(image_path) and os.path.exists(label_path):
                    data_dict = {
                        'image': image_path,
                        'label': label_path,
                        'subject_id': row.get('subject_id', os.path.basename(image_path)),
                        'covid_status': row.get('covid_status', 'unknown')
                    }
                    data_list.append(data_dict)
        else:
            # Scan directory structure
            images_dir = os.path.join(self.data_dir, 'images')
            labels_dir = os.path.join(self.data_dir, 'labels')
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                for image_file in os.listdir(images_dir):
                    if image_file.endswith(('.nii', '.nii.gz')):
                        image_path = os.path.join(images_dir, image_file)
                        label_path = os.path.join(labels_dir, image_file)
                        
                        if os.path.exists(label_path):
                            data_dict = {
                                'image': image_path,
                                'label': label_path,
                                'subject_id': os.path.splitext(image_file)[0]
                            }
                            data_list.append(data_dict)
        
        return data_list
    
    def __getitem__(self, idx):
        if hasattr(self, 'dataset') and self.cache_rate > 0:
            return self.dataset[idx]
        else:
            return super().__getitem__(idx)


class CustomMedicalDataset(MedicalDataset):
    """Custom medical dataset for user-provided data."""
    
    def __init__(self, data_dir: str, transforms=None, cache_rate: float = 0.0,
                 image_pattern: str = "*_image.nii.gz", 
                 label_pattern: str = "*_label.nii.gz"):
        super().__init__(data_dir, transforms, cache_rate)
        self.image_pattern = image_pattern
        self.label_pattern = label_pattern
        self.data_list = self._prepare_data_list()
        
        if cache_rate > 0:
            self.dataset = MonaiDataset(
                data=self.data_list,
                transform=self.transforms,
                cache_rate=cache_rate
            )
    
    def _prepare_data_list(self) -> List[Dict]:
        """Prepare data list for custom dataset."""
        import glob
        
        data_list = []
        image_files = glob.glob(os.path.join(self.data_dir, self.image_pattern))
        
        for image_file in image_files:
            # Derive label file name
            base_name = os.path.basename(image_file).replace('_image', '_label')
            label_file = os.path.join(self.data_dir, base_name)
            
            if os.path.exists(label_file):
                data_dict = {
                    'image': image_file,
                    'label': label_file,
                    'subject_id': os.path.splitext(base_name)[0].replace('_label', '')
                }
                data_list.append(data_dict)
        
        return data_list
    
    def __getitem__(self, idx):
        if hasattr(self, 'dataset') and self.cache_rate > 0:
            return self.dataset[idx]
        else:
            return super().__getitem__(idx)