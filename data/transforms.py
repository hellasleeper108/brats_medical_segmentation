import numpy as np
from monai import transforms as T
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
    ToTensord, Compose
)
from typing import Dict, List, Tuple, Optional


def get_transforms(mode: str = 'train', 
                  roi_size: Tuple[int, int, int] = (128, 128, 128),
                  spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  intensity_range: Tuple[float, float] = (-1000, 1000),
                  num_samples: int = 4) -> Compose:
    """
    Get data transforms for medical image segmentation.
    
    Args:
        mode: 'train', 'val', or 'test'
        roi_size: Size of region of interest for cropping
        spacing: Target voxel spacing
        intensity_range: Intensity range for normalization
        num_samples: Number of samples for positive/negative cropping
    
    Returns:
        Composed transforms
    """
    
    # Common transforms for all modes
    common_transforms = [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        Spacingd(keys=['image', 'label'], pixdim=spacing, mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=['image', 'label'], source_key='image')
    ]
    
    if mode == 'train':
        # Training transforms with augmentation
        train_transforms = common_transforms + [
            RandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key='image',
                image_threshold=0
            ),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3),
            RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.15),
            RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.15),
            RandGaussianNoised(keys=['image'], std=0.01, prob=0.15),
            RandGaussianSmoothd(
                keys=['image'],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15
            ),
            RandAdjustContrastd(keys=['image'], gamma=(0.7, 1.5), prob=0.15),
            ToTensord(keys=['image', 'label'])
        ]
        return Compose(train_transforms)
    
    elif mode in ['val', 'test']:
        # Validation/test transforms without augmentation
        val_transforms = common_transforms + [
            ToTensord(keys=['image', 'label'])
        ]
        return Compose(val_transforms)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")


def get_inference_transforms(roi_size: Tuple[int, int, int] = (128, 128, 128),
                           spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                           intensity_range: Tuple[float, float] = (-1000, 1000)) -> Compose:
    """
    Get transforms for inference (single image).
    
    Args:
        roi_size: Size of region of interest
        spacing: Target voxel spacing
        intensity_range: Intensity range for normalization
    
    Returns:
        Composed transforms for inference
    """
    inference_transforms = [
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=['image'], axcodes='RAS'),
        Spacingd(keys=['image'], pixdim=spacing, mode='bilinear'),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=['image'], source_key='image'),
        ToTensord(keys=['image'])
    ]
    
    return Compose(inference_transforms)


def get_post_transforms() -> Compose:
    """Get post-processing transforms for predictions."""
    from monai.transforms import (
        Activationsd, AsDiscreted, KeepLargestConnectedComponentd,
        SaveImaged
    )
    
    post_transforms = [
        Activationsd(keys='pred', softmax=True),
        AsDiscreted(keys='pred', argmax=True),
        KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 2, 3])
    ]
    
    return Compose(post_transforms)


class CustomTransforms:
    """Custom transforms for specific medical imaging tasks."""
    
    @staticmethod
    def normalize_intensity(image: np.ndarray, 
                          percentiles: Tuple[float, float] = (1, 99)) -> np.ndarray:
        """Normalize intensity using percentiles."""
        p_low, p_high = np.percentile(image, percentiles)
        image = np.clip(image, p_low, p_high)
        image = (image - p_low) / (p_high - p_low)
        return image
    
    @staticmethod
    def pad_to_size(image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Pad image to target size."""
        current_size = image.shape
        pad_width = []
        
        for i in range(3):
            diff = target_size[i] - current_size[i]
            if diff > 0:
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))
        
        return np.pad(image, pad_width, mode='constant', constant_values=0)
    
    @staticmethod
    def crop_to_size(image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Crop image to target size."""
        current_size = image.shape
        start_indices = []
        
        for i in range(3):
            if current_size[i] > target_size[i]:
                start_idx = (current_size[i] - target_size[i]) // 2
                start_indices.append(start_idx)
            else:
                start_indices.append(0)
        
        return image[
            start_indices[0]:start_indices[0] + target_size[0],
            start_indices[1]:start_indices[1] + target_size[1],
            start_indices[2]:start_indices[2] + target_size[2]
        ]