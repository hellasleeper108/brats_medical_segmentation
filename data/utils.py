import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import SimpleITK as sitk
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.utils import first


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Header, np.ndarray]:
    """
    Load NIfTI image file.
    
    Args:
        file_path: Path to NIfTI file
    
    Returns:
        Tuple of (image data, header, affine matrix)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    img = nib.load(file_path)
    data = img.get_fdata()
    header = img.header
    affine = img.affine
    
    return data, header, affine


def save_nifti(data: np.ndarray, 
               affine: np.ndarray, 
               header: nib.Nifti1Header,
               output_path: str) -> None:
    """
    Save data as NIfTI image.
    
    Args:
        data: Image data array
        affine: Affine transformation matrix
        header: NIfTI header
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create NIfTI image
    img = nib.Nifti1Image(data, affine, header)
    
    # Save to file
    nib.save(img, output_path)


def load_dicom_series(dicom_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Load DICOM series from directory.
    
    Args:
        dicom_dir: Directory containing DICOM files
    
    Returns:
        Tuple of (image data, metadata dict)
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    
    # Convert to numpy array
    data = sitk.GetArrayFromImage(image)
    
    # Get metadata
    metadata = {
        'spacing': image.GetSpacing(),
        'origin': image.GetOrigin(),
        'direction': image.GetDirection(),
        'size': image.GetSize()
    }
    
    return data, metadata


def convert_dicom_to_nifti(dicom_dir: str, output_path: str) -> None:
    """
    Convert DICOM series to NIfTI format.
    
    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Output NIfTI file path
    """
    data, metadata = load_dicom_series(dicom_dir)
    
    # Create affine matrix from DICOM metadata
    spacing = metadata['spacing']
    origin = metadata['origin']
    direction = np.array(metadata['direction']).reshape(3, 3)
    
    # Create affine transformation
    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(spacing)
    affine[:3, 3] = origin
    
    # Create minimal header
    header = nib.Nifti1Header()
    header.set_data_shape(data.shape)
    header.set_zooms(spacing)
    
    # Save as NIfTI
    save_nifti(data, affine, header, output_path)


def get_dataloader(dataset, 
                   batch_size: int = 1,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   pin_memory: bool = True,
                   persistent_workers: bool = True) -> DataLoader:
    """
    Create PyTorch DataLoader.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )


def validate_medical_image(data: np.ndarray, 
                          min_dimensions: int = 3,
                          max_dimensions: int = 4) -> bool:
    """
    Validate medical image data.
    
    Args:
        data: Image data array
        min_dimensions: Minimum number of dimensions
        max_dimensions: Maximum number of dimensions
    
    Returns:
        True if valid, False otherwise
    """
    # Check dimensions
    if not (min_dimensions <= data.ndim <= max_dimensions):
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False
    
    # Check if not empty
    if data.size == 0:
        return False
    
    # Check for reasonable intensity range (basic sanity check)
    if np.ptp(data) == 0:  # No variation in intensities
        return False
    
    return True


def normalize_image(image: np.ndarray, 
                   method: str = 'minmax',
                   percentiles: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """
    Normalize image intensities.
    
    Args:
        image: Input image array
        method: Normalization method ('minmax', 'zscore', 'percentile')
        percentiles: Percentile range for percentile normalization
    
    Returns:
        Normalized image array
    """
    if method == 'minmax':
        min_val = np.min(image)
        max_val = np.max(image)
        normalized = (image - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = np.mean(image)
        std_val = np.std(image)
        normalized = (image - mean_val) / (std_val + 1e-8)
    
    elif method == 'percentile':
        p_low, p_high = np.percentile(image, percentiles)
        normalized = np.clip(image, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def resample_image(image: np.ndarray,
                  current_spacing: Tuple[float, float, float],
                  target_spacing: Tuple[float, float, float],
                  interpolation: str = 'linear') -> np.ndarray:
    """
    Resample image to target spacing.
    
    Args:
        image: Input image array
        current_spacing: Current voxel spacing
        target_spacing: Target voxel spacing
        interpolation: Interpolation method
    
    Returns:
        Resampled image array
    """
    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image.SetSpacing(current_spacing)
    
    # Calculate new size
    current_size = sitk_image.GetSize()
    new_size = [
        int(current_size[i] * current_spacing[i] / target_spacing[i])
        for i in range(3)
    ]
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    
    if interpolation == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == 'cubic':
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    # Resample
    resampled_image = resampler.Execute(sitk_image)
    
    # Convert back to numpy
    return sitk.GetArrayFromImage(resampled_image)


def crop_or_pad_to_size(image: np.ndarray, 
                       target_size: Tuple[int, int, int],
                       mode: str = 'constant',
                       constant_value: float = 0.0) -> np.ndarray:
    """
    Crop or pad image to target size.
    
    Args:
        image: Input image array
        target_size: Target size (D, H, W)
        mode: Padding mode
        constant_value: Value for constant padding
    
    Returns:
        Cropped/padded image array
    """
    current_size = image.shape
    
    # Calculate padding/cropping for each dimension
    operations = []
    for i in range(3):
        diff = target_size[i] - current_size[i]
        if diff > 0:
            # Need padding
            pad_before = diff // 2
            pad_after = diff - pad_before
            operations.append(('pad', pad_before, pad_after))
        elif diff < 0:
            # Need cropping
            crop_before = (-diff) // 2
            crop_after = crop_before + target_size[i]
            operations.append(('crop', crop_before, crop_after))
        else:
            # No change needed
            operations.append(('none', 0, current_size[i]))
    
    # Apply operations
    result = image.copy()
    
    # Apply padding first
    pad_width = []
    for op, val1, val2 in operations:
        if op == 'pad':
            pad_width.append((val1, val2))
        else:
            pad_width.append((0, 0))
    
    if any(sum(pw) > 0 for pw in pad_width):
        result = np.pad(result, pad_width, mode=mode, constant_values=constant_value)
    
    # Apply cropping
    slices = []
    for i, (op, val1, val2) in enumerate(operations):
        if op == 'crop':
            slices.append(slice(val1, val2))
        else:
            slices.append(slice(None))
    
    if any(s != slice(None) for s in slices):
        result = result[tuple(slices)]
    
    return result


def split_dataset(data_list: List[Dict], 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.05,
                 random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        data_list: List of data dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(data_list))
    
    n_train = int(len(data_list) * train_ratio)
    n_val = int(len(data_list) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_list = [data_list[i] for i in train_indices]
    val_list = [data_list[i] for i in val_indices]
    test_list = [data_list[i] for i in test_indices]
    
    return train_list, val_list, test_list


def get_image_statistics(image: np.ndarray) -> Dict:
    """
    Calculate comprehensive image statistics.
    
    Args:
        image: Input image array
    
    Returns:
        Dictionary with image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'median': float(np.median(image)),
        'percentile_1': float(np.percentile(image, 1)),
        'percentile_99': float(np.percentile(image, 99)),
        'non_zero_voxels': int(np.count_nonzero(image)),
        'total_voxels': int(image.size)
    }
    
    stats['non_zero_ratio'] = stats['non_zero_voxels'] / stats['total_voxels']
    stats['dynamic_range'] = stats['max'] - stats['min']
    
    return stats