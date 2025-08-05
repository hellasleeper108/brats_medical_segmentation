import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import nibabel as nib
from typing import Tuple, Optional, List, Dict
import os
from pathlib import Path


def plot_segmentation(image: np.ndarray, 
                     segmentation: np.ndarray,
                     slice_idx: int,
                     axis: int = 2,
                     save_path: Optional[str] = None,
                     title: str = "Medical Image Segmentation",
                     cmap_image: str = 'gray',
                     alpha: float = 0.5) -> plt.Figure:
    """
    Plot medical image with segmentation overlay.
    
    Args:
        image: 3D medical image array
        segmentation: 3D segmentation array
        slice_idx: Index of slice to display
        axis: Axis along which to slice (0, 1, or 2)
        save_path: Path to save the plot
        title: Plot title
        cmap_image: Colormap for the image
        alpha: Transparency for segmentation overlay
    
    Returns:
        matplotlib Figure object
    """
    # Extract slices
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        seg_slice = segmentation[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        seg_slice = segmentation[:, slice_idx, :]
    else:
        img_slice = image[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice, cmap=cmap_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation overlay
    axes[1].imshow(img_slice, cmap=cmap_image)
    
    # Create custom colormap for segmentation
    colors = ['transparent', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    seg_cmap = ListedColormap(colors[:len(np.unique(seg_slice))])
    
    masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
    axes[1].imshow(masked_seg, cmap=seg_cmap, alpha=alpha, vmin=0, vmax=len(colors)-1)
    axes[1].set_title('Segmentation Overlay')
    axes[1].axis('off')
    
    # Segmentation only
    axes[2].imshow(seg_slice, cmap=seg_cmap, vmin=0, vmax=len(colors)-1)
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    plt.suptitle(f'{title} - Slice {slice_idx} (Axis {axis})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_slice_comparison(original_path: str, 
                          prediction_path: str,
                          slice_indices: List[int],
                          axis: int = 2,
                          save_dir: str = 'visualizations') -> List[str]:
    """
    Create slice comparison visualizations.
    
    Args:
        original_path: Path to original image
        prediction_path: Path to prediction image
        slice_indices: List of slice indices to visualize
        axis: Axis for slicing
        save_dir: Directory to save visualizations
    
    Returns:
        List of paths to saved visualization files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load images
    original_img = nib.load(original_path).get_fdata()
    prediction_img = nib.load(prediction_path).get_fdata()
    
    saved_files = []
    
    for slice_idx in slice_indices:
        # Create visualization
        fig = plot_segmentation(
            original_img, prediction_img, slice_idx, axis,
            title=f"Segmentation Results"
        )
        
        # Save figure
        filename = f"slice_{axis}_{slice_idx:03d}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        saved_files.append(save_path)
    
    return saved_files


def create_3d_visualization(segmentation: np.ndarray,
                          spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                          save_path: Optional[str] = None) -> Dict:
    """
    Create 3D visualization data for segmentation.
    
    Args:
        segmentation: 3D segmentation array
        spacing: Voxel spacing
        save_path: Path to save visualization data
    
    Returns:
        Dictionary with 3D visualization data
    """
    visualization_data = {}
    
    # Calculate volume statistics for each class
    unique_classes = np.unique(segmentation)
    total_volume = segmentation.size * np.prod(spacing)
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        
        class_mask = (segmentation == class_id)
        class_volume = np.sum(class_mask) * np.prod(spacing)
        
        # Find bounding box
        coords = np.where(class_mask)
        if len(coords[0]) > 0:
            bbox = {
                'min': [int(np.min(coords[i])) for i in range(3)],
                'max': [int(np.max(coords[i])) for i in range(3)]
            }
            
            # Calculate centroid
            centroid = [float(np.mean(coords[i])) for i in range(3)]
            
            visualization_data[f'class_{int(class_id)}'] = {
                'volume_voxels': int(np.sum(class_mask)),
                'volume_mm3': float(class_volume),
                'volume_percentage': float(class_volume / total_volume * 100),
                'bounding_box': bbox,
                'centroid': centroid
            }
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(visualization_data, f, indent=2)
    
    return visualization_data


def create_multi_view_plot(image: np.ndarray,
                          segmentation: Optional[np.ndarray] = None,
                          slice_indices: Optional[Dict[str, int]] = None,
                          save_path: Optional[str] = None,
                          title: str = "Multi-view Visualization") -> plt.Figure:
    """
    Create multi-view (axial, sagittal, coronal) visualization.
    
    Args:
        image: 3D medical image
        segmentation: 3D segmentation (optional)
        slice_indices: Dict with slice indices for each view
        save_path: Path to save the plot
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    if slice_indices is None:
        # Use middle slices
        slice_indices = {
            'axial': image.shape[2] // 2,
            'sagittal': image.shape[0] // 2,
            'coronal': image.shape[1] // 2
        }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    views = [
        ('axial', 2, slice_indices['axial']),
        ('sagittal', 0, slice_indices['sagittal']),
        ('coronal', 1, slice_indices['coronal'])
    ]
    
    for i, (view_name, axis, slice_idx) in enumerate(views):
        # Extract slices
        if axis == 0:
            img_slice = image[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :] if segmentation is not None else None
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :] if segmentation is not None else None
        else:
            img_slice = image[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx] if segmentation is not None else None
        
        # Original image
        axes[0, i].imshow(img_slice, cmap='gray')
        axes[0, i].set_title(f'{view_name.title()} - Original')
        axes[0, i].axis('off')
        
        # With segmentation overlay
        axes[1, i].imshow(img_slice, cmap='gray')
        if seg_slice is not None:
            colors = ['transparent', 'red', 'green', 'blue', 'yellow']
            seg_cmap = ListedColormap(colors[:len(np.unique(seg_slice))])
            masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
            axes[1, i].imshow(masked_seg, cmap=seg_cmap, alpha=0.5)
        
        axes[1, i].set_title(f'{view_name.title()} - With Segmentation')
        axes[1, i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_visualization(fig: plt.Figure, 
                      save_path: str, 
                      dpi: int = 300, 
                      format: str = 'png') -> None:
    """
    Save matplotlib figure with proper formatting.
    
    Args:
        fig: matplotlib Figure object
        save_path: Path to save the figure
        dpi: Resolution in dots per inch
        format: Image format
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, format=format, bbox_inches='tight', 
                facecolor='white', edgecolor='none')


def create_segmentation_summary(original_path: str,
                              prediction_path: str,
                              save_dir: str = 'summary',
                              case_name: str = 'case') -> Dict[str, str]:
    """
    Create a comprehensive segmentation summary with multiple visualizations.
    
    Args:
        original_path: Path to original image
        prediction_path: Path to prediction
        save_dir: Directory to save visualizations
        case_name: Name for this case
    
    Returns:
        Dictionary with paths to created visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load images
    original_img = nib.load(original_path).get_fdata()
    prediction_img = nib.load(prediction_path).get_fdata()
    
    saved_files = {}
    
    # Multi-view visualization
    multiview_fig = create_multi_view_plot(
        original_img, prediction_img,
        title=f"Case: {case_name} - Multi-view Segmentation"
    )
    multiview_path = os.path.join(save_dir, f"{case_name}_multiview.png")
    save_visualization(multiview_fig, multiview_path)
    saved_files['multiview'] = multiview_path
    plt.close(multiview_fig)
    
    # Representative slices for each axis
    slice_paths = {}
    for axis, axis_name in [(0, 'sagittal'), (1, 'coronal'), (2, 'axial')]:
        slice_idx = original_img.shape[axis] // 2
        slice_fig = plot_segmentation(
            original_img, prediction_img, slice_idx, axis,
            title=f"Case: {case_name} - {axis_name.title()} View"
        )
        slice_path = os.path.join(save_dir, f"{case_name}_{axis_name}_slice.png")
        save_visualization(slice_fig, slice_path)
        slice_paths[axis_name] = slice_path
        plt.close(slice_fig)
    
    saved_files['slices'] = slice_paths
    
    # 3D analysis
    analysis_3d = create_3d_visualization(
        prediction_img,
        save_path=os.path.join(save_dir, f"{case_name}_3d_analysis.json")
    )
    saved_files['analysis_3d'] = os.path.join(save_dir, f"{case_name}_3d_analysis.json")
    
    return saved_files