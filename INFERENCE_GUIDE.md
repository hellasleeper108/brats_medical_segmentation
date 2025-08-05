# Model Usage and Inference Guide

## Overview

This guide provides comprehensive instructions for using the trained BraTS 2021 medical image segmentation model that achieved **85.51% Dice score**. The model is optimized for brain tumor segmentation from 3D MRI volumes.

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- Required dependencies installed (`pip install -r requirements.txt`)
- Trained model checkpoint (`checkpoints_brats2021_rescue/best_model_rescue.pth`)

## Quick Start

### Single Image Inference

```bash
# Basic inference with the best model
python inference/run_inference.py \
    --input path/to/your/brain_scan.nii.gz \
    --model checkpoints_brats2021_rescue/best_model_rescue.pth \
    --output predictions/

# With visualization
python inference/run_inference.py \
    --input path/to/your/brain_scan.nii.gz \
    --model checkpoints_brats2021_rescue/best_model_rescue.pth \
    --output predictions/ \
    --visualize
```

### Batch Processing

```bash
# Process entire directory
python inference/run_inference.py \
    --batch path/to/nifti/files/ \
    --model checkpoints_brats2021_rescue/best_model_rescue.pth \
    --output batch_predictions/
```

### Test Time Augmentation (TTA)

```bash
# Enhanced accuracy with TTA
python inference/run_inference.py \
    --input path/to/brain_scan.nii.gz \
    --model checkpoints_brats2021_rescue/best_model_rescue.pth \
    --output predictions/ \
    --tta
```

## Python API Usage

### Basic Prediction

```python
from inference.predictor import MedicalImagePredictor

# Initialize predictor with the best model
predictor = MedicalImagePredictor(
    model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
    model_type='unet'
)

# Perform prediction
prediction, metrics, processing_time = predictor.predict('brain_scan.nii.gz')

print(f"Processing time: {processing_time:.2f}s")
print(f"Classes found: {len(np.unique(prediction))}")
```

### Advanced Usage with Configuration

```python
import yaml
from inference.predictor import MedicalImagePredictor

# Load custom configuration
with open('configs/brats2021_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with configuration
predictor = MedicalImagePredictor(
    model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
    model_type='unet',
    config=config
)

# Get model information
model_info = predictor.get_model_info()
print(f"Model parameters: {model_info['total_parameters']:,}")
print(f"Model size: {model_info['model_size_mb']:.2f} MB")

# Prediction with TTA for higher accuracy
prediction, metrics, time = predictor.predict_with_tta('brain_scan.nii.gz')
```

### Batch Processing API

```python
from inference.predictor import MedicalImagePredictor

predictor = MedicalImagePredictor(
    model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
    model_type='unet'
)

# Process multiple files
image_paths = ['scan1.nii.gz', 'scan2.nii.gz', 'scan3.nii.gz']
results = predictor.batch_predict(image_paths, output_dir='batch_results/')

# Check results
for result in results:
    if result['success']:
        print(f"✅ {result['input_path']}: {result['processing_time']:.2f}s")
    else:
        print(f"❌ {result['input_path']}: {result['error']}")
```

## Input Requirements

### Image Format
- **Format**: NIfTI (.nii or .nii.gz files)
- **Dimensions**: 3D volumes (any size - will be automatically resized)
- **Modalities**: Single-channel (T1, T1ce, T2, or FLAIR)
- **Data Type**: Float32 or Int16

### File Naming
- No specific naming requirements
- Supports standard medical imaging naming conventions
- Examples: `patient_001_t1.nii.gz`, `BraTS_001.nii.gz`

## Output Format

### Segmentation Labels
The model outputs 4-class segmentation masks:
- **0**: Background (healthy tissue)
- **1**: Necrotic and non-enhancing tumor core
- **2**: Peritumoral edema  
- **3**: GD-enhancing tumor

### Output Files
- **Prediction**: `{filename}_prediction.nii.gz`
- **Metrics**: `inference_results.yaml` (detailed metrics)
- **Visualizations**: `visualizations/` directory (if --visualize used)

## Model Performance

### Achieved Metrics (BraTS 2021)
- **Dice Score**: 85.51% (exceeds clinical threshold)
- **Training Dataset**: 1,251 subjects
- **Validation Dataset**: 219 subjects
- **Architecture**: 3D U-Net (22.6M parameters)

### Processing Speed
- **GPU (RTX 4080 Super)**: ~2-5 seconds per volume
- **CPU**: ~30-60 seconds per volume
- **Memory**: ~4-6GB GPU memory per inference

## Hardware Requirements

### Recommended (GPU)
- **GPU**: RTX 3080/4080 or better (≥10GB VRAM)
- **RAM**: 16GB system memory
- **Storage**: 2GB for model + space for predictions

### Minimum (CPU)
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 32GB system memory
- **Processing**: 10x slower than GPU

## Advanced Features

### Test Time Augmentation (TTA)
Improves prediction accuracy by averaging multiple augmented predictions:

```python
# Available TTA transforms
tta_transforms = ['flip_x', 'flip_y', 'flip_z']

prediction, metrics, time = predictor.predict_with_tta(
    'brain_scan.nii.gz', 
    tta_transforms=tta_transforms
)
```

### Custom Preprocessing
```python
from data.transforms import get_inference_transforms

# Custom transforms
transforms = get_inference_transforms(
    roi_size=(128, 128, 128),
    spacing=(1.0, 1.0, 1.0),
    intensity_range=(-1000, 1000)
)
```

### Memory Optimization
```python
# For limited VRAM
predictor = MedicalImagePredictor(
    model_path='checkpoints_brats2021_rescue/best_model_rescue.pth',
    model_type='unet',
    device='cuda'  # or 'cpu' for CPU inference
)

# Clear GPU cache between predictions
torch.cuda.empty_cache()
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Use CPU inference
predictor = MedicalImagePredictor(..., device='cpu')

# Or reduce batch size (contact developers for batch size options)
```

**2. Model File Not Found**
```bash
# Check if model exists
ls -la checkpoints_brats2021_rescue/best_model_rescue.pth

# Download or retrain if missing
python train_brats2021_rescue.py
```

**3. Invalid Input Format**
```bash
# Convert DICOM to NIfTI
dcm2niix input_dicom_folder/

# Check image properties
python -c "import nibabel as nib; img = nib.load('scan.nii.gz'); print(img.shape, img.header.get_data_dtype())"
```

**4. Poor Segmentation Quality**
- Ensure input is brain MRI (not CT or other modalities)
- Check image orientation and spacing
- Try TTA for improved accuracy
- Verify image quality and contrast

### Performance Optimization

**GPU Memory**
```python
# Monitor GPU usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
torch.cuda.empty_cache()  # Clear cache
```

**Batch Processing**
```python
# Process multiple files efficiently
results = predictor.batch_predict(file_list, output_dir='results/')
```

## Clinical Integration

### DICOM Workflow
```bash
# 1. Convert DICOM to NIfTI
dcm2niix -o nifti_output/ dicom_input/

# 2. Run inference
python inference/run_inference.py --input nifti_output/brain.nii.gz --model checkpoints_brats2021_rescue/best_model_rescue.pth

# 3. Convert back to DICOM if needed (using external tools)
```

### Quality Assurance
- **Visual inspection**: Always review segmentation overlays
- **Confidence metrics**: Check prediction confidence scores
- **Clinical validation**: Verify with radiologist review
- **Performance monitoring**: Track inference times and accuracy

## Web Interface

For interactive usage, use the web application:

```bash
cd web_app/
python app.py
# Open http://localhost:5000
```

Features:
- Drag-and-drop file upload
- Real-time segmentation
- Interactive visualization
- Results download

## Model Updates

The current model achieved 85.51% Dice score. For updates or retraining:

```bash
# Retrain with new data
python train_brats2021_rescue.py

# Fine-tune existing model
python train_brats2021_rescue.py --resume checkpoints_brats2021_rescue/best_model_rescue.pth
```

## Support

For technical support or questions:
1. Check this guide and project documentation
2. Review common issues in troubleshooting section
3. Examine inference logs for error details
4. Consider model retraining if performance degrades

## Model Citation

When using this model in research or clinical applications:

```
BraTS 2021 Brain Tumor Segmentation Model
- Architecture: 3D U-Net
- Performance: 85.51% Dice Score
- Training Dataset: BraTS 2021 (1,251 subjects)
- Framework: PyTorch with MONAI transforms
```