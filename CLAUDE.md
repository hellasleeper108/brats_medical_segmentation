# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI-powered medical image segmentation system focused on brain tumor segmentation using deep learning. The project implements U-Net and Vision Transformer (ViT) architectures for 3D medical image analysis, specifically targeting the BraTS (Brain Tumor Segmentation) dataset.

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script (includes dependency installation and directory creation)
python setup.py

# Quick installation test
python setup.py --test-only
```

### Training Models
```bash
# Train U-Net model
python training/train.py --config configs/config.yaml

# Train with specific dataset
python train_brats2021.py
python train_samples.py  # Use sample data for testing

# Quick training test (minimal epochs)
python minimal_train_test.py
python quick_train_test.py
```

### Running Inference
```bash
# Use the BEST trained model (85.51% Dice performance)
python inference/run_inference.py --input path/to/image.nii.gz --model checkpoints_brats2021_rescue/best_model_rescue.pth

# Batch inference on directory
python inference/run_inference.py --batch path/to/nifti/files/ --model checkpoints_brats2021_rescue/best_model_rescue.pth

# Enhanced accuracy with Test Time Augmentation
python inference/run_inference.py --input path/to/image.nii.gz --model checkpoints_brats2021_rescue/best_model_rescue.pth --tta

# With visualization
python inference/run_inference.py --input path/to/image.nii.gz --model checkpoints_brats2021_rescue/best_model_rescue.pth --visualize
```

### Web Application
```bash
# Start Flask web interface for image upload and visualization
python web_app/app.py
# Accessible at http://localhost:5000
```

### Data Management
```bash
# Download sample datasets
python data/download_datasets.py --dataset samples

# Download full BraTS datasets
python auto_download_datasets.py
python acquire_datasets.py

# Setup specific BraTS versions
python setup_brats2021_official.py
python setup_tcia_brats.py
python setup_large_brats.py
```

### Testing and Validation
```bash
# Run simple training test
python test_training_simple.py

# Prepare sample training data
python prepare_sample_training.py
```

### Model Deployment
```bash
# Quantize models for deployment
python deployment/quantization.py --model_path checkpoints/best_model.pth
```

## Project Architecture

### Core Components

1. **Models** (`models/`):
   - `unet.py`: 3D U-Net implementation with skip connections and batch normalization
   - `vit.py`: Vision Transformer for 3D medical images with patch-based attention

2. **Data Pipeline** (`data/`):
   - `dataset.py`: BraTS dataset loader with MONAI transforms
   - `transforms.py`: Medical image preprocessing and augmentation
   - `download_datasets.py`: Automated dataset acquisition

3. **Training System** (`training/`):
   - `train.py`: Main training loop with mixed precision and checkpointing
   - `losses.py`: Medical segmentation losses (Dice, combined losses)

4. **Inference Engine** (`inference/`):
   - `predictor.py`: Model inference wrapper
   - `run_inference.py`: Command-line inference tool

5. **Utilities** (`utils/`):
   - `metrics.py`: Segmentation metrics (Dice, Hausdorff distance)
   - `checkpoint.py`: Model saving/loading utilities
   - `device.py`: CUDA/CPU device management
   - `visualization.py`: Medical image visualization tools

### Data Structure
- **Input**: 3D MRI volumes (T1, T1ce, T2, FLAIR modalities)
- **Output**: 4-class segmentation masks (background, necrotic core, edema, enhancing tumor)
- **Format**: NIfTI (.nii.gz) files, DICOM support available

### Model Architectures
- **U-Net**: 3D encoder-decoder with skip connections, optimized for medical segmentation
- **Vision Transformer**: Patch-based transformer with CNN hybrid architecture
- **Training**: Mixed precision training, gradient clipping, cosine annealing scheduler

## Configuration Management

Primary configuration in `configs/config.yaml`:
- Model parameters (architecture, input size, classes)
- Training settings (batch size, learning rate, epochs)
- Data configuration (dataset paths, splits, augmentation)
- Hardware settings (CUDA, memory optimization for RTX 4080 Super)

## Dataset Information

The project supports multiple medical imaging datasets:
- **BraTS 2021/2023**: Brain tumor segmentation (primary dataset)
- **COVID-CT**: COVID-19 lung segmentation
- **Custom datasets**: Configurable for other medical imaging tasks

Dataset acquisition is automated through dedicated scripts that handle downloads from TCIA, Synapse, and other medical imaging repositories.

## Hardware Optimization

Optimized for RTX 4080 Super (16GB VRAM):
- Batch size 2 for 3D volumes (128x128x128)
- Mixed precision training enabled
- Memory-efficient data loading with caching
- Gradient checkpointing for larger models

## ACHIEVED PERFORMANCE 🏆

**BREAKTHROUGH RESULTS on BraTS 2021:**
- **Final Dice Score**: **85.51%** (EXCEEDS 85% target!)
- **Dataset**: Full BraTS 2021 (1,251 training + 219 validation subjects)
- **Training Time**: 80 epochs in ~6 hours on RTX 4080 Super
- **Architecture**: 3D U-Net with 22.6M parameters
- **Clinical Grade**: Research-quality medical AI performance

## Performance Targets

- **Dice Score**: >0.85 for brain tumor segmentation ✅ **ACHIEVED: 85.51%**
- **Inference Speed**: Real-time processing capabilities
- **Memory Usage**: <16GB VRAM during training ✅ **ACHIEVED: ~12GB peak**

## Critical Lessons Learned

### BraTS Label Remapping (CRITICAL!)
**Issue**: BraTS uses labels [0,1,2,4] but PyTorch expects [0,1,2,3]
**Solution**: Always remap label 4 → 3 in dataset loading
```python
label_remapped[label == 4] = 3  # Essential for BraTS compatibility
```

### Training Stability
**Lesson**: OneCycleLR can cause training collapse with high max_lr
**Solution**: Use conservative learning rates (0.0001) with linear decay for stability
**Best Practice**: Monitor for sudden Dice drops indicating training instability

### Hardware Optimization
**Success**: RTX 4080 Super (16GB) perfectly sized for BraTS training
**Configuration**: Batch size 2, mixed precision, 4 workers
**Memory Usage**: ~12GB peak (75% utilization - optimal)

## Deployment Options

- **Web Interface**: Flask application for clinical use (`python web_app/app.py`)
- **Command Line**: Full-featured inference script (`inference/run_inference.py`)
- **Python API**: Programmatic access via `MedicalImagePredictor` class
- **Batch Processing**: High-throughput inference pipelines with TTA support
- **Model Quantization**: INT8 optimization for edge deployment
- **Docker**: Containerized deployment with GPU support

## Inference Documentation

**📖 Complete inference guide available in: `INFERENCE_GUIDE.md`**

This comprehensive guide covers:
- Quick start examples and command-line usage
- Python API with code examples
- Input/output format specifications
- Performance optimization and troubleshooting
- Clinical integration workflows
- Hardware requirements and recommendations