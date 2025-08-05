#!/usr/bin/env python3
"""
Training script for official BraTS 2021 dataset.
Optimized for multi-modal input (4 channels) and RTX 4080 Super.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BraTS2021Dataset(Dataset):
    """Dataset for official BraTS 2021 multi-modal data."""
    
    def __init__(self, data_dir, target_size=(128, 128, 64), augment=True):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augment = augment
        
        # Find all subjects (nested in cancer center directories)
        self.subjects = []
        
        # First check if subjects are directly in data_dir (flat structure)
        direct_subjects = [d for d in self.data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("BraTS2021_")]
        
        if direct_subjects:
            # Flat structure - subjects directly in data_dir
            for subject_dir in direct_subjects:
                modalities = ['t1', 't1ce', 't2', 'flair']
                files_exist = True
                for mod in modalities:
                    if not (subject_dir / f"{subject_dir.name}_{mod}.nii.gz").exists():
                        files_exist = False
                        break
                
                if files_exist:
                    self.subjects.append(subject_dir)
        else:
            # Nested structure - subjects in cancer center subdirectories
            cancer_centers = ['ACRIN-FMISO-Brain', 'CPTAC-GBM', 'IvyGAP', 'TCGA-GBM', 
                            'TCGA-LGG', 'UCSF-PDGM', 'UPENN-GBM', 'new-not-previously-in-TCIA']
            
            for center_dir in self.data_dir.iterdir():
                if center_dir.is_dir() and center_dir.name in cancer_centers:
                    for subject_dir in center_dir.iterdir():
                        if subject_dir.is_dir() and subject_dir.name.startswith("BraTS2021_"):
                            # Check for required modalities
                            modalities = ['t1', 't1ce', 't2', 'flair']
                            files_exist = True
                            for mod in modalities:
                                if not (subject_dir / f"{subject_dir.name}_{mod}.nii.gz").exists():
                                    files_exist = False
                                    break
                            
                            if files_exist:
                                self.subjects.append(subject_dir)
        
        print(f"Found {len(self.subjects)} subjects in {data_dir}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_dir = self.subjects[idx]
        subject_id = subject_dir.name
        
        try:
            # Load multi-modal images (4 channels)
            modalities = ['t1', 't1ce', 't2', 'flair']
            images = []
            
            for mod in modalities:
                img_path = subject_dir / f"{subject_id}_{mod}.nii.gz"
                img_nii = nib.load(img_path)
                img_data = img_nii.get_fdata().astype(np.float32)
                
                # Normalize
                img_data = self._normalize_image(img_data)
                # Resize
                img_data = self._resize_volume(img_data, self.target_size)
                images.append(img_data)
            
            # Stack modalities (4 channels)
            image = np.stack(images, axis=0)  # Shape: (4, H, W, D)
            
            # Load segmentation (if exists)
            seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
            if seg_path.exists():
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata().astype(np.long)
                seg_data = self._resize_volume(seg_data, self.target_size)
                # Convert BraTS labels to our format
                # BraTS: 0=background, 1=necrotic, 2=edema, 4=enhancing
                # Our format: 0=background, 1=necrotic, 2=edema, 3=enhancing
                seg_data[seg_data == 4] = 3  # Convert enhancing tumor from 4 to 3
                # Ensure all values are in range [0, 3]
                seg_data = np.clip(seg_data, 0, 3)
            else:
                # Create dummy segmentation for validation subjects
                seg_data = np.zeros(self.target_size, dtype=np.long)
            
            # Apply augmentation
            if self.augment:
                image, seg_data = self._augment(image, seg_data)
            
            # Convert to tensors (make copies to ensure contiguous memory)
            image_tensor = torch.FloatTensor(image.copy())
            seg_tensor = torch.LongTensor(seg_data.copy())
            
            return {
                'image': image_tensor,
                'label': seg_tensor,
                'subject_id': subject_id,
                'has_segmentation': seg_path.exists()
            }
            
        except Exception as e:
            self.logger.error(f"Error loading {subject_id}: {e}")
            # Return dummy data
            return {
                'image': torch.zeros(4, *self.target_size),
                'label': torch.zeros(*self.target_size, dtype=torch.long),
                'subject_id': f"error_{subject_id}",
                'has_segmentation': False
            }
    
    def _normalize_image(self, image):
        """Normalize image intensity."""
        # Remove outliers
        p1, p99 = np.percentile(image[image > 0], [1, 99])
        image = np.clip(image, p1, p99)
        
        # Z-score normalization
        mean = np.mean(image[image > 0])
        std = np.std(image[image > 0])
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size."""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        
        if volume.dtype in [np.uint8, np.int32, np.int64]:
            # Nearest neighbor for segmentation
            resized = zoom(volume, zoom_factors, order=0)
        else:
            # Linear interpolation for images
            resized = zoom(volume, zoom_factors, order=1)
        
        return resized
    
    def _augment(self, image, segmentation):
        """Apply data augmentation."""
        if not self.augment:
            return image, segmentation
        
        # Random flip
        if np.random.random() > 0.5:
            image = np.flip(image, axis=1)  # Flip along width
            segmentation = np.flip(segmentation, axis=0)
        
        # Random rotation (small angles)
        if np.random.random() > 0.7:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-5, 5)
            for i in range(image.shape[0]):
                image[i] = rotate(image[i], angle, axes=(0, 1), reshape=False, order=1)
            segmentation = rotate(segmentation, angle, axes=(0, 1), reshape=False, order=0)
        
        # Intensity augmentation for images only
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            image = image * contrast + brightness
        
        return image, segmentation


class BraTS_UNet3D(nn.Module):
    """Simple U-Net optimized for BraTS multi-modal input."""
    
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv3d(64, num_classes, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        x = self.pool1(e1)
        
        e2 = self.enc2(x)
        x = self.pool2(e2)
        
        e3 = self.enc3(x)
        x = self.pool3(e3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        
        # Classifier
        x = self.classifier(x)
        return x


def calculate_dice_score(pred, target, num_classes=4):
    """Calculate Dice score for all classes."""
    dice_scores = []
    
    for class_id in range(1, num_classes):  # Skip background
        pred_class = (pred == class_id).astype(np.float32)
        target_class = (target == class_id).astype(np.float32)
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(float(dice))
        else:
            dice_scores.append(1.0)
    
    return dice_scores


def main():
    print("BraTS 2021 Official Dataset Training")
    print("=" * 50)
    
    # Configuration optimized for BraTS 2021
    config = {
        'model': {
            'num_classes': 4,
            'base_filters': 32,
            'in_channels': 4  # Multi-modal input
        },
        'training': {
            'batch_size': 2,  # Optimized for RTX 4080 Super
            'learning_rate': 0.0005,
            'epochs': 80,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'mixed_precision': True
        },
        'data': {
            'target_size': [128, 128, 64],
            'augment': True
        },
        'hardware': {
            'num_workers': 4,
            'pin_memory': True
        },
        'paths': {
            'dataset': '../PKG - RSNA-ASNR-MICCAI-BraTS-2021/RSNA-ASNR-MICCAI-BraTS-2021',
            'checkpoints': './checkpoints_brats2021',
            'logs': './logs_brats2021'
        }
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Create datasets
    train_dataset = BraTS2021Dataset(
        f"{config['paths']['dataset']}/BraTS2021_TrainingSet",
        target_size=config['data']['target_size'],
        augment=config['data']['augment']
    )
    
    val_dataset = BraTS2021Dataset(
        f"{config['paths']['dataset']}/BraTS2021_ValidationSet", 
        target_size=config['data']['target_size'],
        augment=False
    )
    
    print(f"Training subjects: {len(train_dataset)}")
    print(f"Validation subjects: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    model = BraTS_UNet3D(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.0]).to(device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'] * 10,
        steps_per_epoch=len(train_loader),
        epochs=config['training']['epochs']
    )
    
    # Mixed precision training
    scaler = GradScaler('cuda') if config['training']['mixed_precision'] else None
    
    # Setup logging
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    writer = SummaryWriter(config['paths']['logs'])
    
    # Training loop
    best_dice = 0
    class_names = ['Necrotic Core', 'Peritumoral Edema', 'Enhancing Tumor']
    
    print(f"\nStarting BraTS 2021 training...")
    print(f"Target: Dice score > 0.90 (official dataset)")
    print(f"Training subjects: {len(train_dataset)}")
    print(f"Multi-modal input: T1, T1ce, T2, FLAIR")
    print()
    
    for epoch in range(1, config['training']['epochs'] + 1):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_dice_scores = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            # Skip samples without segmentation during training
            has_seg = batch['has_segmentation']
            if not has_seg.any():
                continue
                
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Filter out samples without segmentation
            seg_mask = has_seg.bool()
            if seg_mask.sum() == 0:
                continue
                
            images = images[seg_mask]
            labels = labels[seg_mask]
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
            
            # Calculate Dice scores
            with torch.no_grad():
                pred_labels = torch.argmax(outputs, dim=1)
                for i in range(images.size(0)):
                    dice_scores = calculate_dice_score(
                        pred_labels[i].cpu().numpy(),
                        labels[i].cpu().numpy(),
                        num_classes=4
                    )
                    train_dice_scores.append(dice_scores)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Skip samples without segmentation during validation
                has_seg = batch['has_segmentation']
                if not has_seg.any():
                    continue
                
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                if scaler:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                pred_labels = torch.argmax(outputs, dim=1)
                for i in range(images.size(0)):
                    dice_scores = calculate_dice_score(
                        pred_labels[i].cpu().numpy(),
                        labels[i].cpu().numpy(),
                        num_classes=4
                    )
                    val_dice_scores.append(dice_scores)
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if train_dice_scores:
            train_dice_per_class = np.mean(train_dice_scores, axis=0)
            train_dice = np.mean(train_dice_per_class)
        else:
            train_dice_per_class = [0, 0, 0]
            train_dice = 0
        
        if val_dice_scores:
            val_dice_per_class = np.mean(val_dice_scores, axis=0)
            val_dice = np.mean(val_dice_per_class)
        else:
            val_dice_per_class = [0, 0, 0]
            val_dice = 0
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch:3d}/{config['training']['epochs']:3d} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Detailed class-wise Dice scores
        if val_dice_scores:
            print(f"  Class Dice - ", end="")
            for i, (class_name, dice) in enumerate(zip(class_names, val_dice_per_class)):
                print(f"{class_name}: {dice:.3f}", end=" | " if i < len(class_names)-1 else "\n")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_dice, epoch)
        writer.add_scalar('Dice/Val', val_dice, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_dice_per_class': val_dice_per_class.tolist() if val_dice_scores else [0, 0, 0],
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['paths']['checkpoints'], 'best_brats2021_model.pth'))
            print(f"  *** NEW BEST MODEL *** Dice: {best_dice:.4f}")
        
        # Check target achievement
        if val_dice > 0.90:
            print(f"  *** OUTSTANDING PERFORMANCE *** Dice: {val_dice:.4f} > 0.90")
        elif val_dice > 0.85:
            print(f"  *** TARGET ACHIEVED *** Dice: {val_dice:.4f} > 0.85")
        
        print()
    
    writer.close()
    
    print("=" * 50)
    print("BRATS 2021 TRAINING COMPLETED!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    
    if best_dice > 0.90:
        print("OUTSTANDING: State-of-the-art performance achieved!")
    elif best_dice > 0.85:
        print("SUCCESS: Target Dice score > 0.85 achieved!")
    else:
        print("Good progress - continue training for better results")
    
    print(f"Best model: {config['paths']['checkpoints']}/best_brats2021_model.pth")
    print("Multi-modal BraTS segmentation system ready for deployment!")


if __name__ == '__main__':
    main()