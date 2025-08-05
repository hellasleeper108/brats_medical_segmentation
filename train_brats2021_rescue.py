#!/usr/bin/env python3
"""
RESCUE VERSION - BraTS 2021 training with safer learning rate parameters
Modified to prevent learning rate explosion and training collapse
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

class UNet3D(nn.Module):
    """3D U-Net for medical image segmentation - STABLE VERSION"""
    
    def __init__(self, in_channels=4, num_classes=4, base_filters=32):
        super(UNet3D, self).__init__()
        
        # Encoder with batch normalization for stability
        self.enc1 = self._make_layer(in_channels, base_filters)
        self.enc2 = self._make_layer(base_filters, base_filters * 2)
        self.enc3 = self._make_layer(base_filters * 2, base_filters * 4)
        self.enc4 = self._make_layer(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = self._make_layer(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, 2)
        self.dec4 = self._make_layer(base_filters * 16, base_filters * 8)
        
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, 2)
        self.dec3 = self._make_layer(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, 2)
        self.dec2 = self._make_layer(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, 2)
        self.dec1 = self._make_layer(base_filters * 2, base_filters)
        
        # Final classification layer
        self.final = nn.Conv3d(base_filters, num_classes, 1)
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)  # Light dropout for stability
        )
    
    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization for stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool3d(2)(e1))
        e3 = self.enc3(nn.MaxPool3d(2)(e2))
        e4 = self.enc4(nn.MaxPool3d(2)(e3))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool3d(2)(e4))
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

class BraTS2021Dataset(Dataset):
    """Dataset for BraTS 2021 with enhanced error handling"""
    
    def __init__(self, data_dir, target_size=(128, 128, 64), augment=True):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augment = augment
        
        # Find all subjects with robust error handling
        self.subjects = []
        
        # Check if subjects are directly in data_dir or in subdirectories
        direct_subjects = [d for d in self.data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("BraTS2021_")]
        
        if direct_subjects:
            # Flat structure
            for subject_dir in direct_subjects:
                if self._validate_subject(subject_dir):
                    self.subjects.append(subject_dir)
        else:
            # Nested structure - subjects in cancer center subdirectories
            cancer_centers = ['ACRIN-FMISO-Brain', 'CPTAC-GBM', 'IvyGAP', 'TCGA-GBM', 
                            'TCGA-LGG', 'UCSF-PDGM', 'UPENN-GBM', 'new-not-previously-in-TCIA']
            
            for center_dir in self.data_dir.iterdir():
                if center_dir.is_dir() and center_dir.name in cancer_centers:
                    for subject_dir in center_dir.iterdir():
                        if subject_dir.is_dir() and subject_dir.name.startswith("BraTS2021_"):
                            if self._validate_subject(subject_dir):
                                self.subjects.append(subject_dir)
        
        print(f"Found {len(self.subjects)} valid subjects in {data_dir}")
        
        if len(self.subjects) == 0:
            raise ValueError(f"No valid BraTS subjects found in {data_dir}")
    
    def _validate_subject(self, subject_dir):
        """Validate that subject has all required modalities"""
        modalities = ['t1', 't1ce', 't2', 'flair']
        for mod in modalities:
            file_path = subject_dir / f"{subject_dir.name}_{mod}.nii.gz"
            if not file_path.exists():
                return False
        return True
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_dir = self.subjects[idx]
        
        try:
            # Load all modalities
            modalities = ['t1', 't1ce', 't2', 'flair']
            images = []
            
            for mod in modalities:
                file_path = subject_dir / f"{subject_dir.name}_{mod}.nii.gz"
                img = nib.load(str(file_path))
                data = img.get_fdata().astype(np.float32)
                
                # Normalize
                if data.std() > 0:
                    data = (data - data.mean()) / data.std()
                
                # Resize to target size
                data = self._resize_volume(data, self.target_size)
                images.append(data)
            
            # Stack modalities
            image = np.stack(images, axis=0)  # Shape: (4, D, H, W)
            
            # Load segmentation if available (for training set)
            seg_path = subject_dir / f"{subject_dir.name}_seg.nii.gz"
            if seg_path.exists():
                seg_img = nib.load(str(seg_path))
                label = seg_img.get_fdata().astype(np.int64)
                label = self._resize_volume(label, self.target_size, is_label=True)
                
                # CRITICAL FIX: Remap BraTS labels [0,1,2,4] to [0,1,2,3]
                label = self._remap_brats_labels(label)
            else:
                # No segmentation (validation set)
                label = np.zeros(self.target_size, dtype=np.int64)
            
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'subject_id': subject_dir.name
            }
            
        except Exception as e:
            print(f"Error loading subject {subject_dir.name}: {e}")
            # Return zero tensors as fallback
            return {
                'image': torch.zeros(4, *self.target_size),
                'label': torch.zeros(self.target_size, dtype=torch.long),
                'subject_id': subject_dir.name
            }
    
    def _resize_volume(self, volume, target_size, is_label=False):
        """Resize volume to target size with proper interpolation"""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        
        if is_label:
            # Use nearest neighbor for labels
            resized = zoom(volume, zoom_factors, order=0)
        else:
            # Use linear interpolation for images
            resized = zoom(volume, zoom_factors, order=1)
        
        return resized
    
    def _remap_brats_labels(self, label):
        """Remap BraTS labels from [0,1,2,4] to [0,1,2,3] for PyTorch compatibility"""
        # BraTS original: 0=background, 1=necrotic, 2=edema, 4=enhancing
        # PyTorch needs: 0=background, 1=necrotic, 2=edema, 3=enhancing
        label_remapped = np.copy(label)
        label_remapped[label == 4] = 3  # Remap enhancing tumor from 4 to 3
        return label_remapped

def calculate_dice_scores(pred, target, num_classes=4):
    """Calculate Dice scores for each class with numerical stability"""
    dice_scores = []
    
    # Convert to numpy for calculation
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    for class_id in range(1, num_classes):  # Skip background
        pred_class = (pred == class_id).astype(np.float32)
        target_class = (target == class_id).astype(np.float32)
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(float(dice))
        else:
            dice_scores.append(1.0)  # Perfect score if both are empty
    
    return dice_scores

def main():
    print("BraTS 2021 RESCUE TRAINING - STABLE VERSION")
    print("=" * 60)
    
    # RESCUE CONFIGURATION - Conservative parameters to prevent collapse
    config = {
        'model': {
            'num_classes': 4,
            'base_filters': 32,
            'in_channels': 4
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 0.0001,  # MUCH LOWER - was 0.0005
            'epochs': 80,
            'weight_decay': 0.01,
            'gradient_clip': 0.5,  # STRICTER - was 1.0
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
            'checkpoints': './checkpoints_brats2021_rescue',
            'logs': './logs_brats2021_rescue'
        }
    }
    
    # Create directories
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    
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
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    model = UNet3D(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        base_filters=config['model']['base_filters']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.0]).to(device))
    
    # RESCUE OPTIMIZER - Much more conservative
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        eps=1e-8  # Numerical stability
    )
    
    # RESCUE SCHEDULER - Linear decay instead of OneCycleLR
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=config['training']['epochs']
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # TensorBoard
    writer = SummaryWriter(config['paths']['logs'])
    
    # Training loop
    best_dice = 0.0
    
    print(f"\nStarting RESCUE training...")
    print(f"Conservative LR: {config['training']['learning_rate']}")
    print(f"Stricter gradient clipping: {config['training']['gradient_clip']}")
    print(f"Linear LR decay (no OneCycleLR)")
    print("=" * 60)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        train_dice_scores = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate Dice scores
                with torch.no_grad():
                    pred = torch.argmax(outputs, dim=1)
                    dice_scores = calculate_dice_scores(pred, labels)
                    train_dice_scores.extend(dice_scores)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{np.mean(dice_scores):.4f}' if dice_scores else '0.0000'
                })
        
        # Step scheduler
        scheduler.step()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = np.mean(train_dice_scores) if train_dice_scores else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Dice', avg_train_dice, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Dice: {avg_train_dice:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint every 10 epochs or if best performance
        if (epoch + 1) % 10 == 0 or avg_train_dice > best_dice:
            if avg_train_dice > best_dice:
                best_dice = avg_train_dice
                checkpoint_name = 'best_model_rescue.pth'
                print(f"  New best Dice: {best_dice:.4f}")
            else:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}_rescue.pth'
            
            checkpoint_path = os.path.join(config['paths']['checkpoints'], checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'config': config
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_name}")
        
        print("-" * 60)
    
    print(f"\nRescue training completed!")
    print(f"Best Dice score: {best_dice:.4f}")
    
    writer.close()

if __name__ == '__main__':
    main()