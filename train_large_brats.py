#!/usr/bin/env python3
"""
Training script optimized for large BraTS dataset (1,479 subjects).
Designed for RTX 4080 Super with advanced training techniques.
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LargeBraTSDataset(Dataset):
    """Optimized dataset for large BraTS dataset."""
    
    def __init__(self, data_dir, target_size=(128, 128, 64), augment=True, cache_size=100):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augment = augment
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Find all subjects
        self.subjects = []
        for subject_dir in self.data_dir.iterdir():
            if subject_dir.is_dir():
                # Check for required files
                modalities = ['t1', 't1ce', 't2', 'flair']
                files_exist = True
                for mod in modalities:
                    if not (subject_dir / f"{subject_dir.name}_{mod}.nii.gz").exists():
                        files_exist = False
                        break
                
                seg_file = subject_dir / f"{subject_dir.name}_seg.nii.gz"
                if files_exist and seg_file.exists():
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
        
        # Check cache first
        if subject_id in self.cache:
            self.cache_hits += 1
            return self.cache[subject_id]
        
        self.cache_misses += 1
        
        try:
            # Load multi-modal image
            modalities = ['t1', 't1ce', 't2', 'flair']
            images = []
            
            for mod in modalities:
                img_path = subject_dir / f"{subject_id}_{mod}.nii.gz"
                if img_path.exists():
                    img_nii = nib.load(img_path)
                    img_data = img_nii.get_fdata().astype(np.float32)
                    # Normalize
                    img_data = self._normalize_image(img_data)
                    # Resize
                    img_data = self._resize_volume(img_data, self.target_size)
                    images.append(img_data)
                else:
                    # Create zero-filled placeholder if modality missing
                    images.append(np.zeros(self.target_size, dtype=np.float32))
            
            # Stack modalities
            image = np.stack(images, axis=0)  # Shape: (4, H, W, D)
            
            # Load segmentation
            seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata().astype(np.long)
            seg_data = self._resize_volume(seg_data, self.target_size)
            
            # Apply augmentation
            if self.augment:
                image, seg_data = self._augment(image, seg_data)
            
            # Convert to tensors
            image_tensor = torch.FloatTensor(image)
            seg_tensor = torch.LongTensor(seg_data)
            
            sample = {
                'image': image_tensor,
                'label': seg_tensor,
                'subject_id': subject_id
            }
            
            # Cache if there's room
            if len(self.cache) < self.cache_size:
                self.cache[subject_id] = sample
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error loading {subject_id}: {e}")
            # Return dummy data
            return {
                'image': torch.zeros((4,) + self.target_size),
                'label': torch.zeros(self.target_size, dtype=torch.long),
                'subject_id': f"error_{subject_id}"
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
    
    def get_cache_stats(self):
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses
        }


class AdvancedUNet3D(nn.Module):
    """Advanced U-Net optimized for large dataset training."""
    
    def __init__(self, in_channels=4, num_classes=4, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, base_filters)
        self.enc2 = self._make_encoder_block(base_filters, base_filters * 2)
        self.enc3 = self._make_encoder_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._make_encoder_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_filters * 8, base_filters * 16, 3, padding=1),
            nn.BatchNorm3d(base_filters * 16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),
            nn.Conv3d(base_filters * 16, base_filters * 16, 3, padding=1),
            nn.BatchNorm3d(base_filters * 16),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = self._make_decoder_block(base_filters * 16, base_filters * 8)
        
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._make_decoder_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._make_decoder_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._make_decoder_block(base_filters * 2, base_filters)
        
        # Final classifier
        self.classifier = nn.Conv3d(base_filters, num_classes, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4[:, :, :d4.size(2), :d4.size(3), :d4.size(4)]], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3[:, :, :d3.size(2), :d3.size(3), :d3.size(4)]], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2[:, :, :d2.size(2), :d2.size(3), :d2.size(4)]], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1[:, :, :d1.size(2), :d1.size(3), :d1.size(4)]], dim=1)
        d1 = self.dec1(d1)
        
        # Classifier
        output = self.classifier(d1)
        return output


def calculate_dice_score(pred, target, num_classes=4):
    """Calculate Dice score for all classes."""
    dice_scores = []
    
    for class_id in range(1, num_classes):  # Skip background
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
        else:
            dice_scores.append(1.0)
    
    return dice_scores


def main():
    print("Large BraTS Dataset Training (1,479 subjects)")
    print("=" * 60)
    
    # Configuration for large dataset
    config = {
        'model': {
            'num_classes': 4,
            'base_filters': 32
        },
        'training': {
            'batch_size': 2,  # Larger batch for better GPU utilization
            'learning_rate': 0.0005,  # Higher LR for large dataset
            'epochs': 100,  # More epochs for large dataset
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'mixed_precision': True
        },
        'data': {
            'target_size': [128, 128, 64],
            'augment': True,
            'cache_size': 200,  # Cache more samples
            'num_workers': 4
        },
        'hardware': {
            'num_workers': 4,
            'pin_memory': True
        },
        'paths': {
            'dataset': 'data/brats_large',
            'checkpoints': './checkpoints_large',
            'logs': './logs_large'
        }
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    # Create datasets
    train_dataset = LargeBraTSDataset(
        f"{config['paths']['dataset']}/train",
        target_size=config['data']['target_size'],
        augment=config['data']['augment'],
        cache_size=config['data']['cache_size']
    )
    
    val_dataset = LargeBraTSDataset(
        f"{config['paths']['dataset']}/val",
        target_size=config['data']['target_size'],
        augment=False,
        cache_size=50
    )
    
    print(f"Training subjects: {len(train_dataset):,}")
    print(f"Validation subjects: {len(val_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validate one at a time for memory
        shuffle=False,
        num_workers=2,
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    model = AdvancedUNet3D(
        in_channels=4,  # Multi-modal input
        num_classes=config['model']['num_classes'],
        base_filters=config['model']['base_filters']
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
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Setup logging
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    writer = SummaryWriter(config['paths']['logs'])
    
    # Training loop
    best_dice = 0
    class_names = ['Necrotic Core', 'Peritumoral Edema', 'Enhancing Tumor']
    
    print(f"\nStarting training on large BraTS dataset...")
    print(f"Target: Dice score > 0.90 (large dataset advantage)")
    print(f"Expected training time: 12-24 hours")
    print()
    
    for epoch in range(1, config['training']['epochs'] + 1):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_dice_scores = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
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
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                if scaler:
                    with autocast():
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
        
        train_dice_per_class = np.mean(train_dice_scores, axis=0)
        val_dice_per_class = np.mean(val_dice_scores, axis=0)
        
        train_dice = np.mean(train_dice_per_class)
        val_dice = np.mean(val_dice_per_class)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch:3d}/{config['training']['epochs']:3d} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Detailed class-wise Dice scores
        print(f"  Class Dice - ", end="")
        for i, (class_name, dice) in enumerate(zip(class_names, val_dice_per_class)):
            print(f"{class_name}: {dice:.3f}", end=" | " if i < len(class_names)-1 else "\n")
        
        # Cache statistics
        cache_stats = train_dataset.get_cache_stats()
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.2f}")
        
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
                'val_dice_per_class': val_dice_per_class.tolist(),
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['paths']['checkpoints'], 'best_large_brats_model.pth'))
            print(f"  *** NEW BEST MODEL *** Dice: {best_dice:.4f}")
        
        # Check target achievement
        if val_dice > 0.90:
            print(f"  *** EXCELLENT PERFORMANCE *** Dice: {val_dice:.4f} > 0.90")
        elif val_dice > 0.85:
            print(f"  *** TARGET ACHIEVED *** Dice: {val_dice:.4f} > 0.85")
        
        print()
    
    writer.close()
    
    print("=" * 60)
    print("LARGE BRATS TRAINING COMPLETED!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Training time: {time.time() - start_time:.1f} seconds")
    
    if best_dice > 0.90:
        print("OUTSTANDING: Achieved state-of-the-art performance!")
    elif best_dice > 0.85:
        print("SUCCESS: Target Dice score > 0.85 achieved!")
    else:
        print("Good progress - large dataset should achieve >0.85 with more training")
    
    print(f"Best model: {config['paths']['checkpoints']}/best_large_brats_model.pth")


if __name__ == '__main__':
    main()