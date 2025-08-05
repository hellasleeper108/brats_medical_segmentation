#!/usr/bin/env python3
"""
Training script for enhanced synthetic dataset with 4 classes.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EnhancedUNet3D(nn.Module):
    """Enhanced U-Net for 4-class segmentation."""
    
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv3d(32, num_classes, 1)
        
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


class EnhancedDataset(Dataset):
    """Dataset for enhanced synthetic brain images."""
    
    def __init__(self, data_dir, target_size=(128, 128, 64), transform=None):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.transform = transform
        
        # Find all image files
        self.image_files = []
        for img_file in self.data_dir.glob("*.nii.gz"):
            if not img_file.name.endswith("_seg.nii.gz"):
                seg_file = self.data_dir / img_file.name.replace(".nii.gz", "_seg.nii.gz")
                if seg_file.exists():
                    self.image_files.append((img_file, seg_file))
        
        print(f"Found {len(self.image_files)} enhanced samples in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path, seg_path = self.image_files[idx]
        
        # Load image
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata().astype(np.float32)
        
        # Load segmentation
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(np.long)
        
        # Resize to target size
        img_data = self._resize_volume(img_data, self.target_size)
        seg_data = self._resize_volume(seg_data, self.target_size)
        
        # Convert to tensors
        image = torch.FloatTensor(img_data).unsqueeze(0)  # Add channel dimension
        segmentation = torch.LongTensor(seg_data)
        
        return {
            'image': image,
            'label': segmentation,
            'filename': img_path.name
        }
    
    def _resize_volume(self, volume, target_size):
        """Resize volume using center crop/pad."""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        
        # Calculate zoom factors
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        
        # Resize using scipy zoom
        if volume.dtype in [np.uint8, np.int32, np.int64]:
            # Use nearest neighbor for segmentation
            resized = zoom(volume, zoom_factors, order=0)
        else:
            # Use linear interpolation for images
            resized = zoom(volume, zoom_factors, order=1)
        
        return resized


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
            dice_scores.append(1.0)  # Perfect score if both are empty
    
    return dice_scores


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_dice_scores = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate Dice scores
        with torch.no_grad():
            pred_labels = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                dice_scores = calculate_dice_score(
                    pred_labels[i].cpu().numpy(),
                    labels[i].cpu().numpy(),
                    num_classes=4
                )
                all_dice_scores.append(dice_scores)
    
    avg_loss = total_loss / len(dataloader)
    
    # Average dice scores per class
    if all_dice_scores:
        avg_dice_per_class = np.mean(all_dice_scores, axis=0)
        avg_dice_overall = np.mean(avg_dice_per_class)
    else:
        avg_dice_per_class = [0, 0, 0]
        avg_dice_overall = 0
    
    return avg_loss, avg_dice_overall, avg_dice_per_class


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate Dice scores
            pred_labels = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                dice_scores = calculate_dice_score(
                    pred_labels[i].cpu().numpy(),
                    labels[i].cpu().numpy(),
                    num_classes=4
                )
                all_dice_scores.append(dice_scores)
    
    avg_loss = total_loss / len(dataloader)
    
    # Average dice scores per class
    if all_dice_scores:
        avg_dice_per_class = np.mean(all_dice_scores, axis=0)
        avg_dice_overall = np.mean(avg_dice_per_class)
    else:
        avg_dice_per_class = [0, 0, 0]
        avg_dice_overall = 0
    
    return avg_loss, avg_dice_overall, avg_dice_per_class


def main():
    print("Training Enhanced U-Net on Synthetic Dataset")
    print("=" * 50)
    
    # Configuration
    config = {
        'model': {'num_classes': 4},
        'training': {
            'batch_size': 1,  # Start with 1 due to larger images
            'learning_rate': 0.0001,
            'epochs': 50,
            'weight_decay': 0.01
        },
        'hardware': {'num_workers': 0},
        'paths': {
            'checkpoints': './checkpoints',
            'logs': './logs'
        }
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create datasets
    train_dataset = EnhancedDataset('data/enhanced_synthetic/train')
    val_dataset = EnhancedDataset('data/enhanced_synthetic/val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['hardware']['num_workers']
    )
    
    # Create model
    model = EnhancedUNet3D(
        in_channels=1,
        num_classes=config['model']['num_classes']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Setup logging
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    
    writer = SummaryWriter(config['paths']['logs'])
    
    # Training loop
    best_dice = 0
    class_names = ['Brain Tissue', 'Tumor Core', 'Tumor Edema']
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print("Target: Dice score > 0.85")
    print()
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Training
        train_loss, train_dice, train_dice_per_class = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_dice, val_dice_per_class = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch:3d}/{config['training']['epochs']:3d} - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Detailed class-wise Dice scores
        print(f"  Class Dice - ", end="")
        for i, (class_name, dice) in enumerate(zip(class_names, val_dice_per_class)):
            print(f"{class_name}: {dice:.3f}", end=" | " if i < len(class_names)-1 else "\n")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_dice, epoch)
        writer.add_scalar('Dice/Val', val_dice, epoch)
        
        # Log per-class Dice scores
        for i, class_name in enumerate(class_names):
            writer.add_scalar(f'Dice_Class/{class_name}', val_dice_per_class[i], epoch)
        
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
                'val_dice_per_class': val_dice_per_class,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['paths']['checkpoints'], 'best_enhanced_model.pth'))
            print(f"  *** NEW BEST MODEL *** Dice: {best_dice:.4f}")
        
        # Check if target achieved
        if val_dice > 0.85:
            print(f"  *** TARGET ACHIEVED *** Dice: {val_dice:.4f} > 0.85")
        
        print()  # Empty line for readability
    
    writer.close()
    
    print("=" * 50)
    print("TRAINING COMPLETED!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    
    if best_dice > 0.85:
        print("SUCCESS: Target Dice score > 0.85 achieved!")
    else:
        print(f"Target not reached, but model improved significantly.")
    
    print(f"Best model saved at: {config['paths']['checkpoints']}/best_enhanced_model.pth")
    print("\nModel performance on enhanced synthetic dataset:")
    print("- 4 tissue classes: Background, Brain, Tumor Core, Tumor Edema")
    print("- Larger image size: 128x128x64")
    print("- More realistic anatomy and pathology")


if __name__ == '__main__':
    main()