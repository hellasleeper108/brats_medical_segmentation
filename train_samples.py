#!/usr/bin/env python3
"""
Training script specifically for sample data.
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

# Simple working U-Net implementation
class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
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
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)
        
        # Classifier
        x = self.classifier(x)
        return x


class SampleDataset(Dataset):
    """Dataset for sample brain images."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all image files
        self.image_files = []
        for img_file in self.data_dir.glob("*.nii.gz"):
            if not img_file.name.endswith("_seg.nii.gz"):
                seg_file = self.data_dir / img_file.name.replace(".nii.gz", "_seg.nii.gz")
                if seg_file.exists():
                    self.image_files.append((img_file, seg_file))
        
        print(f"Found {len(self.image_files)} sample pairs in {data_dir}")
    
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
        
        # Simple preprocessing
        img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)
        
        # Resize to target size (simple center crop/pad)
        target_size = (96, 96, 48)
        img_data = self._resize_volume(img_data, target_size)
        seg_data = self._resize_volume(seg_data, target_size)
        
        # Convert to tensors
        image = torch.FloatTensor(img_data).unsqueeze(0)  # Add channel dimension
        segmentation = torch.LongTensor(seg_data)
        
        return {
            'image': image,
            'label': segmentation,
            'filename': img_path.name
        }
    
    def _resize_volume(self, volume, target_size):
        """Simple resize by center cropping or padding."""
        current_size = volume.shape
        
        # Calculate crop/pad for each dimension
        result = volume
        
        for i in range(3):
            current = current_size[i]
            target = target_size[i]
            
            if current > target:
                # Crop
                start = (current - target) // 2
                if i == 0:
                    result = result[start:start+target, :, :]
                elif i == 1:
                    result = result[:, start:start+target, :]
                else:
                    result = result[:, :, start:start+target]
            elif current < target:
                # Pad
                pad_before = (target - current) // 2
                pad_after = target - current - pad_before
                
                pad_width = [(0, 0)] * 3
                pad_width[i] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode='constant', constant_values=0)
        
        return result


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    dice_scores = []
    
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
        
        # Calculate Dice score
        with torch.no_grad():
            pred_probs = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)
            
            # Calculate Dice for each class (excluding background)
            dice = 0
            num_classes = 0
            for class_id in range(1, outputs.shape[1]):  # Skip background
                pred_class = (pred_labels == class_id).float()
                true_class = (labels == class_id).float()
                
                intersection = (pred_class * true_class).sum()
                union = pred_class.sum() + true_class.sum()
                
                if union > 0:
                    dice += (2.0 * intersection) / (union + 1e-8)
                    num_classes += 1
            
            if num_classes > 0:
                dice_scores.append(dice.item() / num_classes)
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    
    return avg_loss, avg_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    dice_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate Dice score
            pred_probs = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)
            
            dice = 0
            num_classes = 0
            for class_id in range(1, outputs.shape[1]):
                pred_class = (pred_labels == class_id).float()
                true_class = (labels == class_id).float()
                
                intersection = (pred_class * true_class).sum()
                union = pred_class.sum() + true_class.sum()
                
                if union > 0:
                    dice += (2.0 * intersection) / (union + 1e-8)
                    num_classes += 1
            
            if num_classes > 0:
                dice_scores.append(dice.item() / num_classes)
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    
    return avg_loss, avg_dice


def main():
    print("Starting training on sample data...")
    
    # Load configuration
    with open('configs/sample_training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create datasets
    train_dataset = SampleDataset('data/samples/train')
    val_dataset = SampleDataset('data/samples/val')
    
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
    model = SimpleUNet3D(
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
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Training
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch:3d}/{config['training']['epochs']:3d} - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
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
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['paths']['checkpoints'], 'best_model.pth'))
            print(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Regular checkpoint
        if epoch % config['logging']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['paths']['checkpoints'], f'checkpoint_epoch_{epoch}.pth'))
    
    writer.close()
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    
    if best_dice > 0.85:
        print("🎉 SUCCESS: Achieved target Dice score > 0.85!")
    else:
        print(f"Target Dice score (>0.85) not reached, but training completed successfully.")
    
    print(f"Best model saved at: {config['paths']['checkpoints']}/best_model.pth")


if __name__ == '__main__':
    main()