#!/usr/bin/env python3
"""
Minimal training test to verify we can start training with sample data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
from pathlib import Path

def simple_3d_unet():
    """Create a very simple 3D U-Net for testing."""
    class Simple3DUNet(nn.Module):
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
    
    return Simple3DUNet()

def load_sample_data():
    """Load one sample for testing."""
    train_dir = Path("data/samples/train")
    
    # Find first image and segmentation pair
    img_files = [f for f in train_dir.glob("*.nii.gz") if not f.name.endswith("_seg.nii.gz")]
    
    if not img_files:
        raise FileNotFoundError("No sample images found")
    
    img_file = img_files[0]
    seg_file = train_dir / img_file.name.replace(".nii.gz", "_seg.nii.gz")
    
    # Load image
    img_nii = nib.load(img_file)
    img_data = img_nii.get_fdata()
    
    # Load segmentation
    seg_nii = nib.load(seg_file)
    seg_data = seg_nii.get_fdata()
    
    # Normalize image
    img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)
    
    # Convert to tensors
    image = torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    segmentation = torch.LongTensor(seg_data).unsqueeze(0)  # Add batch dim
    
    print(f"Loaded sample: {img_file.name}")
    print(f"Image shape: {image.shape}")
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Segmentation classes: {torch.unique(segmentation)}")
    
    return image, segmentation

def run_training_test():
    """Run a minimal training test."""
    print("Starting minimal training test...")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model
    model = simple_3d_unet()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load data
    try:
        image, segmentation = load_sample_data()
        image = image.to(device)
        segmentation = segmentation.to(device)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return False
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (just a few iterations)
    model.train()
    
    print("\nStarting training...")
    for epoch in range(3):  # Just 3 epochs for testing
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(image)
        loss = criterion(outputs, segmentation)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Test inference
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predictions = torch.argmax(outputs, dim=1)
        
        print(f"\nInference test:")
        print(f"Output shape: {outputs.shape}")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Predicted classes: {torch.unique(predictions)}")
    
    # Save a test checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }
    torch.save(checkpoint, 'checkpoints/minimal_test_model.pth')
    print("Saved test checkpoint: checkpoints/minimal_test_model.pth")
    
    print("\nMINIMAL TRAINING TEST PASSED!")
    return True

def main():
    print("=" * 60)
    print("MINIMAL TRAINING TEST")
    print("=" * 60)
    
    try:
        success = run_training_test()
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS: System is ready for training!")
            print("\nYou can now:")
            print("1. Run full training with the main script")
            print("2. Use the web application")
            print("3. Run inference on medical images")
            return True
        else:
            print("\nFAILED: Please check the errors above")
            return False
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)