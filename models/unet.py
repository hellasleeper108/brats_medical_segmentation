import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock3D(nn.Module):
    """Encoder block with convolution and max pooling."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(EncoderBlock3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock3D(nn.Module):
    """Decoder block with upsampling and skip connections."""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DecoderBlock3D, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(in_channels, out_channels, dropout)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Ensure dimensions match for concatenation
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """3D U-Net for medical image segmentation."""
    
    def __init__(self, in_channels=1, num_classes=4, features=[32, 64, 128, 256, 512], dropout=0.2):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        
        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(len(features)):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoders.append(EncoderBlock3D(in_ch, features[i], dropout))
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2, dropout)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features)):
            in_ch = reversed_features[i] * 2 if i == 0 else reversed_features[i-1]
            out_ch = reversed_features[i]
            self.decoders.append(DecoderBlock3D(in_ch * 2, out_ch, dropout))
        
        # Final classifier
        self.classifier = nn.Conv3d(features[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = list(reversed(skip_connections))
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # Final classification
        x = self.classifier(x)
        return x
    
    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb