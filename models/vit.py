import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class PatchEmbedding3D(nn.Module):
    """3D Patch embedding for Vision Transformer."""
    
    def __init__(self, img_size=(128, 128, 128), patch_size=(8, 8, 8), in_channels=1, embed_dim=768):
        super(PatchEmbedding3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Number of patches
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        # Patch projection
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        # Project patches and flatten
        x = self.proj(x)  # B, embed_dim, D//patch_size[0], H//patch_size[1], W//patch_size[2]
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # B, num_patches, embed_dim
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer for medical image segmentation."""
    
    def __init__(self, img_size=(128, 128, 128), patch_size=(8, 8, 8), in_channels=1, 
                 num_classes=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer3D, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder for segmentation
        self.decoder = self._build_decoder()
        
        # Initialize weights
        self._init_weights()
        
    def _build_decoder(self):
        """Build decoder for dense prediction."""
        patch_dims = [s // p for s, p in zip(self.img_size, self.patch_size)]
        
        decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.ReLU(inplace=True),
            Rearrange('b (d h w) c -> b c d h w', d=patch_dims[0], h=patch_dims[1], w=patch_dims[2]),
            nn.ConvTranspose3d(self.embed_dim // 4, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, self.num_classes, kernel_size=1)
        )
        
        return decoder
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, num_patches, embed_dim
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Remove class token for segmentation
        x = x[:, 1:]
        
        # Decode to segmentation map
        x = self.decoder(x)
        
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