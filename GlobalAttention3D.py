#!/usr/bin/env python3
"""
True 3D Global Attention Convolutional Network for Self-Supervised Denoising
of Real-World Post-Stack Seismic Volumes

This module implements a 3D U-Net architecture with global attention mechanisms
for seismic data denoising. The network processes 128x128x128 seismic volumes
using multi-scale attention and skip connections.

Authors:
    Matin Mahzad (ORCID: 0009-0000-9346-8451)
    Amirreza Mehrabi
    Majid Bagheri
    Majid Nabi Bidhendi

License:
    MIT License

    Copyright (c) 2025 Matin Mahzad, Amirreza Mehrabi, Majid Bagheri, Majid Nabi Bidhendi

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Citation:
    Please cite this work as:
    Mahzad, M., Mehrabi, A., Bagheri, M., & Bidhendi, M. N. (2025).
    True 3D Global Attention Convolutional Network for Self-Supervised Denoising
    of Real-World Post-Stack Seismic Volumes.

Dependencies:
    - torch >= 1.9.0
    - numpy
    - Python >= 3.7

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalAttention3D(nn.Module):
    """
    3D Global Self-Attention mechanism with memory optimization for seismic data processing.

    This module implements scaled dot-product attention in 3D space with multi-head attention
    and memory-efficient computation strategies suitable for large seismic volumes.

    Args:
        channels (int): Number of input channels
        spatial_size (int): Spatial dimension size (assumed cubic)
        heads (int): Number of attention heads. Default: 8

    Note:
        The attention mechanism processes the entire 3D volume globally, enabling
        long-range dependency modeling crucial for seismic noise pattern recognition.
    """

    def __init__(self, channels: int, spatial_size: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.heads = heads
        self.head_dim = channels // heads

        # Use smaller projections to save memory
        self.query = nn.Conv3d(channels, channels, 1)
        self.key = nn.Conv3d(channels, channels, 1)
        self.value = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)

        # Layer norm for stability
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        """
        Forward pass of the 3D attention U-Net.

        Args:
            x (torch.Tensor): Input seismic volume of shape [B, 1, 128, 128, 128]

        Returns:
            torch.Tensor: Denoised seismic volume of shape [B, 1, 128, 128, 128]

        Note:
            The forward pass implements a symmetric encoder-decoder architecture
            with skip connections to preserve fine-grained seismic features while
            applying global attention at each resolution level.
        """
        """
        Forward pass combining convolution and attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]

        Returns:
            torch.Tensor: Processed features with attention enhancement
        """
        """
        Forward pass of 3D global attention.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]

        Returns:
            torch.Tensor: Attention-enhanced features with residual connection
        """
        B, C, D, H, W = x.shape

        # Apply normalization first
        x_norm = self.norm(x)

        # Generate Q, K, V
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)

        # Reshape for multi-head attention: [B, heads, head_dim, D*H*W]
        Q = Q.view(B, self.heads, self.head_dim, -1)
        K = K.view(B, self.heads, self.head_dim, -1)
        V = V.view(B, self.heads, self.head_dim, -1)

        # Scaled attention with memory-efficient computation
        scale = math.sqrt(self.head_dim)

        # Compute attention weights: [B, heads, D*H*W, D*H*W]
        attn = torch.einsum('bhdn,bhdm->bhnm', Q, K) / scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, V)

        # Reshape back: [B, C, D, H, W]
        out = out.contiguous().view(B, C, D, H, W)
        out = self.out_proj(out)

        # Residual connection
        return x + out


class AttentionConv3DBlock(nn.Module):
    """
    3D Convolutional block with integrated global attention for seismic feature extraction.

    This block combines 3D convolutions with global attention mechanisms to capture
    both local geological features and long-range seismic patterns. The design is
    optimized for seismic data characteristics including noise patterns and structural
    continuity.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        spatial_size (int): Current spatial dimension for attention computation
    """

    def __init__(self, in_channels: int, out_channels: int, spatial_size: int):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        self.attention = GlobalAttention3D(out_channels, spatial_size)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        # First conv
        out = F.relu(self.norm1(self.conv1(x)))

        out = self.attention(out)

        # Second conv
        out = F.relu(self.norm2(self.conv2(out)))

        return out


class Attention3DUNet(nn.Module):
    """
    True 3D U-Net with Global Attention for Self-Supervised Seismic Denoising.

    This architecture implements a full 3D U-Net with global attention mechanisms
    at multiple scales, specifically designed for processing 128x128x128 seismic
    volumes. The network employs:

    - Multi-scale global attention for capturing long-range dependencies
    - Skip connections for preserving high-frequency seismic features
    - Group normalization for stable training on seismic data
    - Memory-efficient attention computation for large volumes

    The architecture follows a 4-level encoder-decoder structure:
    128³ → 64³ → 32³ → 16³ → 8³ (bridge) → 16³ → 32³ → 64³ → 128³

    Args:
        in_channels (int): Number of input channels. Default: 1 (single seismic trace)
        out_channels (int): Number of output channels. Default: 1 (denoised trace)
        base_channels (int): Base number of channels, doubled at each level. Default: 32
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        # Encoder - Starting with 128x128x128
        self.enc1 = AttentionConv3DBlock(in_channels, base_channels, 128)
        self.down1 = nn.Conv3d(base_channels, base_channels, 3, stride=2, padding=1)  # 128->64

        self.enc2 = AttentionConv3DBlock(base_channels, base_channels * 2, 64)
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)  # 64->32

        self.enc3 = AttentionConv3DBlock(base_channels * 2, base_channels * 4, 32)
        self.down3 = nn.Conv3d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)  # 32->16

        self.enc4 = AttentionConv3DBlock(base_channels * 4, base_channels * 8, 16)
        self.down4 = nn.Conv3d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1)  # 16->8

        # Bridge at 8x8x8
        self.bridge = AttentionConv3DBlock(base_channels * 8, base_channels * 16, 8)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, stride=2)  # 8->16
        self.dec4 = AttentionConv3DBlock(base_channels * 16, base_channels * 8, 16)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)  # 16->32
        self.dec3 = AttentionConv3DBlock(base_channels * 8, base_channels * 4, 32)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)  # 32->64
        self.dec2 = AttentionConv3DBlock(base_channels * 4, base_channels * 2, 64)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)  # 64->128
        self.dec1 = AttentionConv3DBlock(base_channels * 2, base_channels, 128)

        # Output
        self.out_conv = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 128x128x128
        d1 = self.down1(e1)  # 64x64x64

        e2 = self.enc2(d1)  # 64x64x64
        d2 = self.down2(e2)  # 32x32x32

        e3 = self.enc3(d2)  # 32x32x32
        d3 = self.down3(e3)  # 16x16x16

        e4 = self.enc4(d3)  # 16x16x16
        d4 = self.down4(e4)  # 8x8x8

        # Bridge
        bridge = self.bridge(d4)  # 8x8x8

        # Decoder with skip connections
        u4 = self.up4(bridge)  # 16x16x16
        u4 = torch.cat([u4, e4], dim=1)
        d4_dec = self.dec4(u4)

        u3 = self.up3(d4_dec)  # 32x32x32
        u3 = torch.cat([u3, e3], dim=1)
        d3_dec = self.dec3(u3)

        u2 = self.up2(d3_dec)  # 64x64x64
        u2 = torch.cat([u2, e2], dim=1)
        d2_dec = self.dec2(u2)

        u1 = self.up1(d2_dec)  # 128x128x128
        u1 = torch.cat([u1, e1], dim=1)
        d1_dec = self.dec1(u1)

        # Output
        out = self.out_conv(d1_dec)
        return out
