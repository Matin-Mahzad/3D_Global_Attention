# True 3D Global Attention Convolutional Network for Self-Supervised Denoising of Real-World Post-Stack Seismic Volumes
# Overview
This repository contains the implementation of a novel 3D U-Net architecture with global attention mechanisms specifically designed for self-supervised denoising of post-stack seismic volumes. The network uses multi-scale attention and skip connections to effectively remove noise while preserving geological structures.
Authors
•	Matin Mahzad (ORCID: 0009-0000-9346-8451)
•	Amirreza Mehrabi
•	Majid Bagheri
•	Majid Nabi Bidhendi
Architecture
The network implements a multi-level encoder-decoder architecture with global attention at each scale:
Input → Encoder (4 levels) → Bridge → Decoder (4 levels) → Output
Key Features
•	Global 3D Attention: Captures long-range dependencies crucial for seismic noise pattern recognition
•	Multi-scale Processing: Hierarchical feature extraction at multiple resolutions
•	Skip Connections: Preserves fine-grained seismic features during reconstruction
•	Memory Optimization: Efficient attention computation for large 3D volumes
•	Group Normalization: Stable training on seismic data characteristics
Requirements
•	Python >= 3.7
•	PyTorch >= 1.9.0
•	NumPy
Installation
git clone https://github.com/Matin-Mahzad/3D_Global_Attention
cd 3D_Global_Attention
pip install torch numpy
Model Architecture Details
GlobalAttention3D
•	Implements scaled dot-product attention in 3D space
•	Multi-head attention with memory-efficient computation
•	Processes entire seismic volumes globally
AttentionConv3DBlock
•	Combines 3D convolutions with global attention
•	Captures both local geological features and long-range patterns
•	Optimized for seismic data characteristics
Attention3DUNet
•	Full 3D U-Net with integrated attention mechanisms
•	Hierarchical processing with progressive downsampling and upsampling
•	Self-supervised learning compatible architecture
Usage
import torch
from attention_unet_3d import Attention3DUNet

# Initialize model
model = Attention3DUNet(in_channels=1, out_channels=1, base_channels=32)

# Process seismic volume
seismic_input = torch.randn(1, 1, D, H, W)  # D, H, W are volume dimensions
denoised_output = model(seismic_input)
# License
This project is licensed under the MIT License - see the LICENSE file for details.
# Citation
If you use this code in your research, please cite:
@article{mahzad2025seismic,
  title={True 3D Global Attention Convolutional Network for Self-Supervised Denoising of Real-World Post-Stack Seismic Volumes},
  author={Mahzad, Matin and Mehrabi, Amirreza and Bagheri, Majid and Bidhendi, Majid Nabi},
  year={2025}
}
# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
# Contact
For questions or collaboration opportunities, please contact the corresponding authors.
