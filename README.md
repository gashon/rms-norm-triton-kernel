# RMSNorm Triton Kernel

Custom GPU kernel for Root Mean Square layer normalization layer with fused operations
- Implemented with [Triton compiler](https://github.com/openai/triton) for high performance and parallel computations on GPUs.
- Includes both forward and backward passes of RMS layer normalization.

## Getting Started

**Requirements**

```bash
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
torchvision==0.16.0+cu121
triton==2.1.0
```

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/simudt/Tri-RMSNorm
cd Tri-RMSNorm
pip3 install -e .
```
