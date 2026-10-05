#!/usr/bin/env bash
# Setup for WSL2 + RTX PRO 3000 Blackwell (compute capability 12.0 / sm_120). UNTESTED.
# The PF-RPN README pins torch 2.1 + CUDA 11.8. That build has NO sm_120 kernels.
# Use torch >= 2.7 with CUDA 12.8 wheels, and build mmcv ops from source.
set -euo pipefail
nvidia-smi
conda create -n pf-rpn python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pf-rpn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"  # expect (12, 0)
pip install mmengine "numpy<2" pytest
git clone --depth 1 https://github.com/open-mmlab/mmcv.git -b v2.1.0 /tmp/mmcv
(cd /tmp/mmcv && TORCH_CUDA_ARCH_LIST="12.0" MMCV_WITH_OPS=1 pip install -v -e . --no-build-isolation)
git clone --depth 1 https://github.com/tangqh03/PF-RPN.git ../PF-RPN
(cd ../PF-RPN && pip install "setuptools>=69.0.3,<81" && pip install -v -e . --no-build-isolation && pip install -r requirements.txt)
mkdir -p ../PF-RPN/checkpoints
