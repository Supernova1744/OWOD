#!/usr/bin/env bash
# Setup for WSL2 + RTX PRO 3000 Blackwell (compute capability 12.0 / sm_120). UNTESTED.
# The PF-RPN README pins torch 2.1 + CUDA 11.8. That build has NO sm_120 kernels.
# Use torch >= 2.7 with CUDA 12.8 wheels, and build mmcv ops from source.
set -euo pipefail
nvidia-smi
# No conda needed: uv fetches Python 3.10 and makes a venv.
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv --python 3.10 ~/.venvs/pf-rpn
source ~/.venvs/pf-rpn/bin/activate
uv pip install pip setuptools wheel   # mmcv/PF-RPN build steps call pip and setuptools
# mmcv ops build needs nvcc whose version matches the CUDA of the torch wheel (cu128 -> 12.8).
# A newer system toolkit (for example 13.x) breaks the build. Use 12.8 side by side.
export CUDA_HOME=/usr/local/cuda-12.8
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "CUDA 12.8 toolkit not found at $CUDA_HOME. Install it (toolkit only, no driver):"
  echo "  https://developer.nvidia.com/cuda-12-8-0-download-archive  (Linux > x86_64 > WSL-Ubuntu > deb network)"
  echo "  then: sudo apt-get install -y cuda-toolkit-12-8"
  echo "Your other CUDA versions stay installed."
  exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"  # expect (12, 0)
pip install mmengine "numpy<2" pytest
git clone --depth 1 https://github.com/open-mmlab/mmcv.git -b v2.1.0 /tmp/mmcv
(cd /tmp/mmcv && TORCH_CUDA_ARCH_LIST="12.0" MMCV_WITH_OPS=1 pip install -v -e . --no-build-isolation)
git clone --depth 1 https://github.com/tangqh03/PF-RPN.git ../PF-RPN
(cd ../PF-RPN && pip install "setuptools>=69.0.3,<81" && pip install -v -e . --no-build-isolation && pip install -r requirements.txt)
mkdir -p ../PF-RPN/checkpoints
