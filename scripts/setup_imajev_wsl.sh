#!/usr/bin/env bash
# Prepare imajev-4b on WSL (separate venv; PF-RPN needs Python 3.10, imajev needs >= 3.11). UNTESTED.
# Downloads only. It does NOT start the server and does NOT use the GPU,
# so it is safe to run while SPEC-001 latency benchmarks run. Needs about 10 GB disk.
set -euo pipefail
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv --python 3.11 ~/.venvs/imajev
source ~/.venvs/imajev/bin/activate
uv pip install pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
DIR="${IMAJEV_DIR:-$HOME/imajev}"
[ -d "$DIR" ] || git clone --depth 1 https://github.com/mohit67890/imajev "$DIR"
cd "$DIR"
pip install -e ".[serve,torch]"
python scripts/download_model.py --model 4b                       # base Qwen3.5-4B, pinned revision
hf download mohit67890/imajev-4b --local-dir adapters/imajev-4b   # adapter + calibration files
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(0))"
echo "READY. Start later (not during a latency benchmark):"
echo "  cd $DIR && PYTHONPATH=src:scripts python scripts/playground/server.py --backend torch \\"
echo "    --model-bundle artifacts/model-qwen4b.json --adapter adapters/imajev-4b --rotations 4 \\"
echo "    --calibration adapters/imajev-4b/calibration-rot4.json --model-name imajev-4b --port 8765"
