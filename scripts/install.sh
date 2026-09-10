#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
: "${CUDA_HOME:?Ustaw CUDA_HOME na katalog CUDA Toolkit z nvcc (lokalnie użyto 12.9).}"
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
python -m pip install 'setuptools<81' wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/bevfusion.txt
mkdir -p third_party
if [[ ! -d third_party/mmcv ]]; then
  git clone https://github.com/open-mmlab/mmcv.git third_party/mmcv
fi
(
  cd third_party/mmcv
  git checkout 57c4e25e06e2d4f8a9357c84bcd24089a284dc88
  MMCV_WITH_OPS=1 python -m pip install -v --no-build-isolation --no-deps -e .
)
python -m pip install --no-build-isolation --no-deps -e .
python projects/BEVFusion/setup.py build_ext --inplace
