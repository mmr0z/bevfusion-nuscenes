#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source tools/bevfusion_blackwell_env.sh"
  exit 1
fi

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ "$(type -t conda 2>/dev/null)" != "function" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
  else
    echo "Cannot find conda initialization script: ${CONDA_SH}"
    return 1
  fi
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "bevfusion_blackwell" ]]; then
  conda activate bevfusion_blackwell >/dev/null 2>&1 || {
    echo "Failed to activate conda environment: bevfusion_blackwell"
    return 1
  }
fi

prepend_path_once() {
  local var_name="$1"
  local entry="$2"
  local current_value="${!var_name-}"
  case ":${current_value}:" in
    *":${entry}:"*) ;;
    *) export "${var_name}=${entry}${current_value:+:${current_value}}" ;;
  esac
}

export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="12.0"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${CONDA_PREFIX}/lib/libgcc_s.so.1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

prepend_path_once LD_LIBRARY_PATH "${CONDA_PREFIX}/lib"
prepend_path_once PYTHONPATH "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "BEVFusion environment loaded for ${CONDA_DEFAULT_ENV}"
