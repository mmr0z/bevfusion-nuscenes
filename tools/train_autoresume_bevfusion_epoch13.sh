#!/usr/bin/env bash

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13.py"
WORK_DIR="work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13"

set +u
source ${HOME}/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "bevfusion_blackwell" ]]; then
  conda activate bevfusion_blackwell
fi
set -u

unset CUDA_VISIBLE_DEVICES
unset CUDA_DEVICE_ORDER
unset LD_LIBRARY_PATH

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib

cd ${REPO_ROOT} || exit 1
mkdir -p "$WORK_DIR"

while true; do
  if [[ -f "$WORK_DIR/last_checkpoint" ]]; then
    echo "[auto-resume] Resuming from $WORK_DIR/latest.pth"
    python tools/train.py "$CONFIG" --resume auto "$@"
  else
    echo "[auto-resume] Starting fresh from lidar epoch_13"
    python tools/train.py "$CONFIG" "$@"
  fi

  status=$?
  if [[ $status -eq 0 ]]; then
    echo "[auto-resume] Training finished successfully."
    exit 0
  fi

  if [[ $status -eq 137 || $status -eq 143 ]]; then
    echo "[auto-resume] Process was killed (exit $status). Restarting in 10s."
    sleep 10
    continue
  fi

  echo "[auto-resume] Training exited with code $status. Not restarting."
  exit $status
done
