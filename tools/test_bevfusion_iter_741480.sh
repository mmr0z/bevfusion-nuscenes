#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13.py"
CHECKPOINT="${REPO_ROOT}/work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13/iter_741480.pth"
WORK_DIR="${REPO_ROOT}/work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_1xb1-6e_nus-3d-quality-ft-epoch13/test_iter_741480"
NUSCENES_RESULTS_DIR="${WORK_DIR}/nuscenes_results"
THESIS_RESULTS_DIR="${WORK_DIR}/thesis_metrics"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda initialization script not found: ${CONDA_SH}" >&2
  exit 1
fi

set +u
source "${CONDA_SH}"
conda activate bevfusion_blackwell
set -u

unset CUDA_VISIBLE_DEVICES
unset CUDA_DEVICE_ORDER
unset LD_LIBRARY_PATH

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib

mkdir -p "${WORK_DIR}"
cd "${REPO_ROOT}"

python tools/test.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --work-dir "${WORK_DIR}" \
  "$@" \
  --cfg-options \
  "test_evaluator.jsonfile_prefix=${NUSCENES_RESULTS_DIR}"

METRICS_SUMMARY="${NUSCENES_RESULTS_DIR}/pred_instances_3d/metrics_summary.json"
if [[ ! -f "${METRICS_SUMMARY}" ]]; then
  echo "nuScenes metrics were not produced: ${METRICS_SUMMARY}" >&2
  exit 1
fi

python tools/analysis_tools/summarize_nuscenes_metrics.py \
  "${METRICS_SUMMARY}" \
  --output-dir "${THESIS_RESULTS_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --work-dir "${WORK_DIR}" \
  --validation-samples 6019

echo "Official nuScenes artifacts: ${NUSCENES_RESULTS_DIR}"
echo "Thesis-ready metric tables: ${THESIS_RESULTS_DIR}"
