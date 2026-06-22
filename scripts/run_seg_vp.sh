#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/openai.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"

PLY_PATH="${PLY_PATH:-scene.ply}"
CONFIG_PATH="${CONFIG_PATH:-configs/scenes/outdoor.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/outdoor_town_scene}"
DEVICE="${DEVICE:-cuda}"

VLM_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
VLM_API_KEY="${OPENAI_API_KEY:-}"
VLM_MODEL="${OPENAI_MODEL:-gpt-5.5}"

if [[ -z "${VLM_API_BASE}" || -z "${VLM_API_KEY}" ]]; then
  echo "Missing VLM credentials. Set OPENAI_API_BASE or OPENAI_BASE_URL, and OPENAI_API_KEY." >&2
  exit 1
fi

# chair
# stage1.view_propagation.enabled=true \
# stage1.view_propagation.save_debug=true \
# stage1.view_propagation.fallback_to_sam3=false \
# stage1.view_propagation.candidate_unique_assignment=true \
# stage1.view_propagation.fusion_mode=intersection \
# stage1.view_propagation.max_hops=2 \
# stage1.view_propagation.max_angle_deg=45 \
# stage1.view_propagation.min_view_overlap=0.2 \
# stage1.view_propagation.min_assignment_iou=0.45 \
# stage1.view_propagation.min_assignment_visibility=0.25 \
# stage1.prefilter.enabled=true \
# stage1.vlm.num_keyframes=16 \
# stage1.camera_gen.num_cameras=24 \

python scripts/run_decomposition.py \
  --ply "${PLY_PATH}" \
  --config "${CONFIG_PATH}" \
  --use-vlm \
  --vlm-api-base "${VLM_API_BASE}" \
  --vlm-api-key "${VLM_API_KEY}" \
  --vlm-model "${VLM_MODEL}" \
  --save-keyframes \
  --save-sam3-masks \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
    stage1.view_propagation.enabled=true \
    stage1.view_propagation.save_debug=true \
    stage1.view_propagation.fallback_to_sam3=true \
    stage1.view_propagation.candidate_unique_assignment=true \
    stage1.view_propagation.fusion_mode=intersection \
    stage1.view_propagation.max_hops=4 \
    stage1.view_propagation.max_angle_deg=75 \
    stage1.view_propagation.min_view_overlap=0.08 \
    stage1.view_propagation.min_assignment_iou=0.3 \
    stage1.view_propagation.min_assignment_visibility=0.1 \
    stage1.vlm.num_keyframes=16 \
    stage1.camera_gen.num_cameras=24 \
  "$@"
