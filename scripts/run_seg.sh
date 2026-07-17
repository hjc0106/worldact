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
OBJECTS="${OBJECTS:-}"

VLM_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
VLM_API_KEY="${OPENAI_API_KEY:-}"
VLM_MODEL="${OPENAI_MODEL:-gpt-5.5}"

_vlm_supports_vision() {
  if [[ "${VLM_SUPPORTS_VISION:-}" == "1" ]]; then
    return 0
  fi
  if [[ "${VLM_SUPPORTS_VISION:-}" == "0" ]]; then
    return 1
  fi
  local base_lc
  base_lc="$(printf '%s' "${VLM_API_BASE}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${base_lc}" == *"deepseek.com"* ]]; then
    return 1
  fi
  return 0
}

USE_MANUAL_OBJECTS=0
if [[ -n "${OBJECTS}" ]]; then
  USE_MANUAL_OBJECTS=1
fi
for arg in "$@"; do
  if [[ "${arg}" == "--objects" ]]; then
    USE_MANUAL_OBJECTS=1
    break
  fi
done

VLM_ARGS=()
if [[ "${USE_MANUAL_OBJECTS}" -eq 0 ]]; then
  if [[ -z "${VLM_API_BASE}" || -z "${VLM_API_KEY}" ]]; then
    echo "Missing VLM credentials. Set OPENAI_API_BASE or OPENAI_BASE_URL, and OPENAI_API_KEY." >&2
    exit 1
  fi
  if ! _vlm_supports_vision; then
    cat >&2 <<EOF
Configured VLM endpoint (${VLM_API_BASE}) does not support vision/image_url requests.
DeepSeek chat completions are text-only and cannot analyze rendered keyframes.

Options:
  1) Bypass VLM with manual SAM3 prompts:
     OBJECTS='round table' bash scripts/run_seg.sh --objects "vase on table" "wooden chair"
  2) Use a vision-capable api_base/model (e.g. OpenAI gpt-4o) in your ENV_FILE.
  3) Force attempt anyway: VLM_SUPPORTS_VISION=1 bash scripts/run_seg.sh
EOF
    exit 1
  fi
  VLM_ARGS=(
    --use-vlm
    --vlm-api-base "${VLM_API_BASE}"
    --vlm-api-key "${VLM_API_KEY}"
    --vlm-model "${VLM_MODEL}"
  )
fi
OBJECT_ARGS=()
if [[ -n "${OBJECTS}" ]]; then
  OBJECT_ARGS=(--objects "${OBJECTS}")
fi

python scripts/run_decomposition.py \
  --ply "${PLY_PATH}" \
  --config "${CONFIG_PATH}" \
  "${VLM_ARGS[@]}" \
  "${OBJECT_ARGS[@]}" \
  --save-keyframes \
  --save-sam3-masks \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
