#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANTSPLAT_DIR="${ROOT}/third_party/InstantSplat"
ENV_NAME="${INSTANTSPLAT_ENV_NAME:-instantsplat}"
PYTHON_VERSION="${INSTANTSPLAT_PYTHON_VERSION:-3.10.13}"

export CC="${CC:-/usr/bin/gcc-11}"
export CXX="${CXX:-/usr/bin/g++-11}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"

echo "[setup] InstantSplat root: ${INSTANTSPLAT_DIR}"
echo "[setup] Conda env: ${ENV_NAME}"

if [ ! -d "${INSTANTSPLAT_DIR}/.git" ]; then
  echo "InstantSplat repo not found at ${INSTANTSPLAT_DIR}" >&2
  exit 1
fi

echo "[setup] Initializing git submodules..."
git -C "${INSTANTSPLAT_DIR}" submodule update --init --recursive

mkdir -p "${INSTANTSPLAT_DIR}/mast3r/checkpoints"
CKPT="${INSTANTSPLAT_DIR}/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ ! -f "${CKPT}" ]; then
  echo "[setup] Downloading MASt3R checkpoint..."
  wget -O "${CKPT}" \
    https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Creating conda env ${ENV_NAME} (python=${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" "cmake=3.14.0" -y
fi

ENV_PYTHON="/home/hongjiacheng/miniconda3/envs/${ENV_NAME}/bin/python"
ENV_PIP="/home/hongjiacheng/miniconda3/envs/${ENV_NAME}/bin/pip"

echo "[setup] Installing PyTorch (cu124)..."
"${ENV_PIP}" install --upgrade pip
"${ENV_PIP}" install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[setup] Installing Python requirements..."
"${ENV_PIP}" install -r "${INSTANTSPLAT_DIR}/requirements.txt"

echo "[setup] Building CUDA extensions..."
"${ENV_PIP}" install "${INSTANTSPLAT_DIR}/submodules/simple-knn"
"${ENV_PIP}" install "${INSTANTSPLAT_DIR}/submodules/diff-gaussian-rasterization"
"${ENV_PIP}" install "${INSTANTSPLAT_DIR}/submodules/fused-ssim"

echo "[setup] Building croco RoPE CUDA kernels..."
(
  cd "${INSTANTSPLAT_DIR}/croco/models/curope"
  "${ENV_PYTHON}" setup.py build_ext --inplace
)

echo "[setup] Installing worldact InstantSplat bridge..."
"${ENV_PIP}" install -e "${INSTANTSPLAT_DIR}/worldact_bridge"

echo
echo "[setup] Done."
echo "  Python: ${ENV_PYTHON}"
echo "  Export for worldact:"
echo "    export INSTANT_SPLAT_PYTHON=${ENV_PYTHON}"
echo "    export INSTANT_SPLAT_ROOT=${INSTANTSPLAT_DIR}"
