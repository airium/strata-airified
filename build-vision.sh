#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
VISION_BUILD_DIR=${VISION_BUILD_DIR:-"${ROOT}/build/vision-sm75"}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda-13.4}
BUILD_JOBS=${BUILD_JOBS:-8}
[[ "$VISION_BUILD_DIR" == /* ]] || VISION_BUILD_DIR="${ROOT}/${VISION_BUILD_DIR}"
fresh=()
build_extra=()
for arg in "$@"; do
  case "$arg" in
    --rebuild) fresh=(--fresh); build_extra=(--clean-first) ;;
    -h|--help) printf 'Usage: bash build-vision.sh [--rebuild]\nBuild the pinned CPU/CUDA SM75 picture reader.\n'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
python3 "${ROOT}/tools/apply_patches.py" --check
SOURCE="${ROOT}/vendor/strata/tools/vision"
mkdir -p "$VISION_BUILD_DIR"
VISION_BUILD_DIR=$(cd -- "$VISION_BUILD_DIR" && pwd -P)
if [[ -f "${VISION_BUILD_DIR}/CMakeCache.txt" ]]; then
  while IFS='=' read -r key value; do
    case "$key" in
      CMAKE_HOME_DIRECTORY:INTERNAL) [[ "$value" == "$SOURCE" ]] || fresh=(--fresh) ;;
      CMAKE_CACHEFILE_DIR:INTERNAL) [[ "$value" == "$VISION_BUILD_DIR" ]] || fresh=(--fresh) ;;
    esac
  done < "${VISION_BUILD_DIR}/CMakeCache.txt"
fi
cmake "${fresh[@]}" -S "$SOURCE" -B "$VISION_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE=ON -DSTRATA_VISION_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_TOOLKIT}/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_TOOLKIT" \
  -DCMAKE_CUDA_ARCHITECTURES=75-real -DLLAMA_DIR="${ROOT}/vendor/llama.cpp"
cmake --build "$VISION_BUILD_DIR" "${build_extra[@]}" --parallel "$BUILD_JOBS" \
  --target strata-vision strata-vision-gguf-test strata-vision-arch-test
python3 "${ROOT}/tools/vision_build_manifest.py" "$VISION_BUILD_DIR"
