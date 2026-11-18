#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
BUILD_DIR=${BUILD_DIR:-"${ROOT}/build/sm75"}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda-13.4}
BUILD_JOBS=${BUILD_JOBS:-16}
# Extra engine CMake options, whitespace-separated (e.g. CMAKE_EXTRA_ARGS="-DSTRATA_MMQ_TAB=ON"); empty by default.
read -r -a cmake_extra <<< "${CMAKE_EXTRA_ARGS:-}"
export UPSTREAM_MANIFEST=${UPSTREAM_MANIFEST:-"${ROOT}/upstream.json"}
[[ "$BUILD_DIR" == /* ]] || BUILD_DIR="${ROOT}/${BUILD_DIR}"
configure_extra=()
build_extra=()
for arg in "$@"; do
  case "$arg" in
    --rebuild) configure_extra=(--fresh); build_extra=(--clean-first) ;;
    -h|--help) printf 'Usage: bash build.sh [--rebuild]\nBuild the pinned SM75 engine and K-quant library.\n'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
python3 "${ROOT}/tools/apply_patches.py"
SOURCE=$(python3 "${ROOT}/tools/source_config.py")
geometry_extra=()
if [[ -f "${SOURCE}/cmake/strata_geometry.cmake" ]]; then
  geometry_extra=(-DSTRATA_GEOMETRY=flash_next)
fi
mkdir -p "$BUILD_DIR"
BUILD_DIR=$(cd -- "$BUILD_DIR" && pwd -P)
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  cached_source=
  cached_build=
  while IFS='=' read -r key value; do
    case "$key" in
      CMAKE_HOME_DIRECTORY:INTERNAL) cached_source=$value ;;
      CMAKE_CACHEFILE_DIR:INTERNAL) cached_build=$value ;;
    esac
  done < "${BUILD_DIR}/CMakeCache.txt"
  if [[ "$cached_source" != "$SOURCE" || "$cached_build" != "$BUILD_DIR" ]]; then
    printf 'Moved checkout/build detected; refreshing CMake configuration.\n'
    configure_extra=(--fresh)
  fi
fi
cmake "${configure_extra[@]}" "${geometry_extra[@]}" -S "$SOURCE" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_TOOLKIT}/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_TOOLKIT" \
  -DCMAKE_CUDA_ARCHITECTURES=75-real -DSTRATA_PORTABLE=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_ORCA_Q4KS_MMQ=ON \
  -DSTRATA_MMQ_TAB=ON "-DSTRATA_MMQ_TAB_TYPES=q4_K;q5_0;q8_0" \
  -DSTRATA_GGML_DIR="${ROOT}/vendor/llama.cpp" "${cmake_extra[@]}"
cmake --build "$BUILD_DIR" "${build_extra[@]}" --parallel "$BUILD_JOBS" --target strata strata_kq

python3 "${ROOT}/tools/build_manifest.py" --build-dir "$BUILD_DIR"
