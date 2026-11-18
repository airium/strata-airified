#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CONFIG_FILE=${CONFIG_FILE:-"${ROOT}/config.env"}
set -a
[[ ! -f "$CONFIG_FILE" ]] || source "$CONFIG_FILE"
set +a
# --vision off|cpu|gpu overrides VISION_MODE from the config for this launch; other arguments go to the server.
passthrough=()
while (($#)); do
  case "$1" in
    --vision) [[ $# -ge 2 && "$2" != -* ]] || { echo "serve.sh: --vision needs off, cpu or gpu" >&2; exit 2; }
      VISION_MODE=$2; shift 2 ;;
    --vision=*) VISION_MODE=${1#--vision=}; shift ;;
    *) passthrough+=("$1"); shift ;;
  esac
done
set -- ${passthrough[@]+"${passthrough[@]}"}
case "${VISION_MODE:-}" in
  ""|off|cpu|gpu) ;;
  *) echo "serve.sh: VISION_MODE must be off, cpu or gpu (got '${VISION_MODE}')" >&2; exit 2 ;;
esac
[[ -z "${VISION_MODE:-}" ]] || export VISION_MODE
PYTHON_BIN=${PYTHON_BIN:-python3}
case "$PYTHON_BIN" in
  /*) ;;
  */*) PYTHON_BIN="${ROOT}/${PYTHON_BIN}" ;;
  *) PYTHON_BIN=$(command -v "$PYTHON_BIN") ;;
esac
RUNTIME_DIR=${RUNTIME_DIR:-"${ROOT}/runtime"}
export RUNTIME_DIR
if [[ "${1:-}" == --print-config ]]; then
  exec "$PYTHON_BIN" "${ROOT}/tools/render_config.py"
fi
mkdir -p "$RUNTIME_DIR"
resolved="${RUNTIME_DIR}/resolved.json"
"$PYTHON_BIN" "${ROOT}/tools/render_config.py" --check > "${resolved}.tmp"
mv -- "${resolved}.tmp" "$resolved"
export LD_LIBRARY_PATH=${ENGINE_LIBRARY_PATH:-"${ROOT}/build/sm75"}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
SOURCE=$("$PYTHON_BIN" "${ROOT}/tools/source_config.py")
cd -- "$SOURCE"
exec "$PYTHON_BIN" -m serve.server --engine strata --config "$resolved" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-3893}" --offline --no-fit-max-tokens "$@"
