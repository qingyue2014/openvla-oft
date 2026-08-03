#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pi05_eb_diagnostic}"
case "${MODE}" in
  pi05_eb_diagnostic|smoke_pi05|formal_pi05) ;;
  *)
    echo "Permitted modes: pi05_eb_diagnostic|smoke_pi05|formal_pi05" >&2
    exit 2
    ;;
esac

OPENPI_ROOT="${OPENPI_ROOT:-/home/drwqyhappy/04-mycode/openpi-15a9616}"
PI05_PORT="${PI05_PORT:-8000}"
PI05_SERVER_GPU="${PI05_SERVER_GPU:-0}"
SERVER_PYTHON="${OPENPI_ROOT}/.venv/bin/python"
SERVER_LOG="${SERVER_LOG:-experiments/logs/l1a1_v4_${MODE}_server.log}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/l1a1-v4-pi05-${SLURM_JOB_ID:-local}}"

if [[ ! -x "${SERVER_PYTHON}" ]] \
  || [[ ! -f "${OPENPI_ROOT}/scripts/serve_policy.py" ]]; then
  echo "Missing official OpenPI server environment under ${OPENPI_ROOT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${SERVER_LOG}")"
mkdir -p "${RUNTIME_CACHE_ROOT}/numba" "${RUNTIME_CACHE_ROOT}/triton"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/numba}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export CC="${TRITON_CC:-/usr/bin/gcc}"

(
  cd "${OPENPI_ROOT}"
  CUDA_VISIBLE_DEVICES="${PI05_SERVER_GPU}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    "${SERVER_PYTHON}" scripts/serve_policy.py \
      --env LIBERO \
      --port "${PI05_PORT}"
) >"${SERVER_LOG}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export MODEL_FAMILY=pi05
export CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero
export PI05_HOST=127.0.0.1
export PI05_PORT
export PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-300}"
export PI05_REPLAN_STEPS="${PI05_REPLAN_STEPS:-5}"
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"
export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
export SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-True}"
export EB_NOTE="${EB_NOTE:-L1-A1-v4-ramekin-eb-native-pi05}"
export SMOKE_TRIALS="${SMOKE_TRIALS:-5}"

bash experiments/robot/libero/tasks/run_l1a1_native.sh "${MODE}"
