#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-eb_capability}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/drwqyhappy/04-mycode/openpi-15a9616}"
PI05_PORT="${PI05_PORT:-8000}"
PI05_SERVER_GPU="${PI05_SERVER_GPU:-0}"
SERVER_PYTHON="${OPENPI_ROOT}/.venv/bin/python"
SERVER_LOG="${SERVER_LOG:-experiments/logs/l1a4_spatial_pi05_server.log}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/l1a4-spatial-${SLURM_JOB_ID:-local}}"

if [[ ! -x "${SERVER_PYTHON}" ]] || [[ ! -f "${OPENPI_ROOT}/scripts/serve_policy.py" ]]; then
  echo "Missing official OpenPI server environment under ${OPENPI_ROOT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${SERVER_LOG}")"
mkdir -p "${RUNTIME_CACHE_ROOT}/numba" "${RUNTIME_CACHE_ROOT}/triton"
# Numba otherwise attempts to cache robosuite kernels beside the conda
# environment under /home. A full or read-only home filesystem then aborts
# before the first episode. Bind the cache to compute-local scratch so this
# infrastructure condition cannot be mistaken for a scene/evaluation result.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/numba}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
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
export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-none}"
export SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-True}"
export EB_NOTE="${EB_NOTE:-L1-A4-between-eb-native-pi05}"

bash experiments/robot/libero/tasks/run_l1a4_spatial.sh "${MODE}"
