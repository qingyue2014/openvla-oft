#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/drwqyhappy/04-mycode/openpi-15a9616}"
OPENPI_COMMIT="15a9616a00943ada6c20a0f158e3adb39df2ccac"
PI05_SERVER_GPU="${PI05_SERVER_GPU:-0}"
SERVER_PYTHON="${OPENPI_ROOT}/.venv/bin/python"
SERVER_LOG="${SERVER_LOG:-experiments/logs/l1c1_pi05_server.log}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/l1c1-pi05-${SLURM_JOB_ID:-local}}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FAIL_L1C1_PI05_SUPERPOD_ONLY: this wrapper must run inside a Superpod Slurm job" >&2
  exit 2
fi
if [[ ! -x "${SERVER_PYTHON}" ]] || [[ ! -f "${OPENPI_ROOT}/scripts/serve_policy.py" ]]; then
  echo "FAIL_L1C1_PI05_RUNTIME: missing official OpenPI runtime at ${OPENPI_ROOT}" >&2
  exit 2
fi
actual_openpi_commit="$(git -C "${OPENPI_ROOT}" rev-parse HEAD)"
if [[ "${actual_openpi_commit}" != "${OPENPI_COMMIT}" ]]; then
  echo "FAIL_L1C1_PI05_RUNTIME: OpenPI commit ${actual_openpi_commit} != ${OPENPI_COMMIT}" >&2
  exit 2
fi

# A fixed localhost port can collide with an unrelated job on the same DGX.
# Select an available port immediately before launching the server unless the
# caller deliberately preregistered one.
if [[ -z "${PI05_PORT:-}" ]]; then
  PI05_PORT="$("${SERVER_PYTHON}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi
if [[ ! "${PI05_PORT}" =~ ^[0-9]+$ ]] || (( PI05_PORT < 1024 || PI05_PORT > 65535 )); then
  echo "FAIL_L1C1_PI05_RUNTIME: invalid PI05_PORT=${PI05_PORT}" >&2
  exit 2
fi

case "${MODE}" in
  smoke) task_mode="bowl_stack_smoke" ;;
  formal) task_mode="bowl_stack_eval" ;;
  *)
    echo "Expected smoke or formal, got: ${MODE}" >&2
    exit 2
    ;;
esac

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

# Do not let the evaluator spend its full connection timeout after an early
# server crash (for example, a port bind failure).  The evaluator starts only
# after the exact server process is alive and accepting localhost connections.
server_ready=0
for ((attempt = 0; attempt < 900; attempt++)); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "FAIL_L1C1_PI05_SERVER_START: server exited before readiness on port ${PI05_PORT}" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 2
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PI05_PORT}") 2>/dev/null; then
    server_ready=1
    break
  fi
  sleep 1
done
if (( server_ready != 1 )); then
  echo "FAIL_L1C1_PI05_SERVER_START: timed out waiting for port ${PI05_PORT}" >&2
  tail -n 100 "${SERVER_LOG}" >&2 || true
  exit 2
fi
printf 'PASS_L1C1_PI05_SERVER_READY port=%s pid=%s\n' "${PI05_PORT}" "${server_pid}"

export PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export MODEL_FAMILY=pi05
export MODEL_NAME=pi0.5
export CHECKPOINT=gs://openpi-assets/checkpoints/pi05_libero
export PI05_HOST=127.0.0.1
export PI05_PORT
export PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-900}"
export PI05_REPLAN_STEPS="${PI05_REPLAN_STEPS:-5}"
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"
export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-capped}"
export SAVE_TRAJECTORY=True

if [[ "${MODE}" == "smoke" ]]; then
  export LOG_DIR="${LOG_DIR:-experiments/logs/l1c1_pi05_smoke}"
  export BOWL_STACK_EB_NOTE="${BOWL_STACK_EB_NOTE:-L1-C1-hidden-bowl-stack-eb-pi05-smoke}"
  export BOWL_STACK_ER_NOTE="${BOWL_STACK_ER_NOTE:-L1-C1-hidden-bowl-stack-risk-pi05-smoke}"
  export BOWL_STACK_EC_NOTE="${BOWL_STACK_EC_NOTE:-L1-C1-hidden-bowl-stack-ec-pi05-smoke}"
else
  export LOG_DIR="${LOG_DIR:-experiments/logs/l1c1_pi05_formal}"
  export BOWL_STACK_EB_NOTE="${BOWL_STACK_EB_NOTE:-L1-C1-hidden-bowl-stack-eb-pi05-formal}"
  export BOWL_STACK_ER_NOTE="${BOWL_STACK_ER_NOTE:-L1-C1-hidden-bowl-stack-risk-pi05-formal}"
  export BOWL_STACK_EC_NOTE="${BOWL_STACK_EC_NOTE:-L1-C1-hidden-bowl-stack-ec-pi05-formal}"
fi

bash experiments/robot/libero/tasks/run_l1c1_task2.sh "${task_mode}"
