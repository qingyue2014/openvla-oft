#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:?usage: run_pi05_l1b_smoke.sh FAMILY}"
COUNT="${PI05_SMOKE_TRIALS:-1}"
OPENPI_COMMIT="${OPENPI_COMMIT:-15a9616a00943ada6c20a0f158e3adb39df2ccac}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/drwqyhappy/04-mycode/openpi-${OPENPI_COMMIT:0:7}}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/project/trllmout/models}"
OPENPI_CLIENT_ROOT="${OPENPI_CLIENT_ROOT:-/project/trllmout/models/openpi-client-${OPENPI_COMMIT:0:7}-minimal}"
CHECKPOINT_PATH="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero"
if [[ -n "${PI05_PORT:-}" ]]; then
  PORT="${PI05_PORT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  PORT="$((20000 + SLURM_JOB_ID % 20000))"
else
  PORT=8000
fi
TASKS_DIR="experiments/robot/libero/tasks"
SERVER_LOG="experiments/logs/${FAMILY}_pi05_server.log"
MANIFEST_PATH="experiments/logs/${FAMILY}_pi05_smoke_manifest.json"

case "${FAMILY}" in
  l1b1_native_gripper)
    SOURCE_ROOT="/home/drwqyhappy/04-mycode/openvla-oft/.physcog-agent/worktrees/d219c88cd6a537cdfd3c53ebe74da08ec23f91c4/experiments/robot/libero/tasks"
    SOURCE_FAMILY="l1b1_native_gripper"
    ;;
  l1b2_native_held_object)
    SOURCE_ROOT="/home/drwqyhappy/04-mycode/openvla-oft/.physcog-agent/worktrees/03001b7d2d5e9b549c41de5666417999f863a4ff/experiments/robot/libero/tasks"
    SOURCE_FAMILY="l1b6_native_held_object"
    ;;
  *)
    echo "Unsupported canonical family: ${FAMILY}" >&2
    exit 2
    ;;
esac

if [[ "${COUNT}" -ne 1 ]]; then
  echo "The archived frozen smoke fixtures contain exactly one state per condition." >&2
  exit 2
fi
test -d "${OPENPI_ROOT}/.git"
test "$(git -C "${OPENPI_ROOT}" rev-parse HEAD)" = "${OPENPI_COMMIT}"
test -d "${CHECKPOINT_PATH}/params"
test -d "${CHECKPOINT_PATH}/assets"
test -d "${OPENPI_CLIENT_ROOT}/openpi_client"

mkdir -p experiments/logs
declare -A STATE_HASHES
for condition in eb er ec; do
  source_path="${SOURCE_ROOT}/${SOURCE_FAMILY}_${condition}_states.hdf5"
  destination="${TASKS_DIR}/${FAMILY}_${condition}_states.hdf5"
  test -f "${source_path}"
  cp "${source_path}" "${destination}"
  STATE_HASHES["${condition}"]="$(sha256sum "${destination}" | awk '{print $1}')"
  printf 'Frozen %s state: %s sha256=%s\n' \
    "${condition}" "${source_path}" "${STATE_HASHES[${condition}]}"
done

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif python -m uv --version >/dev/null 2>&1; then
  UV=(python -m uv)
else
  echo "uv is missing; run the registered pi05 setup phase first" >&2
  exit 2
fi

(
  cd "${OPENPI_ROOT}"
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
  CC="${OPENPI_CC:-/usr/bin/gcc}" \
  CXX="${OPENPI_CXX:-/usr/bin/g++}" \
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    "${UV[@]}" run scripts/serve_policy.py \
      --env LIBERO \
      --port "${PORT}" \
      policy:checkpoint \
      --policy.config pi05_libero \
      --policy.dir "${CHECKPOINT_PATH}"
) >"${SERVER_LOG}" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Do not let the websocket client connect to an unrelated service that happens
# to own the requested port. The policy server must both stay alive and open
# this job-specific endpoint before the evaluator starts.
server_ready=false
for _ in $(seq 1 900); do
  if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
    echo "pi0.5 policy server exited before becoming ready" >&2
    tail -200 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if python - "${PORT}" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    server_ready=true
    break
  fi
  sleep 2
done
if [[ "${server_ready}" != "true" ]]; then
  echo "pi0.5 policy server did not become ready within 1800 seconds" >&2
  tail -200 "${SERVER_LOG}" >&2 || true
  exit 1
fi

export PYTHONPATH="${OPENPI_CLIENT_ROOT}:${PYTHONPATH:-}"
export MODEL_FAMILY=pi05
export PI05_HOST=127.0.0.1
export PI05_PORT="${PORT}"
export PI05_REPLAN_STEPS=5
export PI05_CONNECT_TIMEOUT_S="${PI05_CONNECT_TIMEOUT_S:-1800}"
export NUM_TRIALS="${COUNT}"
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"
export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
export MAX_VIOLATION_VIDEOS=1
export MAX_SUCCESS_VIDEOS=1
export MAX_FAILURE_VIDEOS=1
export SAVE_TRAJECTORY=True
export RUN_ID_SUFFIX=pi05-smoke

for condition in eb er ec; do
  STATE_PATH_OVERRIDE="${TASKS_DIR}/${FAMILY}_${condition}_states.hdf5" \
    bash "${TASKS_DIR}/run_l1b_swept.sh" "${FAMILY}" "${condition}"
done

python - "${MANIFEST_PATH}" "${FAMILY}" "${CHECKPOINT_PATH}" \
  "${STATE_HASHES[eb]}" "${STATE_HASHES[er]}" "${STATE_HASHES[ec]}" <<'PY'
import datetime
import json
import pathlib
import sys

manifest_path, family, checkpoint, eb_hash, er_hash, ec_hash = sys.argv[1:]
manifest = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_family": "pi05",
    "policy_config": "pi05_libero",
    "checkpoint": checkpoint,
    "family": family,
    "conditions": ["eb", "er", "ec"],
    "episodes_per_condition": 1,
    "replan_steps": 5,
    "state_sha256": {"eb": eb_hash, "er": er_hash, "ec": ec_hash},
}
pathlib.Path(manifest_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
