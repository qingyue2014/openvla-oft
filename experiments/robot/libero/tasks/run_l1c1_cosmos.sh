#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
COSMOS_CHECKPOINT="${COSMOS_CHECKPOINT:-/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_MODEL_REVISION="${COSMOS_MODEL_REVISION:-cb689ec0e3347c13667d70a78a3447388f5c3bb8}"
COSMOS_SOURCE_ROOT="${COSMOS_SOURCE_ROOT:-/project/trllmout/models/_sources/cosmos-policy}"
COSMOS_SOURCE_REVISION="${COSMOS_SOURCE_REVISION:-18a2accadf4e7a3531e56754102af5a24d2316da}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_SOURCE_ROOT}/.venv/bin/python}"
COSMOS_TOKENIZER_PATH="${COSMOS_TOKENIZER_PATH:-/project/trllmout/models/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/drwqyhappy/04-mycode/LIBERO}"
EVALUATOR_PYTHON="${EVALUATOR_PYTHON:-/home/drwqyhappy/.conda/envs/openvla_oft/bin/python}"
COSMOS_HOST="${COSMOS_HOST:-127.0.0.1}"
COSMOS_PORT="${COSMOS_PORT:-8001}"
COSMOS_AUTHKEY="${COSMOS_AUTHKEY:-l1c1-cosmos-local}"
COSMOS_SERVER_GPU="${COSMOS_SERVER_GPU:-0}"
COSMOS_SERVER_LOG="${COSMOS_SERVER_LOG:-experiments/logs/l1c1_cosmos_server.log}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${TMPDIR:-/tmp}/l1c1-cosmos-${SLURM_JOB_ID:-local}}"
BASE_PATH="${PATH}"
BASE_PYTHONPATH="${PYTHONPATH:-}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FAIL_L1C1_COSMOS_SUPERPOD_ONLY: this wrapper must run inside a Superpod Slurm job" >&2
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

test -s "${COSMOS_CHECKPOINT}/Cosmos-Policy-LIBERO-Predict2-2B.pt"
test -s "${COSMOS_CHECKPOINT}/config.json"
test -s "${COSMOS_CHECKPOINT}/libero_dataset_statistics.json"
test -s "${COSMOS_CHECKPOINT}/libero_t5_embeddings.pkl"
test -s "${COSMOS_TOKENIZER_PATH}"
test -d "${COSMOS_SOURCE_ROOT}/.git"
test "$(git -C "${COSMOS_SOURCE_ROOT}" rev-parse HEAD)" = "${COSMOS_SOURCE_REVISION}"
test -x "${COSMOS_PYTHON}"
test -x "${EVALUATOR_PYTHON}"
test -d "${LIBERO_ROOT}/libero"
test -f "${LIBERO_ROOT}/libero/libero/__init__.py"

EVALUATOR_RUNTIME="$(${EVALUATOR_PYTHON} - <<'PY'
import importlib.metadata as metadata

expected = {
    "libero": "0.1.0",
    "mujoco": "3.9.0",
    "robosuite": "1.4.1",
}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"evaluator runtime mismatch: {actual} != {expected}")
print(",".join(f"{name}={actual[name]}" for name in sorted(actual)))
PY
)"
printf 'PASS_L1C1_COSMOS_EVALUATOR_RUNTIME=%s\n' "${EVALUATOR_RUNTIME}"

COSMOS_SITE_PACKAGES="$("${COSMOS_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
COSMOS_NVRTC_ROOT="${COSMOS_SITE_PACKAGES}/nvidia/cuda_nvrtc"
test -f "${COSMOS_NVRTC_ROOT}/lib/libnvrtc.so.12"
COSMOS_NVIDIA_LIBRARY_PATH="$("${COSMOS_PYTHON}" - "${COSMOS_SITE_PACKAGES}" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib"))))
PY
)"

mkdir -p "$(dirname "${COSMOS_SERVER_LOG}")"
mkdir -p "${RUNTIME_CACHE_ROOT}/numba" "${RUNTIME_CACHE_ROOT}/triton"
(
  export PATH="$(dirname "${COSMOS_PYTHON}"):${BASE_PATH}"
  export CUDA_HOME="${COSMOS_NVRTC_ROOT}"
  export CC="${COSMOS_CC:-/usr/bin/gcc}"
  export CXX="${COSMOS_CXX:-/usr/bin/g++}"
  export LD_LIBRARY_PATH="${COSMOS_NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${PWD}:${COSMOS_SOURCE_ROOT}"
  export NUMBA_CACHE_DIR="${RUNTIME_CACHE_ROOT}/numba"
  export TRITON_CACHE_DIR="${RUNTIME_CACHE_ROOT}/triton"
  CUDA_VISIBLE_DEVICES="${COSMOS_SERVER_GPU}" "${COSMOS_PYTHON}" \
    -m experiments.robot.cosmos_policy_server \
    --host "${COSMOS_HOST}" \
    --port "${COSMOS_PORT}" \
    --authkey "${COSMOS_AUTHKEY}" \
    --pretrained_checkpoint "${COSMOS_CHECKPOINT}" \
    --cosmos_tokenizer_path "${COSMOS_TOKENIZER_PATH}" \
    --cosmos_num_denoising_steps "${COSMOS_NUM_DENOISING_STEPS:-5}" \
    --seed 7 \
    --task_suite_name libero_spatial
) >"${COSMOS_SERVER_LOG}" 2>&1 &
server_pid=$!

cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Keep the exact OpenVLA-approved simulator stack in the evaluator process.
# Only image/proprio requests and 16x7 action chunks cross localhost.
export PATH="$(dirname "${EVALUATOR_PYTHON}"):${BASE_PATH}"
export PYTHONPATH="${LIBERO_ROOT}:${BASE_PYTHONPATH}"
export LIBERO_ROOT

export MODEL_FAMILY=cosmos
export MODEL_NAME=Cosmos-Policy
export CHECKPOINT="${COSMOS_CHECKPOINT}"
export COSMOS_TOKENIZER_PATH
export COSMOS_NUM_DENOISING_STEPS="${COSMOS_NUM_DENOISING_STEPS:-5}"
export COSMOS_HOST
export COSMOS_PORT
export COSMOS_AUTHKEY
export COSMOS_CONNECT_TIMEOUT_S="${COSMOS_CONNECT_TIMEOUT_S:-900}"
export MODEL_OPEN_LOOP_STEPS=16
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"
export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-capped}"
export SAVE_TRAJECTORY=True

if [[ "${MODE}" == "smoke" ]]; then
  export LOG_DIR="${LOG_DIR:-experiments/logs/l1c1_cosmos_smoke}"
  export BOWL_STACK_EB_NOTE="${BOWL_STACK_EB_NOTE:-L1-C1-hidden-bowl-stack-eb-cosmos-smoke}"
  export BOWL_STACK_ER_NOTE="${BOWL_STACK_ER_NOTE:-L1-C1-hidden-bowl-stack-risk-cosmos-smoke}"
  export BOWL_STACK_EC_NOTE="${BOWL_STACK_EC_NOTE:-L1-C1-hidden-bowl-stack-ec-cosmos-smoke}"
else
  export LOG_DIR="${LOG_DIR:-experiments/logs/l1c1_cosmos_formal}"
  export BOWL_STACK_EB_NOTE="${BOWL_STACK_EB_NOTE:-L1-C1-hidden-bowl-stack-eb-cosmos-formal}"
  export BOWL_STACK_ER_NOTE="${BOWL_STACK_ER_NOTE:-L1-C1-hidden-bowl-stack-risk-cosmos-formal}"
  export BOWL_STACK_EC_NOTE="${BOWL_STACK_EC_NOTE:-L1-C1-hidden-bowl-stack-ec-cosmos-formal}"
fi

bash experiments/robot/libero/tasks/run_l1c1_task2.sh "${task_mode}"
