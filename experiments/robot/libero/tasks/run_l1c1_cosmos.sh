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
test -d "${LIBERO_ROOT}/libero"

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

export PATH="$(dirname "${COSMOS_PYTHON}"):${PATH}"
export CUDA_HOME="${COSMOS_NVRTC_ROOT}"
export CC="${COSMOS_CC:-/usr/bin/gcc}"
export CXX="${COSMOS_CXX:-/usr/bin/g++}"
export LD_LIBRARY_PATH="${COSMOS_NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${COSMOS_SOURCE_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export LIBERO_ROOT

export MODEL_FAMILY=cosmos
export MODEL_NAME=Cosmos-Policy
export CHECKPOINT="${COSMOS_CHECKPOINT}"
export COSMOS_TOKENIZER_PATH
export COSMOS_NUM_DENOISING_STEPS="${COSMOS_NUM_DENOISING_STEPS:-5}"
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
