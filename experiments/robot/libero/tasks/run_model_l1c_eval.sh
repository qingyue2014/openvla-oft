#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: run_model_l1c_eval.sh pi05|cosmos l1c1|l1c2|l1c3 smoke|formal}"
SCENARIO="${2:?usage: run_model_l1c_eval.sh pi05|cosmos l1c1|l1c2|l1c3 smoke|formal}"
RUN_KIND="${3:?usage: run_model_l1c_eval.sh pi05|cosmos l1c1|l1c2|l1c3 smoke|formal}"
case "${MODEL}" in
  pi05|cosmos) ;;
  *) echo "Unsupported model: ${MODEL}" >&2; exit 2 ;;
esac
case "${SCENARIO}" in
  l1c1|l1c2|l1c3) ;;
  *) echo "Unsupported scenario: ${SCENARIO}" >&2; exit 2 ;;
esac
case "${RUN_KIND}" in
  smoke)
    COUNT="${L1C_SMOKE_TRIALS:-5}"
    RUN_MODE="smoke"
    ;;
  formal)
    COUNT="${L1C_FORMAL_TRIALS:-50}"
    RUN_MODE="eval"
    ;;
  *) echo "Unsupported evaluation kind: ${RUN_KIND}" >&2; exit 2 ;;
esac
if [[ "${RUN_KIND}" == "smoke" && "${COUNT}" -ne 5 ]]; then
  echo "Registered L1-C smoke evaluation requires exactly 5 episodes." >&2
  exit 2
fi
if [[ "${RUN_KIND}" == "formal" && "${COUNT}" -ne 50 ]]; then
  echo "Registered L1-C formal evaluation requires exactly 50 episodes." >&2
  exit 2
fi

TASKS_DIR="experiments/robot/libero/tasks"
LOG_DIR="experiments/logs"
RUN_SUFFIX="${MODEL}-${RUN_KIND}"
RESULT_PREFIX="${LOG_DIR}/${SCENARIO}_${RUN_SUFFIX}"
RESULTS_JSON="${RESULT_PREFIX}_results.json"
RESULTS_REPORT="${RESULT_PREFIX}_results.md"
MANIFEST_PATH="${RESULT_PREFIX}_manifest.json"
VIDEO_DIR="${RESULT_PREFIX}_videos"
LIBERO_ROOT="${LIBERO_ROOT:-/home/drwqyhappy/04-mycode/LIBERO}"
test -d "${LIBERO_ROOT}/libero"
mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

OPENPI_COMMIT="${OPENPI_COMMIT:-15a9616a00943ada6c20a0f158e3adb39df2ccac}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/drwqyhappy/04-mycode/openpi-${OPENPI_COMMIT:0:7}}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/project/trllmout/models}"
OPENPI_CLIENT_ROOT="${OPENPI_CLIENT_ROOT:-/project/trllmout/models/openpi-client-${OPENPI_COMMIT:0:7}-minimal}"
PI05_CHECKPOINT="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero"

COSMOS_CHECKPOINT="${COSMOS_CHECKPOINT:-/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_MODEL_REVISION="${COSMOS_MODEL_REVISION:-cb689ec0e3347c13667d70a78a3447388f5c3bb8}"
COSMOS_SOURCE_ROOT="${COSMOS_SOURCE_ROOT:-/project/trllmout/models/_sources/cosmos-policy}"
COSMOS_SOURCE_REVISION="${COSMOS_SOURCE_REVISION:-18a2accadf4e7a3531e56754102af5a24d2316da}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_SOURCE_ROOT}/.venv/bin/python}"

export LIBERO_ROOT
export NUM_TRIALS="${COUNT}"
export SMOKE_TRIALS="${COUNT}"
export CALIBRATION_NUM_STATES="${COUNT}"
export PREVIEW_NUM_STATES="$((COUNT < 3 ? COUNT : 3))"
export DEBUG_NUM_DEMOS="${PREVIEW_NUM_STATES}"
export SAVE_VIDEO_MODE=all
export MAX_VIDEOS_PER_OUTCOME=1
export MAX_VIOLATION_VIDEOS=1
export MAX_SUCCESS_VIDEOS=1
export MAX_FAILURE_VIDEOS=1
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"
export RUN_ID_SUFFIX="${RUN_SUFFIX}"
export BOWL_STACK_EB_NOTE="L1-C1-hidden-bowl-stack-eb-${RUN_SUFFIX}"
export BOWL_STACK_ER_NOTE="L1-C1-hidden-bowl-stack-risk-${RUN_SUFFIX}"
export BOWL_STACK_EC_NOTE="L1-C1-hidden-bowl-stack-ec-${RUN_SUFFIX}"
export NATIVE_PREFLIGHT_JSON="${RESULT_PREFIX}_native_preflight.json"
export NATIVE_PREFLIGHT_REPORT="${RESULT_PREFIX}_native_preflight.md"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${MODEL}" == "pi05" ]]; then
  test -d "${OPENPI_ROOT}/.git"
  test "$(git -C "${OPENPI_ROOT}" rev-parse HEAD)" = "${OPENPI_COMMIT}"
  test -d "${PI05_CHECKPOINT}/params"
  test -d "${PI05_CHECKPOINT}/assets"
  test -d "${OPENPI_CLIENT_ROOT}/openpi_client"
  if command -v uv >/dev/null 2>&1; then
    UV=(uv)
  elif python -m uv --version >/dev/null 2>&1; then
    UV=(python -m uv)
  else
    echo "uv is missing; run the registered pi05 setup phase first" >&2
    exit 2
  fi
  if [[ -n "${PI05_PORT:-}" ]]; then
    PORT="${PI05_PORT}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PORT="$((20000 + SLURM_JOB_ID % 20000))"
  else
    PORT=8000
  fi
  SERVER_LOG="${RESULT_PREFIX}_server.log"
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
        --policy.dir "${PI05_CHECKPOINT}"
  ) >"${SERVER_LOG}" 2>&1 &
  server_pid=$!
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
    exit 1
  fi
  export PYTHONPATH="${OPENPI_CLIENT_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
  export MODEL_FAMILY=pi05
  export CHECKPOINT="${PI05_CHECKPOINT}"
  export PI05_HOST=127.0.0.1
  export PI05_PORT="${PORT}"
  export PI05_REPLAN_STEPS=5
  export PI05_CONNECT_TIMEOUT_S=1800
  export MODEL_OPEN_LOOP_STEPS=5
  MODEL_REVISION="pi05_libero"
  SOURCE_REVISION="${OPENPI_COMMIT}"
else
  test -s "${COSMOS_CHECKPOINT}/Cosmos-Policy-LIBERO-Predict2-2B.pt"
  test -s "${COSMOS_CHECKPOINT}/config.json"
  test -d "${COSMOS_SOURCE_ROOT}/.git"
  test "$(git -C "${COSMOS_SOURCE_ROOT}" rev-parse HEAD)" = "${COSMOS_SOURCE_REVISION}"
  test -x "${COSMOS_PYTHON}"
  COSMOS_SITE_PACKAGES="$("${COSMOS_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  COSMOS_NVRTC_ROOT="${COSMOS_SITE_PACKAGES}/nvidia/cuda_nvrtc"
  test -f "${COSMOS_NVRTC_ROOT}/lib/libnvrtc.so.12"
  export PATH="$(dirname "${COSMOS_PYTHON}"):${PATH}"
  export CUDA_HOME="${COSMOS_NVRTC_ROOT}"
  export CC="${COSMOS_CC:-/usr/bin/gcc}"
  export CXX="${COSMOS_CXX:-/usr/bin/g++}"
  COSMOS_NVIDIA_LIBRARY_PATH="$("${COSMOS_PYTHON}" - "${COSMOS_SITE_PACKAGES}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib"))))
PY
)"
  export LD_LIBRARY_PATH="${COSMOS_NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${COSMOS_SOURCE_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
  export MODEL_FAMILY=cosmos
  export CHECKPOINT="${COSMOS_CHECKPOINT}"
  export MODEL_OPEN_LOOP_STEPS=16
  MODEL_REVISION="${COSMOS_MODEL_REVISION}"
  SOURCE_REVISION="${COSMOS_SOURCE_REVISION}"
fi

if [[ "${SCENARIO}" == "l1c1" ]]; then
  if [[ "${RUN_KIND}" == "smoke" ]]; then
    L1C1_MODE="bowl_stack_smoke"
  else
    L1C1_MODE="bowl_stack_eval"
  fi
  bash "${TASKS_DIR}/run_l1c1_task2.sh" "${L1C1_MODE}"
  STATE_EB="${TASKS_DIR}/l1c1_task2_bowl_stack_eb_states.hdf5"
  STATE_ER="${TASKS_DIR}/l1c1_task2_bowl_stack_candidate_states.hdf5"
  STATE_EC="${TASKS_DIR}/l1c1_task2_bowl_stack_ec_states.hdf5"
  CALIBRATION_REPORT="${LOG_DIR}/l1c1_bowl_stack_calibration.md"
  SAFE_REFERENCE_REPORT="${LOG_DIR}/l1c1_safe_reference.md"
  ATTRIBUTION_REPORT="${LOG_DIR}/l1c1_attribution.md"
  PREVIEW_DIR="${TASKS_DIR}/l1c1_implicit_stack_preview"
  ROLLOUT_ROOT="rollouts/libero_spatial"
  NOTES=(
    "${BOWL_STACK_EB_NOTE}"
    "${BOWL_STACK_ER_NOTE}"
    "${BOWL_STACK_EC_NOTE}"
  )
else
  bash "${TASKS_DIR}/run_l1c_occupied.sh" "${SCENARIO}" "${RUN_MODE}"
  STATE_EB="${TASKS_DIR}/${SCENARIO}_eb_states.hdf5"
  STATE_ER="${TASKS_DIR}/${SCENARIO}_er_states.hdf5"
  STATE_EC="${TASKS_DIR}/${SCENARIO}_ec_states.hdf5"
  CALIBRATION_REPORT="${LOG_DIR}/${SCENARIO}_calibration.md"
  SAFE_REFERENCE_REPORT="${LOG_DIR}/${SCENARIO}_safe_reference.md"
  ATTRIBUTION_REPORT="${LOG_DIR}/${SCENARIO}_attribution.md"
  PREVIEW_DIR="${TASKS_DIR}/${SCENARIO}_preview"
  ROLLOUT_ROOT="rollouts/libero_90"
  case "${SCENARIO}" in
    l1c2) SLUG="occupied-tray" ;;
    l1c3) SLUG="occupied-drawer" ;;
  esac
  UPPER_SCENARIO="$(printf '%s' "${SCENARIO}" | tr '[:lower:]' '[:upper:]' | sed 's/C/-C/')"
  NOTES=(
    "${UPPER_SCENARIO}-${SLUG}-eb-${RUN_SUFFIX}"
    "${UPPER_SCENARIO}-${SLUG}-risk-${RUN_SUFFIX}"
    "${UPPER_SCENARIO}-${SLUG}-ec-${RUN_SUFFIX}"
  )
fi

test -s "${STATE_EB}"
test -s "${STATE_ER}"
test -s "${STATE_EC}"
test -s "${CALIBRATION_REPORT}"
test -s "${SAFE_REFERENCE_REPORT}"
test -s "${ATTRIBUTION_REPORT}"
test -s "${NATIVE_PREFLIGHT_JSON}"
test -s "${NATIVE_PREFLIGHT_REPORT}"
grep -Fq 'PASS_NATIVE_TASK_PREFLIGHT' "${NATIVE_PREFLIGHT_REPORT}"
if [[ "${SCENARIO}" == "l1c1" ]]; then
  grep -Fq 'PASS_STACK_PHYSICALLY_FEASIBLE' "${CALIBRATION_REPORT}"
  grep -Fq 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
else
  grep -Fq 'PASS_STATIC_OCCUPANCY_LAYOUT' "${CALIBRATION_REPORT}"
  grep -Fq 'PASS_DYNAMIC_SAFE_REFERENCE' "${SAFE_REFERENCE_REPORT}"
fi

CONDITIONS=(eb er ec)
INDEX_ARGS=()
STATE_ARGS=(
  --state "eb=${STATE_EB}"
  --state "er=${STATE_ER}"
  --state "ec=${STATE_EC}"
)
for index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[${index}]}"
  rollout_dir="${ROLLOUT_ROOT}/${NOTES[${index}]}"
  index_path="${rollout_dir}/trajectories/index.jsonl"
  collected_index="${RESULT_PREFIX}_${condition}_index.jsonl"
  test -s "${index_path}"
  cp "${index_path}" "${collected_index}"
  INDEX_ARGS+=(--index "${condition}=${collected_index}")
  safe_video_path="$(
    find "${rollout_dir}" -maxdepth 1 -type f \
      -name '*--success=True--task=safety=true_*.mp4' -print | sort | sed -n '1p'
  )"
  violation_video_path="$(
    find "${rollout_dir}" -maxdepth 1 -type f \
      -name '*--success=False--task=safety=false_*.mp4' -print | sort | sed -n '1p'
  )"
  failure_video_path="$(
    find "${rollout_dir}" -maxdepth 1 -type f \
      -name '*--success=False--task=safety=true_*.mp4' -print | sort | sed -n '1p'
  )"
  if [[ -n "${safe_video_path}" ]]; then
    cp "${safe_video_path}" "${VIDEO_DIR}/${condition}_safe-success.mp4"
  fi
  if [[ -n "${violation_video_path}" ]]; then
    cp "${violation_video_path}" "${VIDEO_DIR}/${condition}_violation.mp4"
  fi
  if [[ -n "${failure_video_path}" ]]; then
    cp "${failure_video_path}" "${VIDEO_DIR}/${condition}_task-failure.mp4"
  fi
done

python "${TASKS_DIR}/summarize_l1c_model_eval.py" \
  --scenario "${SCENARIO}" \
  --model_family "${MODEL}" \
  --evaluation_kind "${RUN_KIND}" \
  --checkpoint "${CHECKPOINT}" \
  --episodes "${COUNT}" \
  "${INDEX_ARGS[@]}" \
  "${STATE_ARGS[@]}" \
  --model_revision "${MODEL_REVISION}" \
  --source_revision "${SOURCE_REVISION}" \
  --calibration_report "${CALIBRATION_REPORT}" \
  --safe_reference_report "${SAFE_REFERENCE_REPORT}" \
  --out_json "${RESULTS_JSON}" \
  --out_report "${RESULTS_REPORT}" \
  --out_manifest "${MANIFEST_PATH}"

printf 'Attribution report: %s\n' "${ATTRIBUTION_REPORT}"
printf 'Policy-camera previews: %s\n' "${PREVIEW_DIR}"
