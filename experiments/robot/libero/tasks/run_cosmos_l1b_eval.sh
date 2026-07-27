#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:?usage: run_cosmos_l1b_eval.sh FAMILY [smoke|formal]}"
RUN_KIND="${2:-smoke}"
case "${RUN_KIND}" in
  smoke)
    COUNT="${COSMOS_SMOKE_TRIALS:-1}"
    RUN_SUFFIX="cosmos-smoke"
    ;;
  formal)
    COUNT="${COSMOS_FORMAL_TRIALS:-50}"
    RUN_SUFFIX="cosmos-formal"
    ;;
  *)
    echo "Unsupported Cosmos evaluation mode: ${RUN_KIND}" >&2
    exit 2
    ;;
esac

TASKS_DIR="experiments/robot/libero/tasks"
CHECKPOINT="${COSMOS_CHECKPOINT:-/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_MODEL_REVISION="${COSMOS_MODEL_REVISION:-cb689ec0e3347c13667d70a78a3447388f5c3bb8}"
COSMOS_SOURCE_ROOT="${COSMOS_SOURCE_ROOT:-/project/trllmout/models/_sources/cosmos-policy}"
COSMOS_SOURCE_REVISION="${COSMOS_SOURCE_REVISION:-18a2accadf4e7a3531e56754102af5a24d2316da}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_SOURCE_ROOT}/.venv/bin/python}"
RESULTS_JSON="experiments/logs/${FAMILY}_${RUN_SUFFIX}_results.json"
RESULTS_REPORT="experiments/logs/${FAMILY}_${RUN_SUFFIX}_results.md"
MANIFEST_PATH="experiments/logs/${FAMILY}_${RUN_SUFFIX}_manifest.json"
VIDEO_DIR="experiments/logs/${FAMILY}_${RUN_SUFFIX}_videos"

case "${FAMILY}" in
  l1b1_native_gripper)
    SOURCE_ROOT="/home/drwqyhappy/04-mycode/openvla-oft/.physcog-agent/worktrees/d219c88cd6a537cdfd3c53ebe74da08ec23f91c4/experiments/robot/libero/tasks"
    SOURCE_FAMILY="l1b1_native_gripper"
    SOURCE_COMMIT="d219c88cd6a537cdfd3c53ebe74da08ec23f91c4"
    TASK_SUITE="libero_spatial"
    RUN_BASE="L1-B1-task6-native-ramekin-capture-lift-v4"
    ;;
  l1b2_native_held_object)
    SOURCE_ROOT="/home/drwqyhappy/04-mycode/openvla-oft/.physcog-agent/worktrees/03001b7d2d5e9b549c41de5666417999f863a4ff/experiments/robot/libero/tasks"
    SOURCE_FAMILY="l1b6_native_held_object"
    SOURCE_COMMIT="03001b7d2d5e9b549c41de5666417999f863a4ff"
    TASK_SUITE="libero_goal"
    RUN_BASE="L1-B2-goal-cream-cheese-native-wine-bottle-knockdown"
    ;;
  *)
    echo "Unsupported canonical family: ${FAMILY}" >&2
    exit 2
    ;;
esac

if [[ "${RUN_KIND}" == "smoke" && "${COUNT}" -ne 1 ]]; then
  echo "Registered Cosmos smoke requires exactly one episode per condition." >&2
  exit 2
fi
if [[ "${RUN_KIND}" == "formal" && "${COUNT}" -ne 50 ]]; then
  echo "Registered Cosmos formal evaluation requires 50 episodes per condition." >&2
  exit 2
fi

test -s "${CHECKPOINT}/Cosmos-Policy-LIBERO-Predict2-2B.pt"
test -s "${CHECKPOINT}/config.json"
test -s "${CHECKPOINT}/libero_dataset_statistics.json"
test -s "${CHECKPOINT}/libero_t5_embeddings.pkl"
test -d "${COSMOS_SOURCE_ROOT}/.git"
test "$(git -C "${COSMOS_SOURCE_ROOT}" rev-parse HEAD)" = "${COSMOS_SOURCE_REVISION}"
test -x "${COSMOS_PYTHON}"

export PATH="$(dirname "${COSMOS_PYTHON}"):${PATH}"
COSMOS_SITE_PACKAGES="$("${COSMOS_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
COSMOS_NVRTC_ROOT="${COSMOS_SITE_PACKAGES}/nvidia/cuda_nvrtc"
test -f "${COSMOS_NVRTC_ROOT}/lib/libnvrtc.so.12"
export CUDA_HOME="${COSMOS_NVRTC_ROOT}"
COSMOS_NVIDIA_LIBRARY_PATH="$("${COSMOS_PYTHON}" - "${COSMOS_SITE_PACKAGES}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib"))))
PY
)"
export LD_LIBRARY_PATH="${COSMOS_NVIDIA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
"${COSMOS_PYTHON}" - <<'PY'
import torch

major, minor = (int(value) for value in torch.__version__.split("+", 1)[0].split(".")[:2])
if (major, minor) < (2, 6):
    raise SystemExit(f"Cosmos runtime requires torch>=2.6, found {torch.__version__}")
PY

mkdir -p experiments/logs "${VIDEO_DIR}"
for condition in eb er ec; do
  source_path="${SOURCE_ROOT}/${SOURCE_FAMILY}_${condition}_states.hdf5"
  destination="${TASKS_DIR}/${FAMILY}_${condition}_states.hdf5"
  test -f "${source_path}"
  cp "${source_path}" "${destination}"
  printf 'Frozen %s state: %s sha256=%s\n' \
    "${condition}" "${source_path}" "$(sha256sum "${destination}" | awk '{print $1}')"
done

if [[ "${RUN_KIND}" == "formal" ]]; then
  source_pairing="${SOURCE_ROOT}/${SOURCE_FAMILY}_pairing.json"
  destination_pairing="${TASKS_DIR}/${FAMILY}_pairing.json"
  source_logs="${SOURCE_ROOT%/experiments/robot/libero/tasks}/experiments/logs"
  source_safe_reference="${source_logs}/${SOURCE_FAMILY}_safe_reference.md"
  destination_safe_reference="experiments/logs/${FAMILY}_${RUN_SUFFIX}_source_safe_reference.md"
  test -f "${source_pairing}"
  test -f "${source_safe_reference}"
  grep -Fq 'Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**' "${source_safe_reference}"
  cp "${source_pairing}" "${destination_pairing}"
  cp "${source_safe_reference}" "${destination_safe_reference}"
  python - "${destination_pairing}" "${FAMILY}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
family = sys.argv[2]
pairing = json.loads(path.read_text(encoding="utf-8"))
if pairing.get("num_states") != 50 or len(pairing.get("pairs", [])) != 50:
    raise SystemExit("formal Cosmos evaluation requires exactly 50 frozen pairs")
if not pairing.get("spec", {}).get("native_assets_only", False):
    raise SystemExit("formal Cosmos evaluation requires native-only frozen scenes")
pairing["family"] = family
pairing["paths"] = {
    condition: f"experiments/robot/libero/tasks/{family}_{condition}_states.hdf5"
    for condition in ("eb", "er", "ec")
}
path.write_text(json.dumps(pairing, indent=2) + "\n", encoding="utf-8")
PY
  bash "${TASKS_DIR}/run_l1b_swept.sh" "${FAMILY}" check
fi

export PYTHONPATH="${COSMOS_SOURCE_ROOT}:${PYTHONPATH:-}"
export MODEL_FAMILY=cosmos
export CHECKPOINT="${CHECKPOINT}"
export GOAL_CHECKPOINT="${CHECKPOINT}"
export MODEL_OPEN_LOOP_STEPS=16
export NUM_TRIALS="${COUNT}"
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"
export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"
export MAX_VIOLATION_VIDEOS=1
export MAX_SUCCESS_VIDEOS=1
export MAX_FAILURE_VIDEOS=1
export SAVE_TRAJECTORY=True
export RUN_ID_SUFFIX="${RUN_SUFFIX}"

declare -A CONDITION_EXIT_CODES
overall_status=0
for condition in eb er ec; do
  set +e
  STATE_PATH_OVERRIDE="${TASKS_DIR}/${FAMILY}_${condition}_states.hdf5" \
    bash "${TASKS_DIR}/run_l1b_swept.sh" "${FAMILY}" "${condition}"
  condition_status=$?
  set -e
  CONDITION_EXIT_CODES["${condition}"]="${condition_status}"
  if [[ "${condition_status}" -ne 0 ]]; then
    overall_status="${condition_status}"
    echo "Condition ${condition} failed with exit code ${condition_status}; continuing diagnostic sweep." >&2
  fi
  rollout_dir="rollouts/${TASK_SUITE}/${RUN_BASE}-${condition}-${RUN_SUFFIX}"
  index_path="${rollout_dir}/trajectories/index.jsonl"
  test -f "${index_path}"
  cp "${index_path}" "experiments/logs/${FAMILY}_${RUN_SUFFIX}_${condition}_index.jsonl"
  video_path="$(find "${rollout_dir}" -maxdepth 1 -type f -name '*.mp4' -print -quit)"
  test -n "${video_path}"
  cp "${video_path}" "${VIDEO_DIR}/${condition}.mp4"
done

python "${TASKS_DIR}/summarize_l1b_model_eval.py" \
  --family "${FAMILY}" \
  --model_family cosmos \
  --evaluation_kind "${RUN_KIND}" \
  --checkpoint "${CHECKPOINT}" \
  --episodes "${COUNT}" \
  --model_revision "${COSMOS_MODEL_REVISION}" \
  --source_revision "${COSMOS_SOURCE_REVISION}" \
  --frozen_scene_source_commit "${SOURCE_COMMIT}" \
  --index "eb=experiments/logs/${FAMILY}_${RUN_SUFFIX}_eb_index.jsonl" \
  --index "er=experiments/logs/${FAMILY}_${RUN_SUFFIX}_er_index.jsonl" \
  --index "ec=experiments/logs/${FAMILY}_${RUN_SUFFIX}_ec_index.jsonl" \
  --state "eb=${TASKS_DIR}/${FAMILY}_eb_states.hdf5" \
  --state "er=${TASKS_DIR}/${FAMILY}_er_states.hdf5" \
  --state "ec=${TASKS_DIR}/${FAMILY}_ec_states.hdf5" \
  --condition_exit "eb=${CONDITION_EXIT_CODES[eb]}" \
  --condition_exit "er=${CONDITION_EXIT_CODES[er]}" \
  --condition_exit "ec=${CONDITION_EXIT_CODES[ec]}" \
  --out_json "${RESULTS_JSON}" \
  --out_report "${RESULTS_REPORT}" \
  --out_manifest "${MANIFEST_PATH}"

exit "${overall_status}"
