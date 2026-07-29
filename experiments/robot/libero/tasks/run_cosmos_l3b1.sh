#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_cosmos_l3b1.sh prepare|native_cap_smoke|native_cap_formal|smoke|formal|summarize|direct_prepare|direct_smoke|direct_formal|direct_summarize}"
case "${MODE}" in
  prepare|native_cap_smoke|native_cap_formal|smoke|formal|summarize|direct_prepare|direct_smoke|direct_formal|direct_summarize) ;;
  *) echo "Unsupported Cosmos L3-B1 mode: ${MODE}" >&2; exit 2 ;;
esac

COSMOS_CHECKPOINT="${COSMOS_CHECKPOINT:-/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_MODEL_REVISION="${COSMOS_MODEL_REVISION:-cb689ec0e3347c13667d70a78a3447388f5c3bb8}"
COSMOS_SOURCE_ROOT="${COSMOS_SOURCE_ROOT:-/project/trllmout/models/_sources/cosmos-policy}"
COSMOS_SOURCE_REVISION="${COSMOS_SOURCE_REVISION:-18a2accadf4e7a3531e56754102af5a24d2316da}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_SOURCE_ROOT}/.venv/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/drwqyhappy/04-mycode/LIBERO}"
COSMOS_MANIFEST="${COSMOS_MANIFEST:-experiments/logs/l3b1_cosmos_runtime.md}"

test -s "${COSMOS_CHECKPOINT}/Cosmos-Policy-LIBERO-Predict2-2B.pt"
test -s "${COSMOS_CHECKPOINT}/config.json"
test -s "${COSMOS_CHECKPOINT}/libero_dataset_statistics.json"
test -s "${COSMOS_CHECKPOINT}/libero_t5_embeddings.pkl"
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
export PREVIEW_MODEL_FAMILY=cosmos
export CHECKPOINT="${COSMOS_CHECKPOINT}"
export MODEL_OPEN_LOOP_STEPS=16
export DO_SAMPLE=False
export TEMPERATURE=1.0
export TOP_P=1.0
export SAVE_VIDEO_MODE=all
export MAX_VIDEOS_PER_OUTCOME=10
export RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"

export CAP_RUN_NOTE="L3-B1-cosmos-bottle-in-drawer-capability"
export NATIVE_CAP_RUN_NOTE="L3-B1-cosmos-native-wine-bottle-to-rack"
export RISK_RUN_NOTE="L3-B1-cosmos-bottle-in-drawer-risk"
export EB_RUN_NOTE="L3-B1-cosmos-drawer-close-eb-native"
export EC_RUN_NOTE="L3-B1-cosmos-bottle-in-drawer-ec-clearance"
export NATIVE_PREFLIGHT_REPORT="experiments/logs/l3b1_cosmos_native_preflight.md"
export CAPABILITY_PREFLIGHT_REPORT="experiments/logs/l3b1_cosmos_capability_native_preflight.md"
export NATIVE_CAP_STATES_REPORT="experiments/logs/l3b1_cosmos_native_capability_states.md"
export NATIVE_CAP_SMOKE_REPORT="experiments/logs/l3b1_cosmos_native_capability_smoke.md"
export PAIRING_REPORT="experiments/logs/l3b1_cosmos_state_pairing.md"
export REFERENCE_REPORT="experiments/logs/l3b1_cosmos_reference_paths.md"
export SMOKE_REPORT="experiments/logs/l3b1_cosmos_smoke_evidence.md"

if [[ "${MODE}" == direct_* ]]; then
  export RISK_RUN_NOTE="L3-B1-cosmos-direct-bottle-in-drawer-risk"
  export EB_RUN_NOTE="L3-B1-cosmos-direct-drawer-close-eb-native"
  export EC_RUN_NOTE="L3-B1-cosmos-direct-bottle-in-drawer-ec-clearance"
fi

mkdir -p "$(dirname "${COSMOS_MANIFEST}")"
{
  printf '# L3-B1 Cosmos runtime\n\n'
  printf -- '- Checkpoint: `%s`\n' "${COSMOS_CHECKPOINT}"
  printf -- '- Model revision: `%s`\n' "${COSMOS_MODEL_REVISION}"
  printf -- '- Source root: `%s`\n' "${COSMOS_SOURCE_ROOT}"
  printf -- '- Source revision: `%s`\n' "${COSMOS_SOURCE_REVISION}"
  printf -- '- Model family: `cosmos`\n'
  printf -- '- Open-loop steps: `16`\n'
  printf -- '- Prompt/BDDL/assets: enforced by native-only preflight\n'
  if [[ "${MODE}" == direct_* ]]; then
    printf -- '- Native bottle-to-rack competence gate: skipped by explicit user authorization on 2026-07-29\n'
  fi
} >"${COSMOS_MANIFEST}"

BASE="experiments/robot/libero/tasks/run_l3b1_capability_probe.sh"

validate_native_competence_smoke() {
  python experiments/robot/libero/tasks/validate_l3b1_native_capability_smoke.py \
    --index "rollouts/libero_90/${NATIVE_CAP_RUN_NOTE}-smoke/trajectories/index.jsonl" \
    --expected "${SMOKE_TRIALS:-5}" \
    --report "${NATIVE_CAP_SMOKE_REPORT}"
  grep -q PASS_L3B1_NATIVE_CAPABILITY_SMOKE "${NATIVE_CAP_SMOKE_REPORT}"
}

case "${MODE}" in
  prepare)
    bash "${BASE}" native_cap_prepare
    bash "${BASE}" prepare
    bash "${BASE}" preview
    bash "${BASE}" reference
    ;;
  native_cap_smoke)
    bash "${BASE}" native_cap_smoke
    ;;
  native_cap_formal)
    validate_native_competence_smoke
    bash "${BASE}" native_cap_formal
    ;;
  smoke)
    validate_native_competence_smoke
    bash "${BASE}" smoke
    ;;
  formal)
    validate_native_competence_smoke
    bash "${BASE}" formal
    ;;
  summarize)
    bash "${BASE}" summarize
    ;;
  direct_prepare)
    bash "${BASE}" prepare
    bash "${BASE}" preview
    bash "${BASE}" reference
    ;;
  direct_smoke)
    bash "${BASE}" smoke
    ;;
  direct_formal)
    bash "${BASE}" formal
    ;;
  direct_summarize)
    bash "${BASE}" summarize
    ;;
esac
