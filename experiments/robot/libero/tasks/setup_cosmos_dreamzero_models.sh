#!/usr/bin/env bash
set -euo pipefail

# Download only official, revision-pinned checkpoints below the shared
# Superpod model root. This script intentionally does not start evaluation.
#
# Usage:
#   bash experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh cosmos
#   bash experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh dreamzero
#   bash experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh all

MODEL="${1:-all}"
MODEL_ROOT="${MODEL_ROOT:-/project/trllmout/models}"
SOURCE_ROOT="${SOURCE_ROOT:-${MODEL_ROOT}/_sources}"

COSMOS_REPO_ID="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
COSMOS_REVISION="cb689ec0e3347c13667d70a78a3447388f5c3bb8"
COSMOS_DEST="${MODEL_ROOT}/Cosmos-Policy-LIBERO-Predict2-2B"
COSMOS_SOURCE_URL="https://github.com/nvlabs/cosmos-policy.git"
COSMOS_SOURCE_REVISION="18a2accadf4e7a3531e56754102af5a24d2316da"
COSMOS_SOURCE_DEST="${SOURCE_ROOT}/cosmos-policy"

DREAMZERO_REPO_ID="GEAR-Dreams/DreamZero-DROID"
DREAMZERO_REVISION="96ad344138c66e82536422432ad742f015784942"
DREAMZERO_DEST="${MODEL_ROOT}/DreamZero-DROID"
DREAMZERO_SOURCE_URL="https://github.com/dreamzero0/dreamzero.git"
DREAMZERO_SOURCE_REVISION="ab790c198fbce33503358efbbd4187ce9a89adf3"
DREAMZERO_SOURCE_DEST="${SOURCE_ROOT}/dreamzero"

case "${MODEL}" in
  cosmos|dreamzero|all) ;;
  *)
    echo "Usage: $0 {cosmos|dreamzero|all}" >&2
    exit 2
    ;;
esac

mkdir -p "${MODEL_ROOT}" "${SOURCE_ROOT}"

hf_download() {
  local repo_id="$1" revision="$2" destination="$3"
  mkdir -p "${destination}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo_id}" \
      --repo-type model \
      --revision "${revision}" \
      --local-dir "${destination}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" \
      --repo-type model \
      --revision "${revision}" \
      --local-dir "${destination}"
  else
    echo "Neither 'hf' nor 'huggingface-cli' is available." >&2
    echo "Install huggingface_hub in the Superpod setup environment." >&2
    exit 3
  fi
}

checkout_source() {
  local source_url="$1" revision="$2" destination="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${source_url}" "${destination}"
  fi
  git -C "${destination}" fetch origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  test "$(git -C "${destination}" rev-parse HEAD)" = "${revision}"
}

setup_cosmos() {
  hf_download "${COSMOS_REPO_ID}" "${COSMOS_REVISION}" "${COSMOS_DEST}"
  test -s "${COSMOS_DEST}/Cosmos-Policy-LIBERO-Predict2-2B.pt"
  test -s "${COSMOS_DEST}/config.json"
  test -s "${COSMOS_DEST}/libero_dataset_statistics.json"
  test -s "${COSMOS_DEST}/libero_t5_embeddings.pkl"
  checkout_source \
    "${COSMOS_SOURCE_URL}" "${COSMOS_SOURCE_REVISION}" "${COSMOS_SOURCE_DEST}"
  if command -v uv >/dev/null 2>&1; then
    UV=(uv)
  elif python -m uv --version >/dev/null 2>&1; then
    UV=(python -m uv)
  else
    echo "uv is required to create the official Cosmos LIBERO runtime." >&2
    exit 3
  fi
  (
    cd "${COSMOS_SOURCE_DEST}"
    "${UV[@]}" sync --extra cu128 --group libero --python 3.10
  )
  test -x "${COSMOS_SOURCE_DEST}/.venv/bin/python"
  "${COSMOS_SOURCE_DEST}/.venv/bin/python" - <<'PY'
import json
import pathlib
import torch

major, minor = (int(value) for value in torch.__version__.split("+", 1)[0].split(".")[:2])
if (major, minor) < (2, 6):
    raise SystemExit(f"Cosmos runtime requires torch>=2.6, found {torch.__version__}")
import cosmos_policy
import libero

def module_location(module):
    module_file = getattr(module, "__file__", None)
    if module_file:
        return str(pathlib.Path(module_file).resolve())
    module_paths = list(getattr(module, "__path__", ()))
    return str(pathlib.Path(module_paths[0]).resolve()) if module_paths else None

path = pathlib.Path.cwd() / "cosmos_superpod_setup.json"
path.write_text(
    json.dumps(
        {
            "cosmos_policy": module_location(cosmos_policy),
            "libero": module_location(libero),
            "python": str(pathlib.Path(__import__("sys").executable).resolve()),
            "torch": torch.__version__,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  mkdir -p experiments/logs
  cp "${COSMOS_SOURCE_DEST}/cosmos_superpod_setup.json" \
    experiments/logs/cosmos_superpod_setup.json
  printf 'COSMOS_CHECKPOINT=%s\n' "${COSMOS_DEST}"
  printf 'COSMOS_MODEL_REVISION=%s\n' "${COSMOS_REVISION}"
  printf 'COSMOS_SOURCE_REVISION=%s\n' "${COSMOS_SOURCE_REVISION}"
  printf 'COSMOS_PYTHON=%s\n' "${COSMOS_SOURCE_DEST}/.venv/bin/python"
}

setup_dreamzero() {
  hf_download "${DREAMZERO_REPO_ID}" "${DREAMZERO_REVISION}" "${DREAMZERO_DEST}"
  test -s "${DREAMZERO_DEST}/config.json"
  test -s "${DREAMZERO_DEST}/model.safetensors.index.json"
  test -s "${DREAMZERO_DEST}/model-00001-of-00010.safetensors"
  test -s "${DREAMZERO_DEST}/model-00010-of-00010.safetensors"
  checkout_source \
    "${DREAMZERO_SOURCE_URL}" "${DREAMZERO_SOURCE_REVISION}" "${DREAMZERO_SOURCE_DEST}"
  printf 'DREAMZERO_CHECKPOINT=%s\n' "${DREAMZERO_DEST}"
  printf 'DREAMZERO_MODEL_REVISION=%s\n' "${DREAMZERO_REVISION}"
  printf 'DREAMZERO_SOURCE_REVISION=%s\n' "${DREAMZERO_SOURCE_REVISION}"
  printf '%s\n' \
    'DREAMZERO_LIBERO_STATUS=BLOCKED_NO_OFFICIAL_LIBERO_EMBODIMENT_CHECKPOINT'
}

if [[ "${MODEL}" == "cosmos" || "${MODEL}" == "all" ]]; then
  setup_cosmos
fi
if [[ "${MODEL}" == "dreamzero" || "${MODEL}" == "all" ]]; then
  setup_dreamzero
fi
