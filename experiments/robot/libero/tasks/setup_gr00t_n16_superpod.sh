#!/usr/bin/env bash
set -euo pipefail

# Install the revision-pinned NVIDIA N1.6 runtime and a LIBERO-finetuned
# checkpoint below the shared Superpod model root. NVIDIA does not publish an
# N1.6 LIBERO checkpoint, so the default checkpoint is explicitly recorded as
# community-published and can be overridden only together with its revision.

MODEL_ROOT="${MODEL_ROOT:-/project/trllmout/models}"
SOURCE_ROOT="${SOURCE_ROOT:-${MODEL_ROOT}/_sources}"
GR00T_SOURCE_URL="${GR00T_SOURCE_URL:-https://github.com/NVIDIA/Isaac-GR00T.git}"
GR00T_SOURCE_REVISION="${GR00T_SOURCE_REVISION:-9b37aa1ce69c73c6d165233fa88128283bba4508}"
GR00T_SOURCE_DEST="${GR00T_SOURCE_DEST:-${SOURCE_ROOT}/Isaac-GR00T-N1.6}"
GR00T_CHECKPOINT_REPO="${GR00T_CHECKPOINT_REPO:-0xAnkitSingh/GR00T-N1.6-LIBERO}"
GR00T_CHECKPOINT_REVISION="${GR00T_CHECKPOINT_REVISION:-d690a226ad06e81736786f56cf879d2ed1dd3f0f}"
GR00T_CHECKPOINT_DEST="${GR00T_CHECKPOINT_DEST:-${MODEL_ROOT}/GR00T-N1.6-LIBERO}"
GR00T_CLIENT_ROOT="${GR00T_CLIENT_ROOT:-${MODEL_ROOT}/gr00t-n16-client-minimal}"
MANIFEST_PATH="${MANIFEST_PATH:-experiments/logs/gr00t_n16_superpod_setup.json}"

mkdir -p \
  "${MODEL_ROOT}" "${SOURCE_ROOT}" "${GR00T_CLIENT_ROOT}" \
  "$(dirname "${MANIFEST_PATH}")"

if [[ ! -d "${GR00T_SOURCE_DEST}/.git" ]]; then
  git clone --recurse-submodules "${GR00T_SOURCE_URL}" "${GR00T_SOURCE_DEST}"
fi
git -C "${GR00T_SOURCE_DEST}" fetch origin "${GR00T_SOURCE_REVISION}"
git -C "${GR00T_SOURCE_DEST}" checkout --detach "${GR00T_SOURCE_REVISION}"
git -C "${GR00T_SOURCE_DEST}" submodule update --init --recursive
test "$(git -C "${GR00T_SOURCE_DEST}" rev-parse HEAD)" = "${GR00T_SOURCE_REVISION}"

if command -v hf >/dev/null 2>&1; then
  hf download "${GR00T_CHECKPOINT_REPO}" \
    --repo-type model \
    --revision "${GR00T_CHECKPOINT_REVISION}" \
    --local-dir "${GR00T_CHECKPOINT_DEST}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${GR00T_CHECKPOINT_REPO}" \
    --repo-type model \
    --revision "${GR00T_CHECKPOINT_REVISION}" \
    --local-dir "${GR00T_CHECKPOINT_DEST}"
else
  echo "Neither 'hf' nor 'huggingface-cli' is available." >&2
  exit 3
fi

for required_file in \
  config.json processor_config.json statistics.json embodiment_id.json \
  model.safetensors.index.json model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors; do
  test -s "${GR00T_CHECKPOINT_DEST}/${required_file}"
done

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif python -m uv --version >/dev/null 2>&1; then
  UV=(python -m uv)
else
  python -m pip install --user uv
  UV=(python -m uv)
fi

# Follow NVIDIA's N1.6 dGPU installation path. This environment hosts only the
# model server; the existing OpenVLA environment remains the LIBERO evaluator.
(
  cd "${GR00T_SOURCE_DEST}"
  "${UV[@]}" sync --python 3.10
  "${UV[@]}" pip install -e .
)
test -x "${GR00T_SOURCE_DEST}/.venv/bin/python"

# Keep just the wire-protocol dependencies in a PYTHONPATH directory consumable
# by the existing OpenVLA environment.
python -m pip install --upgrade --target "${GR00T_CLIENT_ROOT}" \
  "msgpack==1.1.0" \
  "pyzmq==27.0.1"

"${GR00T_SOURCE_DEST}/.venv/bin/python" - \
  "${GR00T_CHECKPOINT_DEST}" <<'PY'
import json
import pathlib
import sys

import gr00t
import torch

checkpoint = pathlib.Path(sys.argv[1])
processor = json.loads((checkpoint / "processor_config.json").read_text())
configs = processor["processor_kwargs"]["modality_configs"]
if "libero_panda" not in configs:
    raise SystemExit("Checkpoint has no libero_panda modality config")
libero = configs["libero_panda"]
expected = {
    "video": ["image", "wrist_image"],
    "state": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "action": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "language": ["annotation.human.action.task_description"],
}
actual = {
    name: libero[name]["modality_keys"]
    for name in expected
}
if actual != expected:
    raise SystemExit(f"Unexpected LIBERO modality contract: {actual}")
if libero["action"]["delta_indices"] != list(range(16)):
    raise SystemExit("Expected the N1.6 LIBERO checkpoint to emit 16 actions")
print(f"GR00T runtime: {pathlib.Path(gr00t.__file__).resolve()}")
print(f"Torch: {torch.__version__}")
print("PASS_GR00T_N16_LIBERO_MODALITY_CONTRACT")
PY

python - \
  "${MANIFEST_PATH}" \
  "${GR00T_SOURCE_URL}" "${GR00T_SOURCE_REVISION}" "${GR00T_SOURCE_DEST}" \
  "${GR00T_CHECKPOINT_REPO}" "${GR00T_CHECKPOINT_REVISION}" \
  "${GR00T_CHECKPOINT_DEST}" "${GR00T_CLIENT_ROOT}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

(
    manifest_path,
    source_url,
    source_revision,
    source_path,
    checkpoint_repo,
    checkpoint_revision,
    checkpoint_path,
    client_path,
) = sys.argv[1:]

root = pathlib.Path(checkpoint_path)
files = sorted(path for path in root.rglob("*") if path.is_file())
digest = hashlib.sha256()
for path in files:
    digest.update(str(path.relative_to(root)).encode())
    digest.update(str(path.stat().st_size).encode())

manifest = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_family": "gr00t_n16",
    "runtime": {
        "publisher": "NVIDIA",
        "source_url": source_url,
        "source_revision": source_revision,
        "path": source_path,
    },
    "checkpoint": {
        "publisher": "community",
        "repo_id": checkpoint_repo,
        "revision": checkpoint_revision,
        "path": checkpoint_path,
        "file_count": len(files),
        "file_listing_digest": digest.hexdigest(),
    },
    "client_dependencies": {
        "path": client_path,
        "msgpack": "1.1.0",
        "pyzmq": "27.0.1",
    },
    "embodiment_tag": "LIBERO_PANDA",
    "action_horizon": 16,
    "evaluation_open_loop_steps": 8,
    "checkpoint_provenance_note": (
        "NVIDIA publishes the N1.6 runtime and LIBERO recipe but no official "
        "N1.6 LIBERO-finetuned checkpoint; this default checkpoint is community-published."
    ),
}
pathlib.Path(manifest_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
