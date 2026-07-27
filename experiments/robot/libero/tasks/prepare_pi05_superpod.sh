#!/usr/bin/env bash
set -euo pipefail

OPENPI_COMMIT="${OPENPI_COMMIT:-15a9616a00943ada6c20a0f158e3adb39df2ccac}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/drwqyhappy/04-mycode/openpi-${OPENPI_COMMIT:0:7}}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/project/trllmout/models}"
OPENPI_CLIENT_ROOT="${OPENPI_CLIENT_ROOT:-/project/trllmout/models/openpi-client-${OPENPI_COMMIT:0:7}}"
CHECKPOINT_PATH="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero"
MANIFEST_PATH="experiments/logs/pi05_superpod_setup.json"

mkdir -p "$(dirname "${OPENPI_ROOT}")" "${OPENPI_DATA_HOME}" "${OPENPI_CLIENT_ROOT}"

if [[ ! -d "${OPENPI_ROOT}/.git" ]]; then
  git clone --filter=blob:none --recurse-submodules \
    https://github.com/Physical-Intelligence/openpi.git "${OPENPI_ROOT}"
fi

actual_commit="$(git -C "${OPENPI_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${OPENPI_COMMIT}" ]]; then
  git -C "${OPENPI_ROOT}" fetch origin "${OPENPI_COMMIT}"
  git -C "${OPENPI_ROOT}" checkout --detach "${OPENPI_COMMIT}"
  git -C "${OPENPI_ROOT}" submodule update --init --recursive
fi
test "$(git -C "${OPENPI_ROOT}" rev-parse HEAD)" = "${OPENPI_COMMIT}"

if command -v uv >/dev/null 2>&1; then
  UV=(uv)
elif python -m uv --version >/dev/null 2>&1; then
  UV=(python -m uv)
else
  python -m pip install --user uv
  UV=(python -m uv)
fi

# Keep the lightweight websocket client out of the OpenVLA conda environment.
python -m pip install --upgrade --target "${OPENPI_CLIENT_ROOT}" \
  "${OPENPI_ROOT}/packages/openpi-client"

(
  cd "${OPENPI_ROOT}"
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" "${UV[@]}" run python - \
    "${CHECKPOINT_PATH}" <<'PY'
import pathlib
import sys

from openpi.shared import download

expected = pathlib.Path(sys.argv[1]).resolve()
actual = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
if actual.resolve() != expected:
    raise RuntimeError(f"unexpected checkpoint path: {actual} != {expected}")
print(f"Downloaded pi05_libero checkpoint to {actual}")
PY
)

test -d "${CHECKPOINT_PATH}/params"
test -d "${CHECKPOINT_PATH}/assets"
mkdir -p "$(dirname "${MANIFEST_PATH}")"
python - "${MANIFEST_PATH}" "${OPENPI_COMMIT}" "${OPENPI_ROOT}" \
  "${CHECKPOINT_PATH}" "${OPENPI_CLIENT_ROOT}" <<'PY'
import datetime
import json
import pathlib
import sys

manifest_path, commit, runtime, checkpoint, client = sys.argv[1:]
checkpoint_path = pathlib.Path(checkpoint)
files = sorted(
    str(path.relative_to(checkpoint_path))
    for path in checkpoint_path.rglob("*")
    if path.is_file()
)
manifest = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "openpi_commit": commit,
    "openpi_runtime": runtime,
    "checkpoint": checkpoint,
    "openpi_client": client,
    "checkpoint_file_count": len(files),
    "checkpoint_files": files,
}
pathlib.Path(manifest_path).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

