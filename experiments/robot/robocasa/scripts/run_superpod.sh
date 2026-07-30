#!/usr/bin/env bash
#
# Registered SuperPod entry point for native-only RoboCasa validation.
#
# This script is intentionally narrow: it supports environment discovery,
# L2-A1 live native preflight, native-layout diagnostics, and the two-stage
# initial-state gate. It does
# not run a policy, G1/G2/G3, or formal evaluation.

set -uo pipefail

MODE="${1:-probe}"
SCENE="${ROBOCASA_SCENE:-L2-A1}"
SEED="${ROBOCASA_SEED:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ROBOCASA_SOURCE_ROOT="${ROBOCASA_SOURCE_ROOT:-/scratch/trllmout/drwqyhappy/physcog-robocasa/robocasa}"
EVIDENCE_ROOT="${REPO_ROOT}/experiments/logs/robocasa_superpod/${MODE}"
REVIEW_DIR="${REPO_ROOT}/review/${SCENE}_task"

mkdir -p "${EVIDENCE_ROOT}" "${REVIEW_DIR}"

candidate_pythons=()
if [[ -n "${ROBOCASA_PYTHON:-}" ]]; then
  candidate_pythons+=("${ROBOCASA_PYTHON}")
fi
candidate_pythons+=(
  "/scratch/trllmout/drwqyhappy/physcog-robocasa/.venv/bin/python"
  "/scratch/trllmout/drwqyhappy/physcog-robocasa/venv/bin/python"
  "/scratch/trllmout/drwqyhappy/physcog-robocasa/env/bin/python"
  "/home/drwqyhappy/.conda/envs/robocasa/bin/python"
  "/home/drwqyhappy/.conda/envs/robocasa_env/bin/python"
  "/home/drwqyhappy/.conda/envs/openvla_oft/bin/python"
)
while IFS= read -r candidate; do
  candidate_pythons+=("${candidate}")
done < <(
  find /scratch/trllmout/drwqyhappy/physcog-robocasa \
    -maxdepth 5 -type f -path "*/bin/python" 2>/dev/null | sort
)
if command -v python >/dev/null 2>&1; then
  candidate_pythons+=("$(command -v python)")
fi
if command -v python3 >/dev/null 2>&1; then
  candidate_pythons+=("$(command -v python3)")
fi

export PYTHONPATH="${ROBOCASA_SOURCE_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

robosuite_roots=("")
while IFS= read -r robots_init; do
  robosuite_roots+=(
    "$(cd "$(dirname "${robots_init}")/../../.." && pwd)"
  )
done < <(
  find /scratch/trllmout/drwqyhappy/physcog-robocasa \
    -maxdepth 9 -type f -path "*/robosuite/models/robots/__init__.py" \
    2>/dev/null | sort
)

PYTHON_BIN=""
SELECTED_PYTHONPATH=""
environment_file="${EVIDENCE_ROOT}/environment.txt"
{
  printf 'repo=%s\n' "${REPO_ROOT}"
  printf 'repo_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  printf 'robocasa_source_root=%s\n' "${ROBOCASA_SOURCE_ROOT}"
  printf 'mujoco_gl=%s\n' "${MUJOCO_GL}"
  for candidate in "${candidate_pythons[@]}"; do
    [[ -x "${candidate}" ]] || continue
    for robosuite_root in "${robosuite_roots[@]}"; do
      probe_pythonpath="${ROBOCASA_SOURCE_ROOT}:${REPO_ROOT}"
      if [[ -n "${robosuite_root}" ]]; then
        probe_pythonpath="${ROBOCASA_SOURCE_ROOT}:${robosuite_root}:${REPO_ROOT}"
      fi
      probe_pythonpath="${probe_pythonpath}:${PYTHONPATH:-}"
      printf 'candidate=%s\n' "${candidate}"
      printf 'candidate_robosuite_root=%s\n' "${robosuite_root:-<environment>}"
      set +e
      candidate_probe="$(
        PYTHONPATH="${probe_pythonpath}" "${candidate}" -c \
          "import imageio, mujoco, numpy, robocasa, robosuite; from robosuite.models.robots import PandaOmron; print('IMPORT_OK')" \
          2>&1
      )"
      candidate_rc="$?"
      set -e
      printf 'candidate_rc=%s\n' "${candidate_rc}"
      while IFS= read -r diagnostic_line; do
        printf '[candidate] %s\n' "${diagnostic_line}"
      done <<<"${candidate_probe}"
      if [[ "${candidate_rc}" -eq 0 ]]; then
        PYTHON_BIN="${candidate}"
        SELECTED_PYTHONPATH="${probe_pythonpath}"
        printf 'selected_python=%s\n' "${PYTHON_BIN}"
        printf 'selected_robosuite_root=%s\n' "${robosuite_root:-<environment>}"
        break
      fi
    done
    if [[ -n "${PYTHON_BIN}" ]]; then
      break
    fi
  done
} >"${environment_file}"
cat "${environment_file}"

if [[ -z "${PYTHON_BIN}" ]]; then
  printf 'Verdict: FAIL_ROBOCASA_ENVIRONMENT\n' | tee -a "${environment_file}"
  exit 2
fi

export PYTHONPATH="${SELECTED_PYTHONPATH}"

"${PYTHON_BIN}" - "${EVIDENCE_ROOT}/runtime.json" <<'PY'
import importlib
import json
import pathlib
import platform
import sys

output = pathlib.Path(sys.argv[1])
packages = {}
for name in ("imageio", "mujoco", "numpy", "robocasa", "robosuite"):
    module = importlib.import_module(name)
    packages[name] = {
        "file": str(pathlib.Path(module.__file__).resolve()),
        "version": getattr(module, "__version__", None),
    }
output.write_text(
    json.dumps(
        {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "packages": packages,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

case "${MODE}" in
  probe)
    "${PYTHON_BIN}" experiments/robot/robocasa/scripts/static_check.py \
      | tee "${EVIDENCE_ROOT}/static_check.txt"
    probe_rc="${PIPESTATUS[0]}"
    if [[ "${probe_rc}" -eq 0 ]]; then
      printf 'Verdict: PASS_ROBOCASA_ENVIRONMENT\n'
    else
      printf 'Verdict: FAIL_ROBOCASA_STATIC_POLICY\n'
    fi
    exit "${probe_rc}"
    ;;

  static_live)
    manifest="${EVIDENCE_ROOT}/${SCENE}_native_preflight.json"
    "${PYTHON_BIN}" experiments/robot/robocasa/scripts/static_check.py \
      --live --scene "${SCENE}" --manifest "${manifest}" \
      | tee "${EVIDENCE_ROOT}/static_live.txt"
    live_rc="${PIPESTATUS[0]}"
    if [[ "${live_rc}" -eq 0 ]]; then
      printf 'Verdict: PASS_NATIVE_PREFLIGHT\n'
    else
      printf 'Verdict: FAIL_NATIVE_PREFLIGHT\n'
    fi
    exit "${live_rc}"
    ;;

  geometry)
    geometry_output="${EVIDENCE_ROOT}/${SCENE}_geometry.json"
    "${PYTHON_BIN}" \
      experiments/robot/robocasa/scripts/diagnose_l2a1_geometry.py \
      --scene "${SCENE}" --seed "${SEED}" --out "${geometry_output}" \
      | tee "${EVIDENCE_ROOT}/geometry.txt"
    geometry_rc="${PIPESTATUS[0]}"
    if [[ "${geometry_rc}" -eq 0 ]]; then
      printf 'Verdict: PASS_GEOMETRY_DIAGNOSTIC\n'
    else
      printf 'Verdict: FAIL_GEOMETRY_DIAGNOSTIC\n'
    fi
    exit "${geometry_rc}"
    ;;

  layout_scan)
    "${PYTHON_BIN}" \
      experiments/robot/robocasa/scripts/scan_l2a1_layouts.py \
      --scene "${SCENE}" --seed "${SEED}" \
      --layouts 0 1 2 3 4 5 6 7 8 9 \
      --out "${EVIDENCE_ROOT}" \
      | tee "${EVIDENCE_ROOT}/layout_scan.txt"
    scan_rc="${PIPESTATUS[0]}"
    if [[ "${scan_rc}" -eq 0 ]]; then
      printf 'Verdict: PASS_LAYOUT_SCAN_DIAGNOSTIC\n'
    else
      printf 'Verdict: FAIL_LAYOUT_SCAN_DIAGNOSTIC\n'
    fi
    exit "${scan_rc}"
    ;;

  initial_unreviewed|initial_reviewed)
    if [[ "${MODE}" == "initial_reviewed" ]]; then
      visibility="yes"
      pass_verdict="PASS_INITIAL_GATES"
    else
      visibility="unreviewed"
      pass_verdict="NEEDS_HUMAN_VISIBILITY_REVIEW"
    fi
    run_tag="${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)"
    output="${REVIEW_DIR}/${SCENE}_initial_gate_manifest_${run_tag}.json"
    set +e
    "${PYTHON_BIN}" experiments/robot/robocasa/scripts/run_initial_gates.py \
      --scene "${SCENE}" \
      --condition Er \
      --seed "${SEED}" \
      --steps 200 \
      --human-visible "${visibility}" \
      --review "${REVIEW_DIR}" \
      --out "${output}" \
      >"${EVIDENCE_ROOT}/initial_gates.txt" 2>&1
    gate_rc="$?"
    set -e
    sed -n '1,260p' "${EVIDENCE_ROOT}/initial_gates.txt"

    if [[ -f "${output}" ]]; then
      cp "${output}" "${EVIDENCE_ROOT}/"
      [[ -f "${output}.INVALID.json" ]] && cp "${output}.INVALID.json" "${EVIDENCE_ROOT}/"
      "${PYTHON_BIN}" - "${output}" "${EVIDENCE_ROOT}" <<'PY'
import json
import pathlib
import shutil
import sys

manifest_path = pathlib.Path(sys.argv[1])
evidence = pathlib.Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
visibility = manifest.get("gates", {}).get("visibility", {})
paths = [visibility.get("initial_frame")]
paths.extend((visibility.get("paired_initial_frames") or {}).values())
for raw_path in paths:
    if not raw_path:
        continue
    source = pathlib.Path(raw_path)
    if source.is_file():
        shutil.copy2(source, evidence / source.name)
    quarantine = pathlib.Path(f"{source}.INVALID.json")
    if quarantine.is_file():
        shutil.copy2(quarantine, evidence / quarantine.name)
native = manifest_path.parent / f"{manifest.get('scene_id')}_native_preflight.json"
if native.is_file():
    shutil.copy2(native, evidence / native.name)
PY
    fi

    if [[ "${MODE}" == "initial_unreviewed" && "${gate_rc}" -eq 1 ]]; then
      printf 'Verdict: NEEDS_HUMAN_VISIBILITY_REVIEW\n'
      exit 1
    fi
    if [[ "${gate_rc}" -eq 0 ]]; then
      printf 'Verdict: %s\n' "${pass_verdict}"
    else
      printf 'Verdict: FAIL_INITIAL_GATES\n'
    fi
    exit "${gate_rc}"
    ;;

  *)
    printf 'unsupported registered RoboCasa SuperPod mode: %s\n' "${MODE}" >&2
    exit 64
    ;;
esac
