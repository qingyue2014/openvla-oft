#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_l3a1_v2_native.sh edge_sweep|edge_preview}"
case "${MODE}" in
  edge_sweep|edge_preview) ;;
  *) echo "Unsupported L3-A1 V2 mode: ${MODE}" >&2; exit 2 ;;
esac

LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" || ! -d "${LIBERO_ROOT}/libero" ]]; then
  echo "Set LIBERO_ROOT to the native LIBERO checkout" >&2
  exit 2
fi
export LIBERO_ROOT
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

BDDL_BASENAME="KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl"
NATIVE_BDDL="${LIBERO_ROOT}/libero/libero/bddl_files/libero_10/${BDDL_BASENAME}"
TASK_DESCRIPTION="put the black bowl in the bottom drawer of the cabinet and close it"
PREFLIGHT_REPORT="experiments/logs/l3a1_v2_native_preflight.md"

python experiments/robot/libero/tasks/validate_l3a1_native_preflight.py \
  --evaluated_bddl "${NATIVE_BDDL}" \
  --evaluated_prompt "${TASK_DESCRIPTION}" \
  --out_report "${PREFLIGHT_REPORT}"
grep -q PASS_L3A1_V2_NATIVE_ONLY_PREFLIGHT "${PREFLIGHT_REPORT}"

case "${MODE}" in
  edge_sweep)
    python experiments/robot/libero/tasks/sweep_l3a1_edge_geometry.py \
      --bddl "${NATIVE_BDDL}" \
      --out_csv experiments/logs/l3a1_v2_edge_sweep.csv \
      --out_contacts experiments/logs/l3a1_v2_edge_contacts.csv \
      --out_report experiments/logs/l3a1_v2_edge_sweep.md
    ;;
  edge_preview)
    python experiments/robot/libero/tasks/export_l3a1_edge_preview.py \
      --bddl "${NATIVE_BDDL}" \
      --out_dir experiments/logs/l3a1_v2_edge_preview
    ;;
esac
