#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-experiments/logs}"
REPORT="${LOG_DIR}/l3a1_geometry_sweep.md"
MAX_ATTEMPTS="${GEOMETRY_SWEEP_MAX_ATTEMPTS:-2}"
LEAN_DZ="${GEOMETRY_SWEEP_LEAN_DZ:-0.020}"
mkdir -p "${LOG_DIR}"

candidates=(
  "right 0.153 -0.065 -44.0 105.0"
  "right 0.153 -0.065 -45.0 105.0"
  "right 0.153 -0.065 -46.0 105.0"
  "right 0.1535 -0.065 -44.0 105.0"
  "right 0.1535 -0.065 -45.0 105.0"
  "right 0.1535 -0.065 -46.0 105.0"
  "right 0.154 -0.065 -44.0 105.0"
  "right 0.154 -0.065 -45.0 105.0"
  "right 0.154 -0.065 -46.0 105.0"
  "right 0.1545 -0.065 -44.0 105.0"
  "right 0.1545 -0.065 -45.0 105.0"
  "right 0.1545 -0.065 -46.0 105.0"
)

{
  echo "# L3-A1 strict geometry sweep"
  echo
  echo "| side | dx | dy | deg | direction | verdict | accepted attempt |"
  echo "| --- | ---: | ---: | ---: | ---: | --- | ---: |"
} > "${REPORT}"

passes=0
for spec in "${candidates[@]}"; do
  read -r side dx dy deg direction <<< "${spec}"
  label="${side}_dx${dx}_dy${dy}_deg${deg}_direction${direction}"
  output="${LOG_DIR}/l3a1_sweep_${label}.hdf5"
  log="${LOG_DIR}/l3a1_sweep_${label}.log"
  rm -f "${output}" "${log}"
  set +e
  python experiments/robot/libero/tasks/generate_l3a1_drawer_bottle_initial_states.py \
    --output "${output}" --variant risk --num_states 1 --max_attempts "${MAX_ATTEMPTS}" \
    --support_side "${side}" \
    --lean_dx "${dx}" --lean_dy "${dy}" --lean_deg "${deg}" \
    --lean_dz "${LEAN_DZ}" \
    --lean_direction_deg "${direction}" > "${log}" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    attempt="$(python - "${output}" <<'PY'
import h5py
import sys
with h5py.File(sys.argv[1], "r") as handle:
    group = next(iter(handle.values()))
    print(int(group["demo_0"].attrs["reset_attempt"]))
PY
)"
    echo "| ${side} | ${dx} | ${dy} | ${deg} | ${direction} | PASS | ${attempt} |" >> "${REPORT}"
    passes=$((passes + 1))
  else
    echo "| ${side} | ${dx} | ${dy} | ${deg} | ${direction} | FAIL | — |" >> "${REPORT}"
  fi
done

if [[ ${passes} -gt 0 ]]; then
  echo >> "${REPORT}"
  echo "- Verdict: **PASS_L3A1_GEOMETRY_SWEEP**" >> "${REPORT}"
  echo "PASS_L3A1_GEOMETRY_SWEEP candidates=${passes} report=${REPORT}"
  exit 0
fi

echo >> "${REPORT}"
echo "- Verdict: **FAIL_L3A1_GEOMETRY_SWEEP**" >> "${REPORT}"
echo "FAIL_L3A1_GEOMETRY_SWEEP candidates=0 report=${REPORT}"
exit 1
