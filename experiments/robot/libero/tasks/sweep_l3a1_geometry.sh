#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="${LOG_DIR:-experiments/logs}"
REPORT="${LOG_DIR}/l3a1_geometry_sweep.md"
mkdir -p "${LOG_DIR}"

candidates=(
  "-0.040 -0.180 -20.0"
  "-0.050 -0.180 -20.0"
  "-0.060 -0.180 -20.0"
  "-0.060 -0.175 -20.0"
  "-0.060 -0.185 -22.0"
)

{
  echo "# L3-A1 strict geometry sweep"
  echo
  echo "| dx | dy | deg | verdict | accepted attempt |"
  echo "| ---: | ---: | ---: | --- | ---: |"
} > "${REPORT}"

passes=0
for spec in "${candidates[@]}"; do
  read -r dx dy deg <<< "${spec}"
  label="dx${dx}_dy${dy}_deg${deg}"
  output="${LOG_DIR}/l3a1_sweep_${label}.hdf5"
  log="${LOG_DIR}/l3a1_sweep_${label}.log"
  rm -f "${output}" "${log}"
  set +e
  python experiments/robot/libero/tasks/generate_l3a1_drawer_bottle_initial_states.py \
    --output "${output}" --variant risk --num_states 1 --max_attempts 100 \
    --lean_dx "${dx}" --lean_dy "${dy}" --lean_deg "${deg}" > "${log}" 2>&1
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
    echo "| ${dx} | ${dy} | ${deg} | PASS | ${attempt} |" >> "${REPORT}"
    passes=$((passes + 1))
  else
    echo "| ${dx} | ${dy} | ${deg} | FAIL | — |" >> "${REPORT}"
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
