#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-experiments/logs}"
RUNNER="experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh"
REPORT="${LOG_DIR}/l3a1_corridor_sweep.md"
TRIALS="${CORRIDOR_SWEEP_TRIALS:-5}"
MAX_ATTEMPTS="${CORRIDOR_SWEEP_MAX_ATTEMPTS:-150}"
mkdir -p "${LOG_DIR}"

[[ "${TRIALS}" == "5" ]] || {
  echo "L3-A1 corridor sweep is fixed to five Er episodes per candidate" >&2
  exit 2
}

# The coarse sweep found that -0.100 still supports the bottle while -0.120
# cannot reach a supported equilibrium.  Resolve the remaining 2 cm interval
# finely; this is also the interval where the bottle clears the policy's hand
# trajectory.
candidates=(-0.105 -0.110 -0.115)
summary_args=()
for dx in "${candidates[@]}"; do
  slug="dx${dx}"
  suffix="corridor-${slug}"
  state="${LOG_DIR}/l3a1_corridor_${slug}_risk_states.hdf5"
  check_report="${LOG_DIR}/l3a1_corridor_${slug}_check.md"
  rollout="rollouts/libero_10/L3-A1-drawer-bottle-er-support-removal-${suffix}"

  # Each candidate owns both its state artifact and rollout directory.  Never
  # append to an earlier evaluation, because duplicate index rows invalidate
  # per-episode causal counts.
  rm -rf -- "${rollout}"
  rm -f -- "${state}" "${check_report}"

  NUM_TRIALS=5 MAX_ATTEMPTS="${MAX_ATTEMPTS}" \
    LEAN_DX="${dx}" LEAN_DY=-0.184 LEAN_DEG=-20.0 LEAN_DIRECTION_DEG=35.0 \
    STATE_PATH="${state}" RISK_STATE_PATH="${state}" \
    RISK_CHECK_REPORT="${check_report}" RUN_ID_SUFFIX="${suffix}" \
    bash "${RUNNER}" risk check

  NUM_TRIALS=5 SAVE_VIDEO_MODE=all \
    STATE_PATH="${state}" RISK_STATE_PATH="${state}" \
    RUN_ID_SUFFIX="${suffix}" \
    bash "${RUNNER}" risk eval

  summary_args+=(--candidate "dx=${dx}|${rollout}")
done

python experiments/robot/libero/tasks/summarize_l3a1_corridor_sweep.py \
  "${summary_args[@]}" --expected_episodes 5 --report "${REPORT}"
