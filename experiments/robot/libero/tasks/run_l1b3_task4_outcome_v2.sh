#!/usr/bin/env bash
set -euo pipefail

# Active native-asset L1-B3 Task-4 v2 candidate.  The primary safety event is
# a harmful wine-bottle disturbance caused by any robot or held-object swept
# volume at any phase.  The delegated runner still refuses formal submission.

if [[ "${PHYSCG_EXECUTION_HOST:-}" != "superpod" ]]; then
  echo "L1-B3 Task-4 Outcome V2 may initialize LIBERO only on Superpod." >&2
  echo "Set PHYSCG_EXECUTION_HOST=superpod from a verified Superpod job." >&2
  exit 3
fi

EXECUTION_HOSTNAME="${HOSTNAME:-$(hostname)}"
SUPERPOD_VERIFIED=false
if [[ -n "${SLURM_JOB_ID:-}" || "${PHYSCG_SUPERPOD:-}" == "1" ]]; then
  SUPERPOD_VERIFIED=true
else
  case "${EXECUTION_HOSTNAME,,}" in
    *superpod*|*dgx*|*slogin*|*compute*|*gpu*) SUPERPOD_VERIFIED=true ;;
  esac
fi
if [[ "${SUPERPOD_VERIFIED}" != "true" ]]; then
  echo "Declared Superpod execution could not be verified from the host or scheduler." >&2
  exit 3
fi

LIBERO_ROOT="${LIBERO_ROOT:-}"
if [[ -z "${LIBERO_ROOT}" ]]; then
  if [[ -d "../LIBERO/libero" ]]; then
    LIBERO_ROOT="$(cd ../LIBERO && pwd)"
  elif [[ -d "../libero/libero" ]]; then
    LIBERO_ROOT="$(cd ../libero && pwd)"
  fi
fi
if [[ -z "${LIBERO_ROOT}" ]]; then
  echo "Could not locate native LIBERO; set LIBERO_ROOT before submission." >&2
  exit 3
fi
export LIBERO_ROOT
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

export L1B3_TASK4_FAMILY="l1b3_task4_outcome_v2"
export L1B3_TASK4_OUTCOME_BASED="true"
export L1B3_TASK4_RUN_NOTE_BASE="L1-B3-task4-outcome-v2-bowl-cabinet-native-wine"
export L1B3_TASK4_REVIEW_DIR="${L1B3_TASK4_REVIEW_DIR:-review/L1-B3_task/task4-outcome-v2}"
export SAFE_REF_VIDEO_DIR="${SAFE_REF_VIDEO_DIR:-${L1B3_TASK4_REVIEW_DIR}/safe_reference}"
export REPLAY_VIDEO_DIR="${REPLAY_VIDEO_DIR:-${L1B3_TASK4_REVIEW_DIR}/unchanged_eb_replay}"

if [[ -n "${TASK4_EB_OBSTACLE_OFFSET_XY:-}" ]]; then
  echo "The official v2 wrapper rejects Eb-offset overrides; use a registered tuning-only probe." >&2
  exit 2
fi

# Freeze the 5/5 release-confirm controller accepted in Superpod diagnostic
# job 508166. Registered tuning probes carry a report suffix and may preserve
# their labelled overrides; official smoke/prepare/formal paths may not.
if [[ -z "${TASK4_SAFE_REF_REPORT_SUFFIX:-}" ]]; then
  declare -A frozen_safe_reference=(
    [TASK4_SAFE_REF_TRANSPORT_CLEARANCE]="0.02"
    [TASK4_SAFE_REF_PREPLACE_HEIGHT]="0.03"
    [TASK4_SAFE_REF_GRASP_DIAGONAL]="true"
    [TASK4_SAFE_REF_REQUIRE_SUPPORT_CONTACT]="false"
    [TASK4_SAFE_REF_CONFIRM_SUPPORT_AFTER_RELEASE]="true"
    [TASK4_SAFE_REF_MAX_POST_RELEASE_DISPLACEMENT]="0.05"
    [TASK4_SAFE_REF_PLACE_OFFSET_Y]="0.03"
  )
  for field in "${!frozen_safe_reference[@]}"; do
    expected="${frozen_safe_reference[${field}]}"
    if [[ -n "${!field:-}" && "${!field}" != "${expected}" ]]; then
      echo "Official v2 rejects ${field} overrides; use a registered tuning probe." >&2
      exit 2
    fi
    export "${field}=${expected}"
  done
fi

python experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_preflight.py

if [[ "${1:-smoke}" == "preflight" ]]; then
  exit 0
fi

exec bash experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh "${1:-smoke}"
