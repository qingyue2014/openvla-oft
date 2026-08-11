#!/usr/bin/env bash
set -euo pipefail

# Prospective, model-independent construction workflow for Outcome V2 v6.
# This runner never starts pi0.5, OpenVLA-OFT, Cosmos, or formal evaluation.

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|prepare) ;;
  smoke|pi05|openvla|cosmos|formal|all|eval)
    echo "Outcome V2 v6 learned-policy execution is blocked until the exact" >&2
    echo "scene and complete review bundle receive explicit hash-bound approval." >&2
    exit 2
    ;;
  *)
    echo "Expected preflight or prepare; received '${MODE}'." >&2
    exit 2
    ;;
esac

if [[ "${PHYSCG_EXECUTION_HOST:-}" != "superpod" ]]; then
  echo "Outcome V2 v6 construction is Superpod-only." >&2
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
  echo "Declared Superpod execution could not be verified." >&2
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

TASKS_DIR="experiments/robot/libero/tasks"
LOG_DIR="experiments/logs"
FAMILY="l1b3_task4_outcome_v2_v6"
TASK_SUITE="libero_goal"
TASK_ID=4
POOL_COUNT=50
PAIR_COUNT=5
SCENE_SEED=42
SELECTION_SOURCE="model_independent_scripted_osc_ensemble_v2"
SELECTION_PENETRATION=0.001
ROLLOUT_PENETRATION=0.002
PREREG="${TASKS_DIR}/${FAMILY}_design_prereg.json"
CONTROLLER="${TASKS_DIR}/${FAMILY}_scripted_controller_ensemble.json"
PAIRING="${TASKS_DIR}/${FAMILY}_pairing.json"
NATIVE_STATES="${TASKS_DIR}/${FAMILY}_native_source_states.hdf5"
EB_STATES="${TASKS_DIR}/${FAMILY}_eb_states.hdf5"
ER_STATES="${TASKS_DIR}/${FAMILY}_er_states.hdf5"
EC_STATES="${TASKS_DIR}/${FAMILY}_ec_states.hdf5"
PREFLIGHT_MANIFEST="${TASKS_DIR}/${FAMILY}_native_preflight.json"
PREFLIGHT_REPORT="${LOG_DIR}/${FAMILY}_native_preflight.md"
CALIBRATION_CSV="${LOG_DIR}/${FAMILY}_calibration.csv"
CALIBRATION_REPORT="${LOG_DIR}/${FAMILY}_calibration.md"
SELECTION_MANIFEST="${LOG_DIR}/${FAMILY}_selection_manifest.json"
SELECTION_AUDIT="${LOG_DIR}/${FAMILY}_selection_audit.md"
SCENE_REPORT="${LOG_DIR}/${FAMILY}_scene_check.md"
INITIAL_REPORT="${LOG_DIR}/${FAMILY}_initial_gate.md"
SAFE_CSV="${LOG_DIR}/${FAMILY}_safe_reference.csv"
SAFE_REPORT="${LOG_DIR}/${FAMILY}_safe_reference.md"
PREPARE_MANIFEST="${LOG_DIR}/${FAMILY}_prepare_manifest.json"
REVIEW_DIR="review/L1-B3_task/task4-outcome-v2-v6"
INITIAL_MANIFEST="${REVIEW_DIR}/L1-B3-task4-outcome-v2-v6_initial_gate_manifest.json"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"

PROFILES=(canonical_center stress_x_plus stress_x_minus holdout_y_plus holdout_y_minus)
ORDERS=(center_then_axes positive_x_then_center negative_x_then_center positive_y_then_center negative_y_then_center)

mkdir -p "${LOG_DIR}" "${REVIEW_DIR}"

python "${TASKS_DIR}/validate_l1b3_task4_outcome_v2_v6_preflight.py" \
  --manifest "${PREFLIGHT_MANIFEST}" \
  --report "${PREFLIGHT_REPORT}"

if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi

for profile in "${PROFILES[@]}"; do
  attempts="${PHYSCG_SELECTION_ATTEMPTS_ROOT:-${TMPDIR:-/tmp}}/${FAMILY}_${SLURM_JOB_ID:-manual}_${profile}_attempts"
  trajectories="${LOG_DIR}/${FAMILY}_${profile}_trajectories"
  for path in "${attempts}" "${trajectories}"; do
    if [[ -e "${path}" ]]; then
      echo "Refusing stale v6 trajectory directory: ${path}" >&2
      exit 2
    fi
  done
done

python "${TASKS_DIR}/generate_l1b_swept_initial_states.py" \
  --family "${FAMILY}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --num_states "${POOL_COUNT}" \
  --seed "${SCENE_SEED}"

run_profile() {
  local profile="$1"
  local order="$2"
  local attempts="${PHYSCG_SELECTION_ATTEMPTS_ROOT:-${TMPDIR:-/tmp}}/${FAMILY}_${SLURM_JOB_ID:-manual}_${profile}_attempts"
  local trajectories="${LOG_DIR}/${FAMILY}_${profile}_trajectories"
  python "${TASKS_DIR}/validate_l1b_safe_reference.py" \
    --family "${FAMILY}" \
    --state_path "${EB_STATES}" \
    --pairing_json "${PAIRING}" \
    --task_suite_name "${TASK_SUITE}" \
    --task_id "${TASK_ID}" \
    --num_states "${POOL_COUNT}" \
    --seed "${SCENE_SEED}" \
    --approach_height 0.12 \
    --lift_height 0.08 \
    --max_waypoint_steps 400 \
    --transport_max_waypoint_steps 700 \
    --position_tolerance 0.025 \
    --grasp_offset_fractions "0.80" \
    --grasp_candidate_order "${order}" \
    --disable_cross_episode_grasp_cache \
    --transport_obstacle_clearance 0.0 \
    --transport_clearance 0.04 \
    --preplace_height 0.06 \
    --place_offset_x 0.00 \
    --place_offset_y 0.00 \
    --confirm_support_after_release \
    --max_post_release_displacement 0.05 \
    --max_occluder_displacement 0.001 \
    --environment_horizon 2000 \
    --min_safe_reference_rate 0.0 \
    --trajectory_source_label "${SELECTION_SOURCE}" \
    --trajectory_profile_id "${profile}" \
    --trajectory_source_manifest "${CONTROLLER}" \
    --trajectory_track_bodies \
      "akita_black_bowl_1_main,wooden_cabinet_1_main,wine_bottle_1_main,robot0_link5,robot0_link6,robot0_link7" \
    --trajectory_dir "${attempts}" \
    --canonical_success_trajectory_dir "${trajectories}" \
    --out_csv "${LOG_DIR}/${FAMILY}_${profile}.csv" \
    --out_report "${LOG_DIR}/${FAMILY}_${profile}.md"
}

for index in "${!PROFILES[@]}"; do
  run_profile "${PROFILES[$index]}" "${ORDERS[$index]}"
done

CANONICAL_DIR="${LOG_DIR}/${FAMILY}_canonical_center_trajectories"
STRESS_X_PLUS_DIR="${LOG_DIR}/${FAMILY}_stress_x_plus_trajectories"
STRESS_X_MINUS_DIR="${LOG_DIR}/${FAMILY}_stress_x_minus_trajectories"
HOLDOUT_Y_PLUS_DIR="${LOG_DIR}/${FAMILY}_holdout_y_plus_trajectories"
HOLDOUT_Y_MINUS_DIR="${LOG_DIR}/${FAMILY}_holdout_y_minus_trajectories"

python "${TASKS_DIR}/calibrate_l1b3_trajectory_conditioned_states.py" \
  --family "${FAMILY}" \
  --selection_trajectory_provenance "${SELECTION_SOURCE}" \
  --eb_trajectories "${CANONICAL_DIR}" \
  --stress_trajectory "stress_x_plus=${STRESS_X_PLUS_DIR}" \
  --stress_trajectory "stress_x_minus=${STRESS_X_MINUS_DIR}" \
  --holdout_trajectory "holdout_y_plus=${HOLDOUT_Y_PLUS_DIR}" \
  --holdout_trajectory "holdout_y_minus=${HOLDOUT_Y_MINUS_DIR}" \
  --native_source_states "${NATIVE_STATES}" \
  --eb_states "${EB_STATES}" \
  --er_states "${ER_STATES}" \
  --ec_states "${EC_STATES}" \
  --pairing_json "${PAIRING}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --max_goal_region_distance 10.0 \
  --min_obstacle_displacement 0.010 \
  --min_obstacle_tilt_change_deg 30.0 \
  --max_contact_penetration "${SELECTION_PENETRATION}" \
  --radial_distance_candidates "0.012,0.014,0.016,0.018,0.020,0.024,0.028,0.032,0.036,0.040" \
  --angular_candidates_deg "0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5" \
  --refinement_radial_distances "0.0005,0.001,0.0015,0.002,0.003,0.004" \
  --refinement_angular_candidates_deg "0,45,90,135,180,225,270,315" \
  --max_candidates_per_episode 600 \
  --max_refinement_seeds 8 \
  --max_refinement_candidates 256 \
  --min_successful_eb "${PAIR_COUNT}" \
  --min_activation_rate 1.0 \
  --select_count "${PAIR_COUNT}" \
  --out_csv "${CALIBRATION_CSV}" \
  --out_report "${CALIBRATION_REPORT}" \
  --fail_on_invalid

python "${TASKS_DIR}/validate_l1b3_task4_outcome_v2_v6_selection.py" \
  --pairing_json "${PAIRING}" \
  --preregistration "${PREREG}" \
  --controller_manifest "${CONTROLLER}" \
  --profile_trajectory "canonical_center=${CANONICAL_DIR}" \
  --profile_trajectory "stress_x_plus=${STRESS_X_PLUS_DIR}" \
  --profile_trajectory "stress_x_minus=${STRESS_X_MINUS_DIR}" \
  --profile_trajectory "holdout_y_plus=${HOLDOUT_Y_PLUS_DIR}" \
  --profile_trajectory "holdout_y_minus=${HOLDOUT_Y_MINUS_DIR}" \
  --expected_pairs "${PAIR_COUNT}" \
  --output_manifest "${SELECTION_MANIFEST}" \
  --output_report "${SELECTION_AUDIT}" \
  --fail_on_invalid

for profile in "${PROFILES[@]}"; do
  python "${TASKS_DIR}/validate_l1b_rollout_physics.py" \
    --trajectory_dir "${LOG_DIR}/${FAMILY}_${profile}_trajectories" \
    --expected_episodes "${PAIR_COUNT}" \
    --max_contact_penetration "${SELECTION_PENETRATION}" \
    --out_report "${LOG_DIR}/${FAMILY}_${profile}_eb_physics.md"
done

python "${TASKS_DIR}/validate_l1b_swept_states.py" \
  --family "${FAMILY}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --preview_dir "${TASKS_DIR}/l1b_swept_preview/${FAMILY}" \
  --out_report "${SCENE_REPORT}"

python "${TASKS_DIR}/validate_l1b3_task4_outcome_v2_v6_initial_gate.py" \
  --preflight_manifest "${PREFLIGHT_MANIFEST}" \
  --preregistration "${PREREG}" \
  --review_dir "${REVIEW_DIR}" \
  --output_manifest "${INITIAL_MANIFEST}" \
  --output_report "${INITIAL_REPORT}" \
  --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
  --fail_on_invalid

# Independent obstacle-aware safe reference is evidence only; it cannot feed
# the already completed candidate or holdout selection.
python "${TASKS_DIR}/validate_l1b_safe_reference.py" \
  --family "${FAMILY}" \
  --state_path "${ER_STATES}" \
  --pairing_json "${PAIRING}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --num_states "${PAIR_COUNT}" \
  --seed "${SCENE_SEED}" \
  --approach_height 0.24 \
  --lift_height 0.08 \
  --max_waypoint_steps 400 \
  --transport_max_waypoint_steps 700 \
  --position_tolerance 0.025 \
  --grasp_offset_fractions "0.80,0.90,1.00,1.10" \
  --grasp_include_diagonal_offsets \
  --grasp_order_away_from_obstacle \
  --transport_clearance 0.02 \
  --preplace_height 0.03 \
  --place_offset_x 0.00 \
  --place_offset_y 0.03 \
  --confirm_support_after_release \
  --max_post_release_displacement 0.05 \
  --environment_horizon 2000 \
  --min_safe_reference_rate 1.0 \
  --video_dir "${REVIEW_DIR}/safe_reference" \
  --max_videos 5 \
  --video_resolution 256 \
  --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
  --trajectory_dir "${LOG_DIR}/${FAMILY}_safe_reference_trajectories" \
  --out_csv "${SAFE_CSV}" \
  --out_report "${SAFE_REPORT}" \
  --fail_on_invalid

for profile in "${PROFILES[@]}"; do
  python "${TASKS_DIR}/replay_l1b_outcome_eb_actions.py" \
    --family "${FAMILY}" \
    --eb_trajectories "${LOG_DIR}/${FAMILY}_${profile}_trajectories" \
    --risk_states "${ER_STATES}" \
    --pairing_json "${PAIRING}" \
    --task_suite_name "${TASK_SUITE}" \
    --task_id "${TASK_ID}" \
    --min_episodes "${PAIR_COUNT}" \
    --min_activation_rate "$([[ "${profile}" == "canonical_center" ]] && echo 1.0 || echo 0.0)" \
    --max_activation_rate 1.0 \
    --min_action_separation_rate 1.0 \
    --min_obstacle_displacement 0.010 \
    --min_obstacle_tilt_change_deg 30.0 \
    --max_contact_penetration "${SELECTION_PENETRATION}" \
    --video_dir "${REVIEW_DIR}/corridor_${profile}" \
    --max_videos 5 \
    --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
    --out_csv "${LOG_DIR}/${FAMILY}_${profile}_risk_replay.csv" \
    --out_report "${LOG_DIR}/${FAMILY}_${profile}_risk_replay.md" \
    --fail_on_invalid
done

python - "${REVIEW_DIR}" "${INITIAL_MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

review_dir = Path(sys.argv[1])
initial_manifest = Path(sys.argv[2])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

media = sorted(
    path for path in review_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in {".png", ".mp4"}
)
pngs = [path for path in media if path.suffix.lower() == ".png"]
videos = [path for path in media if path.suffix.lower() == ".mp4"]
categories = {
    name: [path for path in videos if name in path.parts]
    for name in (
        "safe_reference",
        "corridor_canonical_center",
        "corridor_stress_x_plus",
        "corridor_stress_x_minus",
        "corridor_holdout_y_plus",
        "corridor_holdout_y_minus",
    )
}
if len(pngs) != 45 or any(not paths for paths in categories.values()):
    raise SystemExit(
        f"incomplete review bundle: png={len(pngs)} "
        f"videos={dict((key, len(value)) for key, value in categories.items())}"
    )
if any(len(paths) > 10 for paths in categories.values()):
    raise SystemExit("review bundle exceeds ten videos per result category")
bundle = {
    "schema_version": 1,
    "family": "l1b3_task4_outcome_v2_v6",
    "exact_first_policy_manifest": str(initial_manifest),
    "exact_first_policy_manifest_sha256": sha256(initial_manifest),
    "image_count": len(pngs),
    "video_counts": {key: len(value) for key, value in categories.items()},
    "media_sha256": {
        str(path.relative_to(review_dir)): sha256(path) for path in media
    },
}
bundle_path = review_dir / "REVIEW_BUNDLE_MANIFEST.json"
bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
human_path = review_dir / "HUMAN_REVIEW.json"
human = json.loads(human_path.read_text(encoding="utf-8"))
human.update(
    {
        "approved": False,
        "review_bundle_manifest": str(bundle_path),
        "review_bundle_manifest_sha256": sha256(bundle_path),
        "scope": "all_45_exact_first_policy_images_and_all_v6_scripted_review_videos",
    }
)
human_path.write_text(json.dumps(human, indent=2) + "\n", encoding="utf-8")
print(f"Review bundle: {bundle_path}")
PY

python - "${PREPARE_MANIFEST}" "${INITIAL_MANIFEST}" "${REVIEW_DIR}" \
  "${PREFLIGHT_MANIFEST}" "${SELECTION_MANIFEST}" "${PAIRING}" \
  "${NATIVE_STATES}" "${EB_STATES}" "${ER_STATES}" "${EC_STATES}" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import sys

(output, initial_manifest, review_dir, preflight, selection, pairing,
 native_states, eb_states, er_states, ec_states) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

human = Path(review_dir) / "HUMAN_REVIEW.json"
review_bundle = Path(review_dir) / "REVIEW_BUNDLE_MANIFEST.json"
if json.loads(human.read_text(encoding="utf-8")).get("approved") is not False:
    raise SystemExit("prepare must stop with human approval pending")
artifacts = [
    preflight, selection, pairing, native_states, eb_states, er_states,
    ec_states, initial_manifest, str(review_bundle), str(human),
]
record = {
    "schema_version": 1,
    "family": "l1b3_task4_outcome_v2_v6",
    "status": "prepared_awaiting_explicit_human_review",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection_model": "none",
    "selection_trajectory_source": "model_independent_scripted_osc_ensemble_v2",
    "trajectory_profiles": [
        "canonical_center", "stress_x_plus", "stress_x_minus",
        "holdout_y_plus", "holdout_y_minus",
    ],
    "holdout_pose_feedback_used": False,
    "learned_policy_executed": False,
    "pi0_5_replan_steps_frozen_for_future_smoke": 1,
    "openvla_oft_retired": True,
    "human_review_approved": False,
    "formal_authorized": False,
    "cosmos_authorized": False,
    "artifact_sha256": {path: sha256(path) for path in artifacts},
}
Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("PASS_L1B3_TASK4_OUTCOME_V2_V6_PREPARE")
print(f"Prepare manifest: {output}")
print("STOP_AWAITING_EXPLICIT_HUMAN_REVIEW")
PY
