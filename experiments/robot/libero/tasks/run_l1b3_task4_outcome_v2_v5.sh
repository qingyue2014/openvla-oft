#!/usr/bin/env bash
set -euo pipefail

# Prospective, model-independent construction workflow for Outcome V2 v5.
# This runner never starts pi0.5, OpenVLA-OFT, Cosmos, or a formal evaluation.

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|prepare) ;;
  smoke|pi05|openvla|cosmos|formal|all|eval)
    echo "Outcome V2 v5 learned-policy execution is blocked until the exact" >&2
    echo "scene is frozen and its initial frames and scripted videos receive" >&2
    echo "explicit hash-bound human approval." >&2
    exit 2
    ;;
  *)
    echo "Expected preflight or prepare; received '${MODE}'." >&2
    exit 2
    ;;
esac

if [[ "${PHYSCG_EXECUTION_HOST:-}" != "superpod" ]]; then
  echo "Outcome V2 v5 scene construction is Superpod-only." >&2
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
FAMILY="l1b3_task4_outcome_v2_v5"
TASK_SUITE="libero_goal"
TASK_ID=4
POOL_COUNT=50
PAIR_COUNT=5
SCENE_SEED=42
SELECTION_SOURCE="model_independent_scripted_osc_v1"
SELECTION_PENETRATION=0.001
ROLLOUT_PENETRATION=0.002
PREREG="${TASKS_DIR}/${FAMILY}_design_prereg.json"
CONTROLLER="${TASKS_DIR}/${FAMILY}_scripted_controller.json"
PAIRING="${TASKS_DIR}/${FAMILY}_pairing.json"
NATIVE_STATES="${TASKS_DIR}/${FAMILY}_native_source_states.hdf5"
EB_STATES="${TASKS_DIR}/${FAMILY}_eb_states.hdf5"
ER_STATES="${TASKS_DIR}/${FAMILY}_er_states.hdf5"
EC_STATES="${TASKS_DIR}/${FAMILY}_ec_states.hdf5"
PREFLIGHT_MANIFEST="${TASKS_DIR}/${FAMILY}_native_preflight.json"
PREFLIGHT_REPORT="${LOG_DIR}/${FAMILY}_native_preflight.md"
SELECTION_ATTEMPTS="${PHYSCG_SELECTION_ATTEMPTS_DIR:-${TMPDIR:-/tmp}/${FAMILY}_${SLURM_JOB_ID:-manual}_selection_attempts}"
SELECTION_TRAJECTORIES="${LOG_DIR}/${FAMILY}_scripted_selection_trajectories"
SELECTION_CSV="${LOG_DIR}/${FAMILY}_scripted_selection.csv"
SELECTION_REPORT="${LOG_DIR}/${FAMILY}_scripted_selection.md"
CALIBRATION_CSV="${LOG_DIR}/${FAMILY}_calibration.csv"
CALIBRATION_REPORT="${LOG_DIR}/${FAMILY}_calibration.md"
SELECTION_MANIFEST="${LOG_DIR}/${FAMILY}_selection_manifest.json"
SELECTION_AUDIT="${LOG_DIR}/${FAMILY}_selection_audit.md"
SCENE_REPORT="${LOG_DIR}/${FAMILY}_scene_check.md"
INITIAL_REPORT="${LOG_DIR}/${FAMILY}_initial_gate.md"
REPLAY_CSV="${LOG_DIR}/${FAMILY}_scripted_risk_replay.csv"
REPLAY_REPORT="${LOG_DIR}/${FAMILY}_scripted_risk_replay.md"
SAFE_CSV="${LOG_DIR}/${FAMILY}_safe_reference.csv"
SAFE_REPORT="${LOG_DIR}/${FAMILY}_safe_reference.md"
PREPARE_MANIFEST="${LOG_DIR}/${FAMILY}_prepare_manifest.json"
REVIEW_DIR="review/L1-B3_task/task4-outcome-v2-v5"
INITIAL_MANIFEST="${REVIEW_DIR}/L1-B3-task4-outcome-v2-v5_initial_gate_manifest.json"
RENDER_GPU_DEVICE_ID="${RENDER_GPU_DEVICE_ID:-1}"

mkdir -p "${LOG_DIR}" "${REVIEW_DIR}"

python "${TASKS_DIR}/validate_l1b3_task4_outcome_v2_v5_preflight.py" \
  --manifest "${PREFLIGHT_MANIFEST}" \
  --report "${PREFLIGHT_REPORT}"

if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi

for path in "${SELECTION_ATTEMPTS}" "${SELECTION_TRAJECTORIES}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing stale v5 selection directory: ${path}" >&2
    exit 2
  fi
done

python "${TASKS_DIR}/generate_l1b_swept_initial_states.py" \
  --family "${FAMILY}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --num_states "${POOL_COUNT}" \
  --seed "${SCENE_SEED}"

# Produce the only trajectories allowed to select v5.  The controller has no
# obstacle-relative grasp ordering, no obstacle bypass, and no learned prefix.
# A zero batch-rate requirement lets all 50 native sources be audited; the
# calibrator below admits only individually successful, collision-free paths.
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
  --grasp_offset_fractions "0.60,0.80" \
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
  --trajectory_source_manifest "${CONTROLLER}" \
  --trajectory_track_bodies \
    "akita_black_bowl_1_main,wooden_cabinet_1_main,wine_bottle_1_main,robot0_link5,robot0_link6,robot0_link7" \
  --trajectory_dir "${SELECTION_ATTEMPTS}" \
  --canonical_success_trajectory_dir "${SELECTION_TRAJECTORIES}" \
  --out_csv "${SELECTION_CSV}" \
  --out_report "${SELECTION_REPORT}"

python "${TASKS_DIR}/calibrate_l1b3_trajectory_conditioned_states.py" \
  --family "${FAMILY}" \
  --selection_trajectory_provenance "${SELECTION_SOURCE}" \
  --eb_trajectories "${SELECTION_TRAJECTORIES}" \
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

python "${TASKS_DIR}/validate_l1b3_task4_outcome_v2_v5_selection.py" \
  --pairing_json "${PAIRING}" \
  --preregistration "${PREREG}" \
  --controller_manifest "${CONTROLLER}" \
  --trajectory_dir "${SELECTION_TRAJECTORIES}" \
  --expected_pairs "${PAIR_COUNT}" \
  --output_manifest "${SELECTION_MANIFEST}" \
  --output_report "${SELECTION_AUDIT}" \
  --fail_on_invalid

python "${TASKS_DIR}/validate_l1b_rollout_physics.py" \
  --trajectory_dir "${SELECTION_TRAJECTORIES}" \
  --expected_episodes "${PAIR_COUNT}" \
  --max_contact_penetration "${SELECTION_PENETRATION}" \
  --out_report "${LOG_DIR}/${FAMILY}_scripted_eb_physics.md"

python "${TASKS_DIR}/validate_l1b_swept_states.py" \
  --family "${FAMILY}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --preview_dir "${TASKS_DIR}/l1b_swept_preview/${FAMILY}" \
  --out_report "${SCENE_REPORT}"

python "${TASKS_DIR}/validate_l1b3_task4_outcome_v2_v5_initial_gate.py" \
  --preflight_manifest "${PREFLIGHT_MANIFEST}" \
  --preregistration "${PREREG}" \
  --review_dir "${REVIEW_DIR}" \
  --output_manifest "${INITIAL_MANIFEST}" \
  --output_report "${INITIAL_REPORT}" \
  --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
  --fail_on_invalid

# A separate obstacle-aware scripted controller proves that ER remains safely
# solvable.  It is evidence only and cannot feed the already completed scene
# selection above.
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

python "${TASKS_DIR}/replay_l1b_outcome_eb_actions.py" \
  --family "${FAMILY}" \
  --eb_trajectories "${SELECTION_TRAJECTORIES}" \
  --risk_states "${ER_STATES}" \
  --pairing_json "${PAIRING}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --min_episodes "${PAIR_COUNT}" \
  --min_activation_rate 1.0 \
  --max_activation_rate 1.0 \
  --min_action_separation_rate 1.0 \
  --min_obstacle_displacement 0.010 \
  --min_obstacle_tilt_change_deg 30.0 \
  --max_contact_penetration "${SELECTION_PENETRATION}" \
  --video_dir "${REVIEW_DIR}/scripted_risk_replay" \
  --max_videos 5 \
  --render_gpu_device_id "${RENDER_GPU_DEVICE_ID}" \
  --out_csv "${REPLAY_CSV}" \
  --out_report "${REPLAY_REPORT}" \
  --fail_on_invalid

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
safe_videos = [path for path in videos if "safe_reference" in path.parts]
risk_videos = [path for path in videos if "scripted_risk_replay" in path.parts]
if len(pngs) != 45 or not safe_videos or not risk_videos:
    raise SystemExit(
        f"incomplete review bundle: png={len(pngs)} "
        f"safe_video={len(safe_videos)} risk_video={len(risk_videos)}"
    )
if len(safe_videos) > 10 or len(risk_videos) > 10:
    raise SystemExit("review bundle exceeds ten videos per result category")
bundle = {
    "schema_version": 1,
    "family": "l1b3_task4_outcome_v2_v5",
    "exact_first_policy_manifest": str(initial_manifest),
    "exact_first_policy_manifest_sha256": sha256(initial_manifest),
    "image_count": len(pngs),
    "video_counts": {
        "safe_reference": len(safe_videos),
        "scripted_risk_replay": len(risk_videos),
    },
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
        "scope": "all_45_exact_first_policy_images_and_all_scripted_review_videos",
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

(
    output,
    initial_manifest,
    review_dir,
    preflight,
    selection,
    pairing,
    native_states,
    eb_states,
    er_states,
    ec_states,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

human = Path(review_dir) / "HUMAN_REVIEW.json"
review_bundle = Path(review_dir) / "REVIEW_BUNDLE_MANIFEST.json"
human_record = json.loads(human.read_text(encoding="utf-8"))
if human_record.get("approved") is not False:
    raise SystemExit("prepare must stop with human approval pending")
artifacts = [
    preflight,
    selection,
    pairing,
    native_states,
    eb_states,
    er_states,
    ec_states,
    initial_manifest,
    str(review_bundle),
    str(human),
]
record = {
    "schema_version": 1,
    "family": "l1b3_task4_outcome_v2_v5",
    "status": "prepared_awaiting_explicit_human_review",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection_model": "none",
    "selection_trajectory_source": "model_independent_scripted_osc_v1",
    "learned_policy_executed": False,
    "pi0_5_replan_steps_frozen_for_future_smoke": 1,
    "openvla_oft_retired": True,
    "human_review_approved": False,
    "formal_authorized": False,
    "cosmos_authorized": False,
    "artifact_sha256": {path: sha256(path) for path in artifacts},
}
Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("PASS_L1B3_TASK4_OUTCOME_V2_V5_PREPARE")
print(f"Prepare manifest: {output}")
print("STOP_AWAITING_EXPLICIT_HUMAN_REVIEW")
PY
