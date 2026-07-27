#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora}"
CANDIDATE_DIR="experiments/logs/l3a3_task59_native_candidate"
OUT_DIR="experiments/logs/l3a3_task59_eb_source"
ROLLOUT_DIR="rollouts/libero_90/L3-A3-task59-eb-source"
REVIEW_JSON="experiments/robot/libero/tasks/L3-A3_TASK59_POLICY_REVIEW.json"

mkdir -p "${OUT_DIR}"
python experiments/robot/libero/tasks/generate_l3a3_task59_native_candidate.py \
  --out_dir "${CANDIDATE_DIR}"
python experiments/robot/libero/tasks/prepare_l3a3_task59_eb_source.py \
  --candidate_dir "${CANDIDATE_DIR}" \
  --review_json "${REVIEW_JSON}" \
  --out_hdf5 "${OUT_DIR}/eb_eval_state.hdf5" \
  --out_report "${OUT_DIR}/input_binding.json"

python -m experiments.robot.libero.run_physcog_libero_l1_eval \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_90 \
  --task_ids 59 \
  --initial_states_path "${OUT_DIR}/eb_eval_state.hdf5" \
  --safety_oracle none \
  --num_trials_per_task 1 \
  --seed 42 \
  --save_video_mode all \
  --save_trajectory True \
  --trajectory_dir "${OUT_DIR}/trajectories" \
  --trajectory_track_bodies \
    "tomato_sauce_1_main,alphabet_soup_1_main,butter_1_main,wooden_tray_1_main" \
  --render_gpu_device_id -1 \
  --run_id_note "L3-A3-task59-eb-source"

python experiments/robot/libero/tasks/validate_l3a3_task59_eb_source.py \
  --eval_hdf5 "${OUT_DIR}/eb_eval_state.hdf5" \
  --candidate_report "${CANDIDATE_DIR}/report.json" \
  --trajectory "${OUT_DIR}/trajectories/task59_ep000.npz" \
  --rollout_dir "${ROLLOUT_DIR}" \
  --out_json "${OUT_DIR}/report.json" \
  --out_md "${OUT_DIR}/report.md"
