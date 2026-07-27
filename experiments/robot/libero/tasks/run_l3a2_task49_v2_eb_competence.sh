#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"
OUT_DIR="experiments/logs/l3a2_task49_v2_eb_competence"
ROLLOUT_DIR="rollouts/libero_90/L3-A2-task49-v2-eb-competence"

python experiments/robot/libero/tasks/task49_v2_eb_competence.py \
  prepare \
  --out_dir "${OUT_DIR}" \
  --rollout_dir "${ROLLOUT_DIR}"

python -m experiments.robot.libero.run_physcog_libero_l1_eval \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_90 \
  --task_ids 49 \
  --initial_states_path "${OUT_DIR}/eb_eval_state.hdf5" \
  --safety_oracle none \
  --num_trials_per_task 1 \
  --num_steps_wait 0 \
  --seed 42 \
  --save_video_mode all \
  --save_trajectory True \
  --trajectory_dir "${OUT_DIR}/trajectories" \
  --trajectory_track_bodies \
    "tomato_sauce_1_main,alphabet_soup_1_main,cream_cheese_1_main,ketchup_1_main,basket_1_main" \
  --render_gpu_device_id -1 \
  --run_id_note "L3-A2-task49-v2-eb-competence"

python experiments/robot/libero/tasks/task49_v2_eb_competence.py \
  validate \
  --out_dir "${OUT_DIR}" \
  --rollout_dir "${ROLLOUT_DIR}"
