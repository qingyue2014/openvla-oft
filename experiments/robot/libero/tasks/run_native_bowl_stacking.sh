#!/usr/bin/env bash
set -euo pipefail

# Native LIBERO-90 pure bowl stacking tasks (IDs 16 and 17).
# The loader auto-detects RLinf's discrete-action + lora_adapter layout.
CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora}"
TASK_IDS="${TASK_IDS:-16,17}"
TRIALS="${TRIALS:-10}"
RUN_NOTE="${RUN_NOTE:-native-bowl-stacking-oft}"

python -m experiments.robot.libero.run_physcog_libero_l1_eval \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_90 \
  --task_ids "${TASK_IDS}" \
  --num_trials_per_task "${TRIALS}" \
  --safety_oracle none \
  --save_video_mode all \
  --save_trajectory False \
  --run_id_note "${RUN_NOTE}"
