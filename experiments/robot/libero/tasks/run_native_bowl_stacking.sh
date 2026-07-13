#!/usr/bin/env bash
set -euo pipefail

# Native LIBERO-90 pure bowl stacking tasks (IDs 16 and 17).
# The loader auto-detects RLinf's discrete-action + lora_adapter layout.
CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora}"
TASK_IDS="${TASK_IDS:-16,17}"
TRIALS="${TRIALS:-10}"
RUN_NOTE="${RUN_NOTE:-native-bowl-stacking-oft}"

LOWER_CHECKPOINT="$(printf '%s' "${CHECKPOINT}" | tr '[:upper:]' '[:lower:]')"
if [[ "${LOWER_CHECKPOINT}" == *"grpo"* ]]; then
  DO_SAMPLE="${DO_SAMPLE:-True}"
  TEMPERATURE="${TEMPERATURE:-1.6}"
  TOP_P="${TOP_P:-1.0}"
else
  DO_SAMPLE="${DO_SAMPLE:-False}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  TOP_P="${TOP_P:-1.0}"
fi

python -m experiments.robot.libero.run_physcog_libero_l1_eval \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_90 \
  --task_ids "${TASK_IDS}" \
  --do_sample "${DO_SAMPLE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --num_trials_per_task "${TRIALS}" \
  --safety_oracle none \
  --save_video_mode all \
  --save_trajectory False \
  --run_id_note "${RUN_NOTE}"
