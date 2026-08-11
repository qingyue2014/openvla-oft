#!/bin/bash
#SBATCH --job-name=pc-l1c5-pi05_smoke_model_informed_v1
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=normal
#SBATCH --account=trllmout
#SBATCH --time=01:00:00
#SBATCH --output=/project/trllmout/physcog-runs/openvla-oft-l1c4/.physcog-agent/worktrees/11613ae279faf0c3c3ea5ac47b7476afb5380736/.physcog-agent/jobs/20260811T103050Z-l1c5-pi05_smoke_model_informed_v1.out
#SBATCH --error=/project/trllmout/physcog-runs/openvla-oft-l1c4/.physcog-agent/worktrees/11613ae279faf0c3c3ea5ac47b7476afb5380736/.physcog-agent/jobs/20260811T103050Z-l1c5-pi05_smoke_model_informed_v1.out
source /etc/profile.d/modules.sh
module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"
set -uo pipefail
cd /project/trllmout/physcog-runs/openvla-oft-l1c4/.physcog-agent/worktrees/11613ae279faf0c3c3ea5ac47b7476afb5380736
export PATH=/home/drwqyhappy/.conda/envs/openvla_oft/bin:$PATH
export LIBERO_ROOT=/home/drwqyhappy/04-mycode/LIBERO
export PYTHONPATH=/home/drwqyhappy/04-mycode/LIBERO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export L1C_SMOKE_TRIALS=5
printf '__PHYSCOG_COMPUTE_NODE__=%s\n' "$(hostname)"
printf '__PHYSCOG_COMMIT__=%s\n' "$(git rev-parse HEAD)"
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_manifest.json
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_results.json
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_results.md
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_native_preflight.json
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_native_preflight.md
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_eb_index.jsonl
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_er_index.jsonl
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_ec_index.jsonl
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_videos
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_server.log
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_gate
rm -rf experiments/logs/l1c5_pi05-smoke-model-informed-v1_review_artifacts.sha256
rm -rf review/L1-C5-MI-v1_task/pi05_smoke
rm -rf rollouts/libero_object/L1-C5-orange-juice-occupied-basket-eb-pi05-smoke-model-informed-v1
rm -rf rollouts/libero_object/L1-C5-orange-juice-occupied-basket-risk-pi05-smoke-model-informed-v1
rm -rf rollouts/libero_object/L1-C5-orange-juice-occupied-basket-ec-pi05-smoke-model-informed-v1
set +e
env L1C_EVAL_VARIANT=model-informed-v1 RENDER_GPU_DEVICE_ID=1 SAVE_VIDEO_MODE=all bash experiments/robot/libero/tasks/run_model_l1c_eval.sh pi05 l1c5 smoke
physcog_rc=$?
printf '__PHYSCOG_EXIT_CODE__=%s\n' "${physcog_rc}"
exit "${physcog_rc}"
