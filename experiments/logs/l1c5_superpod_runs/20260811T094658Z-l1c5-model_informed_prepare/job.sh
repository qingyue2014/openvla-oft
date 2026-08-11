#!/bin/bash
#SBATCH --job-name=pc-l1c5-model_informed_prepare
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=normal
#SBATCH --account=trllmout
#SBATCH --time=01:00:00
#SBATCH --output=/project/trllmout/physcog-runs/openvla-oft-l1c5-formal98-b3bd0486/.physcog-agent/worktrees/8a2a288c935b092095c2df55357e21fe9e70851b/.physcog-agent/jobs/20260811T094658Z-l1c5-model_informed_prepare.out
#SBATCH --error=/project/trllmout/physcog-runs/openvla-oft-l1c5-formal98-b3bd0486/.physcog-agent/worktrees/8a2a288c935b092095c2df55357e21fe9e70851b/.physcog-agent/jobs/20260811T094658Z-l1c5-model_informed_prepare.out
source /etc/profile.d/modules.sh
module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"
set -uo pipefail
cd /project/trllmout/physcog-runs/openvla-oft-l1c5-formal98-b3bd0486/.physcog-agent/worktrees/8a2a288c935b092095c2df55357e21fe9e70851b
export PATH=/home/drwqyhappy/.conda/envs/openvla_oft/bin:$PATH
export LIBERO_ROOT=/home/drwqyhappy/04-mycode/LIBERO
export PYTHONPATH=/home/drwqyhappy/04-mycode/LIBERO:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export L1C_MI_PREPARE_STATES=50
printf '__PHYSCOG_COMPUTE_NODE__=%s\n' "$(hostname)"
printf '__PHYSCOG_COMMIT__=%s\n' "$(git rev-parse HEAD)"
rm -rf experiments/robot/libero/tasks/l1c5_mi_v1_eb_states.hdf5
rm -rf experiments/robot/libero/tasks/l1c5_mi_v1_er_states.hdf5
rm -rf experiments/robot/libero/tasks/l1c5_mi_v1_ec_states.hdf5
rm -rf experiments/robot/libero/tasks/l1c5_mi_v1_source_indices.json
rm -rf experiments/robot/libero/tasks/l1c5_mi_v1_state_bundle.json
rm -rf experiments/logs/l1c5_mi_v1_prepare
rm -rf review/L1-C5-MI-v1_task/initialization
rm -rf review/L1-C5-MI-v1_task/safe_reference
set +e
env RENDER_GPU_DEVICE_ID=1 bash experiments/robot/libero/tasks/run_l1c5_model_informed_prepare.sh
physcog_rc=$?
printf '__PHYSCOG_EXIT_CODE__=%s\n' "${physcog_rc}"
exit "${physcog_rc}"
