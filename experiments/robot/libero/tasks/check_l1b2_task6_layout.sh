#!/usr/bin/env bash
set -euo pipefail

# Run from the OpenVLA-OFT repository root on the server.
# This regenerates the L1-B2 task6 initial states, then prints the saved MuJoCo
# body positions from demo_0 so the layout can be checked before running eval.

STATE_PATH="${1:-experiments/robot/libero/tasks/l1b2_task6_initial_states.hdf5}"
NUM_STATES="${NUM_STATES:-50}"
SEED="${SEED:-42}"

echo "[1/2] Regenerating task6 initial states:"
echo "      ${STATE_PATH}"
rm -f "${STATE_PATH}"

python experiments/robot/libero/tasks/generate_l1b2_initial_states.py \
  --variant task6 \
  --output "${STATE_PATH}" \
  --num_states "${NUM_STATES}" \
  --seed "${SEED}"

echo
echo "[2/2] Inspecting demo_0 body positions from HDF5:"
python - "${STATE_PATH}" <<'PY'
import os
import sys

import h5py

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    benchmark,
    get_libero_path,
)

state_path = sys.argv[1]

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(6)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.reset()

key = task.language.replace(" ", "_")
with h5py.File(state_path, "r") as f:
    print(f"HDF5 key: {key}")
    print(f"num demos: {len(f[key].keys())}")
    state = f[key]["demo_0"]["initial_state"][:]

env.set_init_state(state)

for name in [
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "akita_black_bowl_1_main",
    "plate_1_main",
]:
    bid = env.sim.model.body_name2id(name)
    pos = env.sim.data.body_xpos[bid]
    print(f"{name:42s} x={pos[0]: .4f} y={pos[1]: .4f} z={pos[2]: .4f}")

env.close()

print("\nExpected xy positions:")
print("  cookies_1_main                       x=-0.1300 y= 0.0800")
print("  glazed_rim_porcelain_ramekin_1_main  x= 0.1300 y= 0.0800")
print("  akita_black_bowl_1_main              x=-0.0300 y= 0.0450  +/- 0.005 jitter")
print("  plate_1_main                         x= 0.0000 y= 0.3000  +/- 0.015 jitter")
PY

cat <<'EOF'

To run the L1-B2 task6 eval after the layout check passes:

python -m experiments.robot.libero.run_physcog_libero_l1_eval \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --task_ids 6 \
  --initial_states_path experiments/robot/libero/tasks/l1b2_task6_initial_states.hdf5 \
  --safety_oracle held_object_corridor \
  --held_object_body akita_black_bowl_1_main \
  --corridor_body "cookies_1_main,glazed_rim_porcelain_ramekin_1_main" \
  --num_trials_per_task 50 \
  --run_id_note L1-B2-task6-cookie-ramekin
EOF
