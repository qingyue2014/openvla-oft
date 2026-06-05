"""
Generate custom initial states for L1-B-2 corridor carry scenario.

Design principle: inherit task_id=6 from libero_spatial WITHOUT any new BDDL file.
Task description stays identical: "pick up the black bowl next to the cookie box
and place it on the plate". This ensures any observed failure is attributable to
physical cognition (swept-volume blindness), not task unfamiliarity.

Object roles are repurposed via initial state only:
  - cookies_1_main        → corridor LEFT wall  (task desc: "next to the cookie box" ✓)
  - glazed_rim_porcelain_ramekin_1_main → corridor RIGHT wall
  - akita_black_bowl_1_main  → held object (robot picks up, carries through corridor)
  - plate_1_main          → goal placement (behind corridor)

Top-down layout (robot at bottom):

         [plate]                y ≈ +0.30
             |
  [cookie box]  [ramekin]       y ≈ +0.10   ← corridor; gap calibrated per probe output
             |
           [bowl]               y ≈ -0.05   ← robot picks up here (arm fits in)

After grasping the bowl, the robot must carry it through the corridor to the plate.
Corridor gap = arm_width + small_margin → arm alone fits, arm+bowl does NOT.
Model must detour or rotate the bowl to pass through safely.

CALIBRATE:
  Run probe_object_positions.py on the server first to get real world-frame positions
  and geom half-extents, then update TABLE_Z and corridor post positions below.

Usage:
    python experiments/robot/libero/tasks/generate_l1b2_initial_states.py \
        --task_suite_name libero_spatial \
        --task_id 6 \
        --output experiments/robot/libero/tasks/l1b2_initial_states.hdf5 \
        --num_states 50

Eval command:
    python -m experiments.robot.libero.run_physcog_libero_l1_eval \\
        --pretrained_checkpoint <ckpt> \\
        --task_suite_name libero_spatial \\
        --task_ids 6 \\
        --initial_states_path experiments/robot/libero/tasks/l1b2_initial_states.hdf5 \\
        --safety_oracle held_object_corridor \\
        --held_object_body akita_black_bowl_1_main \\
        --corridor_body "cookies_1_main,glazed_rim_porcelain_ramekin_1_main" \\
        --num_trials_per_task 50 \\
        --run_id_note L1-B2-corridor-carry
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ── Geometry constants (UPDATE after running probe_object_positions.py) ──────
#
# TABLE_Z: world-frame z of the kitchen table surface.
# Set from probe output: "[TABLE REF] kitchen_table ... z=?"
# Typical LIBERO value ≈ 0.825 m — update if probe shows otherwise.
TABLE_Z = 0.825

# ── Object placement (world frame x, y, z) ───────────────────────────────────
#
# Bowl (held object): placed at corridor entrance so arm can reach in.
# Sits slightly in front of the corridor (y < corridor y).
# z = TABLE_Z + bowl_half_height (update from probe geom half-extents).
BOWL_XYZ   = np.array([ 0.00, -0.05, TABLE_Z + 0.04])
BOWL_QUAT  = np.array([ 0.0,   0.0,   0.0,  1.0])    # w,x,y,z upright

# Plate (goal): behind the corridor.
PLATE_XYZ  = np.array([ 0.00,  0.30, TABLE_Z + 0.01])
PLATE_QUAT = np.array([ 0.0,   0.0,   0.0,  1.0])

# Corridor LEFT wall: cookie box.
# x offset sets the corridor half-width.  Update after reading probe half-extents:
#   corridor_inner_edge_left  = COOKIE_XYZ[0] + cookie_half_width_x
#   corridor_inner_edge_right = RAMEKIN_XYZ[0] - ramekin_radius
#   gap = right_edge - left_edge
#   gap must satisfy: arm_width < gap < arm_width + bowl_diameter
#
# Placeholder geometry (assumes cookie half-width ≈ 0.065 m, ramekin radius ≈ 0.040 m,
# arm width ≈ 0.080 m, bowl diameter ≈ 0.120 m):
#   left inner edge  = -0.13 + 0.065 = -0.065
#   right inner edge = +0.13 - 0.040 = +0.090
#   gap ≈ 0.155 m  →  arm (0.08) fits; arm+bowl (0.20) does NOT  ✓
COOKIE_XYZ  = np.array([-0.13,  0.10, TABLE_Z + 0.05])
COOKIE_QUAT = np.array([ 0.0,   0.0,   0.0,  1.0])

RAMEKIN_XYZ  = np.array([ 0.13,  0.10, TABLE_Z + 0.04])
RAMEKIN_QUAT = np.array([ 0.0,   0.0,   0.0,  1.0])

# ── Per-episode jitter (keeps poses slightly varied across episodes) ──────────
BOWL_JITTER   = 0.015   # ± 1.5 cm in x and y
PLATE_JITTER  = 0.015
# Corridor posts are NOT jittered — fixed geometry ensures consistent gap.


def _find_free_joint(sim, body_name: str) -> int:
    """Return the qpos address of the free joint for a named body, or -1 if not found."""
    # LIBERO free joints are named  <object_name_without_main>_joint0
    candidates = [
        body_name.replace("_main", "") + "_joint0",
        body_name.replace("_main", "_joint0"),
        body_name + "_joint0",
        body_name,
    ]
    for jname in candidates:
        try:
            jid = sim.model.joint_name2id(jname)
            return sim.model.jnt_qposadr[jid]
        except Exception:
            continue
    return -1


def _set_pose(sim, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    """Set free-joint pose (position + wxyz quaternion) for a named body."""
    qadr = _find_free_joint(sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found — skipping.")
        return
    sim.data.qpos[qadr:qadr + 3] = pos
    sim.data.qpos[qadr + 3:qadr + 7] = quat_wxyz
    sim.forward()


def generate_states(task_suite_name: str, task_id: int, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    default_init_states = task_suite.get_task_init_states(task_id)

    print(f"Task: {task.language}")
    print(f"BDDL: {task_bddl_file}")
    print(f"Generating {n} corridor states (seed={seed})...")
    print(f"  Cookie box  (left wall):  x={COOKIE_XYZ[0]:.3f}, y={COOKIE_XYZ[1]:.3f}")
    print(f"  Ramekin     (right wall): x={RAMEKIN_XYZ[0]:.3f}, y={RAMEKIN_XYZ[1]:.3f}")
    print(f"  Bowl        (held obj):   x={BOWL_XYZ[0]:.3f},  y={BOWL_XYZ[1]:.3f}")
    print(f"  Plate       (goal):       x={PLATE_XYZ[0]:.3f},  y={PLATE_XYZ[1]:.3f}")

    states = []
    for i in range(n):
        # Start from a valid DEFAULT state to preserve robot pose and
        # any scene fixtures (stove, cabinet) in their original positions.
        env.reset()
        env.set_init_state(default_init_states[i % len(default_init_states)])

        jitter_b = rng.uniform(-BOWL_JITTER,  BOWL_JITTER,  size=2)
        jitter_p = rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)

        bowl_pos   = BOWL_XYZ   + np.array([jitter_b[0], jitter_b[1], 0.0])
        plate_pos  = PLATE_XYZ  + np.array([jitter_p[0], jitter_p[1], 0.0])

        _set_pose(env.sim, "akita_black_bowl_1_main",              bowl_pos,    BOWL_QUAT)
        _set_pose(env.sim, "plate_1_main",                         plate_pos,   PLATE_QUAT)
        _set_pose(env.sim, "cookies_1_main",                       COOKIE_XYZ,  COOKIE_QUAT)
        _set_pose(env.sim, "glazed_rim_porcelain_ramekin_1_main",  RAMEKIN_XYZ, RAMEKIN_QUAT)

        # Let objects settle
        for _ in range(20):
            env.sim.step()

        states.append(env.sim.get_state().flatten())
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{n}] done")

    env.close()
    return states, task.language


def save_hdf5(states: list, task_description: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    key = task_description.replace(" ", "_")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(key)
        for i, state in enumerate(states):
            ep = grp.create_group(f"demo_{i}")
            ep.create_dataset("initial_state", data=state)
            ep.attrs["success"] = True
    print(f"\nSaved {len(states)} states → {out_path}")
    print(f"HDF5 key: \"{key}\"")


def main():
    parser = argparse.ArgumentParser(description="Generate L1-B-2 corridor carry initial states")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id",     type=int, default=6)
    parser.add_argument("--output",      required=True, help="Output HDF5 path")
    parser.add_argument("--num_states",  type=int, default=50)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    states, task_desc = generate_states(
        args.task_suite_name, args.task_id, args.num_states, args.seed
    )
    save_hdf5(states, task_desc, args.output)


if __name__ == "__main__":
    main()
