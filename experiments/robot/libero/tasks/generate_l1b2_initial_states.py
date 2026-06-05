"""
Generate custom initial states for L1-B-2 corridor carry scenario.

Layout (top-down, robot at bottom):

         [plate]          y ≈ +0.28
            |
    [ramekin] [bowl_1]    y ≈ +0.10  (corridor posts, x = ±0.12)
            |
         [cookies]        y ≈ -0.10

Cookie box (~12cm wide) + arm (~8cm) = ~20cm combined width.
Corridor width = 20cm → arm fits, arm+cookies does not fit straight through.
Robot must rotate or detour the cookie box to carry it to the plate.

Usage:
    python generate_l1b2_initial_states.py \
        --bddl tasks/PHYSCOG_L1B2_corridor_carry.bddl \
        --output tasks/l1b2_initial_states.hdf5 \
        --num_states 50
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from libero.libero.envs import OffScreenRenderEnv


# ── Corridor geometry ────────────────────────────────────────────────────────
# All positions are in the simulator world frame (metres).
# Kitchen table surface is at z ≈ 0.825 m in world frame.
# The workspace x-axis runs left-right; y-axis runs near (robot) to far.

TABLE_Z = 0.825          # table surface height

# Cookies box: the object the robot picks up.
# Placed slightly in front of the robot (y negative = nearer to robot).
COOKIES_XYZ   = np.array([ 0.00, -0.05, TABLE_Z + 0.03])
COOKIES_QUAT  = np.array([ 0.0,   0.0,   0.0, 1.0])      # upright, no rotation

# Plate: the goal destination, behind the corridor.
PLATE_XYZ     = np.array([ 0.00,  0.30, TABLE_Z + 0.01])
PLATE_QUAT    = np.array([ 0.0,   0.0,   0.0, 1.0])

# Corridor post LEFT (ramekin): x = -0.12, y = +0.10
# Corridor post RIGHT (bowl):   x = +0.12, y = +0.10
# Gap between inner edges ≈ 0.24 - (2 × post_radius) ≈ 0.24 - 0.08 = 0.16 m
# Arm width ≈ 0.08 m  →  arm fits (0.08 < 0.16)
# Cookie box width ≈ 0.13 m + arm ≈ 0.21 m  →  does NOT fit straight (0.21 > 0.16)
RAMEKIN_XYZ   = np.array([-0.12,  0.10, TABLE_Z + 0.04])
RAMEKIN_QUAT  = np.array([ 0.0,   0.0,   0.0, 1.0])

BOWL_XYZ      = np.array([ 0.12,  0.10, TABLE_Z + 0.04])
BOWL_QUAT     = np.array([ 0.0,   0.0,   0.0, 1.0])

# Jitter range added per episode so each initial state is slightly different.
# Keep cookies and plate jitter small; keep corridor posts fixed.
COOKIES_JITTER = 0.02    # ± 2 cm
PLATE_JITTER   = 0.02


def _set_object_pose(sim, body_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
    """Set the free-joint pose of a named MuJoCo body."""
    joint_name = body_name.replace("_main", "") + "_joint"
    try:
        jid = sim.model.joint_name2id(joint_name)
    except Exception:
        # Fall back: try body name directly as joint name
        try:
            jid = sim.model.joint_name2id(body_name)
        except Exception:
            print(f"  [WARN] Could not find joint for body '{body_name}', skipping.")
            return
    qadr = sim.model.jnt_qposadr[jid]
    sim.data.qpos[qadr:qadr + 3] = pos
    sim.data.qpos[qadr + 3:qadr + 7] = quat   # wxyz in MuJoCo
    sim.forward()


def generate_states(bddl_path: str, n: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)

    states = []
    for i in range(n):
        env.reset()

        # Cookies: add small xy jitter
        c_jitter = rng.uniform(-COOKIES_JITTER, COOKIES_JITTER, size=2)
        cookies_pos = COOKIES_XYZ + np.array([c_jitter[0], c_jitter[1], 0.0])

        # Plate: add small xy jitter
        p_jitter = rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)
        plate_pos = PLATE_XYZ + np.array([p_jitter[0], p_jitter[1], 0.0])

        _set_object_pose(env.sim, "cookies_1_main",                    cookies_pos,  COOKIES_QUAT)
        _set_object_pose(env.sim, "plate_1_main",                      plate_pos,    PLATE_QUAT)
        _set_object_pose(env.sim, "glazed_rim_porcelain_ramekin_1_main", RAMEKIN_XYZ, RAMEKIN_QUAT)
        _set_object_pose(env.sim, "akita_black_bowl_1_main",           BOWL_XYZ,     BOWL_QUAT)

        # Step a few frames to let objects settle
        for _ in range(10):
            env.sim.step()

        states.append(env.sim.get_state().flatten())
        print(f"  Generated state {i + 1}/{n}")

    env.close()
    return states


def save_hdf5(states: list, out_path: str, task_description: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    key = task_description.replace(" ", "_")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(key)
        for i, state in enumerate(states):
            ep = grp.create_group(f"demo_{i}")
            ep.create_dataset("initial_state", data=state)
            ep.attrs["success"] = True   # mark as valid initial state
    print(f"Saved {len(states)} states → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate L1-B-2 corridor carry initial states")
    parser.add_argument("--bddl",    required=True, help="Path to BDDL task file")
    parser.add_argument("--output",  required=True, help="Output HDF5 path")
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.num_states} initial states for L1-B-2...")
    print(f"  BDDL: {args.bddl}")
    print(f"  Corridor: ramekin @ x={RAMEKIN_XYZ[0]:.2f}, bowl @ x={BOWL_XYZ[0]:.2f}, gap ≈ 16 cm")

    states = generate_states(args.bddl, args.num_states, args.seed)

    task_description = "pick up the cookie box and place it on the plate"
    save_hdf5(states, args.output, task_description)


if __name__ == "__main__":
    main()
