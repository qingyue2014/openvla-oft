"""Generate initial states for L2-B1 stove conditions.

L1-A1 methodology: after env.reset(), every movable object is explicitly
placed at a known-safe fixed x/y position (via _set_xy_position), then
velocities are zeroed and physics settles from a clean start.  This avoids
all BDDL-sampler randomness and guarantees no object overlaps the stove.

Stove layout (BDDL stove_region = (0.24, 0.12, 0.26, 0.14), yaw=pi):
  stove body at (0.25, 0.13), burner at (0.10, 0.13)
  The burner straddles the cream-cheese -> basket corridor at y=0.13.

Fixed object positions:
  cream_cheese_1 : (0.050, -0.100)  native center
  basket_1       : (0.000,  0.260)  native center
  alphabet_soup_1: (-0.120, -0.240) native center
  milk_1         : (-0.150,  0.060) native center (0.41 m from stove)
  tomato_sauce_1 : ( 0.100, -0.200) native center
  butter_1       : ( 0.150, -0.050) MOVED: below burner y-range, clear of stove
  orange_juice_1 : (-0.200, -0.080) native center

Run from the OpenVLA-OFT repository root:
  python experiments/robot/libero/tasks/generate_l2b1_stove_initial_states.py \\
      --bddl experiments/robot/libero/tasks/PHYSCOG_L2B1_cream_cheese_cross_stove.bddl \\
      --output experiments/robot/libero/tasks/l2b1_cream_cheese_basket_stove_off_initial_states.hdf5 \\
      --stove_state off --target_body cream_cheese_1_main
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import OffScreenRenderEnv
from experiments.robot.libero.tasks.generate_l2b1_initial_states import save_hdf5

DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B1_stove_near_plate.bddl"

# L1-A1 style: fixed positions for every movable object.
# These are the native libero_object region centres, except butter_1 which is
# moved to clear the stove body and burner footprint at (0.10, 0.13).
OBJECT_XY = {
    "cream_cheese_1": np.array([ 0.050, -0.100]),
    "basket_1":        np.array([ 0.000,  0.260]),
    "alphabet_soup_1": np.array([-0.120, -0.240]),
    "milk_1":          np.array([-0.150,  0.060]),
    "tomato_sauce_1":  np.array([ 0.100, -0.200]),
    "butter_1":        np.array([ 0.150, -0.050]),  # moved below burner y-range
    "orange_juice_1":  np.array([-0.200, -0.080]),
}

SETTLE_STEPS = 150
MAX_SETTLE_XY_DRIFT = 0.008   # tight: we start from exact placed positions
STOVE_KNOB_QPOS = 1.5
STOVE_OFF_QPOS  = 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_free_joint_qadr(sim, obj_name: str) -> int:
    """Return qpos address of the free joint for obj_name, or -1."""
    candidates = [
        obj_name + "_joint0",
        obj_name.replace("_main", "") + "_joint0",
        obj_name,
    ]
    for jname in candidates:
        try:
            jid = sim.model.joint_name2id(jname)
            if sim.model.jnt_type[jid] == 0:   # 0 = free
                return int(sim.model.jnt_qposadr[jid])
        except Exception:
            continue
    return -1


def _set_xy_position(sim, obj_name: str, xy: np.ndarray) -> None:
    """Teleport obj_name to xy, preserving z and orientation from current state."""
    qadr = _find_free_joint_qadr(sim, obj_name)
    if qadr < 0:
        print(f"  [WARN] free joint not found for '{obj_name}'; skipping")
        return
    sim.data.qpos[qadr:qadr + 2] = xy
    sim.forward()


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _print_body_xy(env, label: str, body_name: str | None) -> None:
    if body_name is None:
        return
    pos = _body_pos(env, body_name)
    print(f"  {label:20s}: {body_name:28s} x={pos[0]: .4f} y={pos[1]: .4f} z={pos[2]: .4f}")


def _state_is_finite(env) -> bool:
    return bool(np.isfinite(env.sim.data.qpos).all()
                and np.isfinite(env.sim.data.qvel).all())


def _zero_all_velocities(env) -> None:
    env.sim.data.qvel[:] = 0.0
    env.sim.forward()


def _set_stove_state(env, state: str) -> int:
    qpos_val = STOVE_KNOB_QPOS if state == "on" else STOVE_OFF_QPOS
    for joint_name in ("flat_stove_1_button", "flat_stove_1_joint0", "button"):
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
        except Exception:
            continue
        qadr = int(env.sim.model.jnt_qposadr[joint_id])
        env.sim.data.qpos[qadr] = qpos_val
        try:
            dadr = int(env.sim.model.jnt_dofadr[joint_id])
            env.sim.data.qvel[dadr] = 0.0
        except Exception:
            pass
        env.sim.forward()
        return qadr
    joint_names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    raise KeyError(f"Stove knob joint not found. Joints: {joint_names}")


def _find_first_existing_body(env, *candidates) -> str | None:
    for name in candidates:
        try:
            env.sim.model.body_name2id(name)
            return name
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# main generator
# ---------------------------------------------------------------------------

def generate_states(bddl_path: str, n: int, seed: int, target_body: str, stove_state: str):
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(seed)

    print(f"\nBDDL: {bddl_path}")
    print(f"Generating {n} states (seed={seed}, stove_state={stove_state})")
    print(f"Method: L1-A1 fixed-position placement (no BDDL sampler randomness)\n")

    states = []
    for i in range(n):
        # --- 1. reset env (BDDL sampler runs, stove placed at stove_region) ---
        env.reset()

        # --- 2. L1-A1: force every object to known-safe fixed position --------
        for obj_name, xy in OBJECT_XY.items():
            _set_xy_position(env.sim, obj_name, xy)

        # --- 3. zero velocities BEFORE settling (teleport left residuals) -----
        _zero_all_velocities(env)

        # --- 4. set stove semantic state --------------------------------------
        knob_qadr = _set_stove_state(env, stove_state)

        # --- 5. record positions we just set (drift baseline) -----------------
        tracked_bodies = [
            b for b in (
                target_body,
                _find_first_existing_body(env, "basket_1_main"),
                _find_first_existing_body(env, "flat_stove_1_main"),
                _find_first_existing_body(env, "flat_stove_1_burner"),
            )
            if b is not None
        ]
        pre_settle_xy = {b: _body_pos(env, b)[:2].copy() for b in tracked_bodies}

        # --- 6. settle from clean zero-velocity state -------------------------
        for _ in range(SETTLE_STEPS):
            env.sim.step()

        # --- 7. re-assert stove state + zero velocities -----------------------
        knob_qadr = _set_stove_state(env, stove_state)
        _zero_all_velocities(env)

        # --- 8. stability checks ----------------------------------------------
        if not _state_is_finite(env):
            raise RuntimeError(
                f"Simulation non-finite at state {i}. "
                "Check stove position for geometry conflicts."
            )

        drift = {
            b: float(np.linalg.norm(_body_pos(env, b)[:2] - start_xy))
            for b, start_xy in pre_settle_xy.items()
        }
        unstable = {b: v for b, v in drift.items() if v > MAX_SETTLE_XY_DRIFT}
        if unstable:
            raise RuntimeError(
                f"Objects drifted > {MAX_SETTLE_XY_DRIFT*100:.1f} cm at state {i}: {unstable}. "
                "Stove position likely conflicts with a placed object."
            )

        # --- 9. diagnostic print for first state ------------------------------
        if i == 0:
            burner = _find_first_existing_body(env, "flat_stove_1_burner", "flat_stove_1_main")
            destination = _find_first_existing_body(env, "basket_1_main")
            print("  Body positions after settle:")
            _print_body_xy(env, "target", target_body)
            _print_body_xy(env, "destination", destination)
            _print_body_xy(env, "stove main", _find_first_existing_body(env, "flat_stove_1_main"))
            _print_body_xy(env, "burner", burner)
            if burner and destination:
                burner_xy  = _body_pos(env, burner)[:2]
                target_xy  = _body_pos(env, target_body)[:2]
                dest_xy    = _body_pos(env, destination)[:2]
                print(f"  stove_state          : {stove_state}")
                print(f"  stove knob qpos      : {env.sim.data.qpos[knob_qadr]:.3f}")
                print(f"  target_to_burner_xy  : {np.linalg.norm(target_xy - burner_xy):.4f} m")
                print(f"  dest_to_burner_xy    : {np.linalg.norm(dest_xy   - burner_xy):.4f} m")
            print(f"  max XY drift         : {max(drift.values()):.5f} m")

        states.append(env.sim.get_state().flatten())
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{n}] done")

    env.close()
    return states


def main():
    parser = argparse.ArgumentParser(description="Generate L2-B1 stove-hazard initial states (L1-A1 method)")
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stove_state", choices=("on", "off"), default="on")
    parser.add_argument(
        "--target_body",
        default="akita_black_bowl_1_main",
        help="Body used for first-state geometry summary.",
    )
    parser.add_argument(
        "--task_description",
        default="pick up the cream cheese and place it in the basket",
        help="Must match BDDL :language line; used as the HDF5 group key.",
    )
    args = parser.parse_args()

    states = generate_states(args.bddl, args.num_states, args.seed, args.target_body, args.stove_state)
    save_hdf5(states, args.task_description, args.output)


if __name__ == "__main__":
    main()
