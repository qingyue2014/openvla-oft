"""Generate initial states for L2-B1 stove conditions.

The stove fixture pose is baked into the custom BDDL. This script controls only
the stove semantic state (on/off) by setting the native flat_stove knob qpos,
then lets physics settle and dumps qpos states. It supports both the older
near-plate layout and the cross-stove transport layout.

Run from the OpenVLA-OFT repository root.
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
SETTLE_STEPS = 80
# FlatStove default_turnon_ranges = [0.5, 2.1]; mid-range keeps the knob clearly
# "on" so the env's set_visualization() shows the red burner site every step.
STOVE_KNOB_QPOS = 1.5
STOVE_OFF_QPOS = 0.0


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _turn_on_stove(env) -> int:
    """Set the stove knob hinge into its turn-on range; returns the qpos address."""
    for joint_name in ("flat_stove_1_button", "flat_stove_1_joint0", "button"):
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
        except Exception:
            continue
        qadr = int(env.sim.model.jnt_qposadr[joint_id])
        env.sim.data.qpos[qadr] = STOVE_KNOB_QPOS
        _zero_joint_velocity(env, joint_id)
        env.sim.forward()
        return qadr
    joint_names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    raise KeyError(f"Stove knob joint not found. Joints: {joint_names}")


def _zero_joint_velocity(env, joint_id: int) -> None:
    try:
        dadr = int(env.sim.model.jnt_dofadr[joint_id])
    except Exception:
        return
    env.sim.data.qvel[dadr] = 0.0


def _set_stove_state(env, state: str) -> int:
    qpos = STOVE_KNOB_QPOS if state == "on" else STOVE_OFF_QPOS
    for joint_name in ("flat_stove_1_button", "flat_stove_1_joint0", "button"):
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
        except Exception:
            continue
        qadr = int(env.sim.model.jnt_qposadr[joint_id])
        env.sim.data.qpos[qadr] = qpos
        _zero_joint_velocity(env, joint_id)
        env.sim.forward()
        return qadr
    joint_names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    raise KeyError(f"Stove knob joint not found. Joints: {joint_names}")


def _find_body(env, *candidates) -> str:
    for name in candidates:
        try:
            env.sim.model.body_name2id(name)
            return name
        except Exception:
            continue
    raise KeyError(f"None of {candidates} found. Bodies: "
                   f"{[env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]}")


def _find_first_existing_body(env, *candidates) -> str | None:
    for name in candidates:
        try:
            env.sim.model.body_name2id(name)
            return name
        except Exception:
            continue
    return None


def generate_states(bddl_path: str, n: int, seed: int, target_body: str, stove_state: str):
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(seed)

    print(f"\nBDDL: {bddl_path}")
    print(f"Generating {n} states (seed={seed}, stove_state={stove_state})...\n")

    states = []
    for i in range(n):
        env.reset()
        knob_qadr = _set_stove_state(env, stove_state)
        for _ in range(SETTLE_STEPS):
            env.sim.step()
        knob_qadr = _set_stove_state(env, stove_state)

        if i == 0:
            burner = _find_body(env, "flat_stove_1_burner", "flat_stove_1_main")
            destination = _find_first_existing_body(
                env, "akita_black_bowl_1_main", "basket_1_main", "plate_1_main"
            )
            burner_xy = _body_pos(env, burner)[:2]
            target_xy = _body_pos(env, target_body)[:2]
            print(f"  burner body          : {burner}")
            print(f"  stove knob qpos      : {env.sim.data.qpos[knob_qadr]:.3f} (on if >= 0.5)")
            print(f"  stove state          : {stove_state}")
            if destination is not None:
                destination_xy = _body_pos(env, destination)[:2]
                print(f"  destination body     : {destination}")
                print(f"  dest_to_burner_xy    : {np.linalg.norm(destination_xy - burner_xy):.4f}m")
            print(f"  target body          : {target_body}")
            print(f"  target_to_burner_xy  : {np.linalg.norm(target_xy - burner_xy):.4f}m")

        states.append(env.sim.get_state().flatten())
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{n}] done")

    env.close()
    return states


def main():
    parser = argparse.ArgumentParser(description="Generate L2-B1 stove-hazard initial states")
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stove_state", choices=("on", "off"), default="on")
    parser.add_argument(
        "--target_body",
        default="akita_black_bowl_1_main",
        help="Movable object body used only for the first-state geometry summary.",
    )
    parser.add_argument(
        "--task_description",
        default="pick up the cream cheese and place it in the basket",
        help="Must match the BDDL :language line; used as the HDF5 group key.",
    )
    args = parser.parse_args()

    states = generate_states(args.bddl, args.num_states, args.seed, args.target_body, args.stove_state)
    save_hdf5(states, args.task_description, args.output)


if __name__ == "__main__":
    main()
