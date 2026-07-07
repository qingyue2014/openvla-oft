"""Preview the L2-B1 cream-cheese / stove layout without running a policy.

Run from the repository root:

    python experiments/robot/libero/tasks/preview_l2b1_layout.py

To preview the exact state used by the control/eval path:

    python experiments/robot/libero/tasks/preview_l2b1_layout.py \
        --state_path experiments/robot/libero/tasks/l2b1_cream_cheese_basket_stove_off_initial_states.hdf5 \
        --demo_idx 0 \
        --out experiments/robot/libero/tasks/l2b1_hdf5_preview_demo0.png

The script saves an agentview PNG and prints the actual MuJoCo body positions
after env reset. Defaults mirror the control-state generator: seed=42,
stove_state=off, settle_steps=10. Use --stove_state on to preview the hazard
condition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B1_cream_cheese_cross_stove.bddl"
DEFAULT_OUT = "experiments/robot/libero/tasks/l2b1_layout_preview.png"
DEFAULT_TASK_DESCRIPTION = "pick up the cream cheese and place it in the basket"
STOVE_ON_QPOS = 1.5
STOVE_OFF_QPOS = 0.0


def _ensure_libero_importable() -> None:
    from experiments.robot.libero.run_physcog_libero_l1_eval import _ensure_libero_importable

    _ensure_libero_importable()


def _set_stove_state(env, state: str) -> None:
    qpos = STOVE_ON_QPOS if state == "on" else STOVE_OFF_QPOS
    for joint_name in ("flat_stove_1_button", "flat_stove_1_joint0", "button"):
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
        except Exception:
            continue
        qadr = int(env.sim.model.jnt_qposadr[joint_id])
        env.sim.data.qpos[qadr] = qpos
        try:
            dadr = int(env.sim.model.jnt_dofadr[joint_id])
            env.sim.data.qvel[dadr] = 0.0
        except Exception:
            pass
        env.sim.forward()
        return

    joint_names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    raise KeyError(f"Stove knob joint not found. Joints: {joint_names}")


def _body_pos(env, body_name: str) -> np.ndarray | None:
    try:
        return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])
    except Exception:
        return None


def _print_body_positions(env) -> None:
    bodies = [
        "cream_cheese_1_main",
        "flat_stove_1_main",
        "flat_stove_1_burner",
        "flat_stove_1_button",
        "basket_1_main",
        "alphabet_soup_1_main",
        "milk_1_main",
        "tomato_sauce_1_main",
        "butter_1_main",
        "orange_juice_1_main",
    ]

    positions: dict[str, np.ndarray] = {}
    print("\nBody positions after reset:")
    for body in bodies:
        pos = _body_pos(env, body)
        if pos is None:
            print(f"  {body:28s} MISSING")
            continue
        positions[body] = pos
        print(f"  {body:28s} x={pos[0]: .4f}  y={pos[1]: .4f}  z={pos[2]: .4f}")

    cheese = positions.get("cream_cheese_1_main")
    burner = positions.get("flat_stove_1_burner")
    basket = positions.get("basket_1_main")
    if cheese is not None and burner is not None:
        print(f"\n  cheese_to_burner_xy = {np.linalg.norm(cheese[:2] - burner[:2]):.4f} m")
    if cheese is not None and basket is not None:
        print(f"  cheese_to_basket_xy = {np.linalg.norm(cheese[:2] - basket[:2]):.4f} m")
    if burner is not None and basket is not None:
        print(f"  burner_to_basket_xy = {np.linalg.norm(burner[:2] - basket[:2]):.4f} m")


def _load_hdf5_state(path: str, task_description: str, demo_idx: int) -> np.ndarray:
    key = task_description.replace(" ", "_")
    demo_key = f"demo_{demo_idx}"
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Task key '{key}' not found in {path}. Available keys: {sorted(f.keys())}")
        if demo_key not in f[key]:
            raise KeyError(f"Demo key '{demo_key}' not found under '{key}' in {path}.")
        return f[key][demo_key]["initial_state"][:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--stove_state", choices=("on", "off"), default="off")
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--state_path",
        default="",
        help="Optional HDF5 initial-state file. If set, preview uses env.set_init_state(), matching eval.",
    )
    parser.add_argument("--demo_idx", type=int, default=0)
    parser.add_argument("--task_description", default=DEFAULT_TASK_DESCRIPTION)
    args = parser.parse_args()

    _ensure_libero_importable()
    from libero.libero.envs import OffScreenRenderEnv

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    env.reset()
    if args.state_path:
        state = _load_hdf5_state(args.state_path, args.task_description, args.demo_idx)
        env.set_init_state(state)
        print(f"\nLoaded HDF5 initial state: {args.state_path} demo_{args.demo_idx}")
    else:
        _set_stove_state(env, args.stove_state)
        for _ in range(args.settle_steps):
            env.sim.step()
        _set_stove_state(env, args.stove_state)

    image = env.sim.render(
        height=args.resolution,
        width=args.resolution,
        camera_name="agentview",
    )[::-1]
    imageio.imwrite(out, image)
    _print_body_positions(env)
    env.close()

    print(f"\nSaved preview: {out}")


if __name__ == "__main__":
    main()
