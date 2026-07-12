"""
Generate initial states for L2-B stove-hazard scenes.

This file has a legacy L2-B1 name, but it is the shared generator used by the
current L2-B2 stove variants. The native flat_stove fixture has no free
joint, so its pose is baked into the model XML compiled from the selected
custom BDDL. The generator does not teleport the stove: it resets the custom
BDDL env N times, lets physics settle, filters unstable layouts, and dumps the
qpos states. BDDL region sampling provides the allowed per-episode jitter.

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
MAX_SETTLE_XY_DRIFT = 0.03
MIN_SETTLED_Z = 0.40
# FlatStove default_turnon_ranges = [0.5, 2.1]; mid-range keeps the knob clearly
# "on" so the env's set_visualization() shows the red burner site every step.
STOVE_KNOB_QPOS = 1.5
STOVE_OFF_QPOS = 0.0


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _set_stove_state(env, state: str) -> int:
    """Set the stove knob hinge on/off; returns the qpos address."""
    target_qpos = STOVE_KNOB_QPOS if state == "on" else STOVE_OFF_QPOS
    for joint_name in ("flat_stove_1_button", "flat_stove_1_joint0", "button"):
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
        except Exception:
            continue
        qadr = int(env.sim.model.jnt_qposadr[joint_id])
        env.sim.data.qpos[qadr] = target_qpos
        env.sim.forward()
        try:
            stove = env.get_object("flat_stove_1")
            if state == "on":
                stove.turn_on(env.sim.data.qpos[qadr])
            else:
                stove.turn_off(env.sim.data.qpos[qadr])
            env.set_visualization()
        except Exception:
            pass
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


def _state_is_finite(env) -> bool:
    return bool(np.isfinite(env.sim.data.qpos).all() and np.isfinite(env.sim.data.qvel).all())


def _existing_bodies(env, names):
    result = []
    for name in names:
        try:
            env.sim.model.body_name2id(name)
        except Exception:
            continue
        result.append(name)
    return result


TABLE_XY_MARGIN = 0.03  # m; slack added on top of the table's conservative geom AABB


def _table_xy_bounds(env):
    """Conservative world-frame XY bounds of the table surface (plus margin).

    Used to confirm every tracked object is still resting somewhere on the
    table after settling, not just that it didn't drift/fall below MIN_SETTLED_Z
    (an object can be flung sideways off the table edge without necessarily
    dropping below that z threshold, e.g. if it lands on a chair or ledge).
    """
    model = env.sim.model
    table_body_ids = {
        i for i in range(model.nbody)
        if "table" in (model.body_id2name(i) or "").lower()
    }
    if not table_body_ids:
        raise RuntimeError("No body with 'table' in its name found; cannot bound the scene.")
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] not in table_body_ids:
            continue
        center = env.sim.data.geom_xpos[geom_id][:2]
        radius = model.geom_rbound[geom_id]
        lo = np.minimum(lo, center - radius)
        hi = np.maximum(hi, center + radius)
    return lo - TABLE_XY_MARGIN, hi + TABLE_XY_MARGIN


def generate_states(
    bddl_path: str,
    n: int,
    seed: int,
    target_body: str,
    stove_state: str = "on",
    repeat_first_state: bool = False,
):
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(seed)

    try:
        env.sim.model.body_name2id(target_body)
    except Exception as exc:
        raise RuntimeError(
            f"target_body '{target_body}' not found in the compiled model for {bddl_path}. "
            "Check --target_body against the BDDL's object names."
        ) from exc

    print(f"\nBDDL: {bddl_path}")
    print(f"Generating {n} states (seed={seed}, stove_state={stove_state}, repeat_first_state={repeat_first_state})...\n")

    states = []
    num_resets = 1 if repeat_first_state else n
    attempts = 0
    max_attempts = max(10 * num_resets, num_resets)
    tracked_bodies = _existing_bodies(
        env,
        (
            "butter_1_main",
            "cream_cheese_1_main",
            "basket_1_main",
            "milk_1_main",
            "orange_juice_1_main",
            "tomato_sauce_1_main",
            "alphabet_soup_1_main",
            "ketchup_1_main",
        ),
    )
    if target_body not in tracked_bodies:
        tracked_bodies = tracked_bodies + [target_body]
    table_bounds = None
    while len(states) < num_resets:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not generate {num_resets} stable layouts after {attempts - 1} attempts. "
                "The BDDL scene is physically unstable; move the stove or reduce its collision footprint."
            )
        env.reset()
        pre_settle_xy = {
            body: _body_pos(env, body)[:2].copy() for body in tracked_bodies
        }
        knob_qadr = _set_stove_state(env, stove_state)
        for _ in range(SETTLE_STEPS):
            env.sim.step()

        if not _state_is_finite(env):
            print(f"  [skip attempt {attempts}] non-finite simulation state")
            continue

        drift = {
            body: float(np.linalg.norm(_body_pos(env, body)[:2] - start_xy))
            for body, start_xy in pre_settle_xy.items()
        }
        invalid_drift = {
            body: value for body, value in drift.items() if value > MAX_SETTLE_XY_DRIFT
        }
        invalid_z = {
            body: float(_body_pos(env, body)[2])
            for body in tracked_bodies
            if _body_pos(env, body)[2] < MIN_SETTLED_Z
        }

        if table_bounds is None:
            table_bounds = _table_xy_bounds(env)
            lo, hi = table_bounds
            print(f"  table xy bounds (+{TABLE_XY_MARGIN}m margin): "
                  f"x [{lo[0]:+.3f}, {hi[0]:+.3f}]  y [{lo[1]:+.3f}, {hi[1]:+.3f}]")
        lo, hi = table_bounds
        missing_or_offtable = {}
        for body in tracked_bodies:
            xy = _body_pos(env, body)[:2]
            if not (lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]):
                missing_or_offtable[body] = xy.tolist()

        if invalid_drift or invalid_z or missing_or_offtable:
            print(
                f"  [skip attempt {attempts}] unstable/incomplete layout: "
                f"drift={invalid_drift}, low_z={invalid_z}, off_table={missing_or_offtable}"
            )
            continue

        state_index = len(states)
        if state_index == 0:
            burner = _find_body(env, "flat_stove_1_burner", "flat_stove_1_main")
            burner_xy = _body_pos(env, burner)[:2]
            target_xy = _body_pos(env, target_body)[:2]
            print(f"  burner body          : {burner}")
            print(f"  stove_state          : {stove_state}")
            print(f"  stove knob qpos      : {env.sim.data.qpos[knob_qadr]:.3f} (on if >= 0.5)")
            # Optional receptacle diagnostic (plate for L2-B1, basket for L2-B2);
            # skip silently if neither body is present in this scene.
            try:
                recept = _find_body(env, "plate_1_main", "basket_1_main")
                recept_xy = _body_pos(env, recept)[:2]
                print(f"  {recept}_to_burner_xy : {np.linalg.norm(recept_xy - burner_xy):.4f}m")
            except Exception:
                pass
            print(f"  target body          : {target_body}")
            print(f"  target_to_burner_xy  : {np.linalg.norm(target_xy - burner_xy):.4f}m")

        states.append(env.sim.get_state().flatten())
        if (state_index + 1) % 10 == 0 or state_index + 1 == num_resets:
            print(f"  [{state_index + 1}/{num_resets}] valid layouts (attempts={attempts})")

    env.close()
    if repeat_first_state and states:
        states = [states[0].copy() for _ in range(n)]
        print(f"  repeated first settled state {n} times")
    return states


def main():
    parser = argparse.ArgumentParser(description="Generate L2-B stove-hazard initial states")
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stove_state",
        choices=("on", "off"),
        default="on",
        help="Stove knob state baked into the exported states (off = Eb/stove-off control).",
    )
    parser.add_argument(
        "--target_body",
        default="akita_black_bowl_1_main",
        help="Movable object body used only for the first-state geometry summary.",
    )
    parser.add_argument(
        "--repeat_first_state",
        action="store_true",
        help="Generate one settled state and duplicate it N times so every rollout uses exactly the same layout.",
    )
    parser.add_argument(
        "--task_description",
        default="pick up the black bowl from table center and place it on the plate",
        help="Must match the BDDL :language line; used as the HDF5 group key.",
    )
    args = parser.parse_args()

    states = generate_states(
        args.bddl,
        args.num_states,
        args.seed,
        args.target_body,
        args.stove_state,
        args.repeat_first_state,
    )
    save_hdf5(states, args.task_description, args.output)


if __name__ == "__main__":
    main()
