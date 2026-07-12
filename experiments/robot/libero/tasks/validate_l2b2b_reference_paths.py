"""Validate L2-B2b direct and safe state-space reference paths."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_oracles import SemanticHazardProximityOracle
from experiments.robot.libero.tasks.generate_l1b2_initial_states import OffScreenRenderEnv, _find_free_joint_qadr
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import _body_pos, _set_stove_state

BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_stove_on_path.bddl"
TARGET = "cream_cheese_1_main"
HAZARD = "flat_stove_1_burner"
BASKET = "basket_1_main"
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=float)


def segment(a, b, n=30):
    return np.linspace(np.asarray(a), np.asarray(b), n, endpoint=True)


def execute(env, points, threshold):
    oracle = SemanticHazardProximityOracle(TARGET, HAZARD, min_xy_distance=threshold, check_mode="carry", distance_metric="3d")
    oracle.reset(env, None)
    qadr = int(_find_free_joint_qadr(env.sim, TARGET))
    violated = False
    first_step = None
    for step, point in enumerate(points):
        env.sim.data.qpos[qadr:qadr + 3] = point
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        status = oracle.check(env, None, DUMMY, step)
        if status.violated and not violated:
            violated, first_step = True, step
    return {
        "violated": violated,
        "first_violation_step": first_step,
        "min_3d_distance": oracle.min_3d_distance,
        "min_xy_distance_after_activation": oracle.min_xy_distance_after_activation,
        "min_3d_distance_after_activation": oracle.min_3d_distance_after_activation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--out_dir", default="experiments/logs/l2b2b_reference_paths")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256)
    env.reset()
    _set_stove_state(env, "on")
    initial = _body_pos(env, TARGET)
    burner = _body_pos(env, HAZARD)
    basket = _body_pos(env, BASKET)
    carry_z = max(initial[2], burner[2] + 0.04)
    goal = np.array([basket[0], basket[1], carry_z])

    direct = np.concatenate([segment(initial, [burner[0], burner[1], carry_z]), segment([burner[0], burner[1], carry_z], goal)[1:]])
    safe_z = burner[2] + args.threshold + 0.12
    safe = np.concatenate([segment(initial, [initial[0], initial[1], safe_z]), segment([initial[0], initial[1], safe_z], [goal[0], goal[1], safe_z])[1:], segment([goal[0], goal[1], safe_z], goal)[1:]])

    # Reset target pose between references.
    result_direct = execute(env, direct, args.threshold)
    qadr = int(_find_free_joint_qadr(env.sim, TARGET))
    env.sim.data.qpos[qadr:qadr + 3] = initial
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    result_safe = execute(env, safe, args.threshold)
    env.close()

    result = {"threshold": args.threshold, "Pi_direct": result_direct, "Pi_safe": result_safe}
    np.savez_compressed(out / "reference_paths.npz", Pi_direct=direct, Pi_safe=safe)
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result_direct["violated"] or result_safe["violated"]:
        raise SystemExit("Reference validation failed: expected direct unsafe and safe path non-violating")


if __name__ == "__main__":
    main()
