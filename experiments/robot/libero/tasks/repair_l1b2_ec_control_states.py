"""Move only the selected L1-B2 Ec wine bottle beside its native safe pose."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
    _allowed_obstacle_state_indices,
    _body_pos,
    _changed_state_indices,
    _save_hdf5,
    _settle_and_validate,
)
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states


def repair(args) -> None:
    pairing_path = Path(args.pairing_json)
    pairing = json.loads(pairing_path.read_text())
    eb_states = _load_states(Path(args.eb_states))
    if len(eb_states) != len(pairing["pairs"]):
        raise RuntimeError("Eb states and pairing count differ")

    suite = benchmark.get_benchmark_dict()[pairing["task_suite"]]()
    task = suite.get_task(pairing["task_id"])
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    spec = dict(pairing["spec"])
    control_offset = np.asarray(args.control_offset_from_eb, dtype=float)
    spec.pop("control_lateral", None)
    spec.pop("control_fraction", None)
    spec["control_offset_from_eb"] = control_offset.tolist()
    repaired = []
    try:
        for episode_idx, (eb_state, pair) in enumerate(zip(eb_states, pairing["pairs"])):
            env.reset()
            env.set_init_state(eb_state)
            native_safe_xy = _body_pos(env, pairing["obstacle_body"])[:2]
            placement = native_safe_xy + control_offset
            diagnostics, state = _settle_and_validate(
                env, spec, pairing["obstacle_body"], placement, args.stability_steps
            )
            changed = _changed_state_indices(eb_state, state)
            allowed = _allowed_obstacle_state_indices(
                env.sim, pairing["obstacle_body"], spec
            )
            if not diagnostics["valid"] or not changed or not set(changed).issubset(allowed):
                raise RuntimeError(f"invalid repaired Ec state {episode_idx}: {diagnostics}")
            repaired.append(state)
            pair["ec_placement"] = np.asarray(placement).tolist()
            pair["ec_obstacle_xyz"] = diagnostics["end_xyz"].tolist()
            pair["ec_obstacle_drift_m"] = diagnostics["drift_m"]
            pair["ec_changed_state_indices"] = changed
            pair["only_obstacle_pose_changed"] = True
    finally:
        env.close()

    pairing["spec"] = spec
    pairing["ec_control_repair"] = {
        "placement_mode": "native_safe_offset",
        "control_offset_from_eb": control_offset.tolist(),
        "count": len(repaired),
        "only_obstacle_pose_changed": True,
    }
    _save_hdf5(Path(args.ec_states), pairing["task_language"], repaired)
    pairing_path.write_text(json.dumps(pairing, indent=2) + "\n")
    Path(args.out_report).write_text(
        "# L1-B2 Ec control repair\n\nVerdict: **PASS_EC_CONTROL_REPAIR**\n\n"
        f"- Repaired states: `{len(repaired)}`\n"
        f"- Native-safe XY offset: `{control_offset.tolist()} m`\n"
        "- Changed state fields: protected wine-bottle free joint only.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control_offset_from_eb", type=float, nargs=2, default=(0.0, 0.005)
    )
    parser.add_argument("--stability_steps", type=int, default=10)
    parser.add_argument("--eb_states", default="experiments/robot/libero/tasks/l1b2_native_held_object_eb_states.hdf5")
    parser.add_argument("--ec_states", default="experiments/robot/libero/tasks/l1b2_native_held_object_ec_states.hdf5")
    parser.add_argument("--pairing_json", default="experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json")
    parser.add_argument("--out_report", default="experiments/logs/l1b2_ec_control_repair.md")
    repair(parser.parse_args())


if __name__ == "__main__":
    main()
