"""Calibrate a cross-platform native closed-drawer state for L3-B.

This script never writes an evaluation state bundle.  It restores official
native state 0, changes only the existing bottom-drawer scalar joint, and
reports the exact formal-wait physical gate for a preregistration candidate
grid.  It is a pre-policy calibration tool, not evaluation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks.generate_l3b_bowl_order_states import (
    _fixture_snapshot,
    _formal_gate,
    _restore_fixtures,
    _runtime_task,
    _trusted_native_states,
)
from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    CONSTRUCTION_SETTLE_STEPS,
    DRAWER_JOINT,
    DUMMY_ACTION,
    flat_scalar_joint_indices,
    scalar_joint_addresses,
    scene_measurement,
)


DEFAULT_TARGETS = (
    0.0004,
    0.0006,
    0.0008,
    0.0010,
    0.0012,
    0.0014,
    0.0016,
    0.0018,
    0.0020,
    0.0022,
    0.0024,
    0.0026,
)


def _parse_targets(value: str) -> list[float]:
    targets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not targets:
        raise ValueError("at least one qpos target is required")
    if any(not 0.0 <= target <= 0.005 for target in targets):
        raise ValueError("targets must remain inside the native Close interval")
    return targets


def diagnose(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    task, runtime_bddl = _runtime_task()
    native_states, native_init_path = _trusted_native_states(task)
    base = np.asarray(native_states[args.native_state_index], dtype=float).copy()
    env = OffScreenRenderEnv(
        bddl_file_name=str(runtime_bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(args.seed)
    records = []
    try:
        env.reset()
        fixture_names, fixture_positions, fixture_quaternions = _fixture_snapshot(env)
        for target in args.targets:
            env.reset()
            _restore_fixtures(
                env, fixture_names, fixture_positions, fixture_quaternions
            )
            env.set_init_state(base)
            qpos_address, qvel_address = scalar_joint_addresses(env, DRAWER_JOINT)
            env.sim.data.qpos[qpos_address] = target
            env.sim.data.qvel[qvel_address] = 0.0
            env.sim.forward()
            construction_qpos = [float(env.sim.data.qpos[qpos_address])]
            for _ in range(CONSTRUCTION_SETTLE_STEPS):
                env.step(DUMMY_ACTION)
                construction_qpos.append(float(env.sim.data.qpos[qpos_address]))
            settled = np.asarray(env.sim.get_state().flatten(), dtype=float)
            qpos_index, qvel_index = flat_scalar_joint_indices(env, DRAWER_JOINT)
            candidate = base.copy()
            candidate[qpos_index] = settled[qpos_index]
            candidate[qvel_index] = settled[qvel_index]
            construction_measurement = scene_measurement(env)
            gate, _ = _formal_gate(
                env,
                candidate,
                condition="premature_close",
                fixture_names=fixture_names,
                fixture_positions=fixture_positions,
                fixture_quaternions=fixture_quaternions,
            )
            contact_distances = [
                float(item["distance_m"])
                for item in construction_measurement[
                    "drawer_cabinet_self_contacts"
                ]
            ]
            records.append(
                {
                    "target_qpos": target,
                    "settled_qpos": float(settled[qpos_index]),
                    "settled_qvel": float(settled[qvel_index]),
                    "construction_qpos_min": min(construction_qpos),
                    "construction_qpos_max": max(construction_qpos),
                    "construction_min_drawer_cabinet_contact_distance_m": (
                        min(contact_distances) if contact_distances else None
                    ),
                    "physical_gate_pass": gate["physical_gate_pass"],
                    "failures": gate["failures"],
                    "formal_drawer_joint": gate["formal_window_stats"][
                        "drawer_joint"
                    ],
                    "post_wait_drawer_joint": gate["post_wait_hold_stats"][
                        "drawer_joint"
                    ],
                    "pre_wait_predicates": gate["pre_wait"]["predicates"],
                    "first_policy_predicates": gate["first_policy"][
                        "predicates"
                    ],
                }
            )
    finally:
        env.close()

    result = {
        "kind": "pre_policy_native_er_qpos_calibration",
        "native_state_index": args.native_state_index,
        "native_init_states_path": str(native_init_path),
        "construction_settle_steps": CONSTRUCTION_SETTLE_STEPS,
        "records": records,
        "passing_targets": [
            item["target_qpos"] for item in records if item["physical_gate_pass"]
        ],
        "evaluation_evidence": False,
        "model_rollout_used": False,
        "custom_asset_used": False,
        "custom_bddl_used": False,
        "prompt_changed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PASS_L3B_BOWL_ER_QPOS_DIAGNOSTIC "
        f"tested={len(records)} passing={len(result['passing_targets'])}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=list(DEFAULT_TARGETS),
        help="comma-separated qpos targets inside the native Close interval",
    )
    parser.add_argument("--native-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument(
        "--output",
        default="review/L3-B_bowl_order_calibration_task/er_qpos_sweep.json",
    )
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
