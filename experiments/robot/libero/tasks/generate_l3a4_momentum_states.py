#!/usr/bin/env python3
"""Generate paired L3-A4 Eb/Er/Ec serialized states.

Er places A/B/C beside the native bottom drawer. Ec is derived from the exact
serialized Er state and moves only C to Pi_safe's lateral parking location.
Eb is also derived from Er and parks A/B/C together away from the drawer.
No prompt, native object, fixture, or goal predicate changes between conditions.

Candidate geometry must still pass ``validate_l3a4_scene.py`` on a GPU node.
Generation alone is never a PASS verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import experiments.robot.libero.physcog_objects as physcog_objects  # noqa: F401
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.generate_l2b1_initial_states import save_hdf5
from experiments.robot.libero.tasks.l3a4_momentum import (
    A_BODY,
    B_BODY,
    C_BODY,
    CHAIN_BODIES,
    DEFAULT_BDDL,
    EB_PARK_OFFSETS_XY,
    EC_SENTINEL_PARK_DXY,
    MAX_INITIAL_CHAIN_SPEED_M_S,
    RISK_OFFSETS_XY,
    SCHEMA_VERSION,
    SETTLE_STEPS,
    TASK_DESCRIPTION,
    TOPOLOGY_ID,
    asset_contract,
    contract,
    contract_sha256,
    find_body,
    DRAWER_BODY_CANDIDATES,
)


IDENTITY_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0])


def _body_pos(env, name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(name)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _free_joint_vadr(sim, body_name: str) -> int:
    body_id = int(sim.model.body_name2id(body_name))
    for joint_id in range(sim.model.njnt):
        if int(sim.model.jnt_bodyid[joint_id]) == body_id:
            if int(sim.model.jnt_type[joint_id]) == 0:  # mjJNT_FREE
                return int(sim.model.jnt_dofadr[joint_id])
    raise KeyError(f"no free joint for {body_name}")


def _set_free_pose(env, body_name: str, xy: np.ndarray) -> None:
    qadr = _find_free_joint_qadr(env.sim, body_name)
    vadr = _free_joint_vadr(env.sim, body_name)
    if qadr < 0:
        raise KeyError(f"no free qpos for {body_name}")
    env.sim.data.qpos[qadr:qadr + 2] = np.asarray(xy, dtype=float)
    env.sim.data.qpos[qadr + 3:qadr + 7] = IDENTITY_QUAT_WXYZ
    env.sim.data.qvel[vadr:vadr + 6] = 0.0


def _speed(env, body_name: str) -> float:
    vadr = _free_joint_vadr(env.sim, body_name)
    return float(np.linalg.norm(env.sim.data.qvel[vadr:vadr + 3]))


def _place_risk(env, drawer_body: str, dx: float, dy: float) -> None:
    origin = _body_pos(env, drawer_body)[:2]
    shift = np.asarray([dx, dy], dtype=float)
    for body_name in CHAIN_BODIES:
        _set_free_pose(env, body_name, origin + RISK_OFFSETS_XY[body_name] + shift)
    env.sim.forward()


def _transform_control(env, variant: str) -> list[str]:
    changed = []
    if variant == "stable":
        c_qadr = _find_free_joint_qadr(env.sim, C_BODY)
        c_vadr = _free_joint_vadr(env.sim, C_BODY)
        env.sim.data.qpos[c_qadr:c_qadr + 2] += EC_SENTINEL_PARK_DXY
        env.sim.data.qvel[c_vadr:c_vadr + 6] = 0.0
        changed.append(C_BODY)
    elif variant == "baseline":
        drawer_body = find_body(env, DRAWER_BODY_CANDIDATES)
        origin = _body_pos(env, drawer_body)[:2]
        for body_name in CHAIN_BODIES:
            _set_free_pose(env, body_name, origin + EB_PARK_OFFSETS_XY[body_name])
            changed.append(body_name)
    else:
        raise ValueError(f"control transform does not support {variant!r}")
    env.sim.forward()
    return changed


def _load_paired(path: str, expected_count: int) -> tuple[list[np.ndarray], list[int]]:
    key = TASK_DESCRIPTION.replace(" ", "_")
    with h5py.File(path, "r") as source:
        group = source[key]
        if (
            int(group.attrs.get("l3a4_schema_version", -1)) != SCHEMA_VERSION
            or str(group.attrs.get("l3a4_topology_id", "")) != TOPOLOGY_ID
            or str(group.attrs.get("l3a4_variant", "")) != "risk"
        ):
            raise ValueError("paired source is not the canonical L3-A4 Er artifact")
        states = [
            np.asarray(group[f"demo_{index}"]["initial_state"][:])
            for index in range(len(group))
        ]
        attempts = [
            int(group[f"demo_{index}"].attrs["reset_attempt"])
            for index in range(len(group))
        ]
    if len(states) != expected_count:
        raise ValueError(
            f"paired Er count {len(states)} does not match --num_states {expected_count}"
        )
    return states, attempts


def generate(args):
    paired_states = paired_attempts = None
    if args.variant != "risk":
        if not args.paired_er_states:
            raise ValueError(f"{args.variant} requires --paired_er_states")
        paired_states, paired_attempts = _load_paired(
            args.paired_er_states, args.num_states
        )

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    drawer_body = None
    layout = None
    states = []
    records = []
    for index in range(args.num_states):
        obs = env.reset()
        del obs
        drawer_body = find_body(env, DRAWER_BODY_CANDIDATES)
        if layout is None:
            layout = {
                "nq": int(env.sim.model.nq),
                "nv": int(env.sim.model.nv),
                "qpos": {
                    body: int(_find_free_joint_qadr(env.sim, body))
                    for body in CHAIN_BODIES
                },
                "qvel": {
                    body: int(_free_joint_vadr(env.sim, body))
                    for body in CHAIN_BODIES
                },
            }
        if args.variant == "risk":
            _place_risk(env, drawer_body, args.chain_dx, args.chain_dy)
            reset_attempt = index
        else:
            env.set_init_state(paired_states[index])
            changed = _transform_control(env, args.variant)
            reset_attempt = paired_attempts[index]

        initial_positions = {
            name: _body_pos(env, name).tolist() for name in CHAIN_BODIES
        }
        # Er is rejection-calibrated from a fresh reset and must settle before
        # serialization. Paired controls start from that already-settled Er
        # state; applying another simulation window would advance unrelated
        # qpos/qvel/time and destroy exact pairing. Their stability is tested
        # independently by validate_l3a4_scene.py's open-hold gate.
        if args.variant == "risk":
            for _ in range(args.settle_steps):
                env.sim.step()
        final_positions = {
            name: _body_pos(env, name).tolist() for name in CHAIN_BODIES
        }
        speeds = {name: _speed(env, name) for name in CHAIN_BODIES}
        if max(speeds.values()) > args.max_initial_speed:
            raise RuntimeError(
                f"episode {index} did not settle: speeds={speeds}; "
                "do not serialize a preactivated chain"
            )
        state = np.asarray(env.sim.get_state().flatten()).copy()
        states.append(state)
        records.append(
            {
                "reset_attempt": reset_attempt,
                "initial_positions_json": json.dumps(
                    initial_positions, sort_keys=True
                ),
                "settled_positions_json": json.dumps(
                    final_positions, sort_keys=True
                ),
                "settled_speeds_json": json.dumps(speeds, sort_keys=True),
                "changed_bodies_json": json.dumps(
                    [] if args.variant == "risk" else changed
                ),
                "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            }
        )
    env.close()
    return states, records, drawer_body, layout


def write(args, states, records, drawer_body: str, layout: dict) -> None:
    save_hdf5(states, TASK_DESCRIPTION, args.output)
    key = TASK_DESCRIPTION.replace(" ", "_")
    repo_root = Path(__file__).resolve().parents[4]
    with h5py.File(args.output, "a") as target:
        group = target[key]
        group.attrs["l3a4_schema_version"] = SCHEMA_VERSION
        group.attrs["l3a4_topology_id"] = TOPOLOGY_ID
        group.attrs["l3a4_variant"] = args.variant
        group.attrs["task_description"] = TASK_DESCRIPTION
        group.attrs["native_task_suite"] = "libero_90"
        group.attrs["native_task_id"] = 14
        group.attrs["native_task_name"] = (
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet"
        )
        group.attrs["drawer_body"] = drawer_body
        group.attrs["bddl"] = args.bddl
        group.attrs["bddl_sha256"] = hashlib.sha256(
            Path(args.bddl).read_bytes()
        ).hexdigest()
        group.attrs["contract_json"] = json.dumps(contract(), sort_keys=True)
        group.attrs["contract_sha256"] = contract_sha256()
        group.attrs["asset_contract_json"] = json.dumps(
            asset_contract(repo_root), sort_keys=True
        )
        group.attrs["seed"] = args.seed
        group.attrs["settle_steps"] = (
            args.settle_steps if args.variant == "risk" else 0
        )
        group.attrs["chain_dx"] = args.chain_dx
        group.attrs["chain_dy"] = args.chain_dy
        group.attrs["paired_er_states"] = args.paired_er_states or ""
        group.attrs["pairing_method"] = (
            ""
            if args.variant == "risk"
            else (
                "serialized_er_state_C_only_transform"
                if args.variant == "stable"
                else "serialized_er_state_ABC_parking_transform"
            )
        )
        group.attrs["flattened_state_layout_json"] = json.dumps(
            layout, sort_keys=True
        )
        for index, record in enumerate(records):
            episode = group[f"demo_{index}"]
            for name, value in record.items():
                episode.attrs[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("risk", "stable", "baseline"), required=True)
    parser.add_argument("--paired_er_states")
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chain_dx", type=float, default=0.0)
    parser.add_argument("--chain_dy", type=float, default=0.0)
    parser.add_argument("--settle_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument(
        "--max_initial_speed",
        type=float,
        default=MAX_INITIAL_CHAIN_SPEED_M_S,
    )
    args = parser.parse_args()
    if args.variant == "risk" and args.paired_er_states:
        parser.error("--paired_er_states is only valid for stable/baseline")
    states, records, drawer_body, layout = generate(args)
    write(args, states, records, drawer_body, layout)
    print(
        f"GENERATED_L3A4_{args.variant.upper()} count={len(states)} "
        f"output={args.output} (calibration not yet validated)"
    )


if __name__ == "__main__":
    main()
