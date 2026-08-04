#!/usr/bin/env python3
"""Repair L1-C1 Eb by settling only the native second black bowl.

The paired, already-stable Ec state supplies every invariant field. The old Eb
state supplies only the native second-bowl pose. After isolated settling, the
candidate is rebuilt from Ec with only that bowl's free-joint qpos/qvel changed.
Simulator execution is Superpod-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import h5py
import numpy as np

from experiments.robot.libero.formal_evaluator_state import restore_formal_observation
from experiments.robot.libero.tasks.validate_l1c1_first_policy_frames import (
    CONFIRM_STEPS,
    FORMAL_WAIT_STEPS,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    _evaluate_trace,
    _sample,
)


LOWER_BOWL = "akita_black_bowl_2_main"
SETTLE_STEPS = 150
SETTLE_CONFIRM_STEPS = 50
MAX_NATIVE_XY_SETTLE_DRIFT_M = 0.005
MAX_INVARIANT_QPOS_DIFF = 1e-12
MAX_INVARIANT_QVEL_DIFF = 1e-12
REGISTERED_RADIAL_OFFSETS_M = (0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030)


def _assert_superpod(allow_local_simulator: bool) -> dict[str, object]:
    hostname = socket.gethostname()
    scheduler = bool(os.environ.get("SLURM_JOB_ID"))
    declared = os.environ.get("PHYSCG_SUPERPOD") == "1"
    marker = any(
        token in hostname.lower()
        for token in ("superpod", "dgx", "slogin", "compute", "gpu")
    )
    verified = scheduler or declared or marker
    if not verified and not allow_local_simulator:
        raise RuntimeError("L1-C1 Eb repair is Superpod-only")
    return {
        "hostname": hostname,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "superpod_verified": verified,
        "local_override": bool(allow_local_simulator and not verified),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> tuple[str, list[np.ndarray], dict[str, object]]:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise ValueError(f"expected one prompt group in {path}: {keys}")
        group = handle[keys[0]]
        demos = sorted(group, key=lambda value: int(value.rsplit("_", 1)[1]))
        states = [np.asarray(group[name]["initial_state"]) for name in demos]
        attrs = {str(key): value for key, value in group.attrs.items()}
    return keys[0], states, attrs


def _write(
    path: Path,
    group_key: str,
    states: list[np.ndarray],
    attrs: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(group_key)
        for key, value in attrs.items():
            group.attrs[key] = value
        group.attrs["state_intervention_variant"] = (
            "task2_bowl_stack_benign_settled_paired_v2"
        )
        group.attrs["repair_method"] = (
            "Ec invariant state plus isolated native second-bowl settling"
        )
        for index, state in enumerate(states):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset("initial_state", data=state)
            demo.attrs["success"] = True


def _joint_addresses(env) -> tuple[int, int]:
    model = env.sim.model
    body_id = int(model.body_name2id(LOWER_BOWL))
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != 0:
        raise RuntimeError(f"{LOWER_BOWL} has no free joint")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _outside_max(first: np.ndarray, second: np.ndarray, allowed: set[int]) -> float:
    indices = [index for index in range(len(first)) if index not in allowed]
    if not indices:
        return 0.0
    return float(np.max(np.abs(first[indices] - second[indices])))


def repair(args: argparse.Namespace) -> dict[str, object]:
    host = _assert_superpod(args.allow_local_simulator)
    old_eb_path = Path(args.old_eb_states)
    ec_path = Path(args.ec_states)
    group_key, old_eb_states, old_attrs = _load(old_eb_path)
    ec_group_key, ec_states, ec_attrs = _load(ec_path)
    if group_key != TASK_PROMPT.replace(" ", "_") or ec_group_key != group_key:
        raise ValueError("L1-C1 prompt group mismatch")
    if len(old_eb_states) != len(ec_states):
        raise ValueError("old Eb and Ec state counts differ")
    for key in (
        "native_suite",
        "native_task_id",
        "native_prompt",
        "native_bddl",
        "native_bddl_sha256",
        "native_asset_inventory_sha256",
        "custom_assets",
    ):
        if str(old_attrs[key]) != str(ec_attrs[key]):
            raise ValueError(f"old Eb/Ec metadata mismatch for {key}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != TASK_PROMPT:
        raise ValueError("runtime native task mismatch")
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(0)
    repaired: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    try:
        env.reset()
        qpos_address, qvel_address = _joint_addresses(env)
        qpos_allowed = set(range(qpos_address, qpos_address + 7))
        qvel_allowed = set(range(qvel_address, qvel_address + 6))
        for episode_idx, (old_eb, ec_state) in enumerate(
            zip(old_eb_states, ec_states)
        ):
            env.reset()
            env.set_init_state(old_eb)
            env.sim.forward()
            native_lower_qpos = np.asarray(
                env.sim.data.qpos[qpos_address : qpos_address + 7], dtype=float
            ).copy()
            native_xy = native_lower_qpos[:2].copy()
            plate_id = int(env.sim.model.body_name2id("plate_1_main"))
            plate_xy = np.asarray(env.sim.data.body_xpos[plate_id], dtype=float)[:2]
            direction = native_xy - plate_xy
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-9:
                direction = np.array([0.0, 1.0], dtype=float)
            else:
                direction = direction / direction_norm

            accepted = None
            attempt_failures: list[dict[str, object]] = []
            for radial_offset_m in REGISTERED_RADIAL_OFFSETS_M:
                proposed_qpos = native_lower_qpos.copy()
                proposed_qpos[:2] = native_xy + direction * radial_offset_m
                env.reset()
                env.set_init_state(ec_state)
                env.sim.data.qpos[qpos_address : qpos_address + 7] = proposed_qpos
                env.sim.data.qvel[qvel_address : qvel_address + 6] = 0.0
                env.sim.forward()
                for _ in range(SETTLE_STEPS + SETTLE_CONFIRM_STEPS):
                    env.sim.step()
                settled_lower_qpos = np.asarray(
                    env.sim.data.qpos[qpos_address : qpos_address + 7], dtype=float
                ).copy()

                # Reapply the exact paired invariant state, then change only
                # the lower bowl. The saved qvel is zero so formal no-op
                # waiting begins from a truly settled state.
                env.reset()
                env.set_init_state(ec_state)
                baseline_qpos = np.asarray(env.sim.data.qpos, dtype=float).copy()
                baseline_qvel = np.asarray(env.sim.data.qvel, dtype=float).copy()
                env.sim.data.qpos[
                    qpos_address : qpos_address + 7
                ] = settled_lower_qpos
                env.sim.data.qvel[qvel_address : qvel_address + 6] = 0.0
                env.sim.forward()
                candidate_qpos = np.asarray(env.sim.data.qpos, dtype=float).copy()
                candidate_qvel = np.asarray(env.sim.data.qvel, dtype=float).copy()
                invariant_qpos_diff = _outside_max(
                    baseline_qpos, candidate_qpos, qpos_allowed
                )
                invariant_qvel_diff = _outside_max(
                    baseline_qvel, candidate_qvel, qvel_allowed
                )
                if invariant_qpos_diff > MAX_INVARIANT_QPOS_DIFF:
                    raise RuntimeError(f"ep{episode_idx:03d}: invariant qpos changed")
                if invariant_qvel_diff > MAX_INVARIANT_QVEL_DIFF:
                    raise RuntimeError(f"ep{episode_idx:03d}: invariant qvel changed")
                candidate = env.sim.get_state().flatten().copy()

                observation = restore_formal_observation(env, candidate)
                samples = [_sample(env, "eb")]
                for _ in range(FORMAL_WAIT_STEPS + CONFIRM_STEPS):
                    observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    samples.append(_sample(env, "eb"))
                valid, failures, stability = _evaluate_trace(samples)
                settled_xy = np.asarray(
                    stability[LOWER_BOWL]["pre_wait"]["position"], dtype=float
                )[:2]
                proposed_xy = proposed_qpos[:2]
                settle_xy_drift = float(np.linalg.norm(settled_xy - proposed_xy))
                if settle_xy_drift > MAX_NATIVE_XY_SETTLE_DRIFT_M:
                    failures.append("lower_bowl:proposed_xy_settle_drift")
                    valid = False
                if valid:
                    accepted = {
                        "candidate": candidate,
                        "stability": stability,
                        "settled_xy": settled_xy,
                        "settle_xy_drift": settle_xy_drift,
                        "radial_offset_m": radial_offset_m,
                        "invariant_qpos_diff": invariant_qpos_diff,
                        "invariant_qvel_diff": invariant_qvel_diff,
                    }
                    break
                attempt_failures.append(
                    {
                        "radial_offset_m": radial_offset_m,
                        "failures": sorted(set(failures)),
                    }
                )
            if accepted is None:
                raise RuntimeError(
                    f"ep{episode_idx:03d}: all registered Eb repairs failed: "
                    f"{attempt_failures}"
                )
            candidate = accepted["candidate"]
            stability = accepted["stability"]
            settled_xy = accepted["settled_xy"]
            native_xy_drift = float(np.linalg.norm(settled_xy - native_xy))
            invariant_qpos_diff = float(accepted["invariant_qpos_diff"])
            invariant_qvel_diff = float(accepted["invariant_qvel_diff"])
            repaired.append(candidate)
            records.append(
                {
                    "episode_idx": episode_idx,
                    "native_lower_bowl_xy": native_xy.tolist(),
                    "repaired_lower_bowl_xy": settled_xy.tolist(),
                    "native_to_repaired_xy_m": native_xy_drift,
                    "registered_radial_offset_m": accepted["radial_offset_m"],
                    "proposed_xy_settle_drift_m": accepted["settle_xy_drift"],
                    "rejected_registered_offsets": attempt_failures,
                    "invariant_qpos_max_abs_diff": invariant_qpos_diff,
                    "invariant_qvel_max_abs_diff": invariant_qvel_diff,
                    "first_policy_lower_bowl": stability[LOWER_BOWL][
                        "first_policy"
                    ],
                    "max_lower_bowl_tilt_deg": stability[LOWER_BOWL][
                        "max_tilt_deg"
                    ],
                    "max_lower_bowl_translation_drift_m": stability[LOWER_BOWL][
                        "max_translation_drift_m"
                    ],
                }
            )
            print(f"[repair] episode {episode_idx + 1}/{len(ec_states)} PASS")
    finally:
        env.close()

    output_path = Path(args.output)
    _write(output_path, group_key, repaired, old_attrs)
    output_sha = _sha256(output_path)
    manifest = {
        "schema_version": 1,
        "scenario": "L1-C1",
        "verdict": "PASS_L1C1_EB_REPAIR_BUILD",
        "host_verification": host,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_prompt": TASK_PROMPT,
        "repair_method": (
            "Ec supplies every invariant field; only the native second black "
            "bowl is isolated-settled and substituted"
        ),
        "old_eb_state_sha256": _sha256(old_eb_path),
        "ec_baseline_state_sha256": _sha256(ec_path),
        "output_state_path": str(output_path),
        "output_state_sha256": output_sha,
        "expected_state_sha256": {
            "eb": output_sha,
            "er": args.er_sha256,
            "ec": _sha256(ec_path),
        },
        "intervention_allowlist": {
            "body": LOWER_BOWL,
            "qpos_indices": sorted(qpos_allowed),
            "qvel_indices": sorted(qvel_allowed),
            "max_invariant_qpos_diff": MAX_INVARIANT_QPOS_DIFF,
            "max_invariant_qvel_diff": MAX_INVARIANT_QVEL_DIFF,
        },
        "construction_thresholds": {
            "settle_steps": SETTLE_STEPS,
            "settle_confirmation_steps": SETTLE_CONFIRM_STEPS,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "formal_confirmation_steps": CONFIRM_STEPS,
            "max_native_xy_settle_drift_m": MAX_NATIVE_XY_SETTLE_DRIFT_M,
            "registered_radial_offsets_away_from_plate_m": list(
                REGISTERED_RADIAL_OFFSETS_M
            ),
            "max_receptacle_tilt_deg": 1.0,
        },
        "episode_count": len(records),
        "records": records,
    }
    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Verdict: PASS_L1C1_EB_REPAIR_BUILD")
    print(f"State SHA-256: {output_sha}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_eb_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument(
        "--er_sha256",
        default="66aa396f565c9d80187e756b34a1370f52d1b3e43c87d751dab3e13ce65a326e",
    )
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--allow_local_simulator", action="store_true")
    repair(parser.parse_args())


if __name__ == "__main__":
    main()
