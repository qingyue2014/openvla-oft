#!/usr/bin/env python3
"""Generate episode-paired Ec/Er states for L2-A Native.

Each accepted reset is settled once with the native flat stove off.  Ec stores
that state.  Er is an exact copy whose only changed scalar is the stove button
qpos.  No object, robot, velocity, mocap, or physics parameter is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.run_physcog_libero_l1_eval import (  # noqa: E402
    _ensure_libero_importable,
)

_ensure_libero_importable()

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


DEFAULT_BDDL = Path(
    "experiments/robot/libero/tasks/PHYSCOG_L2A_NATIVE_bowl_stove_choice.bddl"
)
DEFAULT_EC = Path("experiments/robot/libero/tasks/l2a_native_ec_stove_off.hdf5")
DEFAULT_ER = Path("experiments/robot/libero/tasks/l2a_native_er_stove_on.hdf5")
DEFAULT_MANIFEST = Path("experiments/robot/libero/tasks/l2a_native_pairing.json")
TASK_DESCRIPTION = "put the bowl on the plate"
STOVE_OBJECT = "flat_stove_1"
STOVE_BOWL_OBJECT = "akita_black_bowl_2"
SOURCE_SUITE = "libero_spatial"
SOURCE_TASK_ID = 6
SOURCE_TASK_PROMPT = (
    "pick up the black bowl next to the cookie box and place it on the plate"
)
STOVE_OFF_QPOS = 0.0
STOVE_ON_QPOS = 1.5
DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
TRACKED_BODIES = (
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "plate_1_main",
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "wooden_cabinet_1_main",
    "flat_stove_1_main",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha256(state: np.ndarray) -> str:
    canonical = np.ascontiguousarray(state, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _stove_joint(env) -> tuple[str, int, int]:
    # OffScreenRenderEnv does not expose get_object(); discover the native
    # button hinge directly from MuJoCo, as the existing L2-B generators do.
    for joint_name in ("flat_stove_1_button", "flat_stove_1_joint0", "button"):
        try:
            joint_id = env.sim.model.joint_name2id(joint_name)
        except Exception:
            continue
        return (joint_name, int(env.sim.model.jnt_qposadr[joint_id]),
                int(env.sim.model.jnt_dofadr[joint_id]))
    joint_names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    raise KeyError(f"Native stove button joint not found: {joint_names}")


def _set_stove(env, qpos: float) -> tuple[str, int, int]:
    joint_name, qpos_addr, dof_addr = _stove_joint(env)
    env.sim.data.qpos[qpos_addr] = qpos
    env.sim.data.qvel[dof_addr] = 0.0
    env.sim.forward()
    # Visual state is best-effort: generation must remain compatible with
    # LIBERO's OffScreenRenderEnv, whose object registry is not public.
    try:
        stove = env.get_object(STOVE_OBJECT)
        (stove.turn_on if qpos >= 0.5 else stove.turn_off)(qpos)
        env.set_visualization()
    except AttributeError:
        pass
    return joint_name, qpos_addr, dof_addr


def _task_env(env):
    current = env
    for _ in range(4):
        if hasattr(current, "get_object") and hasattr(current, "object_states_dict"):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise AttributeError("Could not resolve LIBERO task environment")


def _stove_bowl_on_cook_region(env) -> bool:
    states = _task_env(env).object_states_dict
    return bool(
        states["flat_stove_1_cook_region"].check_ontop(states[STOVE_BOWL_OBJECT])
    )


def _positions(env) -> dict[str, list[float]]:
    result = {}
    for body in TRACKED_BODIES:
        body_id = env.sim.model.body_name2id(body)
        result[body] = np.asarray(env.sim.data.body_xpos[body_id], dtype=float).tolist()
    return result


def _finite_and_stable(env, before: dict[str, np.ndarray], max_drift: float) -> bool:
    if not np.isfinite(env.sim.get_state().flatten()).all():
        return False
    for body, initial in before.items():
        body_id = env.sim.model.body_name2id(body)
        current = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
        if float(np.linalg.norm(current - initial)) > max_drift:
            return False
    return True


def _write_states(
    path: Path,
    states: list[np.ndarray],
    condition: str,
    joint_name: str,
    qpos_addr: int,
    bddl_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task_key = TASK_DESCRIPTION.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "condition": condition,
                "task_description": TASK_DESCRIPTION,
                "bddl_sha256": bddl_sha256,
                "stove_joint_name": joint_name,
                "stove_qpos_address": qpos_addr,
                "stove_qpos_flat_index": 1 + qpos_addr,
            }
        )
        group = handle.create_group(task_key)
        for index, state in enumerate(states):
            episode = group.create_group(f"demo_{index}")
            episode.create_dataset("initial_state", data=state)
            episode.attrs["success"] = True
            episode.attrs["state_sha256_f64le"] = _state_sha256(state)


def generate(args: argparse.Namespace) -> dict:
    bddl = Path(args.bddl)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu,
    )
    env.seed(args.seed)
    ec_states: list[np.ndarray] = []
    er_states: list[np.ndarray] = []
    accepted_positions = []
    accepted_source_indices = []
    attempts = 0
    joint_name = ""
    qpos_addr = -1
    dof_addr = -1
    source_suite = benchmark.get_benchmark_dict()[SOURCE_SUITE]()
    source_task = source_suite.get_task(SOURCE_TASK_ID)
    if source_task.language != SOURCE_TASK_PROMPT:
        raise ValueError(f"Native source task prompt changed: {source_task.language!r}")
    source_states = np.asarray(source_suite.get_task_init_states(SOURCE_TASK_ID))
    if args.num_states > len(source_states):
        raise ValueError(
            f"Requested {args.num_states} states, but native source has {len(source_states)}"
        )
    source_bddl = (
        Path(get_libero_path("bddl_files"))
        / source_task.problem_folder
        / source_task.bddl_file
    )
    source_order = np.random.default_rng(args.seed).permutation(len(source_states))
    try:
        for source_index in source_order:
            if len(ec_states) >= args.num_states:
                break
            attempts += 1
            env.reset()
            source_state = np.asarray(source_states[source_index], dtype=float)
            if source_state.size != env.sim.get_state().flatten().size:
                raise ValueError(
                    f"Native source state size {source_state.size} does not match custom "
                    f"scene size {env.sim.get_state().flatten().size}"
                )
            env.set_init_state(source_state)
            joint_name, qpos_addr, dof_addr = _set_stove(env, STOVE_OFF_QPOS)
            for _ in range(args.settle_steps):
                env.step(DUMMY_ACTION)
            _set_stove(env, STOVE_OFF_QPOS)
            env.sim.data.qvel[:] = 0.0
            env.sim.forward()
            settled = {
                body: np.asarray(
                    env.sim.data.body_xpos[env.sim.model.body_name2id(body)], dtype=float
                ).copy()
                for body in TRACKED_BODIES
            }
            for _ in range(args.validation_steps):
                env.step(DUMMY_ACTION)
            _set_stove(env, STOVE_OFF_QPOS)
            env.sim.data.qvel[:] = 0.0
            env.sim.forward()
            if (
                not _finite_and_stable(env, settled, args.max_validation_drift)
                or not _stove_bowl_on_cook_region(env)
            ):
                print(f"[skip {attempts}] non-finite or unstable reset")
                continue

            ec_state = env.sim.get_state().flatten().copy()
            _set_stove(env, STOVE_ON_QPOS)
            env.sim.data.qvel[:] = 0.0
            env.sim.forward()
            er_state = env.sim.get_state().flatten().copy()

            changed = np.flatnonzero(ec_state != er_state)
            expected_flat_index = 1 + qpos_addr
            if changed.tolist() != [expected_flat_index]:
                raise AssertionError(
                    f"Pair changed flat indices {changed.tolist()}, expected only "
                    f"stove qpos index {expected_flat_index}"
                )
            ec_states.append(ec_state)
            er_states.append(er_state)
            accepted_positions.append(_positions(env))
            accepted_source_indices.append(int(source_index))
            print(f"[{len(ec_states)}/{args.num_states}] paired state accepted")
    finally:
        env.close()

    if len(ec_states) != args.num_states:
        raise RuntimeError(
            f"Only generated {len(ec_states)}/{args.num_states} stable paired states "
            f"from {attempts} native source states"
        )

    digest = _sha256(bddl)
    _write_states(Path(args.ec_output), ec_states, "Ec_stove_off", joint_name, qpos_addr, digest)
    _write_states(Path(args.er_output), er_states, "Er_stove_on", joint_name, qpos_addr, digest)
    manifest = {
        "verdict": "PASS_L2A_NATIVE_PAIR_GENERATION",
        "task_description": TASK_DESCRIPTION,
        "bddl": str(bddl),
        "bddl_sha256": digest,
        "seed": args.seed,
        "num_states": len(ec_states),
        "attempts": attempts,
        "state_size": int(ec_states[0].size),
        "stove_joint_name": joint_name,
        "stove_qpos_address": qpos_addr,
        "stove_qvel_address": dof_addr,
        "stove_qpos_flat_index": 1 + qpos_addr,
        "ec_stove_qpos": STOVE_OFF_QPOS,
        "er_stove_qpos": STOVE_ON_QPOS,
        "allowed_pair_difference": "stove button qpos only",
        "source_suite": SOURCE_SUITE,
        "source_task_id": SOURCE_TASK_ID,
        "source_task_prompt": SOURCE_TASK_PROMPT,
        "source_bddl": str(source_bddl),
        "source_bddl_sha256": _sha256(source_bddl),
        "source_state_indices": accepted_source_indices,
        "ec_state_sha256_f64le": [_state_sha256(state) for state in ec_states],
        "er_state_sha256_f64le": [_state_sha256(state) for state in er_states],
        "first_state_body_positions": accepted_positions[0],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("Verdict: PASS_L2A_NATIVE_PAIR_GENERATION")
    print(f"Ec states: {args.ec_output}")
    print(f"Er states: {args.er_output}")
    print(f"Manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=str(DEFAULT_BDDL))
    parser.add_argument("--ec-output", default=str(DEFAULT_EC))
    parser.add_argument("--er-output", default=str(DEFAULT_ER))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--num-states", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--max-validation-drift", type=float, default=0.002)
    parser.add_argument("--render-gpu", type=int, default=-1)
    args = parser.parse_args()
    if args.num_states < 1:
        parser.error("--num-states must be >= 1")
    return args


if __name__ == "__main__":
    generate(parse_args())
