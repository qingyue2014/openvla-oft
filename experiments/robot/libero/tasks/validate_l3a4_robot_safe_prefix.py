"""Execute the complete L3-A4 safe reference through robot OSC actions.

After restoring an accepted serialized Er initial state, every task action is
driven by ``env.step``: park the porcelain mug, grasp and place the target mug,
then grasp the native microwave handle and follow its closing arc.  This file
must never write task-object qpos, fixture-joint qpos, or model fixture poses.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.l3a_cascade_oracle import (
    TaskActorCascadeOracle,
)
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.l3a4_microwave_common import (
    DUMMY_ACTION,
    MAX_MUG_TILT_DEG,
    MAX_WAIT_ANGULAR_SPEED_RADPS,
    MAX_WAIT_LINEAR_SPEED_MPS,
    MIN_CASCADE_DISPLACEMENT_M,
    PORCELAIN_BODY,
    TARGET_BODY,
    TASK_KEY,
    body_pose,
    body_speeds,
    body_tilt_deg,
    closest_point_on_oriented_box,
    collision_masks_compatible,
    contact_body_names,
    contacts_between,
    convex_mesh_aabb_distance,
    descendant_body_ids,
    descendant_geom_ids,
    native_site_contains_point,
    oriented_box_separating_clearance,
    planar_park_clearances,
    policy_image,
    resolve_microwave_names,
    segment_aabb_distance,
    triangle_aabb_distance,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)


APPROACH_HEIGHT = 0.16
GRASP_HEIGHT = 0.060
PORCELAIN_GRASP_HEIGHT = 0.080
PORCELAIN_GRASP_CLEARANCE_OFFSET = 0.040
PORCELAIN_CONTACT_SEEK_STEPS = 80
PORCELAIN_CONTACT_SEEK_GAIN = 12.0
PORCELAIN_CONTACT_SEEK_ACTION_LIMIT = 0.25
PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M = 0.030
TARGET_GRASP_CLEARANCE_OFFSET = 0.040
TARGET_CONTACT_SEEK_STEPS = 80
TARGET_CONTACT_SEEK_GAIN = 12.0
TARGET_CONTACT_SEEK_ACTION_LIMIT = 0.25
TARGET_DYNAMIC_AXIS_PROGRESS_EPS_M = 1e-6
TARGET_TRIAL_CANONICAL_REFRESH_MAX_PASSES = 4
SAFE_PARK_MIN_OUTWARD_DISTANCE_M = 0.060
SAFE_PARK_MAX_OUTWARD_DISTANCE_M = 0.400
SAFE_PARK_SEARCH_STEP_M = 0.010
SAFE_PARK_TABLE_EDGE_MARGIN_M = 0.020
SAFE_PARK_DOOR_SWEEP_MARGIN_M = 0.020
SAFE_PARK_STATIC_MARGIN_M = 0.020
SAFE_PARK_DOOR_SWEEP_SAMPLES = 49
TARGET_INSERTION_SEARCH_STEP_M = 0.005
TARGET_INSERTION_SWEEP_STEP_M = 0.005
EEF_POSITION_TOLERANCE = 0.012
MOVE_STEPS = 100
GRIPPER_STEPS = 15
PARK_SETTLE_STEPS = 40
TARGET_SETTLE_STEPS = 60
DOOR_ARC_WAYPOINTS = 24
POST_CLOSE_STEPS = 60


class DeterministicRestoreError(RuntimeError):
    """A counterfactual trial could not restore its exact start state."""


_PLAIN_SCALARS = (str, bytes, bool, int, float, type(None))


def _snapshot_plain_state(value, field_path="state"):
    """Copy state without silently accepting simulator-backed objects."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, _PLAIN_SCALARS):
        return value
    if isinstance(value, list):
        return [
            _snapshot_plain_state(child, f"{field_path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _snapshot_plain_state(child, f"{field_path}[{index}]")
            for index, child in enumerate(value)
        )
    if isinstance(value, dict):
        copied = {}
        for key, child in value.items():
            if not isinstance(key, _PLAIN_SCALARS[:-1]):
                raise DeterministicRestoreError(
                    f"unsupported state key at {field_path}: {type(key)!r}"
                )
            copied[key] = _snapshot_plain_state(
                child, f"{field_path}[{key!r}]"
            )
        return copied
    if isinstance(value, set):
        return {
            _snapshot_plain_state(child, f"{field_path}[]")
            for child in value
        }
    raise DeterministicRestoreError(
        f"unsupported mutable state at {field_path}: {type(value)!r}"
    )


def _plain_state_equal(expected, actual) -> bool:
    if isinstance(expected, np.ndarray):
        return bool(
            isinstance(actual, np.ndarray)
            and expected.dtype == actual.dtype
            and expected.shape == actual.shape
            and np.array_equal(expected, actual)
        )
    if isinstance(expected, np.generic):
        return bool(
            isinstance(actual, np.generic)
            and expected.dtype == actual.dtype
            and expected == actual
        )
    if isinstance(expected, _PLAIN_SCALARS):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (list, tuple)):
        return bool(
            type(expected) is type(actual)
            and len(expected) == len(actual)
            and all(
                _plain_state_equal(left, right)
                for left, right in zip(expected, actual)
            )
        )
    if isinstance(expected, dict):
        return bool(
            isinstance(actual, dict)
            and set(expected) == set(actual)
            and all(
                _plain_state_equal(expected[key], actual[key])
                for key in expected
            )
        )
    if isinstance(expected, set):
        return expected == actual
    return False


def _update_state_digest(digest, value) -> None:
    """Feed a type- and shape-stable plain-state encoding to SHA-256."""
    if isinstance(value, np.ndarray):
        digest.update(b"array\0")
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        _update_state_digest(digest, np.asarray(value))
        return
    if isinstance(value, _PLAIN_SCALARS):
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode())
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for child in value:
            _update_state_digest(digest, child)
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
        return
    if isinstance(value, set):
        digest.update(b"set\0")
        for child in sorted(value, key=lambda item: repr(item)):
            _update_state_digest(digest, child)
        return
    raise DeterministicRestoreError(
        f"cannot hash unsupported state type {type(value)!r}"
    )


def _plain_state_sha256(value) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def _record_from_demo(demo) -> dict:
    record = {"initial_state": demo["initial_state"][:]}
    for key, value in demo.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        record[key] = value
    return record


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eef_body_name(model) -> str:
    for name in ("gripper0_eef", "robot0_gripper0_eef", "robot0_eef"):
        try:
            model.body_name2id(name)
            return name
        except Exception:
            continue
    matches = [
        model.body_id2name(index)
        for index in range(int(model.nbody))
        if (model.body_id2name(index) or "").endswith("gripper0_eef")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve end-effector body; matches={matches}")
    return matches[0]


def _eef_position(env) -> np.ndarray:
    model = env.sim.model
    body_id = int(model.body_name2id(_eef_body_name(model)))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _robot_contact_body_names(env) -> set[str]:
    return {
        pair["other_body_name"]
        for pair in _robot_contact_pairs(env)
        if pair["other_body_name"]
    }


def _robot_contact_pairs(env) -> list[dict]:
    robot_geoms = _robot_geom_ids(env.sim.model)
    contacts = {}
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in robot_geoms:
            robot = int(contact.geom1)
            other = int(contact.geom2)
        elif contact.geom2 in robot_geoms:
            robot = int(contact.geom2)
            other = int(contact.geom1)
        else:
            continue
        robot_body = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[robot])
        ) or ""
        other_body = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[other])
        ) or ""
        if other_body.startswith(("robot0_", "gripper0_")):
            continue
        key = (robot, other)
        contacts[key] = {
            "contact_index": int(index),
            "robot_geom_id": robot,
            "robot_geom_name": _geom_name(env.sim.model, robot),
            "robot_body_name": str(robot_body),
            "other_geom_id": other,
            "other_geom_name": _geom_name(env.sim.model, other),
            "other_body_name": str(other_body),
        }
    return [contacts[key] for key in sorted(contacts)]


def _has_microwave_contact(body_names) -> bool:
    return any("microwave" in name.lower() for name in body_names)


_SIM_RUNTIME_ARRAY_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "eq_active",
)
_ROBOT_RUNTIME_ARRAY_FIELDS = ("torques",)
_ROBOT_RUNTIME_BUFFER_FIELDS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)
_OBSERVABLE_STATIC_FIELDS = (
    "_sensor",
    "_corrupter",
    "_filter",
    "_delayer",
)
_OSC_REQUIRED_RESTORE_FIELDS = {
    "goal_pos",
    "goal_ori",
    "relative_ori",
    "ori_ref",
    "initial_joint",
    "new_update",
}


def _snapshot_object_fields(obj, *, skip=(), label="object") -> dict:
    if not hasattr(obj, "__dict__"):
        raise DeterministicRestoreError(
            f"{label} has no inspectable Python state"
        )
    skipped = set(skip)
    return {
        key: _snapshot_plain_state(value, f"{label}.{key}")
        for key, value in sorted(obj.__dict__.items())
        if key not in skipped
    }


def _restore_object_fields(
    obj, snapshot: dict, *, skip=(), label="object"
) -> None:
    live_keys = set(obj.__dict__) - set(skip)
    if live_keys != set(snapshot):
        raise DeterministicRestoreError(
            f"{label} state fields changed during trial: "
            f"expected={sorted(snapshot)} live={sorted(live_keys)}"
        )
    for key, value in snapshot.items():
        obj.__dict__[key] = _snapshot_plain_state(
            value, f"{label}.{key}.restore"
        )


def _all_contact_signature(env) -> tuple[tuple[int, int], ...]:
    pairs = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        pairs.append((min(geom1, geom2), max(geom1, geom2)))
    return tuple(sorted(pairs))


def _trial_physical_signature(env, names) -> dict:
    body_states = {}
    for body_name in (TARGET_BODY, PORCELAIN_BODY, names["door_body"]):
        body_id = int(env.sim.model.body_name2id(body_name))
        body_states[body_name] = {
            "position": np.asarray(
                env.sim.data.body_xpos[body_id], dtype=float
            ).copy(),
            "rotation": np.asarray(
                env.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3).copy(),
            "quaternion": np.asarray(
                env.sim.data.body_xquat[body_id], dtype=float
            ).copy(),
        }
    door_joint_id = int(
        env.sim.model.joint_name2id(names["door_joint"])
    )
    door_qpos_address = int(env.sim.model.jnt_qposadr[door_joint_id])
    door_qvel_address = int(env.sim.model.jnt_dofadr[door_joint_id])
    eef_body_id = int(
        env.sim.model.body_name2id(_eef_body_name(env.sim.model))
    )
    return {
        "bodies": body_states,
        "door_joint_qpos": float(env.sim.data.qpos[door_qpos_address]),
        "door_joint_qvel": float(env.sim.data.qvel[door_qvel_address]),
        "eef_position": np.asarray(
            env.sim.data.body_xpos[eef_body_id], dtype=float
        ).copy(),
        "eef_rotation": np.asarray(
            env.sim.data.body_xmat[eef_body_id], dtype=float
        ).reshape(3, 3).copy(),
        "all_contact_geom_pairs": _all_contact_signature(env),
        "robot_contact_pairs": tuple(
            (
                pair["robot_geom_id"],
                pair["other_geom_id"],
            )
            for pair in _robot_contact_pairs(env)
        ),
    }


def _target_trial_native_runtime(env):
    """Return the one native OSC runtime or fail before trial mutation."""
    if len(env.robots) != 1:
        raise DeterministicRestoreError(
            f"dynamic reachability requires one robot, got {len(env.robots)}"
        )
    robot = env.robots[0]
    controller = getattr(robot, "controller", None)
    if controller is None or getattr(controller, "name", None) != "OSC_POSE":
        raise DeterministicRestoreError(
            "dynamic reachability requires the native OSC_POSE controller"
        )
    if (
        getattr(controller, "interpolator_pos", None) is not None
        or getattr(controller, "interpolator_ori", None) is not None
    ):
        raise DeterministicRestoreError(
            "cannot deterministically restore non-native OSC interpolators"
        )
    return robot, controller


def _target_trial_canonical_fingerprint(env, names) -> dict:
    """Capture fields changed or derived by MuJoCo / OSC refresh."""
    robot, controller = _target_trial_native_runtime(env)
    sim_arrays = {
        field: np.asarray(getattr(env.sim.data, field)).copy()
        for field in _SIM_RUNTIME_ARRAY_FIELDS
        if hasattr(env.sim.data, field)
    }
    for required in ("qpos", "qvel", "act", "ctrl"):
        if required not in sim_arrays:
            raise DeterministicRestoreError(
                f"MuJoCo runtime field missing: {required}"
            )
    robot_arrays = {}
    for field in _ROBOT_RUNTIME_ARRAY_FIELDS:
        if not hasattr(robot, field):
            raise DeterministicRestoreError(
                f"robot runtime field missing: {field}"
            )
        robot_arrays[field] = _snapshot_plain_state(
            getattr(robot, field), f"robot.{field}.canonical"
        )
    robot_buffers = {}
    for field in _ROBOT_RUNTIME_BUFFER_FIELDS:
        buffer = getattr(robot, field, None)
        if buffer is None:
            raise DeterministicRestoreError(
                f"robot runtime buffer missing: {field}"
            )
        robot_buffers[field] = {
            "class_name": type(buffer).__name__,
            "state": _snapshot_object_fields(
                buffer, label=f"robot.{field}.canonical"
            ),
        }
    return {
        "sim_flat": np.asarray(
            env.sim.get_state().flatten(), dtype=float
        ).copy(),
        "sim_time": float(env.sim.data.time),
        "sim_arrays": sim_arrays,
        "controller_state": _snapshot_object_fields(
            controller, skip=("sim",), label="controller.canonical"
        ),
        "robot_arrays": robot_arrays,
        "robot_buffers": robot_buffers,
        "physical": _trial_physical_signature(env, names),
    }


def _canonical_refresh_target_trial_state(env, names) -> dict:
    """Reach and prove an exact idempotent MuJoCo / OSC boundary.

    Every source snapshot and every restore runs this identical refresh.  The
    fixed-point gate is deliberately bitwise: a merely close EEF or body pose
    cannot authorize a counterfactual trial.
    """
    _, controller = _target_trial_native_runtime(env)
    previous = None
    fingerprint_sha256 = []
    for refresh_pass in range(
        1, TARGET_TRIAL_CANONICAL_REFRESH_MAX_PASSES + 1
    ):
        env.sim.forward()
        controller.update(force=True)
        current = _target_trial_canonical_fingerprint(env, names)
        current_sha256 = _plain_state_sha256(current)
        fingerprint_sha256.append(current_sha256)
        if previous is not None and _plain_state_equal(previous, current):
            return {
                "passed": True,
                "comparison": "bitwise_exact",
                "refresh_sequence": [
                    "sim.forward",
                    "controller.update(force=True)",
                ],
                "passes_executed": int(refresh_pass),
                "stable_fingerprint_sha256": current_sha256,
                "fingerprint_sha256": fingerprint_sha256,
            }
        previous = current
    raise DeterministicRestoreError(
        "MuJoCo / OSC candidate boundary did not reach an exact canonical "
        "fixed point; no tolerance fallback is permitted; "
        f"fingerprint_sha256={fingerprint_sha256}"
    )


def _target_trial_state_sha256(snapshot) -> str:
    """Hash state values while excluding process-local object identities."""
    payload = {
        key: value
        for key, value in snapshot.items()
        if key
        not in {
            "state_sha256",
            "controller_static_ids",
            "canonical_refresh",
        }
    }
    payload["robot_buffers"] = {
        field: {
            "class_name": record["class_name"],
            "state": record["state"],
        }
        for field, record in snapshot["robot_buffers"].items()
    }
    payload["observable_state"] = {
        name: record["state"]
        for name, record in snapshot["observable_state"].items()
    }
    return _plain_state_sha256(payload)


def _snapshot_target_trial_state(env, oracle, names) -> dict:
    """Capture the complete deterministic boundary for one candidate trial."""
    robot, controller = _target_trial_native_runtime(env)
    canonical_refresh = _canonical_refresh_target_trial_state(env, names)

    controller_state = _snapshot_object_fields(
        controller, skip=("sim",), label="controller"
    )
    missing_controller_fields = (
        _OSC_REQUIRED_RESTORE_FIELDS - set(controller_state)
    )
    if missing_controller_fields:
        raise DeterministicRestoreError(
            "OSC restore fields missing: "
            f"{sorted(missing_controller_fields)}"
        )
    controller_static_ids = {"sim": id(controller.sim)}
    robot_arrays = {}
    for field in _ROBOT_RUNTIME_ARRAY_FIELDS:
        if not hasattr(robot, field):
            raise DeterministicRestoreError(
                f"robot runtime field missing: {field}"
            )
        robot_arrays[field] = _snapshot_plain_state(
            getattr(robot, field), f"robot.{field}"
        )
    robot_buffers = {}
    for field in _ROBOT_RUNTIME_BUFFER_FIELDS:
        buffer = getattr(robot, field, None)
        if buffer is None:
            raise DeterministicRestoreError(
                f"robot runtime buffer missing: {field}"
            )
        robot_buffers[field] = {
            "object_id": id(buffer),
            "class_name": type(buffer).__name__,
            "state": _snapshot_object_fields(
                buffer, label=f"robot.{field}"
            ),
        }

    inner_env = getattr(env, "env", None)
    if inner_env is None:
        raise DeterministicRestoreError(
            "LIBERO wrapper does not expose its robosuite environment"
        )
    env_state = {
        field: _snapshot_plain_state(
            getattr(inner_env, field), f"env.{field}"
        )
        for field in ("cur_time", "timestep", "done", "_obs_cache")
        if hasattr(inner_env, field)
    }
    if set(env_state) != {"cur_time", "timestep", "done", "_obs_cache"}:
        raise DeterministicRestoreError(
            "robosuite environment runtime counters are incomplete"
        )
    observables = getattr(inner_env, "_observables", None)
    if not isinstance(observables, dict):
        raise DeterministicRestoreError(
            "robosuite observable registry is not inspectable"
        )
    observable_state = {}
    for observable_name, observable in sorted(observables.items()):
        observable_state[observable_name] = {
            "object_id": id(observable),
            "static_ids": {
                field: id(getattr(observable, field))
                for field in _OBSERVABLE_STATIC_FIELDS
            },
            "state": _snapshot_object_fields(
                observable,
                skip=_OBSERVABLE_STATIC_FIELDS,
                label=f"observable.{observable_name}",
            ),
        }

    sim_arrays = {}
    for field in _SIM_RUNTIME_ARRAY_FIELDS:
        if hasattr(env.sim.data, field):
            sim_arrays[field] = np.asarray(
                getattr(env.sim.data, field)
            ).copy()
    for required in ("qpos", "qvel", "act", "ctrl"):
        if required not in sim_arrays:
            raise DeterministicRestoreError(
                f"MuJoCo runtime field missing: {required}"
            )
    snapshot = {
        "sim_flat": np.asarray(
            env.sim.get_state().flatten(), dtype=float
        ).copy(),
        "sim_time": float(env.sim.data.time),
        "sim_arrays": sim_arrays,
        "controller_state": controller_state,
        "controller_static_ids": controller_static_ids,
        "robot_arrays": robot_arrays,
        "robot_buffers": robot_buffers,
        "env_state": env_state,
        "observable_state": observable_state,
        "oracle_state": _snapshot_object_fields(oracle, label="oracle"),
        "numpy_random_state": _snapshot_plain_state(
            np.random.get_state(), "numpy_random_state"
        ),
        "physical": _trial_physical_signature(env, names),
        "canonical_refresh": canonical_refresh,
    }
    snapshot["state_sha256"] = _target_trial_state_sha256(snapshot)
    return snapshot


def _array_restore_evidence(expected, actual) -> dict:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    same_shape = expected.shape == actual.shape
    exact = bool(same_shape and np.array_equal(expected, actual))
    max_error = None
    if same_shape and expected.size and np.issubdtype(expected.dtype, np.number):
        max_error = float(
            np.max(
                np.abs(
                    expected.astype(float, copy=False)
                    - actual.astype(float, copy=False)
                )
            )
        )
    return {
        "exact": exact,
        "shape": list(expected.shape),
        "max_abs_error": max_error,
    }


def _restore_target_trial_state(env, oracle, names, snapshot) -> dict:
    """Restore a trial and prove every dynamics/controller field matches."""
    robot = env.robots[0]
    controller = robot.controller
    if id(controller.sim) != snapshot["controller_static_ids"]["sim"]:
        raise DeterministicRestoreError(
            "OSC simulator reference changed during candidate trial"
        )
    env.sim.set_state_from_flattened(snapshot["sim_flat"])
    env.sim.data.time = snapshot["sim_time"]
    for field, values in snapshot["sim_arrays"].items():
        live = getattr(env.sim.data, field, None)
        if live is None or np.asarray(live).shape != values.shape:
            raise DeterministicRestoreError(
                f"MuJoCo field cannot be restored exactly: {field}"
            )
        live[...] = values

    _restore_object_fields(
        controller,
        snapshot["controller_state"],
        skip=("sim",),
        label="controller",
    )
    for field, values in snapshot["robot_arrays"].items():
        setattr(
            robot,
            field,
            _snapshot_plain_state(values, f"robot.{field}.restore"),
        )
    for field, record in snapshot["robot_buffers"].items():
        buffer = getattr(robot, field, None)
        if (
            buffer is None
            or id(buffer) != record["object_id"]
            or type(buffer).__name__ != record["class_name"]
        ):
            raise DeterministicRestoreError(
                f"robot buffer identity changed during trial: {field}"
            )
        _restore_object_fields(
            buffer, record["state"], label=f"robot.{field}"
        )

    inner_env = env.env
    for field, value in snapshot["env_state"].items():
        setattr(
            inner_env,
            field,
            _snapshot_plain_state(value, f"env.{field}.restore"),
        )
    live_observables = inner_env._observables
    if set(live_observables) != set(snapshot["observable_state"]):
        raise DeterministicRestoreError(
            "observable inventory changed during candidate trial"
        )
    for observable_name, record in snapshot["observable_state"].items():
        observable = live_observables[observable_name]
        if id(observable) != record["object_id"]:
            raise DeterministicRestoreError(
                f"observable identity changed: {observable_name}"
            )
        for field, expected_id in record["static_ids"].items():
            if id(getattr(observable, field)) != expected_id:
                raise DeterministicRestoreError(
                    f"observable callable changed: {observable_name}.{field}"
                )
        _restore_object_fields(
            observable,
            record["state"],
            skip=_OBSERVABLE_STATIC_FIELDS,
            label=f"observable.{observable_name}",
        )
    _restore_object_fields(
        oracle, snapshot["oracle_state"], label="oracle"
    )
    np.random.set_state(copy.deepcopy(snapshot["numpy_random_state"]))

    restored = _snapshot_target_trial_state(env, oracle, names)
    sim_field_evidence = {
        "flat_state": _array_restore_evidence(
            snapshot["sim_flat"], restored["sim_flat"]
        ),
        "time": {
            "exact": snapshot["sim_time"] == restored["sim_time"],
            "expected": snapshot["sim_time"],
            "actual": restored["sim_time"],
        },
    }
    for field, values in snapshot["sim_arrays"].items():
        sim_field_evidence[field] = _array_restore_evidence(
            values, restored["sim_arrays"][field]
        )
    physical_field_evidence = {}
    for body_name, body_state in snapshot["physical"]["bodies"].items():
        for field, values in body_state.items():
            physical_field_evidence[f"{body_name}.{field}"] = (
                _array_restore_evidence(
                    values,
                    restored["physical"]["bodies"][body_name][field],
                )
            )
    for field in ("eef_position", "eef_rotation"):
        physical_field_evidence[field] = _array_restore_evidence(
            snapshot["physical"][field], restored["physical"][field]
        )
    for field in (
        "door_joint_qpos",
        "door_joint_qvel",
        "all_contact_geom_pairs",
        "robot_contact_pairs",
    ):
        physical_field_evidence[field] = {
            "exact": _plain_state_equal(
                snapshot["physical"][field], restored["physical"][field]
            )
        }
    controller_fields = {
        field: {
            "exact": _plain_state_equal(
                value, restored["controller_state"].get(field)
            )
        }
        for field, value in snapshot["controller_state"].items()
    }
    aggregate_fields = {
        "robot_arrays": _plain_state_equal(
            snapshot["robot_arrays"], restored["robot_arrays"]
        ),
        "robot_buffers": _plain_state_equal(
            snapshot["robot_buffers"], restored["robot_buffers"]
        ),
        "environment": _plain_state_equal(
            snapshot["env_state"], restored["env_state"]
        ),
        "observables": _plain_state_equal(
            snapshot["observable_state"], restored["observable_state"]
        ),
        "oracle": _plain_state_equal(
            snapshot["oracle_state"], restored["oracle_state"]
        ),
        "numpy_random_state": _plain_state_equal(
            snapshot["numpy_random_state"],
            restored["numpy_random_state"],
        ),
    }
    passed = bool(
        all(record["exact"] for record in sim_field_evidence.values())
        and all(
            record["exact"] for record in physical_field_evidence.values()
        )
        and all(record["exact"] for record in controller_fields.values())
        and all(aggregate_fields.values())
        and snapshot["state_sha256"] == restored["state_sha256"]
    )
    proof = {
        "passed": passed,
        "snapshot_sha256": snapshot["state_sha256"],
        "restored_sha256": restored["state_sha256"],
        "sim_fields": sim_field_evidence,
        "physical_fields": physical_field_evidence,
        "controller_fields": controller_fields,
        "aggregate_fields": aggregate_fields,
        "snapshot_canonical_refresh": snapshot["canonical_refresh"],
        "restored_canonical_refresh": restored["canonical_refresh"],
    }
    if not passed:
        raise DeterministicRestoreError(
            "candidate trial exact restore proof failed: "
            f"{proof}"
        )
    return proof


def _geom_name(model, geom_id: int) -> str:
    name = model.geom_id2name(int(geom_id))
    return "" if name is None else str(name)


def _closest_point_on_compiled_geom(env, geom_id: int, point):
    """Return a conservative closest point from compiled MuJoCo geometry."""
    model = env.sim.model
    center = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(
        env.sim.data.geom_xmat[geom_id], dtype=float
    ).reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=float) - center)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    inside = False

    if geom_type == 6:  # mjGEOM_BOX
        return closest_point_on_oriented_box(
            point,
            center,
            rotation,
            size,
        )
    elif geom_type == 2:  # mjGEOM_SPHERE
        norm = float(np.linalg.norm(local))
        inside = norm <= float(size[0])
        direction = (
            local / norm
            if norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0, 0.0])
        )
        closest_local = direction * float(size[0])
    elif geom_type == 3:  # mjGEOM_CAPSULE, local axis is z
        axis_point = np.asarray(
            [0.0, 0.0, np.clip(local[2], -size[1], size[1])]
        )
        radial = local - axis_point
        norm = float(np.linalg.norm(radial))
        inside = norm <= float(size[0])
        direction = (
            radial / norm
            if norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0, 0.0])
        )
        closest_local = axis_point + direction * float(size[0])
    elif geom_type == 5:  # mjGEOM_CYLINDER, local axis is z
        radial = local[:2]
        radial_norm = float(np.linalg.norm(radial))
        radial_direction = (
            radial / radial_norm
            if radial_norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0])
        )
        radial_gap = radial_norm - float(size[0])
        axial_gap = abs(float(local[2])) - float(size[1])
        inside = radial_gap <= 0.0 and axial_gap <= 0.0
        closest_local = local.copy()
        if inside:
            if -radial_gap <= -axial_gap:
                closest_local[:2] = radial_direction * float(size[0])
            else:
                closest_local[2] = (
                    float(size[1]) if local[2] >= 0.0 else -float(size[1])
                )
        else:
            closest_local[:2] = (
                radial_direction * min(radial_norm, float(size[0]))
            )
            closest_local[2] = np.clip(
                local[2], -float(size[1]), float(size[1])
            )
    else:
        # Native microwave collision geoms are boxes.  For any other compiled
        # native geom type, the model's bounding radius gives a conservative
        # direction without consulting or changing source XML.
        radius = float(model.geom_rbound[geom_id])
        norm = float(np.linalg.norm(local))
        inside = norm <= radius
        direction = (
            local / norm
            if norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0, 0.0])
        )
        closest_local = direction * radius

    closest = center + rotation @ closest_local
    return closest, inside


def _collision_compatible_geom_ids(model, candidates, references) -> list[int]:
    references = tuple(int(geom_id) for geom_id in references)
    return sorted(
        int(geom_id)
        for geom_id in candidates
        if any(
            collision_masks_compatible(
                model.geom_contype[geom_id],
                model.geom_conaffinity[geom_id],
                model.geom_contype[reference_id],
                model.geom_conaffinity[reference_id],
            )
            for reference_id in references
        )
    )


def _compiled_microwave_clearance(env, names, mug_position):
    """Resolve an outward XY direction from the compiled static microwave."""
    model = env.sim.model
    fixture_geoms = descendant_geom_ids(model, names["fixture_root"])
    door_geoms = descendant_geom_ids(model, names["door_body"])
    static_fixture_geoms = sorted(fixture_geoms - door_geoms)
    robot_geoms = sorted(_robot_geom_ids(model))
    if not robot_geoms:
        raise RuntimeError("compiled model has no robot geoms")
    collision_geoms = _collision_compatible_geom_ids(
        model,
        static_fixture_geoms,
        robot_geoms,
    )
    if not collision_geoms:
        fixture_masks = sorted(
            {
                (
                    int(model.geom_group[geom_id]),
                    int(model.geom_contype[geom_id]),
                    int(model.geom_conaffinity[geom_id]),
                )
                for geom_id in static_fixture_geoms
            }
        )
        robot_masks = sorted(
            {
                (
                    int(model.geom_group[geom_id]),
                    int(model.geom_contype[geom_id]),
                    int(model.geom_conaffinity[geom_id]),
                )
                for geom_id in robot_geoms
            }
        )
        raise RuntimeError(
            "compiled microwave has no robot-compatible static collision "
            f"geoms; fixture(group,contype,conaffinity)={fixture_masks}; "
            f"robot(group,contype,conaffinity)={robot_masks}"
        )

    mug_position = np.asarray(mug_position, dtype=float)
    candidates = []
    for geom_id in collision_geoms:
        closest, inside = _closest_point_on_compiled_geom(
            env, geom_id, mug_position
        )
        delta = mug_position - closest
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        if horizontal_distance <= np.finfo(float).eps:
            continue
        body_name = model.body_id2name(
            int(model.geom_bodyid[geom_id])
        ) or ""
        candidates.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "body_name": str(body_name),
                "geom_type": int(model.geom_type[geom_id]),
                "geom_group": int(model.geom_group[geom_id]),
                "geom_contype": int(model.geom_contype[geom_id]),
                "geom_conaffinity": int(
                    model.geom_conaffinity[geom_id]
                ),
                "closest_point": closest.tolist(),
                "distance_m": float(np.linalg.norm(delta)),
                "horizontal_distance_m": horizontal_distance,
                "inside": bool(inside),
                "direction_xy": (
                    delta[:2] / horizontal_distance
                ).tolist(),
            }
        )
    if not candidates:
        raise RuntimeError(
            "cannot resolve a horizontal clearance direction from compiled "
            "microwave geometry"
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["distance_m"],
            item["horizontal_distance_m"],
            item["geom_id"],
        ),
    )
    if selected["inside"]:
        raise RuntimeError(
            "porcelain mug center is inside compiled microwave collision "
            f"geom {selected['geom_name'] or selected['geom_id']}"
        )
    direction_xy = np.asarray(selected["direction_xy"], dtype=float)
    hinge_position, _ = body_pose(env.sim, names["door_body"])
    hinge_delta = mug_position[:2] - hinge_position[:2]
    hinge_norm = float(np.linalg.norm(hinge_delta))
    hinge_direction = (
        hinge_delta / hinge_norm
        if hinge_norm > np.finfo(float).eps
        else np.zeros(2, dtype=float)
    )
    return direction_xy, {
        "method": "nearest_compiled_static_microwave_collision_surface",
        "selected": selected,
        "candidate_count": len(candidates),
        "static_fixture_geom_count": len(static_fixture_geoms),
        "robot_geom_count": len(robot_geoms),
        "robot_compatible_collision_geom_count": len(collision_geoms),
        "collision_filter": (
            "(fixture.contype & robot.conaffinity) != 0 or "
            "(robot.contype & fixture.conaffinity) != 0; "
            "geom_group is diagnostic only"
        ),
        "hinge_position": hinge_position.tolist(),
        "hinge_away_direction_xy": hinge_direction.tolist(),
        "surface_hinge_direction_dot": float(
            np.dot(direction_xy, hinge_direction)
        ),
        "requested_outward_offset_m": PORCELAIN_GRASP_CLEARANCE_OFFSET,
        "predicted_eef_surface_horizontal_clearance_m": (
            selected["horizontal_distance_m"]
            + PORCELAIN_GRASP_CLEARANCE_OFFSET
        ),
    }


def _support_contact_box(env, support_body: str):
    model = env.sim.model
    mug_geoms = descendant_geom_ids(model, PORCELAIN_BODY)
    support_geoms = descendant_geom_ids(model, support_body)
    contacted_support_geoms = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in mug_geoms and contact.geom2 in support_geoms:
            contacted_support_geoms.add(int(contact.geom2))
        elif contact.geom2 in mug_geoms and contact.geom1 in support_geoms:
            contacted_support_geoms.add(int(contact.geom1))
    boxes = [
        geom_id
        for geom_id in contacted_support_geoms
        if int(model.geom_type[geom_id]) == 6
    ]
    if not boxes:
        raise RuntimeError(
            "porcelain mug has no compiled box contact with its recorded "
            f"support body {support_body!r}"
        )
    geom_id = max(
        boxes,
        key=lambda candidate: (
            float(
                model.geom_size[candidate][0]
                * model.geom_size[candidate][1]
            ),
            -int(candidate),
        ),
    )
    rotation = np.asarray(
        env.sim.data.geom_xmat[geom_id], dtype=float
    ).reshape(3, 3)
    normal = rotation[:, 2]
    normal_tilt_deg = float(
        np.degrees(
            np.arccos(np.clip(float(normal[2]), -1.0, 1.0))
        )
    )
    if normal_tilt_deg > MAX_MUG_TILT_DEG:
        raise RuntimeError(
            "compiled support contact is not an upright table top; "
            f"normal_tilt_deg={normal_tilt_deg}"
        )
    return geom_id, {
        "support_body": support_body,
        "geom_id": int(geom_id),
        "geom_name": _geom_name(model, geom_id),
        "geom_group": int(model.geom_group[geom_id]),
        "geom_contype": int(model.geom_contype[geom_id]),
        "geom_conaffinity": int(model.geom_conaffinity[geom_id]),
        "center": np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        ).tolist(),
        "rotation": rotation.tolist(),
        "half_size": np.asarray(
            model.geom_size[geom_id], dtype=float
        ).tolist(),
        "normal_tilt_deg": normal_tilt_deg,
        "selection": "largest compiled box in the actual mug-support contact",
    }


def _compiled_mug_horizontal_radius(env, mug_position) -> tuple[float, list[int]]:
    model = env.sim.model
    mug_geoms = sorted(
        geom_id
        for geom_id in descendant_geom_ids(model, PORCELAIN_BODY)
        if int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    )
    if not mug_geoms:
        raise RuntimeError("porcelain mug has no collision-capable compiled geoms")
    mug_position = np.asarray(mug_position, dtype=float)
    radius = max(
        float(
            np.linalg.norm(
                np.asarray(
                    env.sim.data.geom_xpos[geom_id], dtype=float
                )[:2]
                - mug_position[:2]
            )
            + model.geom_rbound[geom_id]
        )
        for geom_id in mug_geoms
    )
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("invalid compiled porcelain horizontal radius")
    return radius, mug_geoms


def _compiled_door_sweep_samples(env, names):
    model = env.sim.model
    robot_geoms = _robot_geom_ids(model)
    door_geoms = _collision_compatible_geom_ids(
        model,
        descendant_geom_ids(model, names["door_body"]),
        robot_geoms,
    )
    if not door_geoms:
        raise RuntimeError("compiled door has no robot-compatible collision geoms")
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_position, hinge_rotation = body_pose(
        env.sim, names["door_body"]
    )
    hinge_axis = hinge_rotation @ np.asarray(
        model.jnt_axis[joint_id], dtype=float
    )
    centers = []
    radii = []
    geom_records = []
    fractions = np.linspace(0.0, 1.0, SAFE_PARK_DOOR_SWEEP_SAMPLES)
    for geom_id in door_geoms:
        initial_center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        radius = float(model.geom_rbound[geom_id])
        if not np.isfinite(radius) or radius <= 0.0:
            raise RuntimeError(
                f"compiled door geom {_geom_name(model, geom_id)!r} "
                "has invalid bounding radius"
            )
        radial_vector = initial_center - hinge_position
        for fraction in fractions:
            center = hinge_position + _rotation_about_axis(
                radial_vector,
                hinge_axis,
                close_angle * float(fraction),
            )
            centers.append(center[:2])
            radii.append(radius)
        geom_records.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "geom_type": int(model.geom_type[geom_id]),
                "geom_rbound_m": radius,
                "geom_contype": int(model.geom_contype[geom_id]),
                "geom_conaffinity": int(
                    model.geom_conaffinity[geom_id]
                ),
            }
        )
    return np.asarray(centers), np.asarray(radii), {
        "door_body": names["door_body"],
        "door_start_qpos": start_qpos,
        "door_closed_qpos": closed_qpos,
        "door_close_angle": close_angle,
        "hinge_position": hinge_position.tolist(),
        "hinge_axis": hinge_axis.tolist(),
        "samples_per_geom": SAFE_PARK_DOOR_SWEEP_SAMPLES,
        "door_collision_geoms": geom_records,
    }


def _compiled_safe_outward_park(
    env,
    names,
    support_body,
    start_mug_position,
    outward_direction_xy,
):
    """Find the nearest table-supported point outside the compiled door sweep."""
    model = env.sim.model
    start = np.asarray(start_mug_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    outward_norm = float(np.linalg.norm(outward))
    if outward.shape != (2,) or outward_norm <= np.finfo(float).eps:
        raise RuntimeError("safe-park outward direction is invalid")
    outward = outward / outward_norm
    support_geom, support_record = _support_contact_box(env, support_body)
    table_center = np.asarray(
        env.sim.data.geom_xpos[support_geom], dtype=float
    )
    table_rotation = np.asarray(
        env.sim.data.geom_xmat[support_geom], dtype=float
    ).reshape(3, 3)
    table_half_size = np.asarray(
        model.geom_size[support_geom], dtype=float
    )
    table_normal = table_rotation[:, 2]
    table_top = table_center + table_normal * table_half_size[2]
    support_offset = float(np.dot(start - table_top, table_normal))
    mug_radius, mug_geoms = _compiled_mug_horizontal_radius(env, start)
    door_centers, door_radii, door_record = _compiled_door_sweep_samples(
        env, names
    )
    fixture_geoms = descendant_geom_ids(model, names["fixture_root"])
    door_geoms = descendant_geom_ids(model, names["door_body"])
    static_geoms = _collision_compatible_geom_ids(
        model,
        fixture_geoms - door_geoms,
        mug_geoms,
    )
    if not static_geoms:
        raise RuntimeError(
            "compiled microwave has no mug-compatible static collision geoms"
        )

    trace = []
    distances = np.arange(
        SAFE_PARK_MIN_OUTWARD_DISTANCE_M,
        SAFE_PARK_MAX_OUTWARD_DISTANCE_M
        + 0.5 * SAFE_PARK_SEARCH_STEP_M,
        SAFE_PARK_SEARCH_STEP_M,
    )
    selected = None
    for distance in distances:
        candidate_xy = start[:2] + outward * float(distance)
        plane_rhs = (
            support_offset
            + float(np.dot(table_normal, table_top))
            - float(np.dot(table_normal[:2], candidate_xy))
        )
        if abs(float(table_normal[2])) <= np.finfo(float).eps:
            raise RuntimeError("compiled table top has a vertical normal")
        candidate = np.asarray(
            [
                candidate_xy[0],
                candidate_xy[1],
                plane_rhs / float(table_normal[2]),
            ]
        )
        planar = planar_park_clearances(
            candidate_xy,
            table_center[:2],
            table_rotation[:2, :2],
            table_half_size[:2],
            mug_radius,
            door_centers,
            door_radii,
        )
        static_candidates = []
        for geom_id in static_geoms:
            closest, inside = _closest_point_on_compiled_geom(
                env, geom_id, candidate
            )
            clearance = (
                -float("inf")
                if inside
                else float(np.linalg.norm(candidate - closest)) - mug_radius
            )
            static_candidates.append((clearance, int(geom_id)))
        static_clearance, closest_static_geom = min(static_candidates)
        passed = bool(
            planar["table_edge_clearance_m"]
            >= SAFE_PARK_TABLE_EDGE_MARGIN_M
            and planar["obstacle_clearance_m"]
            >= SAFE_PARK_DOOR_SWEEP_MARGIN_M
            and static_clearance >= SAFE_PARK_STATIC_MARGIN_M
        )
        record = {
            "outward_distance_m": float(distance),
            "candidate_position": candidate.tolist(),
            "table_edge_clearance_m": planar["table_edge_clearance_m"],
            "door_sweep_clearance_m": planar["obstacle_clearance_m"],
            "static_microwave_clearance_m": static_clearance,
            "closest_static_geom_id": closest_static_geom,
            "closest_static_geom_name": _geom_name(
                model, closest_static_geom
            ),
            "passed": passed,
        }
        trace.append(record)
        if passed:
            selected = record
            break
    if selected is None:
        raise RuntimeError(
            "no compiled table-supported outward park point clears the "
            f"door sweep within {SAFE_PARK_MAX_OUTWARD_DISTANCE_M} m; "
            f"last_candidate={trace[-1] if trace else None}"
        )
    return np.asarray(selected["candidate_position"], dtype=float), {
        "method": (
            "nearest outward point inside actual compiled support box and "
            "outside sampled compiled door sweep/static microwave"
        ),
        "outward_direction_xy": outward.tolist(),
        "selected": selected,
        "mug_horizontal_radius_bound_m": mug_radius,
        "support_offset_along_normal_m": support_offset,
        "support_geometry": support_record,
        "door_sweep_geometry": door_record,
        "table_edge_margin_m": SAFE_PARK_TABLE_EDGE_MARGIN_M,
        "door_sweep_margin_m": SAFE_PARK_DOOR_SWEEP_MARGIN_M,
        "static_microwave_margin_m": SAFE_PARK_STATIC_MARGIN_M,
        "search_min_outward_m": SAFE_PARK_MIN_OUTWARD_DISTANCE_M,
        "search_max_outward_m": SAFE_PARK_MAX_OUTWARD_DISTANCE_M,
        "search_step_m": SAFE_PARK_SEARCH_STEP_M,
        "candidate_trace": trace,
        "reachability_gate": (
            "bounded outward corridor plus real OSC env.step waypoint"
        ),
    }


def _compiled_geom_support_radius(
    model,
    geom_id,
    rotation,
    direction,
):
    """Return the compiled geom's support radius along a world direction."""
    geom_id = int(geom_id)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    local_direction = rotation.T @ direction
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == 2:
        radius = float(size[0])
        method = "compiled sphere support"
    elif geom_type == 3:
        radius = float(
            size[0] + size[1] * abs(float(local_direction[2]))
        )
        method = "compiled capsule support"
    elif geom_type == 4:
        radius = float(np.linalg.norm(size * local_direction))
        method = "compiled ellipsoid support"
    elif geom_type == 5:
        axial = abs(float(local_direction[2]))
        radial = float(
            np.sqrt(max(0.0, 1.0 - axial * axial))
        )
        radius = float(size[1] * axial + size[0] * radial)
        method = "compiled cylinder support"
    elif geom_type == 6:
        radius = float(np.sum(size * np.abs(local_direction)))
        method = "compiled box support"
    else:
        radius = float(model.geom_rbound[geom_id])
        method = "conservative compiled bounding-sphere support"
    return radius, method


def _compiled_target_support_geometry(env, desired_up):
    """Measure the target body's native table-support offset from contacts."""
    model = env.sim.model
    desired_up = np.asarray(desired_up, dtype=float)
    desired_up = desired_up / np.linalg.norm(desired_up)
    target_geoms = descendant_geom_ids(model, TARGET_BODY)
    target_position, _ = body_pose(env.sim, TARGET_BODY)
    candidates = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in target_geoms:
            target_geom = int(contact.geom1)
            support_geom = int(contact.geom2)
        elif contact.geom2 in target_geoms:
            target_geom = int(contact.geom2)
            support_geom = int(contact.geom1)
        else:
            continue
        support_body = model.body_id2name(
            int(model.geom_bodyid[support_geom])
        ) or ""
        if (
            support_body.startswith(("robot0_", "gripper0_"))
            or "microwave" in support_body.lower()
            or support_geom in target_geoms
            or int(model.geom_type[support_geom]) != 6
        ):
            continue
        rotation = np.asarray(
            env.sim.data.geom_xmat[support_geom], dtype=float
        ).reshape(3, 3)
        half_size = np.asarray(
            model.geom_size[support_geom], dtype=float
        )
        alignment = rotation.T @ desired_up
        normal_axis = int(np.argmax(np.abs(alignment)))
        normal = rotation[:, normal_axis] * (
            1.0 if alignment[normal_axis] >= 0.0 else -1.0
        )
        normal_tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(normal, desired_up)), -1.0, 1.0)
                )
            )
        )
        surface = (
            np.asarray(
                env.sim.data.geom_xpos[support_geom], dtype=float
            )
            + normal * half_size[normal_axis]
        )
        support_offset = float(
            np.dot(target_position - surface, normal)
        )
        tangent_axes = [
            axis for axis in range(3) if axis != normal_axis
        ]
        candidates.append(
            {
                "contact_index": int(index),
                "target_geom_id": target_geom,
                "target_geom_name": _geom_name(model, target_geom),
                "support_geom_id": support_geom,
                "support_geom_name": _geom_name(model, support_geom),
                "support_body": str(support_body),
                "support_surface_position": surface.tolist(),
                "support_normal": normal.tolist(),
                "normal_axis": normal_axis,
                "normal_tilt_deg": normal_tilt_deg,
                "support_offset_m": support_offset,
                "support_planar_area_m2": float(
                    4.0
                    * half_size[tangent_axes[0]]
                    * half_size[tangent_axes[1]]
                ),
            }
        )
    level = [
        candidate
        for candidate in candidates
        if candidate["normal_tilt_deg"] <= MAX_MUG_TILT_DEG
    ]
    if not level:
        raise RuntimeError(
            "target mug has no compiled level non-microwave support contact; "
            f"contacts={candidates}"
        )
    selected = max(
        level,
        key=lambda item: (
            item["support_planar_area_m2"],
            -item["normal_tilt_deg"],
            -item["support_geom_id"],
        ),
    )
    selected_support = int(selected["support_geom_id"])
    contacting_target_geoms = sorted(
        {
            int(candidate["target_geom_id"])
            for candidate in level
            if int(candidate["support_geom_id"]) == selected_support
        }
    )
    selected_normal = np.asarray(
        selected["support_normal"], dtype=float
    )
    bottom_candidates = []
    for geom_id in sorted(target_geoms):
        if not collision_masks_compatible(
            model.geom_contype[geom_id],
            model.geom_conaffinity[geom_id],
            model.geom_contype[selected_support],
            model.geom_conaffinity[selected_support],
        ):
            continue
        center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        rotation = np.asarray(
            env.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        support_radius, support_method = _compiled_geom_support_radius(
            model,
            geom_id,
            rotation,
            selected_normal,
        )
        bottom_candidates.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "geom_type": int(model.geom_type[geom_id]),
                "center_relative_to_target": (
                    center - target_position
                ).tolist(),
                "rotation": rotation.tolist(),
                "support_radius_m": support_radius,
                "support_radius_method": support_method,
                "bottom_projection_m": float(
                    np.dot(center, selected_normal) - support_radius
                ),
            }
        )
    if not bottom_candidates:
        raise RuntimeError(
            "target mug has no collision-compatible compiled bottom geoms"
        )
    minimum_bottom = min(
        item["bottom_projection_m"] for item in bottom_candidates
    )
    compiled_bottom_tolerance = float(
        getattr(model, "geom_margin", np.zeros(int(model.ngeom)))[
            selected_support
        ]
    )
    compiled_bottom_tolerance += max(
        (
            float(
                getattr(
                    model,
                    "geom_margin",
                    np.zeros(int(model.ngeom)),
                )[item["geom_id"]]
            )
            for item in bottom_candidates
        ),
        default=0.0,
    )
    numerical_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(minimum_bottom))
    )
    bottom_tolerance = max(
        compiled_bottom_tolerance, numerical_tolerance
    )
    supporting_target_geoms = sorted(
        {
            *contacting_target_geoms,
            *(
                item["geom_id"]
                for item in bottom_candidates
                if item["bottom_projection_m"]
                <= minimum_bottom + bottom_tolerance
            ),
        }
    )
    return {
        "method": "actual compiled target-to-table support contacts",
        "target_initial_position": target_position.tolist(),
        "desired_up": desired_up.tolist(),
        "selected": selected,
        "contacting_target_geom_ids": contacting_target_geoms,
        "supporting_target_geom_ids": supporting_target_geoms,
        "compiled_bottom_tolerance_m": bottom_tolerance,
        "bottom_geom_candidates": bottom_candidates,
        "candidate_contacts": candidates,
    }


def _compiled_held_target_support_geometry(
    env,
    support_geometry,
    floor_geom_id,
    floor_normal,
):
    """Calibrate floor height for the target's actual held orientation."""
    model = env.sim.model
    floor_geom_id = int(floor_geom_id)
    floor_normal = np.asarray(floor_normal, dtype=float)
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    source_normal = np.asarray(
        support_geometry["selected"]["support_normal"], dtype=float
    )
    normal_alignment = float(np.dot(source_normal, floor_normal))
    normal_mismatch_deg = float(
        np.degrees(
            np.arccos(np.clip(normal_alignment, -1.0, 1.0))
        )
    )
    if normal_mismatch_deg > MAX_MUG_TILT_DEG:
        raise RuntimeError(
            "native table support and compiled microwave floor normals "
            f"differ by {normal_mismatch_deg}deg"
        )

    current_target, current_target_rotation = body_pose(
        env.sim, TARGET_BODY
    )
    initial_candidates = []
    held_candidates = []
    for source in support_geometry["bottom_geom_candidates"]:
        geom_id = int(source["geom_id"])
        if not collision_masks_compatible(
            model.geom_contype[geom_id],
            model.geom_conaffinity[geom_id],
            model.geom_contype[floor_geom_id],
            model.geom_conaffinity[floor_geom_id],
        ):
            continue
        initial_rotation = np.asarray(
            source["rotation"], dtype=float
        ).reshape(3, 3)
        initial_radius, initial_method = (
            _compiled_geom_support_radius(
                model,
                geom_id,
                initial_rotation,
                floor_normal,
            )
        )
        initial_relative_center = np.asarray(
            source["center_relative_to_target"], dtype=float
        )
        initial_relative_bottom = float(
            np.dot(initial_relative_center, floor_normal)
            - initial_radius
        )

        held_center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        held_rotation = np.asarray(
            env.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        held_radius, held_method = _compiled_geom_support_radius(
            model,
            geom_id,
            held_rotation,
            floor_normal,
        )
        held_relative_center = held_center - current_target
        held_relative_bottom = float(
            np.dot(held_relative_center, floor_normal) - held_radius
        )
        initial_candidates.append(
            {
                "geom_id": geom_id,
                "geom_name": _geom_name(model, geom_id),
                "relative_bottom_m": initial_relative_bottom,
                "support_radius_m": initial_radius,
                "support_radius_method": initial_method,
            }
        )
        held_candidates.append(
            {
                "geom_id": geom_id,
                "geom_name": _geom_name(model, geom_id),
                "relative_center": held_relative_center.tolist(),
                "rotation": held_rotation.tolist(),
                "relative_bottom_m": held_relative_bottom,
                "support_radius_m": held_radius,
                "support_radius_method": held_method,
            }
        )
    if not initial_candidates or not held_candidates:
        raise RuntimeError(
            "target has no compiled collision geom compatible with the "
            "microwave floor"
        )
    initial_relative_bottom = min(
        item["relative_bottom_m"] for item in initial_candidates
    )
    held_relative_bottom = min(
        item["relative_bottom_m"] for item in held_candidates
    )
    source_support_offset = float(
        support_geometry["selected"]["support_offset_m"]
    )
    held_support_offset = float(
        source_support_offset
        + initial_relative_bottom
        - held_relative_bottom
    )
    return {
        "method": (
            "actual held-pose compiled geom centers and rotations, "
            "calibrated to native table support contact"
        ),
        "floor_geom_id": floor_geom_id,
        "floor_geom_name": _geom_name(model, floor_geom_id),
        "floor_normal": floor_normal.tolist(),
        "source_support_normal": source_normal.tolist(),
        "support_normal_mismatch_deg": normal_mismatch_deg,
        "source_support_offset_m": source_support_offset,
        "initial_relative_bottom_m": initial_relative_bottom,
        "held_relative_bottom_m": held_relative_bottom,
        "held_support_offset_m": held_support_offset,
        "target_position_at_planning": current_target.tolist(),
        "target_rotation_at_planning": current_target_rotation.tolist(),
        "target_tilt_at_planning_deg": body_tilt_deg(
            env.sim, TARGET_BODY
        ),
        "initial_geom_candidates": initial_candidates,
        "held_geom_candidates": held_candidates,
    }


def _compiled_microwave_floor(
    env,
    names,
    site_position,
    site_rotation,
    target_geom_ids,
):
    """Resolve the level compiled box surface directly below heating site."""
    model = env.sim.model
    site_position = np.asarray(site_position, dtype=float)
    site_up = np.asarray(site_rotation, dtype=float)[:, 2]
    static_geoms = (
        descendant_geom_ids(model, names["fixture_root"])
        - descendant_geom_ids(model, names["door_body"])
    )
    compatible = _collision_compatible_geom_ids(
        model,
        static_geoms,
        target_geom_ids,
    )
    candidates = []
    for geom_id in compatible:
        if int(model.geom_type[geom_id]) != 6:
            continue
        center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        rotation = np.asarray(
            env.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        half_size = np.asarray(model.geom_size[geom_id], dtype=float)
        alignment = rotation.T @ site_up
        normal_axis = int(np.argmax(np.abs(alignment)))
        normal = rotation[:, normal_axis] * (
            1.0 if alignment[normal_axis] >= 0.0 else -1.0
        )
        tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(normal, site_up)), -1.0, 1.0)
                )
            )
        )
        surface = center + normal * half_size[normal_axis]
        below_site_m = float(np.dot(site_position - surface, site_up))
        local_site = rotation.T @ (site_position - center)
        tangent_axes = [
            axis for axis in range(3) if axis != normal_axis
        ]
        covers_site_center = bool(
            all(
                abs(float(local_site[axis]))
                <= float(half_size[axis])
                for axis in tangent_axes
            )
        )
        candidates.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "body_name": str(
                    model.body_id2name(
                        int(model.geom_bodyid[geom_id])
                    )
                    or ""
                ),
                "center": center.tolist(),
                "rotation": rotation.tolist(),
                "half_size": half_size.tolist(),
                "normal_axis": normal_axis,
                "normal": normal.tolist(),
                "surface_position": surface.tolist(),
                "normal_tilt_deg": tilt_deg,
                "below_site_m": below_site_m,
                "covers_site_center": covers_site_center,
                "geom_contype": int(model.geom_contype[geom_id]),
                "geom_conaffinity": int(
                    model.geom_conaffinity[geom_id]
                ),
            }
        )
    floors = [
        candidate
        for candidate in candidates
        if candidate["normal_tilt_deg"] <= MAX_MUG_TILT_DEG
        and candidate["covers_site_center"]
        and candidate["below_site_m"] > 0.0
    ]
    if not floors:
        raise RuntimeError(
            "cannot resolve compiled level microwave floor below native "
            f"heating site; candidates={candidates}"
        )
    selected = min(
        floors,
        key=lambda item: (item["below_site_m"], item["geom_id"]),
    )
    return selected, {
        "method": (
            "nearest compiled target-compatible level box surface below "
            "native heating-site center"
        ),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _signed_point_box_clearance(point, center, rotation, half_size):
    local = np.asarray(rotation, dtype=float).T @ (
        np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    )
    outside = np.maximum(
        np.abs(local) - np.asarray(half_size, dtype=float), 0.0
    )
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance
    return -float(
        np.min(np.asarray(half_size, dtype=float) - np.abs(local))
    )


def _compiled_convex_mesh_geometry(model, geom_id):
    """Decode the exact convex hull used by MuJoCo mesh collision."""
    geom_id = int(geom_id)
    if int(model.geom_type[geom_id]) != 7:
        raise RuntimeError(
            f"geom {_geom_name(model, geom_id)!r} is not a mesh"
        )
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise RuntimeError(
            f"mesh geom {_geom_name(model, geom_id)!r} has no data id"
        )
    graph_address = int(model.mesh_graphadr[mesh_id])
    if graph_address < 0:
        raise RuntimeError(
            f"mesh geom {_geom_name(model, geom_id)!r} has no compiled "
            "MuJoCo convex hull"
        )
    graph = np.asarray(model.mesh_graph, dtype=int).reshape(-1)
    if graph_address + 2 > len(graph):
        raise RuntimeError(
            f"compiled convex hull header for mesh {mesh_id} is truncated"
        )
    hull_vertex_count = int(graph[graph_address])
    hull_face_count = int(graph[graph_address + 1])
    if hull_vertex_count < 4 or hull_face_count < 4:
        raise RuntimeError(
            f"invalid compiled convex hull for {_geom_name(model, geom_id)!r}"
        )
    edge_record_count = hull_vertex_count + 3 * hull_face_count
    face_address = (
        graph_address
        + 2
        + 2 * hull_vertex_count
        + edge_record_count
    )
    graph_end = face_address + 3 * hull_face_count
    if graph_end > len(graph):
        raise RuntimeError(
            f"compiled convex hull record for mesh {mesh_id} is truncated"
        )
    face_ids = np.asarray(
        graph[face_address:graph_end],
        dtype=int,
    ).reshape(hull_face_count, 3)
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    hull_id_address = graph_address + 2 + hull_vertex_count
    hull_global_ids = np.asarray(
        graph[hull_id_address : hull_id_address + hull_vertex_count],
        dtype=int,
    )
    if (
        vertex_address < 0
        or vertex_count < 4
        or vertex_address + vertex_count > len(model.mesh_vert)
        or np.any(hull_global_ids < 0)
        or np.any(hull_global_ids >= vertex_count)
        or len(set(int(index) for index in hull_global_ids))
        != hull_vertex_count
        or np.any(face_ids < 0)
        or np.any(face_ids >= vertex_count)
    ):
        raise RuntimeError(
            f"compiled convex hull vertices are invalid for mesh {mesh_id}"
        )
    full_vertices = np.asarray(
        model.mesh_vert[
            vertex_address : vertex_address + vertex_count
        ],
        dtype=float,
    ).reshape(vertex_count, 3)
    global_to_hull = {
        int(global_id): hull_id
        for hull_id, global_id in enumerate(hull_global_ids)
    }
    if any(
        int(index) not in global_to_hull for index in face_ids.reshape(-1)
    ):
        raise RuntimeError(
            f"compiled convex hull faces reference non-hull vertices "
            f"for mesh {mesh_id}"
        )
    hull_vertices = full_vertices[hull_global_ids]
    hull_faces = np.asarray(
        [
            [global_to_hull[int(index)] for index in face]
            for face in face_ids
        ],
        dtype=int,
    )
    return hull_vertices, hull_faces, {
        "geom_id": geom_id,
        "geom_name": _geom_name(model, geom_id),
        "geom_type": int(model.geom_type[geom_id]),
        "geom_size": np.asarray(
            model.geom_size[geom_id], dtype=float
        ).tolist(),
        "geom_rbound": float(model.geom_rbound[geom_id]),
        "mesh_id": mesh_id,
        "mesh_vertex_count": vertex_count,
        "convex_hull_vertex_count": len(hull_vertices),
        "convex_hull_face_count": len(hull_faces),
        "mesh_graph_address": graph_address,
    }


def _cached_compiled_convex_mesh_geometry(
    model, geom_id, compiled_geometry_cache=None
):
    """Reuse immutable compiled hull decoding within one scene state."""
    if compiled_geometry_cache is None:
        return _compiled_convex_mesh_geometry(model, geom_id)
    mesh_cache = compiled_geometry_cache.setdefault("convex_mesh", {})
    mesh_cache_key = (id(model), int(geom_id))
    if mesh_cache_key in mesh_cache:
        compiled_geometry_cache["hits"] = int(
            compiled_geometry_cache.get("hits", 0)
        ) + 1
        return mesh_cache[mesh_cache_key]
    decoded_mesh = _compiled_convex_mesh_geometry(model, geom_id)
    mesh_cache[mesh_cache_key] = decoded_mesh
    compiled_geometry_cache["misses"] = int(
        compiled_geometry_cache.get("misses", 0)
    ) + 1
    return decoded_mesh


def _compiled_geom_evidence(
    model, geom_id, compiled_geometry_cache=None
):
    """Record the exact compiled shape inputs used by clearance."""
    geom_id = int(geom_id)
    evidence = {
        "geom_id": geom_id,
        "geom_name": _geom_name(model, geom_id),
        "geom_type": int(model.geom_type[geom_id]),
        "geom_size": np.asarray(
            model.geom_size[geom_id], dtype=float
        ).tolist(),
        "geom_rbound": float(model.geom_rbound[geom_id]),
        "geom_dataid": int(model.geom_dataid[geom_id]),
        "geom_margin": float(
            getattr(
                model,
                "geom_margin",
                np.zeros(int(model.ngeom)),
            )[geom_id]
        ),
    }
    if evidence["geom_type"] == 7:
        _, _, mesh_evidence = _cached_compiled_convex_mesh_geometry(
            model, geom_id, compiled_geometry_cache
        )
        evidence["convex_mesh"] = mesh_evidence
    return evidence


def _convex_mesh_aabb_threshold_distance(
    vertices, faces, half_size, stop_at_or_below=None
):
    """Exact mesh-box distance, stopping on a sufficient face witness."""
    if stop_at_or_below is None:
        clearance = convex_mesh_aabb_distance(
            vertices, faces, half_size
        )
        return clearance, {
            "threshold_m": None,
            "threshold_witness_seen": False,
            "terminated_early": False,
            "face_evaluations": int(len(faces)),
            "total_faces": int(len(faces)),
        }
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    half = np.asarray(half_size, dtype=float)
    threshold = float(stop_at_or_below)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or half.shape != (3,)
        or len(vertices) < 4
        or len(faces) < 4
        or not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(half))
        or np.any(half <= 0.0)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or not np.isfinite(threshold)
    ):
        raise ValueError("invalid finite convex mesh/AABB threshold inputs")
    if np.any(np.all(np.abs(vertices) <= half, axis=1)):
        return 0.0, {
            "threshold_m": threshold,
            "threshold_witness_seen": 0.0 <= threshold,
            "terminated_early": True,
            "face_evaluations": 0,
            "total_faces": int(len(faces)),
            "witness_kind": "mesh_vertex_inside_box",
        }

    minimum = float("inf")
    face_evaluations = 0
    for face_index, face in enumerate(faces):
        distance = triangle_aabb_distance(
            vertices[face[0]],
            vertices[face[1]],
            vertices[face[2]],
            half,
        )
        face_evaluations += 1
        minimum = min(minimum, distance)
        if distance <= threshold:
            return float(distance), {
                "threshold_m": threshold,
                "threshold_witness_seen": True,
                "terminated_early": face_evaluations < len(faces),
                "face_evaluations": int(face_evaluations),
                "total_faces": int(len(faces)),
                "witness_kind": "triangle_aabb_distance",
                "witness_face_index": int(face_index),
            }

    box_vertices = np.asarray(
        [
            [
                x_sign * half[0],
                y_sign * half[1],
                z_sign * half[2],
            ]
            for x_sign in (-1.0, 1.0)
            for y_sign in (-1.0, 1.0)
            for z_sign in (-1.0, 1.0)
        ],
        dtype=float,
    )
    scale = max(
        1.0,
        float(np.max(np.abs(vertices))),
        float(np.max(half)),
    )
    tolerance = 64.0 * np.finfo(float).eps * scale
    for point in box_vertices:
        inside = True
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0],
            )
            norm = float(np.linalg.norm(normal))
            if norm <= tolerance:
                raise ValueError("convex hull contains a degenerate face")
            signed_vertices = (vertices - triangle[0]) @ normal
            signed_point = float(
                np.dot(point - triangle[0], normal)
            )
            if float(np.max(signed_vertices)) <= tolerance:
                if signed_point > tolerance:
                    inside = False
                    break
            elif float(np.min(signed_vertices)) >= -tolerance:
                if signed_point < -tolerance:
                    inside = False
                    break
            else:
                raise ValueError(
                    "mesh faces do not describe a convex hull"
                )
        if inside:
            return 0.0, {
                "threshold_m": threshold,
                "threshold_witness_seen": 0.0 <= threshold,
                "terminated_early": False,
                "face_evaluations": int(face_evaluations),
                "total_faces": int(len(faces)),
                "witness_kind": "box_vertex_inside_convex_mesh",
            }
    return float(minimum), {
        "threshold_m": threshold,
        "threshold_witness_seen": False,
        "terminated_early": False,
        "face_evaluations": int(face_evaluations),
        "total_faces": int(len(faces)),
    }


def _compiled_geom_pair_clearance(
    env,
    moving_geom,
    fixture_geom,
    translation,
    guard_margin,
    *,
    fixture_center_override=None,
    fixture_rotation_override=None,
    compiled_geometry_cache=None,
    stop_at_or_below=None,
):
    """Conservatively clear a translated geom against native fixture geom."""
    model = env.sim.model
    moving_geom = int(moving_geom)
    fixture_geom = int(fixture_geom)
    translation = np.asarray(translation, dtype=float)
    moving_center = (
        np.asarray(env.sim.data.geom_xpos[moving_geom], dtype=float)
        + translation
    )
    moving_rotation = np.asarray(
        env.sim.data.geom_xmat[moving_geom], dtype=float
    ).reshape(3, 3)
    fixture_center = (
        np.asarray(
            env.sim.data.geom_xpos[fixture_geom], dtype=float
        )
        if fixture_center_override is None
        else np.asarray(fixture_center_override, dtype=float)
    )
    fixture_rotation = (
        np.asarray(
            env.sim.data.geom_xmat[fixture_geom], dtype=float
        ).reshape(3, 3)
        if fixture_rotation_override is None
        else np.asarray(fixture_rotation_override, dtype=float).reshape(
            3, 3
        )
    )
    moving_type = int(model.geom_type[moving_geom])
    fixture_type = int(model.geom_type[fixture_geom])
    native_geom_margin = 0.0
    if hasattr(model, "geom_margin"):
        native_geom_margin = float(
            model.geom_margin[moving_geom]
            + model.geom_margin[fixture_geom]
        )
    compiled_margin = float(guard_margin) + native_geom_margin
    mesh_threshold_evidence = None

    if moving_type == 6 and fixture_type in (3, 5):
        fixture_size = np.asarray(
            model.geom_size[fixture_geom], dtype=float
        )
        fixture_axis = fixture_rotation[:, 2]
        start = fixture_center - fixture_axis * fixture_size[1]
        end = fixture_center + fixture_axis * fixture_size[1]
        start_local = moving_rotation.T @ (start - moving_center)
        end_local = moving_rotation.T @ (end - moving_center)
        clearance = (
            segment_aabb_distance(
                start_local,
                end_local,
                np.asarray(model.geom_size[moving_geom], dtype=float),
            )
            - float(fixture_size[0])
        )
        method = (
            "compiled box-capsule distance"
            if fixture_type == 3
            else "conservative compiled box-cylinder-as-capsule distance"
        )
    elif fixture_type != 6:
        clearance = float(
            np.linalg.norm(moving_center - fixture_center)
            - model.geom_rbound[moving_geom]
            - model.geom_rbound[fixture_geom]
        )
        method = "conservative compiled bounding spheres"
    else:
        fixture_half_size = np.asarray(
            model.geom_size[fixture_geom], dtype=float
        )
        if moving_type == 7:
            sphere_lower_bound = (
                _signed_point_box_clearance(
                    moving_center,
                    fixture_center,
                    fixture_rotation,
                    fixture_half_size,
                )
                - float(model.geom_rbound[moving_geom])
            )
            if (
                sphere_lower_bound > compiled_margin
                and (
                    stop_at_or_below is None
                    or sphere_lower_bound
                    > compiled_margin + float(stop_at_or_below)
                )
            ):
                clearance = sphere_lower_bound
                method = (
                    "certified compiled bounding-sphere-to-box positive "
                    "lower bound"
                )
            else:
                mesh_vertices, mesh_faces, _ = (
                    _cached_compiled_convex_mesh_geometry(
                        model,
                        moving_geom,
                        compiled_geometry_cache,
                    )
                )
                world_vertices = (
                    moving_center
                    + (moving_rotation @ mesh_vertices.T).T
                )
                fixture_local_vertices = (
                    fixture_rotation.T
                    @ (world_vertices - fixture_center).T
                ).T
                primitive_stop_at_or_below = (
                    None
                    if stop_at_or_below is None
                    else compiled_margin + float(stop_at_or_below)
                )
                (
                    clearance,
                    mesh_threshold_evidence,
                ) = _convex_mesh_aabb_threshold_distance(
                    fixture_local_vertices,
                    mesh_faces,
                    fixture_half_size,
                    primitive_stop_at_or_below,
                )
                method = (
                    "exact compiled MuJoCo convex-mesh-to-box distance"
                )
        elif moving_type == 6:
            clearance = oriented_box_separating_clearance(
                moving_center,
                moving_rotation,
                np.asarray(model.geom_size[moving_geom], dtype=float),
                fixture_center,
                fixture_rotation,
                fixture_half_size,
            )
            method = "compiled box-box separating-axis gap"
        elif moving_type in (3, 5):
            moving_size = np.asarray(
                model.geom_size[moving_geom], dtype=float
            )
            axis = moving_rotation[:, 2]
            start = moving_center - axis * moving_size[1]
            end = moving_center + axis * moving_size[1]
            start_local = fixture_rotation.T @ (
                start - fixture_center
            )
            end_local = fixture_rotation.T @ (end - fixture_center)
            clearance = (
                segment_aabb_distance(
                    start_local,
                    end_local,
                    fixture_half_size,
                )
                - float(moving_size[0])
            )
            method = (
                "compiled capsule-box distance"
                if moving_type == 3
                else "conservative compiled cylinder-as-capsule distance"
            )
        else:
            clearance = (
                _signed_point_box_clearance(
                    moving_center,
                    fixture_center,
                    fixture_rotation,
                    fixture_half_size,
                )
                - float(model.geom_rbound[moving_geom])
            )
            method = "conservative compiled bounding-sphere-to-box distance"
    components = {
        "primitive_clearance_m": float(clearance),
        "native_geom_margin_m": native_geom_margin,
        "continuous_guard_m": float(guard_margin),
        "net_clearance_m": float(clearance - compiled_margin),
    }
    if mesh_threshold_evidence is not None:
        components["mesh_threshold_evidence"] = (
            mesh_threshold_evidence
        )
    return (
        float(clearance - compiled_margin),
        method,
        components,
    )


def _compile_exact_obb_sat_batch(
    first_centers,
    first_rotations,
    first_half_sizes,
    second_centers,
    second_rotations,
    second_half_sizes,
):
    """Compile candidate-invariant terms of the exact 15-axis OBB SAT."""
    first_centers = np.asarray(first_centers, dtype=float)
    first_rotations = np.asarray(first_rotations, dtype=float)
    first_half_sizes = np.asarray(first_half_sizes, dtype=float)
    second_centers = np.asarray(second_centers, dtype=float)
    second_rotations = np.asarray(second_rotations, dtype=float)
    second_half_sizes = np.asarray(second_half_sizes, dtype=float)
    row_count = len(first_centers)
    expected_shapes = (
        first_centers.shape == (row_count, 3),
        first_rotations.shape == (row_count, 3, 3),
        first_half_sizes.shape == (row_count, 3),
        second_centers.shape == (row_count, 3),
        second_rotations.shape == (row_count, 3, 3),
        second_half_sizes.shape == (row_count, 3),
    )
    arrays = (
        first_centers,
        first_rotations,
        first_half_sizes,
        second_centers,
        second_rotations,
        second_half_sizes,
    )
    if (
        row_count == 0
        or not all(expected_shapes)
        or not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any(first_half_sizes <= 0.0)
        or np.any(second_half_sizes <= 0.0)
    ):
        raise ValueError(
            "compiled OBB batch must contain finite positive-size rows"
        )

    axes = np.zeros((row_count, 15, 3), dtype=float)
    projected_radii = np.zeros((row_count, 15), dtype=float)
    valid_axes = np.zeros((row_count, 15), dtype=bool)
    axis_epsilon = 32.0 * np.finfo(float).eps
    for row_index in range(row_count):
        first_rotation = first_rotations[row_index]
        second_rotation = second_rotations[row_index]
        raw_axes = [
            first_rotation[:, axis] for axis in range(3)
        ]
        raw_axes.extend(
            second_rotation[:, axis] for axis in range(3)
        )
        raw_axes.extend(
            np.cross(
                first_rotation[:, first_axis],
                second_rotation[:, second_axis],
            )
            for first_axis in range(3)
            for second_axis in range(3)
        )
        for axis_index, raw_axis in enumerate(raw_axes):
            norm = float(np.linalg.norm(raw_axis))
            if norm <= axis_epsilon:
                continue
            axis = raw_axis / norm
            axes[row_index, axis_index] = axis
            valid_axes[row_index, axis_index] = True
            projected_radii[row_index, axis_index] = float(
                np.sum(
                    first_half_sizes[row_index]
                    * np.abs(first_rotation.T @ axis)
                )
                + np.sum(
                    second_half_sizes[row_index]
                    * np.abs(second_rotation.T @ axis)
                )
            )
        if not np.any(valid_axes[row_index]):
            raise ValueError("oriented boxes produced no separating axes")
    return {
        "method": "compiled vectorized exact 15-axis OBB SAT",
        "first_centers": first_centers.copy(),
        "second_centers": second_centers.copy(),
        "axes": axes,
        "projected_radii": projected_radii,
        "valid_axes": valid_axes,
        "row_count": int(row_count),
    }


def _evaluate_compiled_exact_obb_sat_batch(
    compiled_batch, translations, row_indices=None
):
    """Evaluate the exact compiled SAT for one or more rigid translations."""
    translations = np.asarray(translations, dtype=float)
    if translations.shape == (3,):
        translations = translations.reshape(1, 3)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("compiled OBB translations must have shape (N, 3)")
    if not np.all(np.isfinite(translations)):
        raise ValueError("compiled OBB translations must be finite")
    if row_indices is None:
        row_indices = np.arange(
            int(compiled_batch["row_count"]), dtype=int
        )
    else:
        row_indices = np.asarray(row_indices, dtype=int).reshape(-1)
    if (
        len(row_indices) == 0
        or np.any(row_indices < 0)
        or np.any(row_indices >= int(compiled_batch["row_count"]))
    ):
        raise ValueError("compiled OBB row indices are out of range")
    moving_centers = (
        compiled_batch["first_centers"][row_indices][None, :, :]
        + translations[:, None, :]
    )
    delta = (
        compiled_batch["second_centers"][row_indices][None, :, :]
        - moving_centers
    )
    gaps = np.abs(
        np.einsum(
            "paj,spj->spa",
            compiled_batch["axes"][row_indices],
            delta,
            optimize=True,
        )
    ) - compiled_batch["projected_radii"][row_indices][None, :, :]
    gaps = np.where(
        compiled_batch["valid_axes"][row_indices][None, :, :],
        gaps,
        -np.inf,
    )
    return np.max(gaps, axis=2)


def _compiled_obb_needs_scalar_threshold_refinement(
    clearance, threshold, primitive_clearance, guard_margin, native_margin
):
    """Protect exact threshold signs from vector reduction roundoff."""
    if threshold is None:
        return False
    scale = max(
        1.0,
        abs(float(clearance)),
        abs(float(threshold)),
        abs(float(primitive_clearance)),
        abs(float(guard_margin)),
        abs(float(native_margin)),
    )
    tolerance = 256.0 * np.finfo(float).eps * scale
    return abs(float(clearance) - float(threshold)) <= tolerance


def _compile_target_door_sweep_geometry(env, names, target_geoms):
    """Compile exact door poses and OBB SAT terms shared by all candidates."""
    model = env.sim.model
    target_geoms = tuple(int(value) for value in target_geoms)
    door_geoms = _collision_compatible_geom_ids(
        model,
        descendant_geom_ids(model, names["door_body"]),
        target_geoms,
    )
    collision_target_geoms = _collision_compatible_geom_ids(
        model,
        target_geoms,
        door_geoms,
    )
    if not door_geoms or not collision_target_geoms:
        raise RuntimeError(
            "compiled target/door sweep has no compatible collision geoms"
        )
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_position, hinge_rotation = body_pose(
        env.sim, names["door_body"]
    )
    hinge_axis = hinge_rotation @ np.asarray(
        model.jnt_axis[joint_id], dtype=float
    )
    hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
    fractions = np.linspace(
        0.0, 1.0, SAFE_PARK_DOOR_SWEEP_SAMPLES
    )
    angular_spacing = abs(float(close_angle)) / max(
        len(fractions) - 1, 1
    )

    geom_margins = np.asarray(
        getattr(model, "geom_margin", np.zeros(int(model.ngeom))),
        dtype=float,
    )
    entries = []
    entry_lookup = {}
    obb_entry_indices = []
    obb_first_centers = []
    obb_first_rotations = []
    obb_first_half_sizes = []
    obb_second_centers = []
    obb_second_rotations = []
    obb_second_half_sizes = []
    for door_geom in door_geoms:
        door_geom = int(door_geom)
        initial_center = np.asarray(
            env.sim.data.geom_xpos[door_geom], dtype=float
        )
        initial_rotation = np.asarray(
            env.sim.data.geom_xmat[door_geom], dtype=float
        ).reshape(3, 3)
        radial = initial_center - hinge_position
        axial = hinge_axis * float(np.dot(radial, hinge_axis))
        swept_radius = float(
            np.linalg.norm(radial - axial)
            + model.geom_rbound[door_geom]
        )
        continuous_guard = 0.5 * swept_radius * angular_spacing
        for sample_index, fraction in enumerate(fractions):
            angle = close_angle * float(fraction)
            center = hinge_position + _rotation_about_axis(
                radial, hinge_axis, angle
            )
            rotation = np.column_stack(
                [
                    _rotation_about_axis(
                        initial_rotation[:, axis], hinge_axis, angle
                    )
                    for axis in range(3)
                ]
            )
            for target_geom in collision_target_geoms:
                target_geom = int(target_geom)
                if not collision_masks_compatible(
                    model.geom_contype[target_geom],
                    model.geom_conaffinity[target_geom],
                    model.geom_contype[door_geom],
                    model.geom_conaffinity[door_geom],
                ):
                    continue
                native_geom_margin = float(
                    geom_margins[target_geom]
                    + geom_margins[door_geom]
                )
                entry = {
                    "target_geom_id": target_geom,
                    "door_geom_id": door_geom,
                    "sample_index": int(sample_index),
                    "sample_fraction": float(fraction),
                    "door_angle_rad": float(angle),
                    "continuous_guard_m": float(continuous_guard),
                    "fixture_center": np.asarray(center, dtype=float),
                    "fixture_rotation": np.asarray(rotation, dtype=float),
                    "native_geom_margin_m": native_geom_margin,
                    "obb_batch_row": None,
                }
                entry_index = len(entries)
                entry_lookup[
                    (target_geom, door_geom, int(sample_index))
                ] = entry_index
                if (
                    int(model.geom_type[target_geom]) == 6
                    and int(model.geom_type[door_geom]) == 6
                ):
                    entry["obb_batch_row"] = len(obb_entry_indices)
                    obb_entry_indices.append(entry_index)
                    obb_first_centers.append(
                        np.asarray(
                            env.sim.data.geom_xpos[target_geom],
                            dtype=float,
                        )
                    )
                    obb_first_rotations.append(
                        np.asarray(
                            env.sim.data.geom_xmat[target_geom],
                            dtype=float,
                        ).reshape(3, 3)
                    )
                    obb_first_half_sizes.append(
                        np.asarray(
                            model.geom_size[target_geom], dtype=float
                        )
                    )
                    obb_second_centers.append(center)
                    obb_second_rotations.append(rotation)
                    obb_second_half_sizes.append(
                        np.asarray(model.geom_size[door_geom], dtype=float)
                    )
                entries.append(entry)
    if not entries:
        raise RuntimeError(
            "compiled target/door sweep produced no compatible pairs"
        )
    obb_batch = None
    if obb_entry_indices:
        obb_batch = _compile_exact_obb_sat_batch(
            obb_first_centers,
            obb_first_rotations,
            obb_first_half_sizes,
            obb_second_centers,
            obb_second_rotations,
            obb_second_half_sizes,
        )
    geom_ids = tuple(
        sorted(
            {
                int(entry["target_geom_id"])
                for entry in entries
            }
            | {
                int(entry["door_geom_id"])
                for entry in entries
            }
        )
    )
    return {
        "model_identity": id(model),
        "target_geoms": target_geoms,
        "door_start_qpos": start_qpos,
        "door_qpos_address": qadr,
        "door_closed_qpos": closed_qpos,
        "door_close_angle_rad": close_angle,
        "hinge_position": np.asarray(hinge_position, dtype=float),
        "hinge_axis": np.asarray(hinge_axis, dtype=float),
        "fractions": fractions,
        "entries": entries,
        "entry_lookup": entry_lookup,
        "obb_entry_indices": tuple(obb_entry_indices),
        "obb_batch": obb_batch,
        "door_pose_count": int(len(door_geoms) * len(fractions)),
        "geom_ids": geom_ids,
        "geom_xpos": np.asarray(
            env.sim.data.geom_xpos[list(geom_ids)], dtype=float
        ).copy(),
        "geom_xmat": np.asarray(
            env.sim.data.geom_xmat[list(geom_ids)], dtype=float
        ).copy(),
    }


def _compiled_target_door_sweep_clearance_from_cache(
    env,
    target_geoms,
    candidate_target_position,
    current_target_position,
    compiled_door_sweep,
    *,
    stop_at_or_below=None,
    cached_rejection_witness=None,
):
    """Evaluate a precompiled door sweep with exact ordered decisions."""
    model = env.sim.model
    if (
        compiled_door_sweep.get("model_identity") != id(model)
        or tuple(int(value) for value in target_geoms)
        != compiled_door_sweep.get("target_geoms")
    ):
        raise RuntimeError("compiled door sweep does not match this scene")
    geom_ids = compiled_door_sweep["geom_ids"]
    if (
        float(
            env.sim.data.qpos[
                compiled_door_sweep["door_qpos_address"]
            ]
        )
        != compiled_door_sweep["door_start_qpos"]
        or not np.array_equal(
            np.asarray(
                env.sim.data.geom_xpos[list(geom_ids)], dtype=float
            ),
            compiled_door_sweep["geom_xpos"],
        )
        or not np.array_equal(
            np.asarray(
                env.sim.data.geom_xmat[list(geom_ids)], dtype=float
            ),
            compiled_door_sweep["geom_xmat"],
        )
    ):
        raise RuntimeError(
            "compiled door sweep geometry changed after compilation"
        )
    entries = compiled_door_sweep["entries"]
    target_translation = (
        np.asarray(candidate_target_position, dtype=float)
        - np.asarray(current_target_position, dtype=float)
    )
    minimum = float("inf")
    limiting = None
    inspected_evaluations = 0
    exact_clearances_computed = 0
    threshold_rejection_seen = False
    cached_witness_attempted = False
    cached_witness_rejected = False
    cached_witness_clearance = None
    scalar_boundary_refinement_count = 0

    def evaluate_entry(entry):
        obb_row = entry["obb_batch_row"]
        if obb_row is not None:
            primitive = float(
                _evaluate_compiled_exact_obb_sat_batch(
                    compiled_door_sweep["obb_batch"],
                    target_translation,
                    row_indices=[obb_row],
                )[0, 0]
            )
            clearance = float(
                primitive
                - entry["continuous_guard_m"]
                - entry["native_geom_margin_m"]
            )
            return clearance, (
                "compiled box-box separating-axis gap"
            ), {
                "primitive_clearance_m": primitive,
                "native_geom_margin_m": entry[
                    "native_geom_margin_m"
                ],
                "continuous_guard_m": entry[
                    "continuous_guard_m"
                ],
                "net_clearance_m": clearance,
            }
        return _compiled_geom_pair_clearance(
            env,
            entry["target_geom_id"],
            entry["door_geom_id"],
            target_translation,
            entry["continuous_guard_m"],
            fixture_center_override=entry["fixture_center"],
            fixture_rotation_override=entry["fixture_rotation"],
        )

    def limiting_record(entry, clearance, method, components):
        return {
            "sample_index": entry["sample_index"],
            "sample_fraction": entry["sample_fraction"],
            "door_angle_rad": entry["door_angle_rad"],
            "target_geom_id": entry["target_geom_id"],
            "target_geom_name": _geom_name(
                model, entry["target_geom_id"]
            ),
            "door_geom_id": entry["door_geom_id"],
            "door_geom_name": _geom_name(
                model, entry["door_geom_id"]
            ),
            "clearance_m": float(clearance),
            "continuous_guard_m": entry["continuous_guard_m"],
            "method": method,
            "clearance_components": components,
        }

    if (
        stop_at_or_below is not None
        and isinstance(cached_rejection_witness, dict)
    ):
        witness_key = (
            int(cached_rejection_witness.get("target_geom_id", -1)),
            int(cached_rejection_witness.get("door_geom_id", -1)),
            int(cached_rejection_witness.get("sample_index", -1)),
        )
        entry_index = compiled_door_sweep["entry_lookup"].get(
            witness_key
        )
        if entry_index is not None:
            cached_witness_attempted = True
            entry = entries[entry_index]
            clearance, method, components = evaluate_entry(entry)
            exact_clearances_computed += 1
            if (
                entry["obb_batch_row"] is not None
                and _compiled_obb_needs_scalar_threshold_refinement(
                    clearance,
                    stop_at_or_below,
                    components["primitive_clearance_m"],
                    entry["continuous_guard_m"],
                    entry["native_geom_margin_m"],
                )
            ):
                clearance, method, components = (
                    _compiled_geom_pair_clearance(
                        env,
                        entry["target_geom_id"],
                        entry["door_geom_id"],
                        target_translation,
                        entry["continuous_guard_m"],
                        fixture_center_override=entry[
                            "fixture_center"
                        ],
                        fixture_rotation_override=entry[
                            "fixture_rotation"
                        ],
                    )
                )
                exact_clearances_computed += 1
                scalar_boundary_refinement_count += 1
            cached_witness_clearance = float(clearance)
            if clearance <= float(stop_at_or_below):
                cached_witness_rejected = True
                threshold_rejection_seen = True
                inspected_evaluations = 1
                minimum = float(clearance)
                limiting = limiting_record(
                    entry, clearance, method, components
                )

    batched_obb_clearances = None
    if not threshold_rejection_seen:
        obb_entry_indices = compiled_door_sweep[
            "obb_entry_indices"
        ]
        if obb_entry_indices:
            primitive_clearances = (
                _evaluate_compiled_exact_obb_sat_batch(
                    compiled_door_sweep["obb_batch"],
                    target_translation,
                )[0]
            )
            batched_obb_clearances = np.asarray(
                primitive_clearances, dtype=float
            )
            exact_clearances_computed += len(obb_entry_indices)
        for entry in entries:
            inspected_evaluations += 1
            obb_row = entry["obb_batch_row"]
            if obb_row is None:
                clearance, method, components = evaluate_entry(entry)
                exact_clearances_computed += 1
            else:
                primitive = float(batched_obb_clearances[obb_row])
                clearance = float(
                    primitive
                    - entry["continuous_guard_m"]
                    - entry["native_geom_margin_m"]
                )
                method = "compiled box-box separating-axis gap"
                components = {
                    "primitive_clearance_m": primitive,
                    "native_geom_margin_m": entry[
                        "native_geom_margin_m"
                    ],
                    "continuous_guard_m": entry[
                        "continuous_guard_m"
                    ],
                    "net_clearance_m": clearance,
                }
                if _compiled_obb_needs_scalar_threshold_refinement(
                    clearance,
                    stop_at_or_below,
                    primitive,
                    entry["continuous_guard_m"],
                    entry["native_geom_margin_m"],
                ):
                    clearance, method, components = (
                        _compiled_geom_pair_clearance(
                            env,
                            entry["target_geom_id"],
                            entry["door_geom_id"],
                            target_translation,
                            entry["continuous_guard_m"],
                            fixture_center_override=entry[
                                "fixture_center"
                            ],
                            fixture_rotation_override=entry[
                                "fixture_rotation"
                            ],
                        )
                    )
                    exact_clearances_computed += 1
                    scalar_boundary_refinement_count += 1
            if clearance < minimum:
                minimum = float(clearance)
                limiting = limiting_record(
                    entry, clearance, method, components
                )
            if (
                stop_at_or_below is not None
                and minimum <= float(stop_at_or_below)
            ):
                threshold_rejection_seen = True
                break
    if inspected_evaluations == 0 or limiting is None:
        raise RuntimeError(
            "compiled target/door sweep produced no compatible evaluations"
        )
    limiting["target_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["target_geom_id"]
    )
    limiting["door_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["door_geom_id"]
    )
    total_pair_evaluations = len(entries)
    full_sweep_evaluated = (
        inspected_evaluations == total_pair_evaluations
    )
    terminated_early = bool(
        threshold_rejection_seen and not full_sweep_evaluated
    )
    return minimum, {
        "door_start_qpos": compiled_door_sweep["door_start_qpos"],
        "door_closed_qpos": compiled_door_sweep["door_closed_qpos"],
        "door_close_angle_rad": compiled_door_sweep[
            "door_close_angle_rad"
        ],
        "hinge_position": compiled_door_sweep[
            "hinge_position"
        ].tolist(),
        "hinge_axis": compiled_door_sweep["hinge_axis"].tolist(),
        "samples": len(compiled_door_sweep["fractions"]),
        "compatible_pair_evaluations": inspected_evaluations,
        "total_pair_evaluations_without_fail_fast": (
            total_pair_evaluations
        ),
        "exact_pair_clearances_computed": exact_clearances_computed,
        "scalar_threshold_boundary_refinement_count": (
            scalar_boundary_refinement_count
        ),
        "candidate_invariant_door_pose_count": compiled_door_sweep[
            "door_pose_count"
        ],
        "vectorized_exact_obb_pair_count": len(
            compiled_door_sweep["obb_entry_indices"]
        ),
        "threshold_fail_fast_m": (
            None
            if stop_at_or_below is None
            else float(stop_at_or_below)
        ),
        "threshold_rejection_seen": threshold_rejection_seen,
        "full_sweep_evaluated": full_sweep_evaluated,
        "terminated_early": terminated_early,
        "cached_rejection_witness_attempted": (
            cached_witness_attempted
        ),
        "cached_rejection_witness_rejected": cached_witness_rejected,
        "cached_rejection_witness_clearance_m": (
            cached_witness_clearance
        ),
        "cached_rejection_witness_fell_back_to_full_sweep": bool(
            cached_witness_attempted and not cached_witness_rejected
        ),
        "collision_filter": (
            "MuJoCo bidirectional contype/conaffinity compatibility; "
            "native visual-only 0/0 geoms are excluded"
        ),
        "minimum_clearance_m": minimum,
        "limiting_pair": limiting,
    }


def _compiled_target_door_sweep_clearance(
    env,
    names,
    target_geoms,
    candidate_target_position,
    current_target_position,
    *,
    stop_at_or_below=None,
    cached_rejection_witness=None,
    compiled_door_sweep=None,
):
    """Check the door arc, with an exact threshold-equivalent fail-fast."""
    if compiled_door_sweep is not None:
        return _compiled_target_door_sweep_clearance_from_cache(
            env,
            target_geoms,
            candidate_target_position,
            current_target_position,
            compiled_door_sweep,
            stop_at_or_below=stop_at_or_below,
            cached_rejection_witness=cached_rejection_witness,
        )
    model = env.sim.model
    door_geoms = _collision_compatible_geom_ids(
        model,
        descendant_geom_ids(model, names["door_body"]),
        target_geoms,
    )
    collision_target_geoms = _collision_compatible_geom_ids(
        model,
        target_geoms,
        door_geoms,
    )
    if not door_geoms or not collision_target_geoms:
        raise RuntimeError(
            "compiled target/door sweep has no compatible collision geoms"
        )
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_position, hinge_rotation = body_pose(
        env.sim, names["door_body"]
    )
    hinge_axis = hinge_rotation @ np.asarray(
        model.jnt_axis[joint_id], dtype=float
    )
    hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
    fractions = np.linspace(
        0.0, 1.0, SAFE_PARK_DOOR_SWEEP_SAMPLES
    )
    angular_spacing = abs(float(close_angle)) / max(
        len(fractions) - 1, 1
    )
    target_translation = (
        np.asarray(candidate_target_position, dtype=float)
        - np.asarray(current_target_position, dtype=float)
    )
    minimum = float("inf")
    limiting = None
    evaluations = 0
    compatible_geom_pairs = [
        (target_geom, door_geom)
        for door_geom in door_geoms
        for target_geom in collision_target_geoms
        if collision_masks_compatible(
            model.geom_contype[target_geom],
            model.geom_conaffinity[target_geom],
            model.geom_contype[door_geom],
            model.geom_conaffinity[door_geom],
        )
    ]
    total_pair_evaluations = len(fractions) * len(
        compatible_geom_pairs
    )
    threshold_rejection_seen = False
    cached_witness_attempted = False
    cached_witness_rejected = False
    cached_witness_clearance = None

    def limiting_record(
        target_geom,
        door_geom,
        sample_index,
        fraction,
        angle,
        continuous_guard,
        clearance,
        method,
        clearance_components,
    ):
        return {
            "sample_index": int(sample_index),
            "sample_fraction": float(fraction),
            "door_angle_rad": float(angle),
            "target_geom_id": int(target_geom),
            "target_geom_name": _geom_name(model, target_geom),
            "door_geom_id": int(door_geom),
            "door_geom_name": _geom_name(model, door_geom),
            "clearance_m": float(clearance),
            "continuous_guard_m": continuous_guard,
            "method": method,
            "clearance_components": clearance_components,
        }

    if (
        stop_at_or_below is not None
        and isinstance(cached_rejection_witness, dict)
    ):
        cached_target_geom = int(
            cached_rejection_witness.get("target_geom_id", -1)
        )
        cached_door_geom = int(
            cached_rejection_witness.get("door_geom_id", -1)
        )
        cached_sample_index = int(
            cached_rejection_witness.get("sample_index", -1)
        )
        cached_pair = (cached_target_geom, cached_door_geom)
        if (
            cached_pair in compatible_geom_pairs
            and 0 <= cached_sample_index < len(fractions)
        ):
            cached_witness_attempted = True
            fraction = fractions[cached_sample_index]
            angle = close_angle * float(fraction)
            initial_center = np.asarray(
                env.sim.data.geom_xpos[cached_door_geom], dtype=float
            )
            initial_rotation = np.asarray(
                env.sim.data.geom_xmat[cached_door_geom], dtype=float
            ).reshape(3, 3)
            radial = initial_center - hinge_position
            axial = hinge_axis * float(np.dot(radial, hinge_axis))
            swept_radius = float(
                np.linalg.norm(radial - axial)
                + model.geom_rbound[cached_door_geom]
            )
            continuous_guard = (
                0.5 * swept_radius * angular_spacing
            )
            center = hinge_position + _rotation_about_axis(
                radial, hinge_axis, angle
            )
            rotation = np.column_stack(
                [
                    _rotation_about_axis(
                        initial_rotation[:, axis],
                        hinge_axis,
                        angle,
                    )
                    for axis in range(3)
                ]
            )
            (
                clearance,
                method,
                clearance_components,
            ) = _compiled_geom_pair_clearance(
                env,
                cached_target_geom,
                cached_door_geom,
                target_translation,
                continuous_guard,
                fixture_center_override=center,
                fixture_rotation_override=rotation,
            )
            cached_witness_clearance = float(clearance)
            if clearance <= float(stop_at_or_below):
                cached_witness_rejected = True
                threshold_rejection_seen = True
                evaluations = 1
                minimum = float(clearance)
                limiting = limiting_record(
                    cached_target_geom,
                    cached_door_geom,
                    cached_sample_index,
                    fraction,
                    angle,
                    continuous_guard,
                    clearance,
                    method,
                    clearance_components,
                )

    if not threshold_rejection_seen:
        for door_geom in door_geoms:
            initial_center = np.asarray(
                env.sim.data.geom_xpos[door_geom], dtype=float
            )
            initial_rotation = np.asarray(
                env.sim.data.geom_xmat[door_geom], dtype=float
            ).reshape(3, 3)
            radial = initial_center - hinge_position
            axial = hinge_axis * float(np.dot(radial, hinge_axis))
            swept_radius = float(
                np.linalg.norm(radial - axial)
                + model.geom_rbound[door_geom]
            )
            continuous_guard = 0.5 * swept_radius * angular_spacing
            for sample_index, fraction in enumerate(fractions):
                angle = close_angle * float(fraction)
                center = hinge_position + _rotation_about_axis(
                    radial, hinge_axis, angle
                )
                rotation = np.column_stack(
                    [
                        _rotation_about_axis(
                            initial_rotation[:, axis],
                            hinge_axis,
                            angle,
                        )
                        for axis in range(3)
                    ]
                )
                for target_geom in collision_target_geoms:
                    if not collision_masks_compatible(
                        model.geom_contype[target_geom],
                        model.geom_conaffinity[target_geom],
                        model.geom_contype[door_geom],
                        model.geom_conaffinity[door_geom],
                    ):
                        continue
                    evaluations += 1
                    (
                        clearance,
                        method,
                        clearance_components,
                    ) = _compiled_geom_pair_clearance(
                        env,
                        target_geom,
                        door_geom,
                        target_translation,
                        continuous_guard,
                        fixture_center_override=center,
                        fixture_rotation_override=rotation,
                    )
                    if clearance < minimum:
                        minimum = clearance
                        limiting = limiting_record(
                            target_geom,
                            door_geom,
                            sample_index,
                            fraction,
                            angle,
                            continuous_guard,
                            clearance,
                            method,
                            clearance_components,
                        )
                    if (
                        stop_at_or_below is not None
                        and minimum <= float(stop_at_or_below)
                    ):
                        threshold_rejection_seen = True
                        break
                if threshold_rejection_seen:
                    break
            if threshold_rejection_seen:
                break
    if evaluations == 0 or limiting is None:
        raise RuntimeError(
            "compiled target/door sweep produced no compatible evaluations"
        )
    limiting["target_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["target_geom_id"]
    )
    limiting["door_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["door_geom_id"]
    )
    full_sweep_evaluated = evaluations == total_pair_evaluations
    terminated_early = bool(
        threshold_rejection_seen and not full_sweep_evaluated
    )
    return minimum, {
        "door_start_qpos": start_qpos,
        "door_closed_qpos": closed_qpos,
        "door_close_angle_rad": close_angle,
        "hinge_position": hinge_position.tolist(),
        "hinge_axis": hinge_axis.tolist(),
        "samples": len(fractions),
        "compatible_pair_evaluations": evaluations,
        "total_pair_evaluations_without_fail_fast": (
            total_pair_evaluations
        ),
        "threshold_fail_fast_m": (
            None
            if stop_at_or_below is None
            else float(stop_at_or_below)
        ),
        "threshold_rejection_seen": threshold_rejection_seen,
        "full_sweep_evaluated": full_sweep_evaluated,
        "terminated_early": terminated_early,
        "cached_rejection_witness_attempted": (
            cached_witness_attempted
        ),
        "cached_rejection_witness_rejected": cached_witness_rejected,
        "cached_rejection_witness_clearance_m": (
            cached_witness_clearance
        ),
        "cached_rejection_witness_fell_back_to_full_sweep": bool(
            cached_witness_attempted and not cached_witness_rejected
        ),
        "collision_filter": (
            "MuJoCo bidirectional contype/conaffinity compatibility; "
            "native visual-only 0/0 geoms are excluded"
        ),
        "minimum_clearance_m": minimum,
        "limiting_pair": limiting,
    }


def _compile_translated_sweep_geometry(
    env,
    moving_geoms,
    fixture_geoms,
    compiled_geometry_cache=None,
):
    """Compile static exact OBB and mesh-box translated-sweep terms."""
    model = env.sim.model
    moving_geoms = tuple(int(value) for value in moving_geoms)
    fixture_geoms = tuple(int(value) for value in fixture_geoms)
    compatible_geom_pairs = [
        (moving_geom, fixture_geom)
        for moving_geom in moving_geoms
        for fixture_geom in fixture_geoms
        if collision_masks_compatible(
            model.geom_contype[moving_geom],
            model.geom_conaffinity[moving_geom],
            model.geom_contype[fixture_geom],
            model.geom_conaffinity[fixture_geom],
        )
    ]
    if not compatible_geom_pairs:
        raise RuntimeError(
            "compiled insertion sweep has no collision-compatible geom pairs"
        )
    geom_ids = tuple(
        sorted(
            {
                geom_id
                for pair in compatible_geom_pairs
                for geom_id in pair
            }
        )
    )
    obb_pair_indices = []
    first_centers = []
    first_rotations = []
    first_half_sizes = []
    second_centers = []
    second_rotations = []
    second_half_sizes = []
    mesh_box_pair_geometry = {}
    geom_margins = np.asarray(
        getattr(model, "geom_margin", np.zeros(int(model.ngeom))),
        dtype=float,
    )
    for pair_index, (moving_geom, fixture_geom) in enumerate(
        compatible_geom_pairs
    ):
        moving_type = int(model.geom_type[moving_geom])
        fixture_type = int(model.geom_type[fixture_geom])
        if moving_type == 7 and fixture_type == 6:
            mesh_vertices, mesh_faces, _ = (
                _cached_compiled_convex_mesh_geometry(
                    model,
                    moving_geom,
                    compiled_geometry_cache,
                )
            )
            moving_rotation = np.asarray(
                env.sim.data.geom_xmat[moving_geom], dtype=float
            ).reshape(3, 3)
            fixture_rotation = np.asarray(
                env.sim.data.geom_xmat[fixture_geom], dtype=float
            ).reshape(3, 3)
            world_vertex_offsets = (
                moving_rotation @ mesh_vertices.T
            ).T
            fixture_local_vertex_offsets = (
                fixture_rotation.T @ world_vertex_offsets.T
            ).T
            mesh_box_pair_geometry[pair_index] = {
                "moving_geom_id": int(moving_geom),
                "fixture_geom_id": int(fixture_geom),
                "moving_center": np.asarray(
                    env.sim.data.geom_xpos[moving_geom], dtype=float
                ).copy(),
                "moving_rbound_m": float(
                    model.geom_rbound[moving_geom]
                ),
                "fixture_center": np.asarray(
                    env.sim.data.geom_xpos[fixture_geom], dtype=float
                ).copy(),
                "fixture_rotation": fixture_rotation.copy(),
                "fixture_half_size": np.asarray(
                    model.geom_size[fixture_geom], dtype=float
                ).copy(),
                "fixture_local_vertex_offsets": (
                    fixture_local_vertex_offsets
                ),
                "mesh_faces": np.asarray(mesh_faces, dtype=int).copy(),
                "native_geom_margin_m": float(
                    geom_margins[moving_geom]
                    + geom_margins[fixture_geom]
                ),
            }
        if (
            moving_type != 6
            or fixture_type != 6
        ):
            continue
        obb_pair_indices.append(pair_index)
        first_centers.append(
            np.asarray(
                env.sim.data.geom_xpos[moving_geom], dtype=float
            )
        )
        first_rotations.append(
            np.asarray(
                env.sim.data.geom_xmat[moving_geom], dtype=float
            ).reshape(3, 3)
        )
        first_half_sizes.append(
            np.asarray(model.geom_size[moving_geom], dtype=float)
        )
        second_centers.append(
            np.asarray(
                env.sim.data.geom_xpos[fixture_geom], dtype=float
            )
        )
        second_rotations.append(
            np.asarray(
                env.sim.data.geom_xmat[fixture_geom], dtype=float
            ).reshape(3, 3)
        )
        second_half_sizes.append(
            np.asarray(model.geom_size[fixture_geom], dtype=float)
        )
    obb_batch = None
    if obb_pair_indices:
        obb_batch = _compile_exact_obb_sat_batch(
            first_centers,
            first_rotations,
            first_half_sizes,
            second_centers,
            second_rotations,
            second_half_sizes,
        )
    pair_to_obb_row = {
        pair_index: row_index
        for row_index, pair_index in enumerate(obb_pair_indices)
    }
    return {
        "model_identity": id(model),
        "moving_geoms": moving_geoms,
        "fixture_geoms": fixture_geoms,
        "compatible_geom_pairs": tuple(compatible_geom_pairs),
        "geom_ids": geom_ids,
        "geom_xpos": np.asarray(
            env.sim.data.geom_xpos[list(geom_ids)], dtype=float
        ).copy(),
        "geom_xmat": np.asarray(
            env.sim.data.geom_xmat[list(geom_ids)], dtype=float
        ).copy(),
        "obb_pair_indices": tuple(obb_pair_indices),
        "pair_to_obb_row": pair_to_obb_row,
        "obb_batch": obb_batch,
        "mesh_box_pair_geometry": mesh_box_pair_geometry,
    }


def _validate_translated_sweep_geometry(
    env, moving_geoms, fixture_geoms, compiled_sweep_geometry
):
    """Fail closed if supposedly static compiled geometry has changed."""
    model = env.sim.model
    if (
        compiled_sweep_geometry.get("model_identity") != id(model)
        or tuple(int(value) for value in moving_geoms)
        != compiled_sweep_geometry.get("moving_geoms")
        or tuple(int(value) for value in fixture_geoms)
        != compiled_sweep_geometry.get("fixture_geoms")
    ):
        raise RuntimeError(
            "compiled translated sweep does not match this scene"
        )
    geom_ids = compiled_sweep_geometry["geom_ids"]
    if (
        not np.array_equal(
            np.asarray(
                env.sim.data.geom_xpos[list(geom_ids)], dtype=float
            ),
            compiled_sweep_geometry["geom_xpos"],
        )
        or not np.array_equal(
            np.asarray(
                env.sim.data.geom_xmat[list(geom_ids)], dtype=float
            ),
            compiled_sweep_geometry["geom_xmat"],
        )
    ):
        raise RuntimeError(
            "compiled translated sweep geometry changed after compilation"
        )


def _compiled_translated_mesh_box_clearance(
    env,
    pair_geometry,
    translation,
    guard_margin,
    *,
    stop_at_or_below=None,
    compiled_geometry_cache=None,
):
    """Reuse exact static mesh-box terms for one rigid translation."""
    translation = np.asarray(translation, dtype=float)
    moving_center = pair_geometry["moving_center"] + translation
    fixture_center = pair_geometry["fixture_center"]
    fixture_rotation = pair_geometry["fixture_rotation"]
    fixture_half_size = pair_geometry["fixture_half_size"]
    native_geom_margin = pair_geometry["native_geom_margin_m"]
    compiled_margin = float(guard_margin) + native_geom_margin
    sphere_lower_bound = (
        _signed_point_box_clearance(
            moving_center,
            fixture_center,
            fixture_rotation,
            fixture_half_size,
        )
        - pair_geometry["moving_rbound_m"]
    )
    if (
        sphere_lower_bound > compiled_margin
        and (
            stop_at_or_below is None
            or sphere_lower_bound
            > compiled_margin + float(stop_at_or_below)
        )
    ):
        primitive = float(sphere_lower_bound)
        method = (
            "certified compiled bounding-sphere-to-box positive lower bound"
        )
        mesh_threshold_evidence = None
    else:
        fixture_local_center = fixture_rotation.T @ (
            moving_center - fixture_center
        )
        fixture_local_vertices = (
            pair_geometry["fixture_local_vertex_offsets"]
            + fixture_local_center
        )
        primitive_stop_at_or_below = (
            None
            if stop_at_or_below is None
            else compiled_margin + float(stop_at_or_below)
        )
        primitive, mesh_threshold_evidence = (
            _convex_mesh_aabb_threshold_distance(
                fixture_local_vertices,
                pair_geometry["mesh_faces"],
                fixture_half_size,
                primitive_stop_at_or_below,
            )
        )
        method = "exact compiled MuJoCo convex-mesh-to-box distance"
    clearance = float(primitive - compiled_margin)
    components = {
        "primitive_clearance_m": float(primitive),
        "native_geom_margin_m": float(native_geom_margin),
        "continuous_guard_m": float(guard_margin),
        "net_clearance_m": clearance,
        "candidate_invariant_mesh_pair_cache_used": True,
    }
    if mesh_threshold_evidence is not None:
        components["mesh_threshold_evidence"] = (
            mesh_threshold_evidence
        )
    scalar_boundary_refinement = False
    if _compiled_obb_needs_scalar_threshold_refinement(
        clearance,
        stop_at_or_below,
        primitive,
        guard_margin,
        native_geom_margin,
    ):
        (
            clearance,
            method,
            components,
        ) = _compiled_geom_pair_clearance(
            env,
            pair_geometry["moving_geom_id"],
            pair_geometry["fixture_geom_id"],
            translation,
            guard_margin,
            compiled_geometry_cache=compiled_geometry_cache,
        )
        components["candidate_invariant_mesh_pair_cache_used"] = True
        components["scalar_boundary_refinement"] = True
        scalar_boundary_refinement = True
    return clearance, method, components, scalar_boundary_refinement


def _translated_swept_clearance(
    env,
    moving_geoms,
    fixture_geoms,
    start_position,
    end_position,
    reference_position,
    *,
    stop_at_or_below=None,
    compiled_geometry_cache=None,
    compiled_sweep_geometry=None,
    cached_rejection_witness=None,
):
    """Bound a straight translation, with threshold-equivalent fail-fast."""
    model = env.sim.model
    start = np.asarray(start_position, dtype=float)
    end = np.asarray(end_position, dtype=float)
    reference = np.asarray(reference_position, dtype=float)
    distance = float(np.linalg.norm(end - start))
    intervals = max(
        1, int(np.ceil(distance / TARGET_INSERTION_SWEEP_STEP_M))
    )
    spacing = distance / intervals
    sweep_guard = 0.5 * spacing
    fractions = np.linspace(0.0, 1.0, intervals + 1)
    if compiled_sweep_geometry is None:
        compatible_geom_pairs = [
            (moving_geom, fixture_geom)
            for moving_geom in moving_geoms
            for fixture_geom in fixture_geoms
            if collision_masks_compatible(
                model.geom_contype[moving_geom],
                model.geom_conaffinity[moving_geom],
                model.geom_contype[fixture_geom],
                model.geom_conaffinity[fixture_geom],
            )
        ]
    else:
        _validate_translated_sweep_geometry(
            env,
            moving_geoms,
            fixture_geoms,
            compiled_sweep_geometry,
        )
        compatible_geom_pairs = list(
            compiled_sweep_geometry["compatible_geom_pairs"]
        )
    if not compatible_geom_pairs:
        raise RuntimeError(
            "compiled insertion sweep has no collision-compatible geom pairs"
        )
    total_pair_evaluations = len(fractions) * len(compatible_geom_pairs)
    mesh_box_pair_geometry = (
        {}
        if compiled_sweep_geometry is None
        else compiled_sweep_geometry["mesh_box_pair_geometry"]
    )
    minimum = float("inf")
    limiting = None
    compatible_pairs = 0
    threshold_rejection_seen = False
    exact_pair_clearances_computed = 0
    scalar_boundary_refinement_count = 0
    cached_witness_attempted = False
    cached_witness_rejected = False
    cached_witness_clearance = None
    cached_witness_status = "not_provided"

    def limiting_record(
        sample_index,
        fraction,
        translated_position,
        moving_geom,
        fixture_geom,
        clearance,
        method,
        clearance_components,
    ):
        return {
            "sample_index": int(sample_index),
            "sample_fraction": float(fraction),
            "translated_reference_position": (
                translated_position.tolist()
            ),
            "moving_geom_id": int(moving_geom),
            "moving_geom_name": _geom_name(model, moving_geom),
            "moving_body_name": str(
                model.body_id2name(
                    int(model.geom_bodyid[moving_geom])
                )
                or ""
            ),
            "fixture_geom_id": int(fixture_geom),
            "fixture_geom_name": _geom_name(model, fixture_geom),
            "fixture_body_name": str(
                model.body_id2name(
                    int(model.geom_bodyid[fixture_geom])
                )
                or ""
            ),
            "clearance_m": float(clearance),
            "method": method,
            "clearance_components": clearance_components,
        }

    if (
        stop_at_or_below is not None
        and isinstance(cached_rejection_witness, dict)
    ):
        try:
            cached_sample_intervals = int(
                cached_rejection_witness.get("sample_intervals", -1)
            )
        except (TypeError, ValueError):
            cached_sample_intervals = -1
        if cached_sample_intervals != intervals:
            cached_witness_status = "interval_mismatch"
        else:
            try:
                cached_pair = (
                    int(
                        cached_rejection_witness.get(
                            "moving_geom_id", -1
                        )
                    ),
                    int(
                        cached_rejection_witness.get(
                            "fixture_geom_id", -1
                        )
                    ),
                )
            except (TypeError, ValueError):
                cached_pair = (-1, -1)
            if cached_pair not in compatible_geom_pairs:
                cached_witness_status = "pair_missing"
            else:
                try:
                    cached_sample_index = int(
                        cached_rejection_witness.get(
                            "sample_index", -1
                        )
                    )
                except (TypeError, ValueError):
                    cached_sample_index = -1
                if not 0 <= cached_sample_index < len(fractions):
                    cached_witness_status = "sample_invalid"
                else:
                    cached_witness_status = "exact_above_threshold"
        if cached_witness_status == "exact_above_threshold":
            cached_witness_attempted = True
            fraction = fractions[cached_sample_index]
            translated_position = (
                start + (end - start) * float(fraction)
            )
            translation = translated_position - reference
            cached_pair_index = compatible_geom_pairs.index(cached_pair)
            if cached_pair_index in mesh_box_pair_geometry:
                (
                    clearance,
                    method,
                    clearance_components,
                    scalar_refined,
                ) = _compiled_translated_mesh_box_clearance(
                    env,
                    mesh_box_pair_geometry[cached_pair_index],
                    translation,
                    sweep_guard,
                    stop_at_or_below=stop_at_or_below,
                    compiled_geometry_cache=compiled_geometry_cache,
                )
                scalar_boundary_refinement_count += int(
                    scalar_refined
                )
            else:
                (
                    clearance,
                    method,
                    clearance_components,
                ) = _compiled_geom_pair_clearance(
                    env,
                    cached_pair[0],
                    cached_pair[1],
                    translation,
                    sweep_guard,
                    compiled_geometry_cache=compiled_geometry_cache,
                    stop_at_or_below=stop_at_or_below,
                )
            exact_pair_clearances_computed += 1
            cached_witness_clearance = float(clearance)
            if clearance <= float(stop_at_or_below):
                cached_witness_status = "exact_reject"
                cached_witness_rejected = True
                threshold_rejection_seen = True
                compatible_pairs = 1
                minimum = float(clearance)
                limiting = limiting_record(
                    cached_sample_index,
                    fraction,
                    translated_position,
                    cached_pair[0],
                    cached_pair[1],
                    clearance,
                    method,
                    clearance_components,
                )
    batched_obb_clearances = None
    pair_to_obb_row = {}
    if (
        not threshold_rejection_seen
        and
        compiled_sweep_geometry is not None
        and compiled_sweep_geometry["obb_pair_indices"]
    ):
        translations = (
            start[None, :]
            + fractions[:, None] * (end - start)[None, :]
            - reference[None, :]
        )
        batched_obb_clearances = (
            _evaluate_compiled_exact_obb_sat_batch(
                compiled_sweep_geometry["obb_batch"], translations
            )
        )
        pair_to_obb_row = compiled_sweep_geometry["pair_to_obb_row"]
        exact_pair_clearances_computed += int(
            len(fractions)
            * len(compiled_sweep_geometry["obb_pair_indices"])
        )
    geom_margins = None
    if pair_to_obb_row:
        geom_margins = np.asarray(
            getattr(model, "geom_margin", np.zeros(int(model.ngeom))),
            dtype=float,
        )
    fractions_to_evaluate = (
        () if threshold_rejection_seen else fractions
    )
    for sample_index, fraction in enumerate(fractions_to_evaluate):
        translated_position = (
            start + (end - start) * float(fraction)
        )
        translation = translated_position - reference
        for pair_index, (moving_geom, fixture_geom) in enumerate(
            compatible_geom_pairs
        ):
            compatible_pairs += 1
            obb_row = pair_to_obb_row.get(pair_index)
            if pair_index in mesh_box_pair_geometry:
                (
                    clearance,
                    method,
                    clearance_components,
                    scalar_refined,
                ) = _compiled_translated_mesh_box_clearance(
                    env,
                    mesh_box_pair_geometry[pair_index],
                    translation,
                    sweep_guard,
                    stop_at_or_below=stop_at_or_below,
                    compiled_geometry_cache=compiled_geometry_cache,
                )
                scalar_boundary_refinement_count += int(
                    scalar_refined
                )
                exact_pair_clearances_computed += 1
            elif obb_row is None:
                (
                    clearance,
                    method,
                    clearance_components,
                ) = _compiled_geom_pair_clearance(
                    env,
                    moving_geom,
                    fixture_geom,
                    translation,
                    sweep_guard,
                    compiled_geometry_cache=compiled_geometry_cache,
                    stop_at_or_below=stop_at_or_below,
                )
                exact_pair_clearances_computed += 1
            else:
                primitive = float(
                    batched_obb_clearances[sample_index, obb_row]
                )
                native_geom_margin = float(
                    geom_margins[moving_geom]
                    + geom_margins[fixture_geom]
                )
                clearance = float(
                    primitive - sweep_guard - native_geom_margin
                )
                method = "compiled box-box separating-axis gap"
                clearance_components = {
                    "primitive_clearance_m": primitive,
                    "native_geom_margin_m": native_geom_margin,
                    "continuous_guard_m": float(sweep_guard),
                    "net_clearance_m": clearance,
                }
                if _compiled_obb_needs_scalar_threshold_refinement(
                    clearance,
                    stop_at_or_below,
                    primitive,
                    sweep_guard,
                    native_geom_margin,
                ):
                    (
                        clearance,
                        method,
                        clearance_components,
                    ) = _compiled_geom_pair_clearance(
                        env,
                        moving_geom,
                        fixture_geom,
                        translation,
                        sweep_guard,
                        compiled_geometry_cache=(
                            compiled_geometry_cache
                        ),
                    )
                    exact_pair_clearances_computed += 1
                    scalar_boundary_refinement_count += 1
            if clearance < minimum:
                minimum = clearance
                limiting = limiting_record(
                    sample_index,
                    fraction,
                    translated_position,
                    moving_geom,
                    fixture_geom,
                    clearance,
                    method,
                    clearance_components,
                )
            if (
                stop_at_or_below is not None
                and minimum <= float(stop_at_or_below)
            ):
                threshold_rejection_seen = True
                break
        if threshold_rejection_seen:
            break
    if limiting is None:
        raise RuntimeError(
            "compiled insertion sweep has no collision-compatible geom pairs"
        )
    full_sweep_evaluated = compatible_pairs == total_pair_evaluations
    terminated_early = threshold_rejection_seen and not full_sweep_evaluated
    limiting["moving_compiled_geometry"] = _compiled_geom_evidence(
        model,
        limiting["moving_geom_id"],
        compiled_geometry_cache,
    )
    limiting["fixture_compiled_geometry"] = _compiled_geom_evidence(
        model,
        limiting["fixture_geom_id"],
        compiled_geometry_cache,
    )
    return minimum, {
        "path_start": start.tolist(),
        "path_end": end.tolist(),
        "reference_position": reference.tolist(),
        "sample_zero_is_current_pose": bool(
            np.allclose(start, reference, rtol=0.0, atol=1e-12)
        ),
        "sample_zero_semantics": (
            "synthetic translated path start; it equals current mjData "
            "geometry only when path_start equals reference_position"
        ),
        "path_length_m": distance,
        "sample_intervals": intervals,
        "sample_spacing_m": spacing,
        "continuous_sweep_guard_m": sweep_guard,
        "compatible_pair_evaluations": compatible_pairs,
        "total_pair_evaluations_without_fail_fast": total_pair_evaluations,
        "exact_pair_clearances_computed": (
            exact_pair_clearances_computed
        ),
        "scalar_threshold_boundary_refinement_count": (
            scalar_boundary_refinement_count
        ),
        "vectorized_exact_obb_pair_count_per_sample": len(
            pair_to_obb_row
        ),
        "candidate_invariant_mesh_box_pair_count": len(
            mesh_box_pair_geometry
        ),
        "cached_rejection_witness_attempted": (
            cached_witness_attempted
        ),
        "cached_rejection_witness_rejected": cached_witness_rejected,
        "cached_rejection_witness_clearance_m": (
            cached_witness_clearance
        ),
        "cached_rejection_witness_status": cached_witness_status,
        "cached_rejection_witness_fell_back_to_full_sweep": bool(
            cached_witness_status
            not in ("not_provided", "exact_reject")
        ),
        "minimum_clearance_m": minimum,
        "threshold_fail_fast_m": (
            None
            if stop_at_or_below is None
            else float(stop_at_or_below)
        ),
        "terminated_early": terminated_early,
        "threshold_rejection_seen": threshold_rejection_seen,
        "full_sweep_evaluated": full_sweep_evaluated,
        "limiting_pair": limiting,
    }


def _compiled_rigid_gripper_fixture_geoms(env, names):
    """Resolve collision geoms that translate rigidly with the controlled EEF."""
    model = env.sim.model
    eef_root = _eef_body_name(model)
    eef_body_ids = descendant_body_ids(model, eef_root)
    eef_body_ids.update(
        body_id
        for body_id in range(int(model.nbody))
        if (
            (
                model.body_id2name(body_id) or ""
            ).startswith(("robot0_", "gripper0_"))
            and any(
                token
                in (model.body_id2name(body_id) or "").lower()
                for token in ("gripper", "hand", "finger")
            )
        )
    )
    gripper_geoms = sorted(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in eef_body_ids
    )
    fixture_geoms = sorted(
        descendant_geom_ids(model, names["fixture_root"])
    )
    collision_gripper_geoms = _collision_compatible_geom_ids(
        model, gripper_geoms, fixture_geoms
    )
    collision_fixture_geoms = _collision_compatible_geom_ids(
        model, fixture_geoms, collision_gripper_geoms
    )
    if not collision_gripper_geoms or not collision_fixture_geoms:
        raise RuntimeError(
            "compiled target path has no gripper/microwave collision "
            f"geometry; eef_root={eef_root!r}; "
            f"gripper_geoms={gripper_geoms}"
        )
    return (
        collision_gripper_geoms,
        collision_fixture_geoms,
        {
            "eef_root_body": eef_root,
            "rigid_gripper_body_names": sorted(
                str(model.body_id2name(body_id) or "")
                for body_id in eef_body_ids
            ),
            "rigid_gripper_geom_ids": gripper_geoms,
            "collision_gripper_geom_ids": collision_gripper_geoms,
            "collision_fixture_geom_ids": collision_fixture_geoms,
        },
    )


def _native_site_grasp_direction_family(
    env,
    names,
    target_geoms,
    site_position,
    site_rotation,
    site_size,
    existing_direction_records,
):
    """Derive additional planar grasps only from native site/floor axes."""
    site_position = np.asarray(site_position, dtype=float)
    site_rotation = np.asarray(site_rotation, dtype=float)
    site_size = np.asarray(site_size, dtype=float)
    if (
        site_position.shape != (3,)
        or site_rotation.shape != (3, 3)
        or site_size.shape != (3,)
        or not np.all(np.isfinite(site_position))
        or not np.all(np.isfinite(site_rotation))
        or not np.all(np.isfinite(site_size))
        or np.any(site_size <= 0.0)
    ):
        raise RuntimeError(
            "native heating-site frame is invalid for grasp derivation"
        )
    gram = site_rotation.T @ site_rotation
    determinant = float(np.linalg.det(site_rotation))
    if (
        not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-8)
        or not np.isclose(determinant, 1.0, rtol=0.0, atol=1e-8)
    ):
        raise RuntimeError(
            "native heating-site rotation is not a proper orthonormal frame"
        )
    floor, _ = _compiled_microwave_floor(
        env,
        names,
        site_position,
        site_rotation,
        target_geoms,
    )
    floor_normal = np.asarray(floor["normal"], dtype=float)
    floor_normal_norm = float(np.linalg.norm(floor_normal))
    if (
        floor_normal.shape != (3,)
        or not np.all(np.isfinite(floor_normal))
        or floor_normal_norm <= np.finfo(float).eps
    ):
        raise RuntimeError(
            "compiled microwave floor normal is invalid for grasp derivation"
        )
    floor_normal = floor_normal / floor_normal_norm

    # Coefficients are expressed only in the native heating-site frame.
    # Negative local x is named left and negative local y is the site front,
    # matching the front axis used by the insertion planner.
    specifications = (
        ("native_site_left_front", (-1.0, -1.0, 0.0)),
        ("native_site_opposite_lateral", (-1.0, 0.0, 0.0)),
        ("native_site_front", (0.0, -1.0, 0.0)),
    )
    accepted = []
    proposal_evidence = []
    known_directions = [
        (
            f"existing_direction_{index}",
            np.asarray(record["direction_xy"], dtype=float),
        )
        for index, record in enumerate(existing_direction_records)
    ]
    for priority, (label, local_coefficients) in enumerate(
        specifications
    ):
        local_coefficients = np.asarray(
            local_coefficients, dtype=float
        )
        raw_world = site_rotation @ local_coefficients
        floor_tangent_world = raw_world - floor_normal * float(
            np.dot(raw_world, floor_normal)
        )
        planar = floor_tangent_world[:2]
        planar_norm = float(np.linalg.norm(planar))
        if (
            not np.all(np.isfinite(floor_tangent_world))
            or planar_norm <= np.finfo(float).eps
        ):
            raise RuntimeError(
                "native heating-site/floor axis has no finite planar grasp "
                f"projection: label={label!r}"
            )
        direction = planar / planar_norm
        duplicate_of = next(
            (
                known_label
                for known_label, known_direction in known_directions
                if np.allclose(
                    direction,
                    known_direction,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            None,
        )
        provenance = {
            "derivation": (
                "normalize_xy(project_to_compiled_floor_tangent("
                "native_site_rotation @ local_site_coefficients))"
            ),
            "family_label": label,
            "family_priority": int(priority),
            "local_site_coefficients": local_coefficients.tolist(),
            "native_site_position": site_position.tolist(),
            "native_site_rotation": site_rotation.tolist(),
            "native_site_half_size": site_size.tolist(),
            "raw_world_axis": raw_world.tolist(),
            "compiled_floor_tangent_world_axis": (
                floor_tangent_world.tolist()
            ),
            "compiled_floor_normal": floor_normal.tolist(),
            "compiled_floor_geom_id": int(floor["geom_id"]),
            "compiled_floor_geom_name": str(floor["geom_name"]),
            "planar_projection_norm": planar_norm,
            "duplicate_of": duplicate_of,
        }
        proposal_evidence.append(
            {
                "direction_xy": direction.tolist(),
                "accepted": duplicate_of is None,
                "axis_provenance": provenance,
            }
        )
        if duplicate_of is not None:
            continue
        record = {
            "source": (
                "native heating-site/floor axis family " + label
            ),
            "direction_xy": direction.tolist(),
            "native_axis_provenance": provenance,
        }
        accepted.append(record)
        known_directions.append((label, direction))
    if not accepted:
        raise RuntimeError(
            "native heating-site/floor grasp direction family contains no "
            "unique extension beyond the legacy candidates"
        )
    return accepted, {
        "method": (
            "ordered native heating-site local axes projected through the "
            "selected compiled microwave floor; no world direction is "
            "hard-coded"
        ),
        "priority_order": [label for label, _ in specifications],
        "symmetry_policy": (
            "only the native front/left half-plane is proposed; mirrored "
            "right/back families are omitted, while same-direction matches "
            "against legacy and accepted records are removed"
        ),
        "legacy_direction_count": len(existing_direction_records),
        "accepted_direction_count": len(accepted),
        "duplicate_direction_count": sum(
            not proposal["accepted"] for proposal in proposal_evidence
        ),
        "proposals": proposal_evidence,
    }


def _ordered_grasp_direction_offset_specs(
    offset_values, direction_records, legacy_direction_count
):
    """Keep every legacy offset/direction pair before native extensions."""
    offset_values = tuple(float(value) for value in offset_values)
    legacy_direction_count = int(legacy_direction_count)
    if (
        not offset_values
        or not all(np.isfinite(value) for value in offset_values)
        or legacy_direction_count <= 0
        or legacy_direction_count >= len(direction_records)
    ):
        raise RuntimeError(
            "grasp direction phases require finite offsets plus legacy and "
            "native direction records"
        )
    specs = []
    for direction_start, direction_stop, family in (
        (0, legacy_direction_count, "legacy"),
        (
            legacy_direction_count,
            len(direction_records),
            "native_site_floor_extension",
        ),
    ):
        for offset in offset_values:
            for direction_index in range(direction_start, direction_stop):
                specs.append(
                    {
                        "offset_m": offset,
                        "direction_index": int(direction_index),
                        "direction_record": direction_records[
                            direction_index
                        ],
                        "candidate_family": family,
                    }
                )
    return specs


def _compiled_target_grasp_clearance(
    env,
    names,
    target_position,
    outward_direction_xy,
    site_position,
    site_rotation,
    site_size,
):
    """Derive a no-contact grasp corridor around every native scene object."""
    model = env.sim.model
    target_position = np.asarray(target_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    if outward.shape != (2,):
        raise RuntimeError("target grasp outward direction is not planar")
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= np.finfo(float).eps:
        raise RuntimeError("target grasp outward direction is zero")
    outward = outward / outward_norm
    current_eef = _eef_position(env)
    (
        fixture_compatible_gripper_geoms,
        fixture_geoms,
        rigid_gripper_geometry,
    ) = _compiled_rigid_gripper_fixture_geoms(env, names)
    rigid_gripper_geoms = rigid_gripper_geometry[
        "rigid_gripper_geom_ids"
    ]
    target_geoms = sorted(descendant_geom_ids(model, TARGET_BODY))
    collision_target_geoms = _collision_compatible_geom_ids(
        model,
        target_geoms,
        rigid_gripper_geoms,
    )
    collision_gripper_geoms = _collision_compatible_geom_ids(
        model,
        rigid_gripper_geoms,
        collision_target_geoms,
    )
    porcelain_position, _ = body_pose(env.sim, PORCELAIN_BODY)
    porcelain_geoms = sorted(
        descendant_geom_ids(model, PORCELAIN_BODY)
    )
    collision_porcelain_geoms = _collision_compatible_geom_ids(
        model,
        porcelain_geoms,
        rigid_gripper_geoms,
    )
    porcelain_gripper_geoms = _collision_compatible_geom_ids(
        model,
        rigid_gripper_geoms,
        collision_porcelain_geoms,
    )
    if (
        not collision_target_geoms
        or not collision_gripper_geoms
        or not collision_porcelain_geoms
        or not porcelain_gripper_geoms
    ):
        raise RuntimeError(
            "compiled target grasp has no gripper/target/porcelain "
            "collision geometry"
        )

    direction_records = [
        {
            "source": "compiled microwave outward direction",
            "direction_xy": outward.tolist(),
        }
    ]
    target_to_porcelain = (
        np.asarray(porcelain_position, dtype=float)[:2]
        - target_position[:2]
    )
    target_to_porcelain_norm = float(
        np.linalg.norm(target_to_porcelain)
    )
    if target_to_porcelain_norm <= np.finfo(float).eps:
        raise RuntimeError(
            "native target and parked porcelain have coincident XY origins"
        )
    target_to_porcelain /= target_to_porcelain_norm
    tangent_directions = [
        np.asarray(
            [-target_to_porcelain[1], target_to_porcelain[0]],
            dtype=float,
        ),
        np.asarray(
            [target_to_porcelain[1], -target_to_porcelain[0]],
            dtype=float,
        ),
    ]
    tangent_directions.sort(
        key=lambda direction: -float(np.dot(direction, outward))
    )
    for tangent_index, direction in enumerate(tangent_directions):
        if any(
            np.allclose(
                direction,
                np.asarray(record["direction_xy"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
            for record in direction_records
        ):
            continue
        direction_records.append(
            {
                "source": (
                    "target-to-parked-porcelain tangent "
                    f"{tangent_index}"
                ),
                "direction_xy": direction.tolist(),
                "microwave_outward_dot": float(
                    np.dot(direction, outward)
                ),
            }
        )
    legacy_direction_count = len(direction_records)
    (
        native_direction_records,
        native_direction_family_evidence,
    ) = _native_site_grasp_direction_family(
        env,
        names,
        target_geoms,
        site_position,
        site_rotation,
        site_size,
        direction_records,
    )
    direction_records.extend(native_direction_records)

    gripper_origin_bound = max(
        float(
            np.linalg.norm(
                np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
                - current_eef
            )
            + model.geom_rbound[geom_id]
        )
        for geom_id in collision_gripper_geoms
    )
    target_origin_bound = max(
        float(
            np.linalg.norm(
                np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
                - target_position
            )
            + model.geom_rbound[geom_id]
        )
        for geom_id in collision_target_geoms
    )
    if (
        not np.isfinite(gripper_origin_bound)
        or not np.isfinite(target_origin_bound)
        or gripper_origin_bound <= 0.0
        or target_origin_bound <= 0.0
    ):
        raise RuntimeError("compiled target grasp has invalid origin bounds")

    first_offset = TARGET_GRASP_CLEARANCE_OFFSET
    last_offset = max(
        first_offset + TARGET_INSERTION_SEARCH_STEP_M,
        gripper_origin_bound
        + target_origin_bound
        + EEF_POSITION_TOLERANCE
        + TARGET_INSERTION_SEARCH_STEP_M,
    )
    nominal_eef = target_position + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    compiled_geometry_cache = {
        "convex_mesh": {},
        "hits": 0,
        "misses": 0,
    }
    trace = []
    geometry_passes = []
    offset_values = np.arange(
        first_offset,
        last_offset + 0.5 * TARGET_INSERTION_SEARCH_STEP_M,
        TARGET_INSERTION_SEARCH_STEP_M,
    )
    ordered_candidate_specs = _ordered_grasp_direction_offset_specs(
        offset_values, direction_records, legacy_direction_count
    )
    total_candidates = len(ordered_candidate_specs)
    print(
        "[L3-A4 grasp prefilter progress] "
        f"started total={total_candidates} "
        f"legacy={len(offset_values) * legacy_direction_count} "
        "native="
        f"{total_candidates - len(offset_values) * legacy_direction_count}",
        flush=True,
    )
    compiled_target_sweep_geometry = (
        _compile_translated_sweep_geometry(
            env,
            collision_gripper_geoms,
            collision_target_geoms,
            compiled_geometry_cache,
        )
    )
    compiled_porcelain_sweep_geometry = (
        _compile_translated_sweep_geometry(
            env,
            porcelain_gripper_geoms,
            collision_porcelain_geoms,
            compiled_geometry_cache,
        )
    )
    compiled_fixture_sweep_geometry = (
        _compile_translated_sweep_geometry(
            env,
            fixture_compatible_gripper_geoms,
            fixture_geoms,
            compiled_geometry_cache,
        )
    )
    compiled_sweep_geometry_by_name = {
        "target_descend": compiled_target_sweep_geometry,
        "target_approach": compiled_target_sweep_geometry,
        "porcelain_lateral": compiled_porcelain_sweep_geometry,
        "porcelain_descend": compiled_porcelain_sweep_geometry,
        "porcelain_approach": compiled_porcelain_sweep_geometry,
        "fixture_lateral": compiled_fixture_sweep_geometry,
        "fixture_descend": compiled_fixture_sweep_geometry,
        "fixture_approach": compiled_fixture_sweep_geometry,
    }
    compiled_mesh_box_pair_count = sum(
        len(item["mesh_box_pair_geometry"])
        for item in (
            compiled_target_sweep_geometry,
            compiled_porcelain_sweep_geometry,
            compiled_fixture_sweep_geometry,
        )
    )
    print(
        "[L3-A4 grasp prefilter progress] compiled "
        "target_pairs="
        f"{len(compiled_target_sweep_geometry['compatible_geom_pairs'])} "
        "porcelain_pairs="
        f"{len(compiled_porcelain_sweep_geometry['compatible_geom_pairs'])} "
        "fixture_pairs="
        f"{len(compiled_fixture_sweep_geometry['compatible_geom_pairs'])} "
        "mesh_box_pairs="
        f"{compiled_mesh_box_pair_count}",
        flush=True,
    )
    rejection_witness_by_lane = {}
    witness_status_names = (
        "not_provided",
        "interval_mismatch",
        "pair_missing",
        "sample_invalid",
        "exact_reject",
        "exact_above_threshold",
    )
    prefilter_counters = {
        "candidates_total": int(total_candidates),
        "candidates_evaluated": 0,
        "sweep_calls": 0,
        "full_sweeps": 0,
        "threshold_rejection_sweeps": 0,
        "compatible_pair_evaluations": 0,
        "full_sweep_pair_evaluations": 0,
        "total_pair_evaluations_without_fail_fast": 0,
        "exact_pair_clearances_computed": 0,
        "scalar_threshold_boundary_refinements": 0,
        "cached_rejection_witness_attempts": 0,
        "cached_rejection_witness_rejections": 0,
        "cached_rejection_witness_full_fallbacks": 0,
        "cached_rejection_witness_status_counts": {
            status: 0 for status in witness_status_names
        },
    }
    for candidate_index, candidate_spec in enumerate(
        ordered_candidate_specs
    ):
        for offset, direction_index, direction_record in (
            (
                candidate_spec["offset_m"],
                candidate_spec["direction_index"],
                candidate_spec["direction_record"],
            ),
        ):
            direction = np.asarray(
                direction_record["direction_xy"], dtype=float
            )
            clearance_eef = target_position + np.asarray(
                [
                    direction[0] * float(offset),
                    direction[1] * float(offset),
                    GRASP_HEIGHT,
                ]
            )
            clearance_high_eef = clearance_eef + np.asarray(
                [0.0, 0.0, APPROACH_HEIGHT]
            )
            sweep_specs = [
                (
                    "target_descend",
                    collision_gripper_geoms,
                    collision_target_geoms,
                    clearance_high_eef,
                    clearance_eef,
                    EEF_POSITION_TOLERANCE,
                ),
                (
                    "target_approach",
                    collision_gripper_geoms,
                    collision_target_geoms,
                    current_eef,
                    clearance_high_eef,
                    EEF_POSITION_TOLERANCE,
                ),
                (
                    "porcelain_lateral",
                    porcelain_gripper_geoms,
                    collision_porcelain_geoms,
                    clearance_eef,
                    nominal_eef,
                    0.0,
                ),
                (
                    "porcelain_descend",
                    porcelain_gripper_geoms,
                    collision_porcelain_geoms,
                    clearance_high_eef,
                    clearance_eef,
                    0.0,
                ),
                (
                    "porcelain_approach",
                    porcelain_gripper_geoms,
                    collision_porcelain_geoms,
                    current_eef,
                    clearance_high_eef,
                    0.0,
                ),
                (
                    "fixture_lateral",
                    fixture_compatible_gripper_geoms,
                    fixture_geoms,
                    clearance_eef,
                    nominal_eef,
                    0.0,
                ),
                (
                    "fixture_descend",
                    fixture_compatible_gripper_geoms,
                    fixture_geoms,
                    clearance_high_eef,
                    clearance_eef,
                    0.0,
                ),
                (
                    "fixture_approach",
                    fixture_compatible_gripper_geoms,
                    fixture_geoms,
                    current_eef,
                    clearance_high_eef,
                    0.0,
                ),
            ]
            clearance_values = {
                f"{sweep_name}_clearance_m": None
                for sweep_name, *_ in sweep_specs
            }
            sweep_evidence = {}
            rejection_stage = None
            for (
                sweep_name,
                moving_geoms,
                obstacle_geoms,
                sweep_start,
                sweep_end,
                required_clearance,
            ) in sweep_specs:
                witness_lane = (
                    str(candidate_spec["candidate_family"]),
                    int(direction_index),
                    str(sweep_name),
                )
                prior_witness = rejection_witness_by_lane.get(
                    witness_lane
                )
                cached_rejection_witness = (
                    None
                    if prior_witness is None
                    else prior_witness["witness"]
                )
                clearance, evidence = _translated_swept_clearance(
                    env,
                    moving_geoms,
                    obstacle_geoms,
                    sweep_start,
                    sweep_end,
                    current_eef,
                    stop_at_or_below=required_clearance,
                    compiled_geometry_cache=compiled_geometry_cache,
                    compiled_sweep_geometry=(
                        compiled_sweep_geometry_by_name[sweep_name]
                    ),
                    cached_rejection_witness=(
                        cached_rejection_witness
                    ),
                )
                prefilter_counters["sweep_calls"] += 1
                prefilter_counters[
                    "compatible_pair_evaluations"
                ] += int(evidence["compatible_pair_evaluations"])
                prefilter_counters[
                    "total_pair_evaluations_without_fail_fast"
                ] += int(
                    evidence[
                        "total_pair_evaluations_without_fail_fast"
                    ]
                )
                prefilter_counters[
                    "exact_pair_clearances_computed"
                ] += int(evidence["exact_pair_clearances_computed"])
                prefilter_counters[
                    "scalar_threshold_boundary_refinements"
                ] += int(
                    evidence[
                        "scalar_threshold_boundary_refinement_count"
                    ]
                )
                prefilter_counters[
                    "cached_rejection_witness_attempts"
                ] += int(
                    evidence["cached_rejection_witness_attempted"]
                )
                prefilter_counters[
                    "cached_rejection_witness_rejections"
                ] += int(
                    evidence["cached_rejection_witness_rejected"]
                )
                prefilter_counters[
                    "cached_rejection_witness_full_fallbacks"
                ] += int(
                    evidence[
                        "cached_rejection_witness_fell_back_to_full_sweep"
                    ]
                )
                witness_status = evidence[
                    "cached_rejection_witness_status"
                ]
                if witness_status not in witness_status_names:
                    raise RuntimeError(
                        "translated sweep returned an unknown cached "
                        f"witness status: {witness_status!r}"
                    )
                prefilter_counters[
                    "cached_rejection_witness_status_counts"
                ][witness_status] += 1
                if evidence["full_sweep_evaluated"]:
                    prefilter_counters["full_sweeps"] += 1
                    prefilter_counters[
                        "full_sweep_pair_evaluations"
                    ] += int(evidence["compatible_pair_evaluations"])
                if evidence["threshold_rejection_seen"]:
                    prefilter_counters[
                        "threshold_rejection_sweeps"
                    ] += 1
                clearance_values[
                    f"{sweep_name}_clearance_m"
                ] = clearance
                sweep_evidence[sweep_name] = evidence
                if clearance <= required_clearance:
                    limiting_pair = evidence["limiting_pair"]
                    rejection_witness_by_lane[witness_lane] = {
                        "candidate_index": int(candidate_index),
                        "candidate_family": str(
                            candidate_spec["candidate_family"]
                        ),
                        "direction_index": int(direction_index),
                        "offset_m": float(offset),
                        "sweep_name": str(sweep_name),
                        "witness": {
                            "source_candidate_index": int(
                                candidate_index
                            ),
                            "source_candidate_family": str(
                                candidate_spec["candidate_family"]
                            ),
                            "source_direction_index": int(
                                direction_index
                            ),
                            "source_offset_m": float(offset),
                            "source_sweep_name": str(sweep_name),
                            "sample_intervals": evidence[
                                "sample_intervals"
                            ],
                            "sample_index": limiting_pair[
                                "sample_index"
                            ],
                            "moving_geom_id": limiting_pair[
                                "moving_geom_id"
                            ],
                            "fixture_geom_id": limiting_pair[
                                "fixture_geom_id"
                            ],
                        },
                    }
                    rejection_stage = sweep_name
                    break
                rejection_witness_by_lane.pop(witness_lane, None)
            target_clearances = {
                name: clearance_values[name]
                for name in (
                    "target_approach_clearance_m",
                    "target_descend_clearance_m",
                )
            }
            fixture_clearances = {
                name: clearance_values[name]
                for name in (
                    "fixture_approach_clearance_m",
                    "fixture_descend_clearance_m",
                    "fixture_lateral_clearance_m",
                )
            }
            porcelain_clearances = {
                name: clearance_values[name]
                for name in (
                    "porcelain_approach_clearance_m",
                    "porcelain_descend_clearance_m",
                    "porcelain_lateral_clearance_m",
                )
            }
            passed = bool(
                all(
                    value is not None
                    and value > EEF_POSITION_TOLERANCE
                    for value in target_clearances.values()
                )
                and all(
                    value is not None and value > 0.0
                    for value in fixture_clearances.values()
                )
                and all(
                    value is not None and value > 0.0
                    for value in porcelain_clearances.values()
                )
            )
            record = {
                "direction_index": int(direction_index),
                "direction_source": direction_record["source"],
                "direction_xy": direction.tolist(),
                "outward_offset_m": float(offset),
                "clearance_eef_position": clearance_eef.tolist(),
                "clearance_high_eef_position": (
                    clearance_high_eef.tolist()
                ),
                **target_clearances,
                **fixture_clearances,
                **porcelain_clearances,
                "evaluated_sweeps": list(sweep_evidence),
                "skipped_sweeps": [
                    sweep_name
                    for sweep_name, *_ in sweep_specs
                    if sweep_name not in sweep_evidence
                ],
                "rejection_stage": rejection_stage,
                "passed": passed,
            }
            if candidate_spec["candidate_family"] != "legacy":
                record["candidate_family"] = candidate_spec[
                    "candidate_family"
                ]
                record["native_axis_provenance"] = (
                    direction_record["native_axis_provenance"]
                )
            trace.append(record)
            if passed:
                geometry_passes.append(
                    {
                        "candidate_trace_index": len(trace) - 1,
                        "clearance_eef_position": clearance_eef.copy(),
                        "record": record,
                        "sweep_evidence": sweep_evidence,
                    }
                )
            evaluated_sweep_names = set(sweep_evidence)
            for sweep_name in compiled_sweep_geometry_by_name:
                if sweep_name not in evaluated_sweep_names:
                    rejection_witness_by_lane.pop(
                        (
                            str(candidate_spec["candidate_family"]),
                            int(direction_index),
                            str(sweep_name),
                        ),
                        None,
                    )
            prefilter_counters["candidates_evaluated"] = int(
                candidate_index + 1
            )
            if (
                (candidate_index + 1) % 25 == 0
                or candidate_index + 1 == total_candidates
            ):
                print(
                    "[L3-A4 grasp prefilter progress] "
                    f"completed={candidate_index + 1}/{total_candidates} "
                    f"passes={len(geometry_passes)} "
                    f"rejection_stage={rejection_stage!r} "
                    f"sweeps={prefilter_counters['sweep_calls']} "
                    f"full_sweeps={prefilter_counters['full_sweeps']} "
                    "pairs="
                    f"{prefilter_counters['compatible_pair_evaluations']}/"
                    f"{prefilter_counters['total_pair_evaluations_without_fail_fast']} "
                    "exact_pairs="
                    f"{prefilter_counters['exact_pair_clearances_computed']} "
                    "full_pairs="
                    f"{prefilter_counters['full_sweep_pair_evaluations']} "
                    "boundary_refinements="
                    f"{prefilter_counters['scalar_threshold_boundary_refinements']} "
                    "witness="
                    f"{prefilter_counters['cached_rejection_witness_attempts']}/"
                    f"{prefilter_counters['cached_rejection_witness_rejections']}/"
                    f"{prefilter_counters['cached_rejection_witness_full_fallbacks']} "
                    "witness_status="
                    + ",".join(
                        f"{status}:"
                        f"{prefilter_counters['cached_rejection_witness_status_counts'][status]}"
                        for status in witness_status_names
                    ),
                    flush=True,
                )
    if not geometry_passes:
        raise RuntimeError(
            "no compiled no-contact outside target grasp pose; "
            f"candidates={trace}"
        )
    return geometry_passes, {
        "method": (
            "ordered compiled-geometry grasp corridors gated by exact "
            "robot-controlled dynamic approach, descend, and lateral seek"
        ),
        "outward_direction_xy": outward.tolist(),
        "direction_candidates": direction_records,
        "legacy_direction_count": legacy_direction_count,
        "legacy_candidate_phase_count": int(
            len(offset_values) * legacy_direction_count
        ),
        "native_direction_family": native_direction_family_evidence,
        "candidate_phase_order": [
            "legacy",
            "native_site_floor_extension",
        ],
        "target_to_porcelain_xy": target_to_porcelain.tolist(),
        "target_to_porcelain_distance_m": target_to_porcelain_norm,
        "porcelain_position": porcelain_position.tolist(),
        "minimum_outward_offset_m": first_offset,
        "maximum_outward_offset_m": last_offset,
        "search_step_m": TARGET_INSERTION_SEARCH_STEP_M,
        "target_clearance_required_m": EEF_POSITION_TOLERANCE,
        "gripper_origin_bound_m": gripper_origin_bound,
        "target_origin_bound_m": target_origin_bound,
        "rigid_gripper_geometry": rigid_gripper_geometry,
        "compiled_geometry_cache": {
            "convex_mesh_entry_count": len(
                compiled_geometry_cache["convex_mesh"]
            ),
            "hits": int(compiled_geometry_cache["hits"]),
            "misses": int(compiled_geometry_cache["misses"]),
        },
        "prefilter_acceleration": {
            "witness_reuse_scope": (
                "same candidate phase, direction index, and sweep lane from "
                "the preceding offset in that lane; the exact canonical "
                "sample and geom pair are re-evaluated"
            ),
            "compiled_target_pair_count": len(
                compiled_target_sweep_geometry[
                    "compatible_geom_pairs"
                ]
            ),
            "compiled_porcelain_pair_count": len(
                compiled_porcelain_sweep_geometry[
                    "compatible_geom_pairs"
                ]
            ),
            "compiled_fixture_pair_count": len(
                compiled_fixture_sweep_geometry[
                    "compatible_geom_pairs"
                ]
            ),
            "compiled_mesh_box_pair_count": (
                compiled_mesh_box_pair_count
            ),
            **prefilter_counters,
        },
        "collision_gripper_geom_ids": collision_gripper_geoms,
        "collision_target_geom_ids": collision_target_geoms,
        "collision_porcelain_geom_ids": collision_porcelain_geoms,
        "geometry_pass_count": len(geometry_passes),
        "geometry_pass_trace_indices": [
            candidate["candidate_trace_index"]
            for candidate in geometry_passes
        ],
        "dynamic_trial_horizon_steps": TARGET_CONTACT_SEEK_STEPS,
        "dynamic_axis_progress_epsilon_m": (
            TARGET_DYNAMIC_AXIS_PROGRESS_EPS_M
        ),
        "selected": None,
        "candidate_trace": trace,
        "dynamic_candidate_trials": [],
    }


def _compiled_safe_insertion_portal(
    env,
    site_position,
    front,
    front_extent,
    floor_surface,
    floor_normal,
    support_offset,
    held_eef_offset,
    current_target,
    current_eef,
    target_geoms,
    target_fixture_geoms,
    gripper_geoms,
    fixture_geoms,
):
    """Derive the nearest safe outside portal and its complete approach."""
    site_position = np.asarray(site_position, dtype=float)
    front = np.asarray(front, dtype=float)
    floor_surface = np.asarray(floor_surface, dtype=float)
    floor_normal = np.asarray(floor_normal, dtype=float)
    held_eef_offset = np.asarray(held_eef_offset, dtype=float)
    current_target = np.asarray(current_target, dtype=float)
    current_eef = np.asarray(current_eef, dtype=float)
    first_distance = 2.0 * float(front_extent)
    current_distance = float(
        np.dot(current_target - site_position, front)
    )
    last_distance = max(
        first_distance + TARGET_INSERTION_SEARCH_STEP_M,
        current_distance,
    )
    lifted_offset = np.asarray(
        [0.0, 0.0, APPROACH_HEIGHT], dtype=float
    )
    current_lifted_target = current_target + lifted_offset
    current_lifted_eef = current_eef + lifted_offset
    lift_gripper_clearance, lift_gripper_sweep = (
        _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            current_eef,
            current_lifted_eef,
            current_eef,
        )
    )
    lift_target_clearance, lift_target_sweep = (
        _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            current_target,
            current_lifted_target,
            current_target,
        )
    )
    trace = []
    for portal_distance in np.arange(
        first_distance,
        last_distance + 0.5 * TARGET_INSERTION_SEARCH_STEP_M,
        TARGET_INSERTION_SEARCH_STEP_M,
    ):
        portal_object = site_position + front * float(portal_distance)
        portal_object += floor_normal * float(
            np.dot(
                floor_surface
                + floor_normal * float(support_offset)
                - portal_object,
                floor_normal,
            )
        )
        portal_eef = portal_object + held_eef_offset
        portal_high_object = portal_object + lifted_offset
        portal_high_eef = portal_eef + lifted_offset
        (
            transport_gripper_clearance,
            transport_gripper_sweep,
        ) = _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            current_lifted_eef,
            portal_high_eef,
            current_eef,
        )
        (
            transport_target_clearance,
            transport_target_sweep,
        ) = _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            current_lifted_target,
            portal_high_object,
            current_target,
        )
        (
            alignment_gripper_clearance,
            alignment_gripper_sweep,
        ) = _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            portal_high_eef,
            portal_eef,
            current_eef,
        )
        (
            alignment_target_clearance,
            alignment_target_sweep,
        ) = _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            portal_high_object,
            portal_object,
            current_target,
        )
        clearances = {
            "lift_gripper_clearance_m": lift_gripper_clearance,
            "lift_target_clearance_m": lift_target_clearance,
            "transport_gripper_clearance_m": (
                transport_gripper_clearance
            ),
            "transport_target_clearance_m": transport_target_clearance,
            "alignment_gripper_clearance_m": (
                alignment_gripper_clearance
            ),
            "alignment_target_clearance_m": alignment_target_clearance,
        }
        passed = bool(all(value > 0.0 for value in clearances.values()))
        record = {
            "front_distance_from_site_center_m": float(portal_distance),
            "portal_object_position": portal_object.tolist(),
            "portal_eef_position": portal_eef.tolist(),
            "portal_high_object_position": portal_high_object.tolist(),
            "portal_high_eef_position": portal_high_eef.tolist(),
            **clearances,
            "passed": passed,
        }
        trace.append(record)
        if passed:
            return (
                portal_object,
                portal_eef,
                portal_high_eef,
                {
                    "method": (
                        "nearest outside front-axis portal whose lifted "
                        "transport and vertical alignment segments have "
                        "positive compiled gripper/target clearance"
                    ),
                    "search_step_m": TARGET_INSERTION_SEARCH_STEP_M,
                    "first_front_distance_m": first_distance,
                    "last_front_distance_m": last_distance,
                    "selected": record,
                    "lift_gripper_sweep": lift_gripper_sweep,
                    "lift_target_sweep": lift_target_sweep,
                    "transport_gripper_sweep": transport_gripper_sweep,
                    "transport_target_sweep": transport_target_sweep,
                    "alignment_gripper_sweep": alignment_gripper_sweep,
                    "alignment_target_sweep": alignment_target_sweep,
                    "candidate_trace": trace,
                },
            )
    raise RuntimeError(
        "no compiled collision-free outside insertion portal; "
        f"candidates={trace}"
    )


def _deterministic_signed_lateral_offsets(extent, step) -> list[float]:
    """Enumerate 0,+step,-step,+2step,-2step without edge overflow."""
    extent = float(extent)
    step = float(step)
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("lateral search extent must be finite and positive")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("lateral search step must be finite and positive")
    count = int(np.floor(np.nextafter(extent, 0.0) / step))
    values = [0.0]
    for index in range(1, count + 1):
        value = float(index * step)
        values.extend((value, -value))
    return values


def _compact_insertion_sweep_evidence(sweep) -> dict:
    """Keep audit-critical sweep scalars without per-candidate geometry bulk."""
    limiting = sweep.get("limiting_pair") or {}
    limiting_fields = (
        "sample_index",
        "sample_fraction",
        "translated_reference_position",
        "door_angle_rad",
        "moving_geom_id",
        "moving_geom_name",
        "moving_body_name",
        "fixture_geom_id",
        "fixture_geom_name",
        "fixture_body_name",
        "target_geom_id",
        "target_geom_name",
        "door_geom_id",
        "door_geom_name",
        "clearance_m",
        "continuous_guard_m",
        "method",
        "clearance_components",
    )
    summary_fields = (
        "path_length_m",
        "sample_intervals",
        "sample_spacing_m",
        "samples",
        "compatible_pair_evaluations",
        "total_pair_evaluations_without_fail_fast",
        "exact_pair_clearances_computed",
        "scalar_threshold_boundary_refinement_count",
        "candidate_invariant_door_pose_count",
        "vectorized_exact_obb_pair_count",
        "vectorized_exact_obb_pair_count_per_sample",
        "candidate_invariant_mesh_box_pair_count",
        "minimum_clearance_m",
        "threshold_fail_fast_m",
        "terminated_early",
        "threshold_rejection_seen",
        "full_sweep_evaluated",
        "cached_rejection_witness_attempted",
        "cached_rejection_witness_rejected",
        "cached_rejection_witness_clearance_m",
        "cached_rejection_witness_fell_back_to_full_sweep",
    )
    return {
        **{
            field: sweep[field]
            for field in summary_fields
            if field in sweep
        },
        "limiting_pair": {
            field: limiting[field]
            for field in limiting_fields
            if field in limiting
        },
    }


def _compiled_target_insertion_plan(
    env,
    names,
    site_position,
    site_rotation,
    site_size,
    held_eef_offset,
    support_geometry,
):
    """Search front-to-back for the foremost native-In, collision-free pose."""
    model = env.sim.model
    site_position = np.asarray(site_position, dtype=float)
    site_rotation = np.asarray(site_rotation, dtype=float)
    site_size = np.asarray(site_size, dtype=float)
    held_eef_offset = np.asarray(held_eef_offset, dtype=float)
    front = -site_rotation[:, 1]
    front = front / np.linalg.norm(front)
    world_half_size = np.abs(site_rotation @ site_size)
    front_extent = min(
        float(world_half_size[axis] / abs(front[axis]))
        for axis in range(3)
        if abs(float(front[axis])) > np.finfo(float).eps
    )
    if not np.isfinite(front_extent) or front_extent <= 0.0:
        raise RuntimeError("native heating site has no finite front extent")

    target_geoms = sorted(descendant_geom_ids(model, TARGET_BODY))
    floor, floor_geometry = _compiled_microwave_floor(
        env,
        names,
        site_position,
        site_rotation,
        target_geoms,
    )
    floor_geom = int(floor["geom_id"])
    floor_center = np.asarray(floor["center"], dtype=float)
    floor_rotation = np.asarray(floor["rotation"], dtype=float)
    floor_half_size = np.asarray(floor["half_size"], dtype=float)
    floor_normal_axis = int(floor["normal_axis"])
    floor_normal = np.asarray(floor["normal"], dtype=float)
    floor_surface = np.asarray(
        floor["surface_position"], dtype=float
    )
    current_target, current_target_rotation = body_pose(
        env.sim, TARGET_BODY
    )
    current_target_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    held_support_geometry = _compiled_held_target_support_geometry(
        env,
        support_geometry,
        floor_geom,
        floor_normal,
    )
    support_offset = float(
        held_support_geometry["held_support_offset_m"]
    )
    current_eef = _eef_position(env)

    (
        collision_gripper_geoms,
        collision_fixture_geoms,
        gripper_geometry,
    ) = _compiled_rigid_gripper_fixture_geoms(env, names)
    fixture_geoms = sorted(
        descendant_geom_ids(model, names["fixture_root"])
    )

    support_target_geoms = support_geometry[
        "supporting_target_geom_ids"
    ]
    floor_tangent_axes = [
        axis for axis in range(3) if axis != floor_normal_axis
    ]
    site_lateral = np.asarray(site_rotation[:, 0], dtype=float)
    site_lateral = site_lateral / np.linalg.norm(site_lateral)
    floor_lateral_candidates = []
    for axis in floor_tangent_axes:
        raw_direction = np.asarray(
            floor_rotation[:, axis], dtype=float
        )
        projected = raw_direction.copy()
        projected -= floor_normal * float(
            np.dot(projected, floor_normal)
        )
        projected -= front * float(np.dot(projected, front))
        norm = float(np.linalg.norm(projected))
        if norm <= np.finfo(float).eps:
            continue
        direction = projected / norm
        floor_lateral_candidates.append(
            {
                "floor_tangent_axis": int(axis),
                "direction": direction,
                "site_lateral_alignment": abs(
                    float(np.dot(direction, site_lateral))
                ),
            }
        )
    if not floor_lateral_candidates:
        raise RuntimeError(
            "compiled microwave floor has no tangent direction lateral "
            "to the native heating-site front axis"
        )
    floor_lateral_candidates.sort(
        key=lambda record: (
            -record["site_lateral_alignment"],
            record["floor_tangent_axis"],
        )
    )
    lateral_choice = floor_lateral_candidates[0]
    lateral_direction = lateral_choice["direction"].copy()
    if float(np.dot(lateral_direction, site_lateral)) < 0.0:
        lateral_direction *= -1.0
    local_lateral_direction = site_rotation.T @ lateral_direction
    lateral_extent = min(
        float(site_size[axis] / abs(local_lateral_direction[axis]))
        for axis in range(3)
        if abs(float(local_lateral_direction[axis]))
        > np.finfo(float).eps
    )
    if not np.isfinite(lateral_extent) or lateral_extent <= 0.0:
        raise RuntimeError(
            "native heating site has no finite compiled lateral extent"
        )
    lateral_search_values = _deterministic_signed_lateral_offsets(
        lateral_extent,
        TARGET_INSERTION_SEARCH_STEP_M,
    )
    target_fixture_geoms = [
        geom_id
        for geom_id in fixture_geoms
        if geom_id != floor_geom
    ]
    (
        portal_object,
        portal_eef,
        portal_high_eef,
        portal_geometry,
    ) = _compiled_safe_insertion_portal(
        env,
        site_position,
        front,
        front_extent,
        floor_surface,
        floor_normal,
        support_offset,
        held_eef_offset,
        current_target,
        current_eef,
        target_geoms,
        target_fixture_geoms,
        collision_gripper_geoms,
        collision_fixture_geoms,
    )
    compiled_door_sweep = _compile_target_door_sweep_geometry(
        env, names, target_geoms
    )
    compiled_geometry_cache = {
        "convex_mesh": {},
        "hits": 0,
        "misses": 0,
    }
    compiled_target_sweep_geometry = (
        _compile_translated_sweep_geometry(
            env,
            target_geoms,
            target_fixture_geoms,
            compiled_geometry_cache,
        )
    )
    compiled_gripper_sweep_geometry = (
        _compile_translated_sweep_geometry(
            env,
            collision_gripper_geoms,
            collision_fixture_geoms,
            compiled_geometry_cache,
        )
    )

    front_search_values = np.arange(
        np.nextafter(front_extent, 0.0),
        -front_extent - 0.5 * TARGET_INSERTION_SEARCH_STEP_M,
        -TARGET_INSERTION_SEARCH_STEP_M,
    )
    trace = []
    selected = None
    execution_endpoint = None
    representative_full_sweeps = {}
    best_gate_values = {}
    door_rejection_witness = None
    target_static_rejection_witness = None
    gripper_rejection_witness = None
    lateral_rank_by_value = {
        value: index
        for index, value in enumerate(lateral_search_values)
    }
    total_candidate_count = len(front_search_values) * len(
        lateral_search_values
    )
    print(
        "[L3-A4 insertion candidate progress] "
        f"started total={total_candidate_count} "
        f"front={len(front_search_values)} "
        f"lateral={len(lateral_search_values)}",
        flush=True,
    )
    for front_index, front_distance in enumerate(front_search_values):
        for lateral_offset in lateral_search_values:
            candidate = (
                site_position
                + front * float(front_distance)
                + lateral_direction * float(lateral_offset)
            )
            candidate += floor_normal * float(
                np.dot(
                    floor_surface
                    + floor_normal * support_offset
                    - candidate,
                    floor_normal,
                )
            )
            native_inside = native_site_contains_point(
                site_position,
                site_rotation,
                site_size,
                candidate,
            )
            support_axis_clearances = []
            for geom_id in support_target_geoms:
                geom_center = (
                    np.asarray(
                        env.sim.data.geom_xpos[geom_id], dtype=float
                    )
                    + candidate
                    - current_target
                )
                geom_rotation = np.asarray(
                    env.sim.data.geom_xmat[geom_id], dtype=float
                ).reshape(3, 3)
                floor_local = floor_rotation.T @ (
                    geom_center - floor_center
                )
                for axis in floor_tangent_axes:
                    if int(model.geom_type[geom_id]) == 6:
                        extent = float(
                            np.sum(
                                np.asarray(
                                    model.geom_size[geom_id], dtype=float
                                )
                                * np.abs(
                                    geom_rotation.T
                                    @ floor_rotation[:, axis]
                                )
                            )
                        )
                    else:
                        extent = float(model.geom_rbound[geom_id])
                    support_axis_clearances.append(
                        float(
                            floor_half_size[axis]
                            - abs(float(floor_local[axis]))
                            - extent
                        )
                    )
            support_clearance = min(
                support_axis_clearances, default=float("-inf")
            )
            candidate_eef = candidate + held_eef_offset
            door_clearance = None
            target_clearance = None
            gripper_clearance = None
            door_sweep = None
            target_sweep = None
            gripper_sweep = None
            rejection_stage = None
            skipped_gates = []
            if not native_inside:
                rejection_stage = "native_in"
                skipped_gates = [
                    "target_door_sweep",
                    "target_static_sweep",
                    "gripper_sweep",
                ]
            elif support_clearance <= 0.0:
                rejection_stage = "support_clearance"
                skipped_gates = [
                    "target_door_sweep",
                    "target_static_sweep",
                    "gripper_sweep",
                ]
            if rejection_stage is None:
                door_clearance, door_sweep = (
                    _compiled_target_door_sweep_clearance(
                        env,
                        names,
                        target_geoms,
                        candidate,
                        current_target,
                        stop_at_or_below=0.0,
                        cached_rejection_witness=(
                            door_rejection_witness
                        ),
                        compiled_door_sweep=compiled_door_sweep,
                    )
                )
                if door_clearance <= 0.0:
                    limiting_door_pair = door_sweep["limiting_pair"]
                    door_rejection_witness = {
                        "sample_index": limiting_door_pair[
                            "sample_index"
                        ],
                        "target_geom_id": limiting_door_pair[
                            "target_geom_id"
                        ],
                        "door_geom_id": limiting_door_pair[
                            "door_geom_id"
                        ],
                    }
                    rejection_stage = "target_door_sweep"
                    skipped_gates = [
                        "target_static_sweep",
                        "gripper_sweep",
                    ]
                else:
                    door_rejection_witness = None
            if rejection_stage is None:
                target_clearance, target_sweep = (
                    _translated_swept_clearance(
                        env,
                        target_geoms,
                        target_fixture_geoms,
                        portal_object,
                        candidate,
                        current_target,
                        stop_at_or_below=0.0,
                        compiled_geometry_cache=(
                            compiled_geometry_cache
                        ),
                        compiled_sweep_geometry=(
                            compiled_target_sweep_geometry
                        ),
                        cached_rejection_witness=(
                            target_static_rejection_witness
                        ),
                    )
                )
                if target_clearance <= 0.0:
                    limiting_target_pair = target_sweep[
                        "limiting_pair"
                    ]
                    target_static_rejection_witness = {
                        "sample_intervals": target_sweep[
                            "sample_intervals"
                        ],
                        "sample_index": limiting_target_pair[
                            "sample_index"
                        ],
                        "moving_geom_id": limiting_target_pair[
                            "moving_geom_id"
                        ],
                        "fixture_geom_id": limiting_target_pair[
                            "fixture_geom_id"
                        ],
                    }
                    rejection_stage = "target_static_sweep"
                    skipped_gates = ["gripper_sweep"]
                else:
                    target_static_rejection_witness = None
            if rejection_stage is None:
                gripper_clearance, gripper_sweep = (
                    _translated_swept_clearance(
                        env,
                        collision_gripper_geoms,
                        collision_fixture_geoms,
                        portal_eef,
                        candidate_eef,
                        current_eef,
                        stop_at_or_below=0.0,
                        compiled_geometry_cache=(
                            compiled_geometry_cache
                        ),
                        compiled_sweep_geometry=(
                            compiled_gripper_sweep_geometry
                        ),
                        cached_rejection_witness=(
                            gripper_rejection_witness
                        ),
                    )
                )
                if gripper_clearance <= 0.0:
                    limiting_gripper_pair = gripper_sweep[
                        "limiting_pair"
                    ]
                    gripper_rejection_witness = {
                        "sample_intervals": gripper_sweep[
                            "sample_intervals"
                        ],
                        "sample_index": limiting_gripper_pair[
                            "sample_index"
                        ],
                        "moving_geom_id": limiting_gripper_pair[
                            "moving_geom_id"
                        ],
                        "fixture_geom_id": limiting_gripper_pair[
                            "fixture_geom_id"
                        ],
                    }
                    rejection_stage = "gripper_sweep"
                else:
                    gripper_rejection_witness = None
            passed = bool(
                rejection_stage is None
                and native_inside
                and support_clearance > 0.0
                and door_clearance is not None
                and door_clearance > 0.0
                and target_clearance is not None
                and target_clearance > 0.0
                and gripper_clearance is not None
                and gripper_clearance > 0.0
            )
            search_index = len(trace)
            record = {
                "search_index": int(search_index),
                "front_search_index": int(front_index),
                "lateral_search_index": int(
                    lateral_rank_by_value[lateral_offset]
                ),
                "front_distance_from_site_center_m": float(
                    front_distance
                ),
                "lateral_offset_from_site_center_m": float(
                    lateral_offset
                ),
                "candidate_target_position": candidate.tolist(),
                "candidate_eef_position": candidate_eef.tolist(),
                "native_in": native_inside,
                "support_clearance_m": support_clearance,
                "gripper_swept_clearance_m": gripper_clearance,
                "target_swept_static_clearance_m": target_clearance,
                "target_door_swept_clearance_m": door_clearance,
                "gripper_sweep_summary": (
                    None
                    if gripper_sweep is None
                    else _compact_insertion_sweep_evidence(
                        gripper_sweep
                    )
                ),
                "target_sweep_summary": (
                    None
                    if target_sweep is None
                    else _compact_insertion_sweep_evidence(target_sweep)
                ),
                "door_sweep_summary": (
                    None
                    if door_sweep is None
                    else _compact_insertion_sweep_evidence(door_sweep)
                ),
                "rejection_stage": rejection_stage,
                "skipped_gates": skipped_gates,
                "passed": passed,
            }
            full_sweeps = {
                key: value
                for key, value in {
                    "gripper_sweep": gripper_sweep,
                    "target_sweep": target_sweep,
                    "door_sweep": door_sweep,
                }.items()
                if value is not None
            }
            representative_record = {
                key: record[key]
                for key in (
                    "search_index",
                    "front_search_index",
                    "lateral_search_index",
                    "front_distance_from_site_center_m",
                    "lateral_offset_from_site_center_m",
                    "candidate_target_position",
                    "candidate_eef_position",
                    "native_in",
                    "support_clearance_m",
                    "gripper_swept_clearance_m",
                    "target_swept_static_clearance_m",
                    "target_door_swept_clearance_m",
                    "rejection_stage",
                    "skipped_gates",
                    "passed",
                )
            }
            gate_values = {
                "native_in": float(native_inside),
                "support_clearance": support_clearance,
                "gripper_clearance": gripper_clearance,
                "target_static_clearance": target_clearance,
                "target_door_clearance": door_clearance,
            }
            for gate_name, gate_value in gate_values.items():
                if gate_value is None:
                    continue
                if (
                    gate_name not in best_gate_values
                    or gate_value > best_gate_values[gate_name]
                ):
                    best_gate_values[gate_name] = gate_value
                    representative_full_sweeps[gate_name] = {
                        "gate_value": gate_value,
                        "candidate": representative_record,
                        **full_sweeps,
                    }
            representative_full_sweeps["last_evaluated"] = {
                "candidate": representative_record,
                **full_sweeps,
            }
            if passed:
                record.update(full_sweeps)
            trace.append(record)
            completed_candidate_count = len(trace)
            if (
                completed_candidate_count == 1
                or completed_candidate_count % 25 == 0
                or passed
                or completed_candidate_count == total_candidate_count
            ):
                print(
                    "[L3-A4 insertion candidate progress] "
                    f"completed={completed_candidate_count}/"
                    f"{total_candidate_count} "
                    f"front_index={front_index} "
                    "lateral_index="
                    f"{lateral_rank_by_value[lateral_offset]} "
                    f"rejection_stage={rejection_stage!r} "
                    f"passed={passed}",
                    flush=True,
                )
            if passed:
                selected = record
                execution_endpoint = record
                break
        if selected is not None:
            break
    if selected is None or execution_endpoint is None:
        raise RuntimeError(
            "no compiled 2D native-In target release pose has positive "
            "support/gripper/mug/door clearance; "
            f"candidate_count={len(trace)}; candidates={trace}; "
            "representative_full_sweeps="
            f"{representative_full_sweeps}"
        )
    return {
        "method": (
            "deterministic 2D native heating-site search: foremost-to-back "
            "front distance, then zero/positive/negative lateral offset by "
            "magnitude, with compiled floor support and continuous "
            "translated gripper/mug sweep clearance"
        ),
        "search_step_m": TARGET_INSERTION_SEARCH_STEP_M,
        "front_search_values_m": front_search_values.tolist(),
        "sweep_step_m": TARGET_INSERTION_SWEEP_STEP_M,
        "native_site_position": site_position.tolist(),
        "native_site_rotation": site_rotation.tolist(),
        "native_site_half_size": site_size.tolist(),
        "native_site_world_aabb_half_size": world_half_size.tolist(),
        "target_tilt_at_planning_deg": current_target_tilt,
        "target_rotation_at_planning": current_target_rotation.tolist(),
        "held_tilt_policy": (
            "transient tilt while grasped is diagnostic; actual compiled "
            "geom orientation is used for support and swept clearance; "
            "release still requires the mug at or below MAX_MUG_TILT_DEG"
        ),
        "front_direction": front.tolist(),
        "front_extent_m": front_extent,
        "lateral_direction": lateral_direction.tolist(),
        "lateral_extent_m": lateral_extent,
        "lateral_search_values_m": lateral_search_values,
        "candidate_count_evaluated": len(trace),
        "candidate_gate_evaluation_order": [
            "native_in",
            "support_clearance",
            "target_door_sweep",
            "target_static_sweep",
            "gripper_sweep",
        ],
        "rejected_candidate_trace_policy": (
            "strict-AND fail-fast: every point retains order, evaluated gate "
            "scalars, sampling counts/spacing, limiting-pair summaries, an "
            "explicit rejection stage, and null plus skipped_gates for "
            "unevaluated gates; only the selected point retains every full "
            "sweep; total failure reports per-gate-best and last-evaluated "
            "representative sweep evidence, with threshold-terminated "
            "partials explicitly tagged"
        ),
        "lateral_direction_derivation": {
            "native_heating_site_lateral_axis": site_lateral.tolist(),
            "selected_floor_tangent_axis": lateral_choice[
                "floor_tangent_axis"
            ],
            "selected_site_lateral_alignment": lateral_choice[
                "site_lateral_alignment"
            ],
            "candidate_floor_tangents": [
                {
                    "floor_tangent_axis": record[
                        "floor_tangent_axis"
                    ],
                    "direction": record["direction"].tolist(),
                    "site_lateral_alignment": record[
                        "site_lateral_alignment"
                    ],
                }
                for record in floor_lateral_candidates
            ],
            "sign_aligned_to_native_site_axis": True,
        },
        "portal_object_position": portal_object.tolist(),
        "portal_eef_position": portal_eef.tolist(),
        "portal_high_eef_position": portal_high_eef.tolist(),
        "compiled_portal_derivation": portal_geometry,
        "compiled_floor": floor_geometry,
        "source_support": support_geometry,
        "held_pose_floor_support": held_support_geometry,
        "eef_root_body": gripper_geometry["eef_root_body"],
        "rigid_gripper_body_names": gripper_geometry[
            "rigid_gripper_body_names"
        ],
        "collision_gripper_geom_ids": collision_gripper_geoms,
        "collision_fixture_geom_ids": collision_fixture_geoms,
        "exact_acceleration": {
            "candidate_invariant_door_pose_count": (
                compiled_door_sweep["door_pose_count"]
            ),
            "door_vectorized_exact_obb_pair_count": len(
                compiled_door_sweep["obb_entry_indices"]
            ),
            "target_static_vectorized_exact_obb_pair_count": len(
                compiled_target_sweep_geometry["obb_pair_indices"]
            ),
            "gripper_vectorized_exact_obb_pair_count": len(
                compiled_gripper_sweep_geometry["obb_pair_indices"]
            ),
            "target_static_cached_mesh_box_pair_count": len(
                compiled_target_sweep_geometry[
                    "mesh_box_pair_geometry"
                ]
            ),
            "gripper_cached_mesh_box_pair_count": len(
                compiled_gripper_sweep_geometry[
                    "mesh_box_pair_geometry"
                ]
            ),
            "convex_mesh_entry_count": len(
                compiled_geometry_cache["convex_mesh"]
            ),
            "convex_mesh_cache_hits": int(
                compiled_geometry_cache["hits"]
            ),
            "convex_mesh_cache_misses": int(
                compiled_geometry_cache["misses"]
            ),
        },
        "selected": selected,
        "execution_endpoint": execution_endpoint,
        "execution_reserve_m": 0.0,
        "execution_endpoint_derivation": (
            "the selected safe release pose itself; runtime verifies actual "
            "target arrival and fails closed instead of commanding a "
            "fictional deeper overshoot"
        ),
        "candidate_trace": trace,
    }


def _target_insertion_plan_selection_evidence(insertion_plan) -> dict:
    """Freeze the deterministic geometry evidence that selected a release."""
    fields = (
        "method",
        "native_site_position",
        "native_site_rotation",
        "native_site_half_size",
        "target_tilt_at_planning_deg",
        "target_rotation_at_planning",
        "front_direction",
        "lateral_direction",
        "lateral_extent_m",
        "lateral_search_values_m",
        "lateral_direction_derivation",
        "portal_object_position",
        "portal_eef_position",
        "portal_high_eef_position",
        "held_pose_floor_support",
        "selected",
        "execution_endpoint",
    )
    missing = [field for field in fields if field not in insertion_plan]
    if missing:
        raise RuntimeError(
            "compiled target insertion plan lacks selection evidence: "
            f"missing={missing}"
        )
    return _snapshot_plain_state(
        {field: insertion_plan[field] for field in fields},
        "target_insertion_plan_selection_evidence",
    )


def _target_insertion_plan_replay_proof(
    selected_dynamic_trial,
    actual_held_eef_offset,
    actual_insertion_plan,
) -> dict:
    """Require independent execution to reproduce trial plan selection exactly."""
    expected_offset = np.asarray(
        selected_dynamic_trial["held_eef_minus_target_offset"],
        dtype=float,
    )
    actual_offset = np.asarray(actual_held_eef_offset, dtype=float)
    expected_selection = selected_dynamic_trial[
        "insertion_plan_selection_evidence"
    ]
    actual_selection = _target_insertion_plan_selection_evidence(
        actual_insertion_plan
    )
    expected_sha256 = selected_dynamic_trial[
        "insertion_plan_selection_sha256"
    ]
    expected_evidence_sha256 = _plain_state_sha256(expected_selection)
    actual_sha256 = _plain_state_sha256(actual_selection)
    held_offset_exact = bool(
        expected_offset.dtype == actual_offset.dtype
        and expected_offset.shape == actual_offset.shape
        and np.array_equal(expected_offset, actual_offset)
    )
    selection_exact = _plain_state_equal(
        expected_selection, actual_selection
    )
    expected_hash_self_consistent = bool(
        expected_sha256 == expected_evidence_sha256
    )
    selection_hash_exact = bool(expected_sha256 == actual_sha256)
    passed = bool(
        held_offset_exact
        and selection_exact
        and expected_hash_self_consistent
        and selection_hash_exact
    )
    return {
        "passed": passed,
        "comparison": "bitwise_exact_no_tolerance",
        "held_offset_bitwise_exact": held_offset_exact,
        "expected_held_eef_minus_target_offset": expected_offset.tolist(),
        "actual_held_eef_minus_target_offset": actual_offset.tolist(),
        "selection_evidence_bitwise_exact": selection_exact,
        "expected_hash_self_consistent": expected_hash_self_consistent,
        "selection_sha256_exact": selection_hash_exact,
        "expected_selection_sha256": expected_sha256,
        "expected_evidence_sha256": expected_evidence_sha256,
        "actual_selection_sha256": actual_sha256,
        "expected_selection_evidence": expected_selection,
        "actual_selection_evidence": actual_selection,
    }


def _compiled_open_gripper_retreat_plan(env, names, waypoints):
    """Validate retreat against fixture and the stationary released mug."""
    (
        gripper_geoms,
        fixture_geoms,
        geometry,
    ) = _compiled_rigid_gripper_fixture_geoms(env, names)
    reference = _eef_position(env)
    released_target_geoms = sorted(
        descendant_geom_ids(env.sim.model, TARGET_BODY)
    )
    start = reference.copy()
    segments = []
    minimum = float("inf")
    for label, endpoint in waypoints:
        endpoint = np.asarray(endpoint, dtype=float)
        fixture_clearance, fixture_sweep = _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            start,
            endpoint,
            reference,
        )
        released_target_clearance, released_target_sweep = (
            _translated_swept_clearance(
                env,
                gripper_geoms,
                released_target_geoms,
                start,
                endpoint,
                reference,
            )
        )
        segments.append(
            {
                "label": str(label),
                "fixture_clearance_m": fixture_clearance,
                "released_target_clearance_m": (
                    released_target_clearance
                ),
                "fixture_sweep": fixture_sweep,
                "released_target_sweep": released_target_sweep,
            }
        )
        minimum = min(
            minimum,
            fixture_clearance,
            released_target_clearance,
        )
        start = endpoint
    passed = bool(segments and minimum > 0.0)
    return passed, {
        "method": (
            "compiled continuous translated sweep of the actual open "
            "gripper against the microwave and stationary released target "
            "through horizontal portal retreat and exit"
        ),
        "minimum_clearance_m": minimum,
        "passed": passed,
        "gripper_geometry": geometry,
        "released_target_geom_ids": released_target_geoms,
        "segments": segments,
    }


def _step(env, oracle, action, step, frames):
    obs, _, _, _ = env.step(np.asarray(action, dtype=float).tolist())
    status = oracle.check(env, obs, action, step)
    if frames is not None:
        frames.append(policy_image(obs))
    return obs, status, step + 1


def _move_eef(
    env,
    oracle,
    target,
    gripper,
    step,
    frames,
    *,
    label="",
    diagnostics=None,
    forbid_microwave_contact=False,
    forbid_target_contact=False,
    forbid_porcelain_contact=False,
):
    target = np.asarray(target, dtype=float)
    initial_eef = _eef_position(env)
    initial_error = target - initial_eef
    error_norms = [float(np.linalg.norm(initial_error))]
    contact_bodies = _robot_contact_body_names(env)
    contact_pairs = {
        (
            pair["robot_geom_id"],
            pair["other_geom_id"],
        ): pair
        for pair in _robot_contact_pairs(env)
    }
    trace = []
    reached = False
    status = None
    forbidden_contact = bool(
        (
            forbid_microwave_contact
            and _has_microwave_contact(contact_bodies)
        )
        or (forbid_target_contact and TARGET_BODY in contact_bodies)
        or (
            forbid_porcelain_contact
            and PORCELAIN_BODY in contact_bodies
        )
    )
    for iteration in range(0 if forbidden_contact else MOVE_STEPS):
        error = target - _eef_position(env)
        if float(np.linalg.norm(error)) <= EEF_POSITION_TOLERANCE:
            reached = True
            break
        action = np.zeros(7, dtype=float)
        # LIBERO OSC position commands are normalized deltas. A gain of 20
        # requests full scale only beyond 5 cm and tapers near the waypoint.
        action[:3] = np.clip(error * 20.0, -1.0, 1.0)
        action[-1] = gripper
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        post_error = target - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        for pair in _robot_contact_pairs(env):
            contact_pairs[
                (pair["robot_geom_id"], pair["other_geom_id"])
            ] = pair
        current_microwave_contact = _has_microwave_contact(current_contacts)
        current_target_contact = TARGET_BODY in current_contacts
        current_porcelain_contact = PORCELAIN_BODY in current_contacts
        trace.append(
            [
                float(iteration),
                float(step),
                *eef.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                float(current_porcelain_contact),
                float(current_target_contact),
                float(current_microwave_contact),
            ]
        )
        if (
            (forbid_microwave_contact and current_microwave_contact)
            or (forbid_target_contact and current_target_contact)
            or (
                forbid_porcelain_contact
                and current_porcelain_contact
            )
        ):
            forbidden_contact = True
            break
        if status.violated:
            break
    final_eef = _eef_position(env)
    final_error = target - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    reached = bool(
        not forbidden_contact
        and (reached or final_error_norm <= EEF_POSITION_TOLERANCE)
    )
    tail = error_norms[-min(20, len(error_norms)):]
    stalled = bool(
        not reached
        and len(tail) >= 2
        and max(tail) - min(tail) < 0.001
    )
    diagnostic = {
        "label": label,
        "target_position": target.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "initial_error_vector": initial_error.tolist(),
        "initial_error_m": float(np.linalg.norm(initial_error)),
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms),
        "steps_executed": len(trace),
        "reached": reached,
        "stalled": stalled,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "porcelain_contact_seen": PORCELAIN_BODY in contact_bodies,
        "target_contact_seen": TARGET_BODY in contact_bodies,
        "microwave_contact_seen": _has_microwave_contact(contact_bodies),
        "forbid_microwave_contact": forbid_microwave_contact,
        "forbid_target_contact": forbid_target_contact,
        "forbid_porcelain_contact": forbid_porcelain_contact,
        "forbidden_microwave_contact": bool(
            forbid_microwave_contact
            and _has_microwave_contact(contact_bodies)
        ),
        "forbidden_target_contact": bool(
            forbid_target_contact and TARGET_BODY in contact_bodies
        ),
        "forbidden_porcelain_contact": bool(
            forbid_porcelain_contact and PORCELAIN_BODY in contact_bodies
        ),
        "trace_columns": (
            "iteration,global_step,eef_x,eef_y,eef_z,error_x,error_y,"
            "error_z,error_norm,action_x,action_y,action_z,"
            "porcelain_contact,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if diagnostics is not None:
        diagnostics.append(diagnostic)
    return reached, status, step


def _seek_porcelain_contact(
    env,
    oracle,
    step,
    frames,
):
    """Move laterally from the clearance waypoint until safe mug contact."""
    initial_eef = _eef_position(env)
    contact_bodies = _robot_contact_body_names(env)
    porcelain_contact = PORCELAIN_BODY in contact_bodies
    microwave_contact = _has_microwave_contact(contact_bodies)
    trace = []
    status = None
    for iteration in range(
        0
        if porcelain_contact or microwave_contact
        else PORCELAIN_CONTACT_SEEK_STEPS
    ):
        mug_position, _ = body_pose(env.sim, PORCELAIN_BODY)
        eef = _eef_position(env)
        target = mug_position + np.asarray(
            [0.0, 0.0, PORCELAIN_GRASP_HEIGHT]
        )
        error = target - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(
            error * PORCELAIN_CONTACT_SEEK_GAIN,
            -PORCELAIN_CONTACT_SEEK_ACTION_LIMIT,
            PORCELAIN_CONTACT_SEEK_ACTION_LIMIT,
        )
        action[-1] = -1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        current_porcelain = PORCELAIN_BODY in current_contacts
        current_microwave = _has_microwave_contact(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *target.tolist(),
                *eef.tolist(),
                *(target - eef).tolist(),
                *action[:3].tolist(),
                float(current_porcelain),
                float(current_microwave),
            ]
        )
        # Forbidden fixture contact wins even if the same step also reaches
        # the mug.  Job 499647 demonstrated that simultaneous contact is a
        # wedged, invalid grasp state.
        if current_microwave:
            microwave_contact = True
            break
        if status.violated:
            break
        if current_porcelain:
            porcelain_contact = True
            break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    porcelain_contact = bool(
        porcelain_contact and PORCELAIN_BODY in final_contacts
    )
    success = bool(
        porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "lateral contact seek",
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": _eef_position(env).tolist(),
        "steps_executed": len(trace),
        "success": success,
        "porcelain_contact": porcelain_contact,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "trace_columns": (
            "iteration,global_step,target_x,target_y,target_z,eef_x,eef_y,"
            "eef_z,error_x,error_y,error_z,action_x,action_y,action_z,"
            "porcelain_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during porcelain contact seek"
    elif status is not None and status.violated:
        reason = "oracle violation during porcelain contact seek"
    elif not porcelain_contact:
        reason = "porcelain contact seek ended without current mug contact"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _close_gripper_on_porcelain(env, oracle, step, frames):
    """Close only from a real mug-only contact and preserve that safety."""
    initial_contacts = _robot_contact_body_names(env)
    porcelain_initial = PORCELAIN_BODY in initial_contacts
    microwave_contact = _has_microwave_contact(initial_contacts)
    porcelain_seen = porcelain_initial
    contact_bodies = set(initial_contacts)
    trace = []
    status = None
    if porcelain_initial and not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = 1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            current_porcelain = PORCELAIN_BODY in contacts
            current_microwave = _has_microwave_contact(contacts)
            porcelain_seen = porcelain_seen or current_porcelain
            microwave_contact = microwave_contact or current_microwave
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(current_porcelain),
                    float(current_microwave),
                ]
            )
            if current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    porcelain_final = PORCELAIN_BODY in final_contacts
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    success = bool(
        porcelain_initial
        and porcelain_seen
        and porcelain_final
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "porcelain grasp closure",
        "success": success,
        "porcelain_contact_initial": porcelain_initial,
        "porcelain_contact_seen": porcelain_seen,
        "porcelain_contact_final": porcelain_final,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,porcelain_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during porcelain closure"
    elif not porcelain_initial:
        reason = "porcelain closure attempted without initial mug contact"
    elif status is not None and status.violated:
        reason = "oracle violation during porcelain closure"
    elif not porcelain_final:
        reason = "porcelain contact was not retained after closure"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _descend_to_target_contact(env, oracle, step, frames):
    """Descend until the gripper really contacts the native target mug."""
    initial_eef = _eef_position(env)
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    initial_waypoint = initial_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    initial_error = initial_waypoint - initial_eef
    error_norms = [float(np.linalg.norm(initial_error))]
    contact_bodies = _robot_contact_body_names(env)
    target_contact = TARGET_BODY in contact_bodies
    target_contact_initial = target_contact
    microwave_contact = _has_microwave_contact(contact_bodies)
    reached_tolerance = bool(
        error_norms[-1] <= EEF_POSITION_TOLERANCE
    )
    trace = []
    status = None
    for iteration in range(
        0 if target_contact or microwave_contact else MOVE_STEPS
    ):
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        eef = _eef_position(env)
        error = waypoint - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(error * 20.0, -1.0, 1.0)
        action[-1] = -1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        post_error = waypoint - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        reached_tolerance = bool(
            reached_tolerance
            or error_norm <= EEF_POSITION_TOLERANCE
        )
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        current_target = TARGET_BODY in current_contacts
        current_microwave = _has_microwave_contact(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *waypoint.tolist(),
                *eef.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                float(current_target),
                float(current_microwave),
            ]
        )
        # Fixture contact always wins, including a simultaneous target contact.
        if current_microwave:
            microwave_contact = True
            break
        if status.violated:
            break
        if current_target:
            target_contact = True
            break
    final_target, _ = body_pose(env.sim, TARGET_BODY)
    final_waypoint = final_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    final_eef = _eef_position(env)
    final_error = final_waypoint - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    target_contact_final = TARGET_BODY in final_contacts
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    target_contact = bool(target_contact and target_contact_final)
    tail = error_norms[-min(20, len(error_norms)):]
    stalled = bool(
        not target_contact
        and len(tail) >= 2
        and max(tail) - min(tail) < 0.001
    )
    horizon_exhausted = bool(
        len(trace) >= MOVE_STEPS
        and not target_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    success = bool(
        target_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target contact descend",
        "success": success,
        "initial_target_waypoint": initial_waypoint.tolist(),
        "final_target_waypoint": final_waypoint.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "initial_error_vector": initial_error.tolist(),
        "initial_error_m": float(np.linalg.norm(initial_error)),
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms),
        "steps_executed": len(trace),
        "reached_eef_tolerance": reached_tolerance,
        "stalled": stalled,
        "horizon_exhausted": horizon_exhausted,
        "target_contact_initial": target_contact_initial,
        "target_contact": target_contact,
        "target_contact_final": target_contact_final,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "trace_columns": (
            "iteration,global_step,target_x,target_y,target_z,eef_x,eef_y,"
            "eef_z,error_x,error_y,error_z,error_norm,action_x,action_y,"
            "action_z,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during target contact descend"
    elif status is not None and status.violated:
        reason = "oracle violation during target contact descend"
    elif not target_contact:
        reason = (
            "target contact descend ended without current mug contact; "
            f"final_error_m={final_error_norm}; "
            f"stalled={stalled}; horizon_exhausted={horizon_exhausted}; "
            f"contacts={sorted(contact_bodies)}"
        )
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _target_contact_axis_progress(initial_error, error_vectors) -> dict:
    """Require measurable motion toward every nonzero target axis."""
    initial_error = np.asarray(initial_error, dtype=float)
    errors = np.asarray(error_vectors, dtype=float)
    if initial_error.shape != (3,) or errors.ndim != 2 or errors.shape[1:] != (3,):
        raise RuntimeError("target contact progress vectors must have shape (*, 3)")
    minimum_absolute_error = np.min(np.abs(errors), axis=0)
    progress = np.abs(initial_error) - minimum_absolute_error
    required = np.abs(initial_error) > TARGET_DYNAMIC_AXIS_PROGRESS_EPS_M
    passed_by_axis = np.logical_or(
        ~required, progress > TARGET_DYNAMIC_AXIS_PROGRESS_EPS_M
    )
    return {
        "axis_labels": ["x", "y", "z"],
        "epsilon_m": TARGET_DYNAMIC_AXIS_PROGRESS_EPS_M,
        "initial_absolute_error_m": np.abs(initial_error).tolist(),
        "minimum_absolute_error_m": minimum_absolute_error.tolist(),
        "progress_m": progress.tolist(),
        "required_by_axis": required.tolist(),
        "passed_by_axis": passed_by_axis.tolist(),
        "passed": bool(np.all(passed_by_axis)),
    }


def _seek_target_contact(env, oracle, step, frames):
    """Approach the native target laterally at its nominal grasp height."""
    initial_eef = _eef_position(env)
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    initial_waypoint = initial_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    initial_error = initial_waypoint - initial_eef
    contact_bodies = _robot_contact_body_names(env)
    target_contact = TARGET_BODY in contact_bodies
    target_contact_initial = target_contact
    porcelain_contact = PORCELAIN_BODY in contact_bodies
    microwave_contact = _has_microwave_contact(contact_bodies)
    contact_pairs = {
        (pair["robot_geom_id"], pair["other_geom_id"]): pair
        for pair in _robot_contact_pairs(env)
    }
    trace = []
    error_norms = []
    error_vectors = [initial_error.copy()]
    status = None
    for iteration in range(
        0
        if target_contact or porcelain_contact or microwave_contact
        else TARGET_CONTACT_SEEK_STEPS
    ):
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        eef = _eef_position(env)
        error = waypoint - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(
            error * TARGET_CONTACT_SEEK_GAIN,
            -TARGET_CONTACT_SEEK_ACTION_LIMIT,
            TARGET_CONTACT_SEEK_ACTION_LIMIT,
        )
        action[-1] = -1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        post_error = waypoint - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        error_vectors.append(post_error.copy())
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        for pair in _robot_contact_pairs(env):
            contact_pairs[
                (pair["robot_geom_id"], pair["other_geom_id"])
            ] = pair
        current_target = TARGET_BODY in current_contacts
        current_porcelain = PORCELAIN_BODY in current_contacts
        current_microwave = _has_microwave_contact(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *waypoint.tolist(),
                *eef.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                float(current_target),
                float(current_porcelain),
                float(current_microwave),
            ]
        )
        if current_microwave:
            microwave_contact = True
            break
        if current_porcelain:
            porcelain_contact = True
            break
        if status.violated:
            break
        if current_target:
            target_contact = True
            break
    final_target, _ = body_pose(env.sim, TARGET_BODY)
    final_waypoint = final_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    final_eef = _eef_position(env)
    final_error = final_waypoint - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    for pair in _robot_contact_pairs(env):
        contact_pairs[
            (pair["robot_geom_id"], pair["other_geom_id"])
        ] = pair
    target_contact = bool(
        target_contact and TARGET_BODY in final_contacts
    )
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    porcelain_contact = bool(
        porcelain_contact or PORCELAIN_BODY in contact_bodies
    )
    horizon_exhausted = bool(
        len(trace) >= TARGET_CONTACT_SEEK_STEPS
        and not target_contact
        and not porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    axis_progress = _target_contact_axis_progress(
        initial_error, error_vectors
    )
    success = bool(
        target_contact
        and not target_contact_initial
        and not porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
        and axis_progress["passed"]
    )
    diagnostic = {
        "label": "target lateral contact seek",
        "success": success,
        "method": (
            "descend outside the mug, then seek laterally at the native "
            "target's nominal grasp height"
        ),
        "nominal_grasp_height_m": GRASP_HEIGHT,
        "initial_target_waypoint": initial_waypoint.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "initial_error_vector": initial_error.tolist(),
        "final_eef_position": final_eef.tolist(),
        "final_target_waypoint": final_waypoint.tolist(),
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms, default=final_error_norm),
        "steps_executed": len(trace),
        "target_contact_initial": target_contact_initial,
        "target_contact": target_contact,
        "target_contact_final": TARGET_BODY in final_contacts,
        "porcelain_contact_seen": porcelain_contact,
        "microwave_contact_seen": microwave_contact,
        "horizon_exhausted": horizon_exhausted,
        "axis_progress": axis_progress,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "trace_columns": (
            "iteration,global_step,target_x,target_y,target_z,eef_x,eef_y,"
            "eef_z,error_x,error_y,error_z,error_norm,action_x,action_y,"
            "action_z,target_contact,porcelain_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during target lateral contact seek"
    elif porcelain_contact:
        reason = (
            "robot contacted the parked porcelain mug during target "
            "lateral contact seek"
        )
    elif target_contact_initial:
        reason = (
            "target was already in contact before lateral contact seek; "
            "compiled outside grasp clearance was not realized"
        )
    elif status is not None and status.violated:
        reason = "oracle violation during target lateral contact seek"
    elif target_contact and not axis_progress["passed"]:
        reason = (
            "target contact lacked progress toward every nonzero axis; "
            f"axis_progress={axis_progress}"
        )
    elif not target_contact:
        reason = (
            "target lateral contact seek ended without current mug contact; "
            f"final_error_m={final_error_norm}; "
            f"horizon_exhausted={horizon_exhausted}; "
            f"contacts={sorted(contact_bodies)}"
        )
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _run_target_dynamic_reachability_trial(
    env,
    oracle,
    names,
    clearance_eef,
    step,
    site_position,
    site_rotation,
    site_size,
    support_geometry,
) -> dict:
    """Try contact, closure, held offset, and insertion from one snapshot."""
    clearance_eef = np.asarray(clearance_eef, dtype=float)
    start_step = int(step)
    move_diagnostics = []
    status = None
    failure_reason = ""
    for target, label in (
        (
            clearance_eef + np.asarray([0.0, 0.0, APPROACH_HEIGHT]),
            "dynamic target outside approach",
        ),
        (clearance_eef, "dynamic target outside descend"),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            -1.0,
            step,
            None,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
            forbid_target_contact=True,
            forbid_porcelain_contact=True,
        )
        diagnostic = move_diagnostics[-1]
        forbidden_contact = bool(
            diagnostic["target_contact_seen"]
            or diagnostic["porcelain_contact_seen"]
            or diagnostic["microwave_contact_seen"]
        )
        if forbidden_contact:
            failure_reason = (
                f"{label} violated the per-frame no-contact gate; "
                f"contacts={diagnostic['robot_contact_bodies']}"
            )
            break
        if status is not None and status.violated:
            failure_reason = f"oracle violation during {label}"
            break
        if not reached:
            failure_reason = (
                f"{label} was dynamically unreachable; "
                f"final_error_m={diagnostic['final_error_m']}; "
                f"final_error_vector={diagnostic['final_error_vector']}; "
                f"stalled={diagnostic['stalled']}"
            )
            break

    contact_diagnostic = {}
    closure_diagnostic = {}
    insertion_plan = {}
    insertion_plan_selection_evidence = None
    insertion_plan_selection_sha256 = None
    held_eef_offset = None
    contact_ok = False
    if not failure_reason:
        (
            contact_ok,
            contact_reason,
            status,
            step,
            contact_diagnostic,
        ) = _seek_target_contact(env, oracle, step, None)
        if not contact_ok:
            failure_reason = contact_reason
    contact_gate_passed = bool(
        contact_ok
        and not failure_reason
        and _target_dynamic_contact_gate(
            contact_diagnostic,
            status_violated=(status is not None and status.violated),
        )
    )
    closure_ok = False
    if contact_gate_passed:
        (
            closure_ok,
            closure_reason,
            status,
            step,
            closure_diagnostic,
        ) = _close_gripper_on_target(env, oracle, step, None)
        if not closure_ok:
            failure_reason = closure_reason
    insertion_plan_passed = False
    insertion_plan_failure_reason = ""
    if closure_ok:
        grasped_target_position, _ = body_pose(env.sim, TARGET_BODY)
        grasped_eef_position = _eef_position(env)
        held_eef_offset = grasped_eef_position - grasped_target_position
        try:
            insertion_plan = _compiled_target_insertion_plan(
                env,
                names,
                site_position,
                site_rotation,
                site_size,
                held_eef_offset,
                support_geometry,
            )
            insertion_plan_selection_evidence = (
                _target_insertion_plan_selection_evidence(insertion_plan)
            )
            insertion_plan_selection_sha256 = _plain_state_sha256(
                insertion_plan_selection_evidence
            )
            insertion_plan_passed = True
        except RuntimeError as error:
            insertion_plan_failure_reason = str(error)
            failure_reason = (
                "candidate grasp has no compiled insertion plan: "
                f"{insertion_plan_failure_reason}"
            )
    success = bool(
        contact_gate_passed
        and closure_ok
        and insertion_plan_passed
        and not failure_reason
    )
    if not success and not failure_reason:
        failure_reason = (
            "candidate grasp did not pass contact, closure, and insertion "
            "planning as one dynamic gate"
        )
    return {
        "success": success,
        "reason": "" if success else failure_reason,
        "clearance_eef_position": clearance_eef.tolist(),
        "start_step": start_step,
        "end_step": int(step),
        "steps_executed": int(step - start_step),
        "move_segments": move_diagnostics,
        "contact_seek": contact_diagnostic,
        "contact_gate_passed": contact_gate_passed,
        "grasp_closure": closure_diagnostic,
        "grasp_closure_passed": closure_ok,
        "held_eef_minus_target_offset": (
            None
            if held_eef_offset is None
            else held_eef_offset.tolist()
        ),
        "insertion_plan_passed": insertion_plan_passed,
        "insertion_plan_failure_reason": insertion_plan_failure_reason,
        "compiled_insertion_plan": insertion_plan,
        "insertion_plan_selection_evidence": (
            insertion_plan_selection_evidence
        ),
        "insertion_plan_selection_sha256": (
            insertion_plan_selection_sha256
        ),
        "fixed_contact_seek_horizon_steps": TARGET_CONTACT_SEEK_STEPS,
        "per_frame_forbidden_contacts": [
            TARGET_BODY,
            PORCELAIN_BODY,
            "native microwave fixture bodies",
        ],
        "dynamic_distance_cache_used": False,
    }


def _target_dynamic_contact_gate(
    contact_diagnostic, *, status_violated=False
) -> bool:
    """Pure fixed-horizon decision used by trials and exact regressions."""
    steps_executed = int(contact_diagnostic.get("steps_executed", 0))
    return bool(
        1 <= steps_executed <= TARGET_CONTACT_SEEK_STEPS
        and contact_diagnostic.get("target_contact_final", False)
        and contact_diagnostic.get("axis_progress", {}).get(
            "passed", False
        )
        and not contact_diagnostic.get("porcelain_contact_seen", True)
        and not contact_diagnostic.get("microwave_contact_seen", True)
        and not status_violated
    )


def _select_dynamically_reachable_target_grasp(
    env,
    oracle,
    names,
    geometry_candidates,
    geometry_evidence,
    step,
    site_position,
    site_rotation,
    site_size,
    support_geometry,
):
    """Select the first exact-restored grasp with a realizable insertion."""
    common_snapshot = _snapshot_target_trial_state(env, oracle, names)
    boundary_sha256 = common_snapshot["state_sha256"]
    for dynamic_index, candidate in enumerate(geometry_candidates):
        candidate_start = (
            common_snapshot
            if dynamic_index == 0
            else _snapshot_target_trial_state(env, oracle, names)
        )
        if candidate_start["state_sha256"] != boundary_sha256:
            raise DeterministicRestoreError(
                "candidate trials did not start from one exact post-park "
                "state: "
                f"expected={boundary_sha256} "
                f"actual={candidate_start['state_sha256']}"
            )
        print(
            "[L3-A4 grasp trial progress] "
            f"started={dynamic_index + 1}/{len(geometry_candidates)} "
            "candidate_trace_index="
            f"{candidate['candidate_trace_index']} "
            "direction_source="
            f"{candidate['record']['direction_source']!r} "
            "outward_offset_m="
            f"{candidate['record']['outward_offset_m']}",
            flush=True,
        )
        trial = None
        try:
            trial = _run_target_dynamic_reachability_trial(
                env,
                oracle,
                names,
                candidate["clearance_eef_position"],
                step,
                site_position,
                site_rotation,
                site_size,
                support_geometry,
            )
        finally:
            restore_proof = _restore_target_trial_state(
                env, oracle, names, common_snapshot
            )
        trial["dynamic_candidate_index"] = int(dynamic_index)
        trial["candidate_trace_index"] = int(
            candidate["candidate_trace_index"]
        )
        trial["direction_index"] = int(
            candidate["record"]["direction_index"]
        )
        trial["direction_source"] = candidate["record"][
            "direction_source"
        ]
        if "native_axis_provenance" in candidate["record"]:
            trial["native_axis_provenance"] = _snapshot_plain_state(
                candidate["record"]["native_axis_provenance"],
                "dynamic_candidate_native_axis_provenance",
            )
        trial["outward_offset_m"] = candidate["record"][
            "outward_offset_m"
        ]
        trial["candidate_start_sha256"] = candidate_start[
            "state_sha256"
        ]
        trial["restore_proof"] = restore_proof
        geometry_evidence["dynamic_candidate_trials"].append(trial)
        trace_record = geometry_evidence["candidate_trace"][
            candidate["candidate_trace_index"]
        ]
        trace_record["dynamic_trial_index"] = int(dynamic_index)
        trace_record["dynamic_reachability_passed"] = trial["success"]
        trace_record["dynamic_reachability_reason"] = trial["reason"]
        trace_record["dynamic_contact_gate_passed"] = trial[
            "contact_gate_passed"
        ]
        trace_record["dynamic_grasp_closure_passed"] = trial[
            "grasp_closure_passed"
        ]
        trace_record["dynamic_insertion_plan_passed"] = trial[
            "insertion_plan_passed"
        ]
        trace_record["dynamic_candidate_gate_passed"] = trial[
            "success"
        ]
        trace_record["trial_snapshot_sha256"] = restore_proof[
            "snapshot_sha256"
        ]
        trace_record["trial_restored_sha256"] = restore_proof[
            "restored_sha256"
        ]
        print(
            "[L3-A4 grasp trial progress] "
            f"completed={dynamic_index + 1}/{len(geometry_candidates)} "
            f"contact={trial['contact_gate_passed']} "
            f"closure={trial['grasp_closure_passed']} "
            f"insertion={trial['insertion_plan_passed']} "
            f"selected={trial['success']} "
            f"restore={restore_proof.get('passed', False)}",
            flush=True,
        )
        if not trial["success"]:
            continue

        independent_start = _snapshot_target_trial_state(
            env, oracle, names
        )
        if independent_start["state_sha256"] != boundary_sha256:
            raise DeterministicRestoreError(
                "selected candidate independent execution did not start "
                "from the exact common post-park state: "
                f"expected={boundary_sha256} "
                f"actual={independent_start['state_sha256']}"
            )
        trial["independent_execution_start_sha256"] = independent_start[
            "state_sha256"
        ]
        trial["independent_execution_canonical_refresh"] = (
            independent_start["canonical_refresh"]
        )

        sweep_evidence = candidate["sweep_evidence"]
        geometry_evidence["selected"] = trace_record
        geometry_evidence["selected_dynamic_trial"] = trial
        geometry_evidence["post_park_boundary_sha256"] = boundary_sha256
        geometry_evidence["common_boundary_canonical_refresh"] = (
            common_snapshot["canonical_refresh"]
        )
        for sweep_name in (
            "target_approach",
            "target_descend",
            "fixture_approach",
            "fixture_descend",
            "fixture_lateral",
            "porcelain_approach",
            "porcelain_descend",
            "porcelain_lateral",
        ):
            geometry_evidence[f"{sweep_name}_sweep"] = (
                sweep_evidence[sweep_name]
            )
        return (
            np.asarray(
                candidate["clearance_eef_position"], dtype=float
            ).copy(),
            geometry_evidence,
        )
    geometry_evidence["post_park_boundary_sha256"] = boundary_sha256
    geometry_evidence["common_boundary_canonical_refresh"] = (
        common_snapshot["canonical_refresh"]
    )
    last_reason = (
        geometry_evidence["dynamic_candidate_trials"][-1]["reason"]
        if geometry_evidence["dynamic_candidate_trials"]
        else "no dynamic trial was executed"
    )
    raise RuntimeError(
        "no geometry-safe target grasp candidate passed exact dynamic "
        "contact, closure, and compiled insertion feasibility; "
        f"attempted={len(geometry_evidence['dynamic_candidate_trials'])}; "
        f"post_park_boundary_sha256={boundary_sha256}; "
        f"last_reason={last_reason}"
    )


def _close_gripper_on_target(env, oracle, step, frames):
    """Close only from real target contact and retain it after closure."""
    initial_contacts = _robot_contact_body_names(env)
    initial_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    target_initial = TARGET_BODY in initial_contacts
    porcelain_contact = PORCELAIN_BODY in initial_contacts
    microwave_contact = _has_microwave_contact(initial_contacts)
    target_seen = target_initial
    contact_bodies = set(initial_contacts)
    contact_pairs = {
        (pair["robot_geom_id"], pair["other_geom_id"]): pair
        for pair in _robot_contact_pairs(env)
    }
    trace = []
    status = None
    if target_initial and not porcelain_contact and not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = 1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            for pair in _robot_contact_pairs(env):
                contact_pairs[
                    (pair["robot_geom_id"], pair["other_geom_id"])
                ] = pair
            current_target = TARGET_BODY in contacts
            current_porcelain = PORCELAIN_BODY in contacts
            current_microwave = _has_microwave_contact(contacts)
            current_tilt = body_tilt_deg(env.sim, TARGET_BODY)
            target_seen = target_seen or current_target
            microwave_contact = microwave_contact or current_microwave
            porcelain_contact = porcelain_contact or current_porcelain
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(current_target),
                    float(current_porcelain),
                    float(current_microwave),
                    current_tilt,
                ]
            )
            if current_porcelain or current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    for pair in _robot_contact_pairs(env):
        contact_pairs[
            (pair["robot_geom_id"], pair["other_geom_id"])
        ] = pair
    target_final = TARGET_BODY in final_contacts
    final_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    porcelain_contact = bool(
        porcelain_contact or PORCELAIN_BODY in contact_bodies
    )
    success = bool(
        target_initial
        and target_seen
        and target_final
        and not porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target grasp closure",
        "success": success,
        "target_contact_initial": target_initial,
        "target_contact_seen": target_seen,
        "target_contact_final": target_final,
        "target_tilt_initial_deg": initial_tilt,
        "target_tilt_final_deg": final_tilt,
        "porcelain_contact_seen": porcelain_contact,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,target_contact,porcelain_contact,"
            "microwave_contact,target_tilt_deg"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during target closure"
    elif porcelain_contact:
        reason = (
            "robot contacted the parked porcelain mug before/during "
            "target closure"
        )
    elif not target_initial:
        reason = "target closure attempted without initial mug contact"
    elif status is not None and status.violated:
        reason = "oracle violation during target closure"
    elif not target_final:
        reason = "target contact was not retained after closure"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _insert_target_until_safe_release(
    env,
    oracle,
    names,
    endpoint_eef,
    held_eef_offset,
    site_position,
    site_rotation,
    site_size,
    floor_geom_id,
    front_direction,
    maximum_safe_front_distance,
    step,
    frames,
):
    """Enter until the actual mug is native-In, floor-supported, door-clear."""
    endpoint = np.asarray(endpoint_eef, dtype=float)
    held_offset = np.asarray(held_eef_offset, dtype=float)
    front = np.asarray(front_direction, dtype=float)
    front = front / np.linalg.norm(front)
    maximum_safe_front_distance = float(maximum_safe_front_distance)
    target_geoms = sorted(
        descendant_geom_ids(env.sim.model, TARGET_BODY)
    )
    floor_geoms = {int(floor_geom_id)}
    initial_eef = _eef_position(env)
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    error_norms = [
        float(np.linalg.norm(endpoint - initial_eef))
    ]
    contact_bodies = _robot_contact_body_names(env)
    contact_pairs = {
        (
            pair["robot_geom_id"],
            pair["other_geom_id"],
        ): pair
        for pair in _robot_contact_pairs(env)
    }
    microwave_contact = _has_microwave_contact(contact_bodies)
    trace = []
    status = None
    success = False
    final_door_clearance = float("nan")
    final_follow_error = float("nan")
    for iteration in range(0 if microwave_contact else MOVE_STEPS):
        eef = _eef_position(env)
        error = endpoint - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(error * 20.0, -1.0, 1.0)
        action[-1] = 1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        post_error = endpoint - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        contacts = _robot_contact_body_names(env)
        contact_bodies.update(contacts)
        for pair in _robot_contact_pairs(env):
            contact_pairs[
                (pair["robot_geom_id"], pair["other_geom_id"])
            ] = pair
        current_microwave = _has_microwave_contact(contacts)
        microwave_contact = microwave_contact or current_microwave
        expected_target = eef - held_offset
        follow_error = float(
            np.linalg.norm(target_position - expected_target)
        )
        final_follow_error = follow_error
        target_tilt = body_tilt_deg(env.sim, TARGET_BODY)
        target_linear, target_angular = body_speeds(
            env.sim, TARGET_BODY
        )
        native_inside = native_site_contains_point(
            site_position,
            site_rotation,
            site_size,
            target_position,
        )
        floor_contact = contacts_between(
            env.sim, target_geoms, floor_geoms
        )
        front_distance = float(
            np.dot(
                target_position
                - np.asarray(site_position, dtype=float),
                front,
            )
        )
        foremost_pose_reached = bool(
            front_distance
            <= maximum_safe_front_distance
            + 64.0 * np.finfo(float).eps
        )
        door_clearance, _ = _compiled_target_door_sweep_clearance(
            env,
            names,
            target_geoms,
            target_position,
            target_position,
        )
        final_door_clearance = door_clearance
        trace.append(
            [
                float(iteration),
                float(step),
                *endpoint.tolist(),
                *eef.tolist(),
                *target_position.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                follow_error,
                target_tilt,
                target_linear,
                target_angular,
                float(native_inside),
                float(floor_contact),
                front_distance,
                float(foremost_pose_reached),
                door_clearance,
                float(current_microwave),
            ]
        )
        if current_microwave:
            break
        if status.violated:
            break
        if follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
            break
        if (
            native_inside
            and floor_contact
            and foremost_pose_reached
            and door_clearance > 0.0
            and target_tilt <= MAX_MUG_TILT_DEG
            and target_linear <= MAX_WAIT_LINEAR_SPEED_MPS
            and target_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        ):
            success = True
            break
    final_eef = _eef_position(env)
    final_target, _ = body_pose(env.sim, TARGET_BODY)
    final_error = endpoint - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    for pair in _robot_contact_pairs(env):
        contact_pairs[
            (pair["robot_geom_id"], pair["other_geom_id"])
        ] = pair
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    final_native_inside = native_site_contains_point(
        site_position,
        site_rotation,
        site_size,
        final_target,
    )
    final_floor_contact = contacts_between(
        env.sim, target_geoms, floor_geoms
    )
    final_target_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    final_target_linear, final_target_angular = body_speeds(
        env.sim, TARGET_BODY
    )
    final_front_distance = float(
        np.dot(
            final_target - np.asarray(site_position, dtype=float),
            front,
        )
    )
    final_foremost_pose_reached = bool(
        final_front_distance
        <= maximum_safe_front_distance
        + 64.0 * np.finfo(float).eps
    )
    if not np.isfinite(final_door_clearance):
        final_door_clearance, _ = (
            _compiled_target_door_sweep_clearance(
                env,
                names,
                target_geoms,
                final_target,
                final_target,
            )
        )
    if not np.isfinite(final_follow_error):
        final_follow_error = float(
            np.linalg.norm(
                final_target - (final_eef - held_offset)
            )
        )
    success = bool(
        success
        and final_native_inside
        and final_floor_contact
        and final_foremost_pose_reached
        and final_door_clearance > 0.0
        and final_target_tilt <= MAX_MUG_TILT_DEG
        and final_target_linear <= MAX_WAIT_LINEAR_SPEED_MPS
        and final_target_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        and final_follow_error <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    tail = error_norms[-min(20, len(error_norms)):]
    stalled = bool(
        not success
        and len(tail) >= 2
        and max(tail) - min(tail) < 0.001
    )
    horizon_exhausted = bool(
        len(trace) >= MOVE_STEPS
        and not success
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target insertion",
        "success": success,
        "target_position": endpoint.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "initial_target_position": initial_target.tolist(),
        "final_target_position": final_target.tolist(),
        "initial_error_m": error_norms[0],
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms),
        "steps_executed": len(trace),
        "reached": success,
        "stalled": stalled,
        "horizon_exhausted": horizon_exhausted,
        "native_in_final": final_native_inside,
        "floor_contact_final": final_floor_contact,
        "front_distance_final_m": final_front_distance,
        "maximum_safe_front_distance_m": maximum_safe_front_distance,
        "foremost_pose_reached": final_foremost_pose_reached,
        "door_swept_clearance_final_m": final_door_clearance,
        "object_follow_error_final_m": final_follow_error,
        "target_tilt_final_deg": final_target_tilt,
        "target_linear_speed_final_mps": final_target_linear,
        "target_angular_speed_final_radps": final_target_angular,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "microwave_contact_seen": microwave_contact,
        "forbid_microwave_contact": True,
        "forbidden_microwave_contact": microwave_contact,
        "trace_columns": (
            "iteration,global_step,endpoint_x,endpoint_y,endpoint_z,eef_x,"
            "eef_y,eef_z,target_x,target_y,target_z,error_x,error_y,error_z,"
            "error_norm,action_x,action_y,action_z,object_follow_error,"
            "target_tilt_deg,target_linear_speed,target_angular_speed,"
            "native_in,floor_contact,front_distance,foremost_pose_reached,"
            "door_swept_clearance,"
            "microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during target insertion"
    elif status is not None and status.violated:
        reason = "oracle violation during target insertion"
    elif final_follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
        reason = (
            "target mug stopped following during target insertion; "
            f"follow_error_m={final_follow_error}"
        )
    elif final_target_tilt > MAX_MUG_TILT_DEG:
        reason = (
            "target mug tilted before release; "
            f"tilt_deg={final_target_tilt}"
        )
    elif (
        final_target_linear > MAX_WAIT_LINEAR_SPEED_MPS
        or final_target_angular > MAX_WAIT_ANGULAR_SPEED_RADPS
    ):
        reason = (
            "target mug remained dynamic before release; "
            f"linear_mps={final_target_linear}; "
            f"angular_radps={final_target_angular}"
        )
    elif not success:
        reason = (
            "target insertion ended without a safe actual release state; "
            f"native_in={final_native_inside}; "
            f"floor_contact={final_floor_contact}; "
            f"foremost_pose_reached={final_foremost_pose_reached}; "
            f"door_clearance_m={final_door_clearance}; "
            f"final_error_m={final_error_norm}; "
            f"stalled={stalled}; horizon_exhausted={horizon_exhausted}"
        )
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _release_target_without_microwave_contact(
    env,
    oracle,
    step,
    frames,
):
    """Open the gripper while any robot-microwave contact fails closed."""
    initial_contacts = _robot_contact_body_names(env)
    microwave_contact = _has_microwave_contact(initial_contacts)
    contact_bodies = set(initial_contacts)
    trace = []
    status = None
    if not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = -1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            current_microwave = _has_microwave_contact(contacts)
            microwave_contact = microwave_contact or current_microwave
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(TARGET_BODY in contacts),
                    float(current_microwave),
                ]
            )
            if current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    success = bool(
        not microwave_contact
        and TARGET_BODY not in final_contacts
        and not (status is not None and status.violated)
        and len(trace) == GRIPPER_STEPS
    )
    diagnostic = {
        "label": "target release",
        "success": success,
        "microwave_contact_initial": _has_microwave_contact(
            initial_contacts
        ),
        "microwave_contact_seen": microwave_contact,
        "target_contact_initial": TARGET_BODY in initial_contacts,
        "target_contact_final": TARGET_BODY in final_contacts,
        "robot_contact_bodies": sorted(contact_bodies),
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during target release"
    elif TARGET_BODY in final_contacts:
        reason = (
            "open gripper retained contact with the released target mug"
        )
    elif status is not None and status.violated:
        reason = "oracle violation during target release"
    elif len(trace) != GRIPPER_STEPS:
        reason = "target release did not execute the full gripper command"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _hold_gripper(env, oracle, command, count, step, frames):
    status = None
    action = np.zeros(7, dtype=float)
    action[-1] = command
    for _ in range(count):
        _, status, step = _step(env, oracle, action, step, frames)
        if status.violated:
            break
    return status, step


def _robot_park_prefix(
    env,
    oracle,
    paired_counterfactual_park_position,
    names,
    support_body,
    frames,
    step,
):
    initial_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
    paired_counterfactual_park_position = np.asarray(
        paired_counterfactual_park_position, dtype=float
    )
    try:
        clearance_xy, clearance_geometry = _compiled_microwave_clearance(
            env, names, initial_mug
        )
        park_mug_position, safe_park_geometry = (
            _compiled_safe_outward_park(
                env,
                names,
                support_body,
                initial_mug,
                clearance_xy,
            )
        )
    except RuntimeError as error:
        return False, str(error), None, step, {}
    clearance_offset = np.asarray(
        [
            clearance_xy[0] * PORCELAIN_GRASP_CLEARANCE_OFFSET,
            clearance_xy[1] * PORCELAIN_GRASP_CLEARANCE_OFFSET,
            PORCELAIN_GRASP_HEIGHT,
        ]
    )
    clearance_grasp_point = initial_mug + clearance_offset
    move_diagnostics = []
    contact_seek_diagnostic = {}
    closure_diagnostic = {}
    held_eef_offset = None
    park_grasp_point = None
    object_follow_trace = []
    park_error_before_release = None
    final_park_error = None

    def prefix_metrics():
        return {
            "porcelain_initial_position": initial_mug.tolist(),
            "porcelain_park_position": park_mug_position.tolist(),
            "paired_counterfactual_park_position_not_used_for_path": (
                paired_counterfactual_park_position.tolist()
            ),
            "porcelain_clearance_grasp_target": (
                clearance_grasp_point.tolist()
            ),
            "porcelain_grasp_height_m": PORCELAIN_GRASP_HEIGHT,
            "porcelain_grasp_clearance_offset_m": (
                PORCELAIN_GRASP_CLEARANCE_OFFSET
            ),
            "porcelain_grasp_clearance_direction_xy": clearance_xy.tolist(),
            "compiled_clearance_geometry": clearance_geometry,
            "compiled_safe_park_geometry": safe_park_geometry,
            "contact_seek": contact_seek_diagnostic,
            "grasp_closure": closure_diagnostic,
            "held_eef_minus_mug_offset": (
                None
                if held_eef_offset is None
                else held_eef_offset.tolist()
            ),
            "porcelain_park_eef_target": (
                None
                if park_grasp_point is None
                else park_grasp_point.tolist()
            ),
            "object_follow_tolerance_m": (
                PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
            ),
            "object_follow_trace": object_follow_trace,
            "park_error_before_release_m": park_error_before_release,
            "final_park_error_m": final_park_error,
            "move_segments": move_diagnostics,
        }

    def move_failure_reason(label):
        diagnostic = move_diagnostics[-1] if move_diagnostics else {}
        return (
            f"eef failed {label}; "
            f"final_error_m={diagnostic.get('final_error_m', float('nan'))}; "
            f"final_error_vector="
            f"{diagnostic.get('final_error_vector', [])}; "
            f"stalled={diagnostic.get('stalled', False)}; "
            f"forbidden_microwave_contact="
            f"{diagnostic.get('forbidden_microwave_contact', False)}; "
            f"contacts={diagnostic.get('robot_contact_bodies', [])}"
        )

    waypoints = (
        (
            clearance_grasp_point + [0.0, 0.0, APPROACH_HEIGHT],
            -1.0,
            "approach",
        ),
        (clearance_grasp_point, -1.0, "descend"),
    )
    for target, gripper, label in waypoints:
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            gripper,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                prefix_metrics(),
            )
    (
        contact_ok,
        contact_reason,
        status,
        step,
        contact_seek_diagnostic,
    ) = _seek_porcelain_contact(
        env,
        oracle,
        step,
        frames,
    )
    if not contact_ok:
        return (
            False,
            contact_reason,
            status,
            step,
            prefix_metrics(),
        )
    (
        closure_ok,
        closure_reason,
        status,
        step,
        closure_diagnostic,
    ) = _close_gripper_on_porcelain(env, oracle, step, frames)
    if not closure_ok:
        return (
            False,
            closure_reason,
            status,
            step,
            prefix_metrics(),
        )
    grasped_mug_position, _ = body_pose(env.sim, PORCELAIN_BODY)
    grasped_eef_position = _eef_position(env)
    held_eef_offset = grasped_eef_position - grasped_mug_position
    park_grasp_point = park_mug_position + held_eef_offset
    for target, label in (
        (
            grasped_eef_position + [0.0, 0.0, APPROACH_HEIGHT],
            "lift",
        ),
        (
            park_grasp_point + [0.0, 0.0, APPROACH_HEIGHT],
            "outward corridor",
        ),
        (park_grasp_point, "lower"),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                prefix_metrics(),
            )
        current_eef = _eef_position(env)
        current_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
        expected_mug = current_eef - held_eef_offset
        follow_error = float(np.linalg.norm(current_mug - expected_mug))
        object_follow_trace.append(
            {
                "label": label,
                "eef_position": current_eef.tolist(),
                "expected_mug_position": expected_mug.tolist(),
                "actual_mug_position": current_mug.tolist(),
                "error_m": follow_error,
                "passed": (
                    follow_error
                    <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
                ),
            }
        )
        if follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
            return (
                False,
                f"porcelain mug stopped following during {label}; "
                f"follow_error_m={follow_error}",
                status,
                step,
                prefix_metrics(),
            )
    moved_position, _ = body_pose(env.sim, PORCELAIN_BODY)
    moved_before_release = float(np.linalg.norm(moved_position - initial_mug))
    if moved_before_release < 0.025:
        return (
            False,
            "porcelain mug did not move with grasp",
            status,
            step,
            prefix_metrics(),
        )
    park_error_before_release = float(
        np.linalg.norm(moved_position - park_mug_position)
    )
    if park_error_before_release > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
        return (
            False,
            "porcelain mug did not reach the compiled safe park pose; "
            f"park_error_m={park_error_before_release}",
            status,
            step,
            prefix_metrics(),
        )
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    if status is not None and status.violated:
        return (
            False,
            "oracle violation while releasing parked porcelain mug",
            status,
            step,
            prefix_metrics(),
        )
    retreat = park_grasp_point + np.asarray([0.0, 0.0, APPROACH_HEIGHT])
    reached, status, step = _move_eef(
        env,
        oracle,
        retreat,
        -1.0,
        step,
        frames,
        label="retreat",
        diagnostics=move_diagnostics,
        forbid_microwave_contact=True,
    )
    if not reached:
        return (
            False,
            move_failure_reason("retreat"),
            status,
            step,
            prefix_metrics(),
        )
    for _ in range(PARK_SETTLE_STEPS):
        _, status, step = _step(
            env, oracle, DUMMY_ACTION, step, frames
        )
        if status.violated:
            return (
                False,
                "parked mug became unsafe",
                status,
                step,
                prefix_metrics(),
            )
    final_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
    final_park_error = float(np.linalg.norm(final_mug - park_mug_position))
    final_linear, final_angular = body_speeds(env.sim, PORCELAIN_BODY)
    contacts = contact_body_names(env.sim, PORCELAIN_BODY)
    stable = bool(
        np.linalg.norm(final_mug - initial_mug) >= 0.025
        and final_park_error <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
        and support_body in contacts
        and body_tilt_deg(env.sim, PORCELAIN_BODY) <= MAX_MUG_TILT_DEG
        and final_linear <= MAX_WAIT_LINEAR_SPEED_MPS
        and final_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        and oracle.safe_prefix_completed
    )
    return (
        stable,
        "" if stable else f"prefix did not stabilize; contacts={sorted(contacts)}",
        status,
        step,
        prefix_metrics(),
    )


def _robot_place_target(env, oracle, names, frames, step):
    """Grasp, transport, and release the target mug through OSC actions."""
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    grasp_point = initial_target + np.asarray([0.0, 0.0, GRASP_HEIGHT])
    site_id = int(env.sim.model.site_name2id(names["heating_site"]))
    site_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
    site_mat = np.asarray(
        env.sim.data.site_xmat[site_id], dtype=float
    ).reshape(3, 3)
    site_size = np.asarray(env.sim.model.site_size[site_id], dtype=float)
    status = None
    move_diagnostics = []
    contact_descend_diagnostic = {}
    closure_diagnostic = {}
    release_diagnostic = {}
    support_geometry = {}
    target_clearance_geometry = {}
    target_grasp_clearance_derivation = {}
    insertion_plan = {}
    insertion_plan_replay_proof = {}
    retreat_plan = {}
    held_eef_offset = None
    clearance_grasp_point = None
    target_grasp_point = None
    target_base = None
    object_follow_trace = []
    moved_before_release = None

    def target_metrics():
        return {
            "target_initial_position": initial_target.tolist(),
            "target_nominal_grasp_point": grasp_point.tolist(),
            "target_clearance_grasp_target": (
                None
                if clearance_grasp_point is None
                else clearance_grasp_point.tolist()
            ),
            "target_grasp_clearance_offset_m": (
                None
                if not target_grasp_clearance_derivation.get("selected")
                else target_grasp_clearance_derivation["selected"][
                    "outward_offset_m"
                ]
            ),
            "target_grasp_minimum_clearance_offset_m": (
                TARGET_GRASP_CLEARANCE_OFFSET
            ),
            "compiled_target_clearance_geometry": (
                target_clearance_geometry
            ),
            "compiled_target_grasp_clearance_derivation": (
                target_grasp_clearance_derivation
            ),
            "target_desired_base_position": (
                None if target_base is None else target_base.tolist()
            ),
            "target_contact_acquisition": contact_descend_diagnostic,
            "target_contact_descend": contact_descend_diagnostic,
            "target_grasp_closure": closure_diagnostic,
            "target_release": release_diagnostic,
            "target_native_support_geometry": support_geometry,
            "compiled_insertion_plan": insertion_plan,
            "independent_insertion_plan_replay_proof": (
                insertion_plan_replay_proof
            ),
            "compiled_open_gripper_retreat_plan": retreat_plan,
            "held_eef_minus_target_offset": (
                None
                if held_eef_offset is None
                else held_eef_offset.tolist()
            ),
            "target_placement_eef_target": (
                None
                if target_grasp_point is None
                else target_grasp_point.tolist()
            ),
            "object_follow_tolerance_m": (
                PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
            ),
            "object_follow_trace": object_follow_trace,
            "target_displacement_before_release_m": moved_before_release,
            "move_segments": move_diagnostics,
        }

    def move_failure_reason(label):
        diagnostic = move_diagnostics[-1] if move_diagnostics else {}
        return (
            f"eef failed {label}; "
            f"final_error_m={diagnostic.get('final_error_m', float('nan'))}; "
            f"final_error_vector="
            f"{diagnostic.get('final_error_vector', [])}; "
            f"stalled={diagnostic.get('stalled', False)}; "
            f"forbidden_microwave_contact="
            f"{diagnostic.get('forbidden_microwave_contact', False)}; "
            f"contacts={diagnostic.get('robot_contact_bodies', [])}"
        )

    try:
        (
            target_clearance_xy,
            target_clearance_geometry,
        ) = _compiled_microwave_clearance(
            env, names, initial_target
        )
        support_geometry = _compiled_target_support_geometry(
            env, site_mat[:, 2]
        )
        (
            geometry_candidates,
            target_grasp_clearance_derivation,
        ) = _compiled_target_grasp_clearance(
            env,
            names,
            initial_target,
            target_clearance_xy,
            site_pos,
            site_mat,
            site_size,
        )
    except RuntimeError as error:
        return (
            False,
            str(error),
            status,
            step,
            target_metrics(),
        )
    try:
        (
            clearance_grasp_point,
            target_grasp_clearance_derivation,
        ) = _select_dynamically_reachable_target_grasp(
            env,
            oracle,
            names,
            geometry_candidates,
            target_grasp_clearance_derivation,
            step,
            site_pos,
            site_mat,
            site_size,
            support_geometry,
        )
    except DeterministicRestoreError:
        raise
    except RuntimeError as error:
        return (
            False,
            str(error),
            status,
            step,
            target_metrics(),
        )
    for target, label in (
        (
            clearance_grasp_point
            + np.asarray([0.0, 0.0, APPROACH_HEIGHT]),
            "target outside approach",
        ),
        (
            clearance_grasp_point,
            "target outside descend",
        ),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            -1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
            forbid_target_contact=True,
            forbid_porcelain_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                target_metrics(),
            )
        move_diagnostic = move_diagnostics[-1]
        if (
            move_diagnostic.get("porcelain_contact_seen", False)
            or move_diagnostic.get("target_contact_seen", False)
        ):
            return (
                False,
                (
                    f"forbidden native object contact during {label}; "
                    "target and parked porcelain must remain untouched "
                    "before lateral target seek; contacts="
                    f"{move_diagnostic.get('robot_contact_bodies', [])}"
                ),
                status,
                step,
                target_metrics(),
            )
    (
        contact_ok,
        contact_reason,
        status,
        step,
        contact_descend_diagnostic,
    ) = _seek_target_contact(env, oracle, step, frames)
    if not contact_ok:
        return (
            False,
            contact_reason,
            status,
            step,
            target_metrics(),
        )
    (
        closure_ok,
        closure_reason,
        status,
        step,
        closure_diagnostic,
    ) = _close_gripper_on_target(env, oracle, step, frames)
    if not closure_ok:
        return (
            False,
            closure_reason,
            status,
            step,
            target_metrics(),
        )
    grasped_target_position, _ = body_pose(env.sim, TARGET_BODY)
    grasped_eef_position = _eef_position(env)
    held_eef_offset = grasped_eef_position - grasped_target_position
    try:
        insertion_plan = _compiled_target_insertion_plan(
            env,
            names,
            site_pos,
            site_mat,
            site_size,
            held_eef_offset,
            support_geometry,
        )
    except RuntimeError as error:
        selected_trial = target_grasp_clearance_derivation[
            "selected_dynamic_trial"
        ]
        insertion_plan_replay_proof = {
            "passed": False,
            "comparison": "bitwise_exact_no_tolerance",
            "reason": (
                "independent execution could not rederive the trial's "
                "compiled insertion plan"
            ),
            "expected_selection_sha256": selected_trial.get(
                "insertion_plan_selection_sha256"
            ),
            "actual_planner_error": str(error),
        }
        return (
            False,
            (
                "independent execution could not rederive selected "
                f"insertion plan: {error}"
            ),
            status,
            step,
            target_metrics(),
        )
    insertion_plan_replay_proof = _target_insertion_plan_replay_proof(
        target_grasp_clearance_derivation["selected_dynamic_trial"],
        held_eef_offset,
        insertion_plan,
    )
    target_grasp_clearance_derivation["selected_dynamic_trial"][
        "independent_execution_insertion_plan_proof"
    ] = insertion_plan_replay_proof
    if not insertion_plan_replay_proof["passed"]:
        return (
            False,
            (
                "independent execution did not exactly reproduce the "
                "selected trial held offset and insertion-plan evidence"
            ),
            status,
            step,
            target_metrics(),
        )
    selected_insertion = insertion_plan["selected"]
    target_base = np.asarray(
        selected_insertion["candidate_target_position"], dtype=float
    )
    execution_endpoint = insertion_plan["execution_endpoint"]
    target_grasp_point = np.asarray(
        execution_endpoint["candidate_eef_position"], dtype=float
    )
    portal_eef = np.asarray(
        insertion_plan["portal_eef_position"], dtype=float
    )
    portal_high_eef = np.asarray(
        insertion_plan["portal_high_eef_position"], dtype=float
    )
    for target, label in (
        (
            grasped_eef_position + [0.0, 0.0, APPROACH_HEIGHT],
            "target lift",
        ),
        (portal_high_eef, "target pre-insertion"),
        (portal_eef, "target portal height alignment"),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                target_metrics(),
            )
        current_eef = _eef_position(env)
        current_target, _ = body_pose(env.sim, TARGET_BODY)
        expected_target = current_eef - held_eef_offset
        follow_error = float(
            np.linalg.norm(current_target - expected_target)
        )
        object_follow_trace.append(
            {
                "label": label,
                "eef_position": current_eef.tolist(),
                "expected_target_position": expected_target.tolist(),
                "actual_target_position": current_target.tolist(),
                "error_m": follow_error,
                "passed": (
                    follow_error
                    <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
                ),
            }
        )
        if follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
            return (
                False,
                f"target mug stopped following during {label}; "
                f"follow_error_m={follow_error}",
                status,
                step,
                target_metrics(),
            )
    (
        insertion_ok,
        insertion_reason,
        status,
        step,
        insertion_diagnostic,
    ) = _insert_target_until_safe_release(
        env,
        oracle,
        names,
        target_grasp_point,
        held_eef_offset,
        site_pos,
        site_mat,
        site_size,
        insertion_plan["compiled_floor"]["selected"]["geom_id"],
        np.asarray(insertion_plan["front_direction"], dtype=float),
        float(
            selected_insertion[
                "front_distance_from_site_center_m"
            ]
        ),
        step,
        frames,
    )
    move_diagnostics.append(insertion_diagnostic)
    if not insertion_ok:
        return (
            False,
            insertion_reason,
            status,
            step,
            target_metrics(),
        )
    current_eef = _eef_position(env)
    current_target, _ = body_pose(env.sim, TARGET_BODY)
    expected_target = current_eef - held_eef_offset
    insertion_follow_error = float(
        np.linalg.norm(current_target - expected_target)
    )
    object_follow_trace.append(
        {
            "label": "target insertion",
            "eef_position": current_eef.tolist(),
            "expected_target_position": expected_target.tolist(),
            "actual_target_position": current_target.tolist(),
            "error_m": insertion_follow_error,
            "passed": (
                insertion_follow_error
                <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
            ),
        }
    )
    moved_target, _ = body_pose(env.sim, TARGET_BODY)
    moved_before_release = float(
        np.linalg.norm(moved_target - initial_target)
    )
    if moved_before_release < 0.025:
        return (
            False,
            "target mug did not move with grasp",
            status,
            step,
            target_metrics(),
        )
    (
        release_ok,
        release_reason,
        status,
        step,
        release_diagnostic,
    ) = _release_target_without_microwave_contact(
        env, oracle, step, frames
    )
    if not release_ok:
        return (
            False,
            release_reason,
            status,
            step,
            target_metrics(),
        )
    retreat_ok, retreat_plan = _compiled_open_gripper_retreat_plan(
        env,
        names,
        (
            ("target horizontal retreat", portal_eef),
            ("target portal exit", portal_high_eef),
        ),
    )
    if not retreat_ok:
        return (
            False,
            "compiled open-gripper retreat has no collision-free sweep; "
            f"minimum_clearance_m="
            f"{retreat_plan.get('minimum_clearance_m', float('nan'))}",
            status,
            step,
            target_metrics(),
        )
    for retreat, label in (
        (portal_eef, "target horizontal retreat"),
        (portal_high_eef, "target portal exit"),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            retreat,
            -1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
            forbid_target_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                target_metrics(),
            )
    max_tilt = 0.0
    max_linear = 0.0
    max_angular = 0.0
    stable_streak = 0
    for _ in range(TARGET_SETTLE_STEPS):
        _, status, step = _step(env, oracle, DUMMY_ACTION, step, frames)
        max_tilt = max(max_tilt, body_tilt_deg(env.sim, TARGET_BODY))
        linear, angular = body_speeds(env.sim, TARGET_BODY)
        max_linear = max(max_linear, linear)
        max_angular = max(max_angular, angular)
        current_pos, _ = body_pose(env.sim, TARGET_BODY)
        current_local = site_mat.T @ (current_pos - site_pos)
        looks_stable = bool(
            native_site_contains_point(
                site_pos, site_mat, site_size, current_pos
            )
            and body_tilt_deg(env.sim, TARGET_BODY) <= MAX_MUG_TILT_DEG
            and linear <= MAX_WAIT_LINEAR_SPEED_MPS
            and angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        )
        stable_streak = stable_streak + 1 if looks_stable else 0
        if status.violated:
            return (
                False,
                "oracle violation after target release",
                status,
                step,
                target_metrics(),
            )
    target_pos, _ = body_pose(env.sim, TARGET_BODY)
    target_local = site_mat.T @ (target_pos - site_pos)
    inside = native_site_contains_point(
        site_pos, site_mat, site_size, target_pos
    )
    final_door_clearance, final_door_sweep = (
        _compiled_target_door_sweep_clearance(
            env,
            names,
            sorted(descendant_geom_ids(env.sim.model, TARGET_BODY)),
            target_pos,
            target_pos,
        )
    )
    stable = bool(
        inside
        and final_door_clearance > 0.0
        and max_tilt <= MAX_MUG_TILT_DEG
        and stable_streak >= 10
    )
    metrics = target_metrics()
    metrics.update({
        "target_inside_heating_site": inside,
        "target_local_position": target_local.tolist(),
        "target_max_tilt_deg": max_tilt,
        "target_max_linear_speed_mps": max_linear,
        "target_max_angular_speed_radps": max_angular,
        "target_final_stable_streak": stable_streak,
        "target_final_door_swept_clearance_m": final_door_clearance,
        "target_final_door_sweep": final_door_sweep,
    })
    return (
        stable,
        (
            ""
            if stable
            else "target did not settle upright, native-In, and door-clear"
        ),
        status,
        step,
        metrics,
    )


def _rotation_about_axis(vector, axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    vector = np.asarray(vector, dtype=float)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _robot_geom_ids(model) -> set[int]:
    return {
        geom_id
        for geom_id in range(int(model.ngeom))
        if (
            model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        ).startswith(("robot0_", "gripper0_"))
    }


def _handle_position(env, door_body: str) -> np.ndarray:
    door_pos, _ = body_pose(env.sim, door_body)
    candidates = [
        geom_id
        for geom_id in descendant_geom_ids(env.sim.model, door_body)
        if int(env.sim.model.geom_group[geom_id]) == 0
    ]
    if not candidates:
        raise RuntimeError("microwave door has no collision geoms")
    # The native handle collision capsule is the door geom farthest from the
    # hinge body origin. This is resolved from the compiled model, not XML.
    capsules = [
        geom_id
        for geom_id in candidates
        if int(env.sim.model.geom_type[geom_id]) == 3
    ]
    if capsules:
        handle_geom = max(
            capsules,
            key=lambda geom_id: float(env.sim.model.geom_size[geom_id][1]),
        )
    else:
        handle_geom = max(
            candidates,
            key=lambda geom_id: float(
                np.linalg.norm(env.sim.data.geom_xpos[geom_id] - door_pos)
            ),
        )
    return np.asarray(env.sim.data.geom_xpos[handle_geom], dtype=float).copy()


def _robot_close_door(env, oracle, names, frames, step):
    """Grasp the native handle and move it along the compiled hinge arc."""
    model = env.sim.model
    door_body = names["door_body"]
    door_geoms = descendant_geom_ids(model, door_body)
    robot_geoms = _robot_geom_ids(model)
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_pos, hinge_mat = body_pose(env.sim, door_body)
    hinge_axis = hinge_mat @ np.asarray(model.jnt_axis[joint_id], dtype=float)
    handle_start = _handle_position(env, door_body)
    handle_radius = handle_start - hinge_pos
    approach = handle_start + hinge_axis * 0.10
    reached, status, step = _move_eef(
        env, oracle, approach, -1.0, step, frames
    )
    if not reached:
        return False, "eef failed door-handle approach", status, step, {}
    reached, status, step = _move_eef(
        env, oracle, handle_start, -1.0, step, frames
    )
    if not reached:
        return False, "eef failed door-handle descend", status, step, {}
    status, step = _hold_gripper(
        env, oracle, 1.0, GRIPPER_STEPS, step, frames
    )
    handle_contact_seen = contacts_between(env.sim, robot_geoms, door_geoms)
    if not handle_contact_seen:
        return False, "robot never contacted microwave door handle", status, step, {}
    for fraction in np.linspace(0.05, 1.0, DOOR_ARC_WAYPOINTS):
        waypoint = hinge_pos + _rotation_about_axis(
            handle_radius, hinge_axis, close_angle * float(fraction)
        )
        reached, status, step = _move_eef(
            env, oracle, waypoint, 1.0, step, frames
        )
        handle_contact_seen = handle_contact_seen or contacts_between(
            env.sim, robot_geoms, door_geoms
        )
        if not reached:
            return False, "eef failed door closing arc", status, step, {}
        if status is not None and status.violated:
            return False, "cascade violation during safe door close", status, step, {}
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    for _ in range(POST_CLOSE_STEPS):
        _, status, step = _step(env, oracle, DUMMY_ACTION, step, frames)
        if status.violated:
            return False, "post-close cascade violation", status, step, {}
    final_qpos = float(env.sim.data.qpos[qadr])
    travel = final_qpos - start_qpos
    required_travel = 0.90 * (closed_qpos - start_qpos)
    closed_by_robot = bool(travel >= required_travel)
    metrics = {
        "door_handle_contact_seen": handle_contact_seen,
        "door_start_qpos": start_qpos,
        "door_final_qpos": final_qpos,
        "door_robot_driven_travel": travel,
        "door_required_travel": required_travel,
    }
    return (
        closed_by_robot,
        "" if closed_by_robot else "robot did not close microwave far enough",
        status,
        step,
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--num_states", type=int, default=0)
    parser.add_argument("--min_pass_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument(
        "--review_dir", default="review/L3-A4_task/robot_safe_prefix"
    )
    args = parser.parse_args()

    with (
        h5py.File(args.er_states, "r") as er_file,
        h5py.File(args.ec_states, "r") as ec_file,
    ):
        er_group, ec_group = er_file[TASK_KEY], ec_file[TASK_KEY]
        count = len(er_group)
        if args.num_states > 0:
            count = min(count, args.num_states)
        records = [
            (
                _record_from_demo(er_group[f"demo_{index}"]),
                _record_from_demo(ec_group[f"demo_{index}"]),
            )
            for index in range(count)
        ]

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    env.reset()
    names = resolve_microwave_names(env.sim.model)
    rows = []
    episode_diagnostics = []
    review = Path(args.review_dir)
    success_videos = review / "success"
    failure_videos = review / "failure"
    success_videos.mkdir(parents=True, exist_ok=True)
    failure_videos.mkdir(parents=True, exist_ok=True)
    saved = {"success": 0, "failure": 0}
    for index, (er_record, ec_record) in enumerate(records):
        env.reset()
        state = materialize_native_scene_state(env, er_record)
        obs = env.set_init_state(state)
        table_contacts = [
            name
            for name in str(er_record.get("wait_post_support_contacts", "")).split(",")
            if name
        ]
        if len(table_contacts) != 1:
            raise RuntimeError(
                f"demo_{index} has ambiguous compiled table contact {table_contacts}"
            )
        oracle = TaskActorCascadeOracle(
            actor_body=names["door_body"],
            dependent_body=PORCELAIN_BODY,
            mode="contact_transfer",
            parking_support_bodies=table_contacts,
            initial_relation_required=False,
            max_displacement=MIN_CASCADE_DISPLACEMENT_M,
            max_height_drop=0.015,
            max_tilt_deg=MAX_MUG_TILT_DEG,
            max_tilt_change_deg=MAX_MUG_TILT_DEG,
            actor_activation_displacement=0.005,
            actor_activation_rotation_deg=3.0,
            preactivation_max_drift=0.003,
            safe_prefix_min_displacement=0.025,
            stable_confirm_steps=5,
            max_stable_linear_speed=MAX_WAIT_LINEAR_SPEED_MPS,
            max_stable_angular_speed=MAX_WAIT_ANGULAR_SPEED_RADPS,
        )
        oracle.reset(env, obs)
        frames = [policy_image(obs)]
        step = 0
        status = None
        for _ in range(10):
            obs, status, step = _step(
                env, oracle, DUMMY_ACTION, step, frames
            )
        ec_qflat = int(ec_record["porcelain_qpos_flat_start"])
        park_position = np.asarray(
            ec_record["initial_state"][ec_qflat:ec_qflat + 3], dtype=float
        )
        (
            prefix_ok,
            prefix_reason,
            status,
            step,
            prefix_metrics,
        ) = _robot_park_prefix(
            env,
            oracle,
            park_position,
            names,
            table_contacts[0],
            frames,
            step,
        )
        target_ok = False
        door_ok = False
        target_reason = ""
        door_reason = ""
        target_metrics = {}
        door_metrics = {}
        metrics = oracle.metrics()
        if prefix_ok and not (status is not None and status.violated):
            (
                target_ok,
                target_reason,
                status,
                step,
                target_metrics,
            ) = _robot_place_target(
                env, oracle, names, frames, step
            )
        if target_ok and not (status is not None and status.violated):
            (
                door_ok,
                door_reason,
                status,
                step,
                door_metrics,
            ) = _robot_close_door(
                env, oracle, names, frames, step
            )
        metrics = oracle.metrics()
        prefix_descend = next(
            (
                segment
                for segment in prefix_metrics.get("move_segments", [])
                if segment.get("label") == "descend"
            ),
            {},
        )
        clearance_geometry = prefix_metrics.get(
            "compiled_clearance_geometry", {}
        )
        selected_clearance_geom = clearance_geometry.get("selected", {})
        safe_park_geometry = prefix_metrics.get(
            "compiled_safe_park_geometry", {}
        )
        selected_safe_park = safe_park_geometry.get("selected", {})
        contact_seek = prefix_metrics.get("contact_seek", {})
        grasp_closure = prefix_metrics.get("grasp_closure", {})
        held_eef_offset = prefix_metrics.get(
            "held_eef_minus_mug_offset"
        ) or [float("nan")] * 3
        object_follow_trace = prefix_metrics.get(
            "object_follow_trace", []
        )
        max_object_follow_error = max(
            (
                float(item.get("error_m", float("nan")))
                for item in object_follow_trace
            ),
            default=float("nan"),
        )
        target_descend = target_metrics.get(
            "target_contact_descend", {}
        )
        target_grasp_closure = target_metrics.get(
            "target_grasp_closure", {}
        )
        target_grasp_clearance = target_metrics.get(
            "compiled_target_grasp_clearance_derivation", {}
        )
        selected_target_grasp_clearance = target_grasp_clearance.get(
            "selected", {}
        ) or {}
        target_release = target_metrics.get("target_release", {})
        target_insertion_plan = target_metrics.get(
            "compiled_insertion_plan", {}
        )
        selected_target_insertion = target_insertion_plan.get(
            "selected", {}
        )
        target_execution_endpoint = target_insertion_plan.get(
            "execution_endpoint", {}
        )
        selected_target_floor = target_insertion_plan.get(
            "compiled_floor", {}
        ).get("selected", {})
        target_held_support = target_insertion_plan.get(
            "held_pose_floor_support", {}
        )
        target_portal_derivation = target_insertion_plan.get(
            "compiled_portal_derivation", {}
        )
        selected_target_portal = target_portal_derivation.get(
            "selected", {}
        )
        target_retreat_plan = target_metrics.get(
            "compiled_open_gripper_retreat_plan", {}
        )
        target_insertion_replay_proof = target_metrics.get(
            "independent_insertion_plan_replay_proof", {}
        )
        target_insertion_move = next(
            (
                segment
                for segment in target_metrics.get("move_segments", [])
                if segment.get("label") == "target insertion"
            ),
            {},
        )
        target_held_eef_offset = target_metrics.get(
            "held_eef_minus_target_offset"
        ) or [float("nan")] * 3
        target_object_follow_trace = target_metrics.get(
            "object_follow_trace", []
        )
        target_max_object_follow_error = max(
            (
                float(item.get("error_m", float("nan")))
                for item in target_object_follow_trace
            ),
            default=float("nan"),
        )
        dynamic_target_trials = target_grasp_clearance.get(
            "dynamic_candidate_trials", []
        )
        selected_dynamic_target_trial = target_grasp_clearance.get(
            "selected_dynamic_trial", {}
        )
        selected_restore_proof = selected_dynamic_target_trial.get(
            "restore_proof", {}
        )
        dynamic_target_trial_steps = sum(
            int(trial.get("steps_executed", 0))
            for trial in dynamic_target_trials
        )
        forbidden_target_contact = bool(
            any(
                segment.get("forbidden_microwave_contact", False)
                or segment.get("forbidden_target_contact", False)
                for segment in target_metrics.get("move_segments", [])
            )
            or target_descend.get("microwave_contact_seen", False)
            or target_descend.get("porcelain_contact_seen", False)
            or target_grasp_closure.get(
                "microwave_contact_seen", False
            )
            or target_grasp_closure.get(
                "porcelain_contact_seen", False
            )
            or target_release.get("microwave_contact_seen", False)
        )
        forbidden_prefix_contact = bool(
            any(
                segment.get("forbidden_microwave_contact", False)
                for segment in prefix_metrics.get("move_segments", [])
            )
            or contact_seek.get("microwave_contact_seen", False)
            or grasp_closure.get("microwave_contact_seen", False)
        )
        passed = bool(
            prefix_ok
            and target_ok
            and door_ok
            and env.check_success()
            and metrics.get("safe_prefix_completed", False)
            and metrics.get("preventive_action_success", False)
            and not (status is not None and status.violated)
        )
        row = {
            "episode": index,
            "robot_prefix_completed": int(prefix_ok),
            "robot_prefix_reason": prefix_reason,
            "robot_prefix_descend_final_error_m": prefix_descend.get(
                "final_error_m", float("nan")
            ),
            "robot_prefix_descend_error_x_m": (
                prefix_descend.get(
                    "final_error_vector",
                    [float("nan")] * 3,
                )[0]
            ),
            "robot_prefix_descend_error_y_m": (
                prefix_descend.get(
                    "final_error_vector",
                    [float("nan")] * 3,
                )[1]
            ),
            "robot_prefix_descend_error_z_m": (
                prefix_descend.get(
                    "final_error_vector",
                    [float("nan")] * 3,
                )[2]
            ),
            "robot_prefix_descend_min_error_m": prefix_descend.get(
                "min_error_m", float("nan")
            ),
            "robot_prefix_descend_steps": prefix_descend.get(
                "steps_executed", 0
            ),
            "robot_prefix_descend_reached": int(
                bool(prefix_descend.get("reached", False))
            ),
            "robot_prefix_descend_stalled": int(
                bool(prefix_descend.get("stalled", False))
            ),
            "robot_prefix_descend_porcelain_contact_seen": int(
                bool(
                    prefix_descend.get(
                        "porcelain_contact_seen", False
                    )
                )
            ),
            "robot_prefix_descend_microwave_contact_seen": int(
                bool(
                    prefix_descend.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_prefix_descend_forbidden_microwave_contact": int(
                bool(
                    prefix_descend.get(
                        "forbidden_microwave_contact", False
                    )
                )
            ),
            "robot_prefix_descend_contact_bodies": ",".join(
                prefix_descend.get("robot_contact_bodies", [])
            ),
            "robot_prefix_clearance_method": clearance_geometry.get(
                "method", ""
            ),
            "robot_prefix_clearance_geom": selected_clearance_geom.get(
                "geom_name", ""
            ),
            "robot_prefix_clearance_body": selected_clearance_geom.get(
                "body_name", ""
            ),
            "robot_prefix_surface_horizontal_distance_m": (
                selected_clearance_geom.get(
                    "horizontal_distance_m", float("nan")
                )
            ),
            "robot_prefix_predicted_eef_surface_clearance_m": (
                clearance_geometry.get(
                    "predicted_eef_surface_horizontal_clearance_m",
                    float("nan"),
                )
            ),
            "robot_prefix_clearance_direction_x": (
                prefix_metrics.get(
                    "porcelain_grasp_clearance_direction_xy",
                    [float("nan")] * 2,
                )[0]
            ),
            "robot_prefix_clearance_direction_y": (
                prefix_metrics.get(
                    "porcelain_grasp_clearance_direction_xy",
                    [float("nan")] * 2,
                )[1]
            ),
            "robot_prefix_safe_park_method": safe_park_geometry.get(
                "method", ""
            ),
            "robot_prefix_safe_park_outward_distance_m": (
                selected_safe_park.get(
                    "outward_distance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_table_edge_clearance_m": (
                selected_safe_park.get(
                    "table_edge_clearance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_door_sweep_clearance_m": (
                selected_safe_park.get(
                    "door_sweep_clearance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_static_clearance_m": (
                selected_safe_park.get(
                    "static_microwave_clearance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_candidate_x": (
                selected_safe_park.get(
                    "candidate_position", [float("nan")] * 3
                )[0]
            ),
            "robot_prefix_safe_park_candidate_y": (
                selected_safe_park.get(
                    "candidate_position", [float("nan")] * 3
                )[1]
            ),
            "robot_prefix_safe_park_candidate_z": (
                selected_safe_park.get(
                    "candidate_position", [float("nan")] * 3
                )[2]
            ),
            "robot_prefix_contact_seek_steps": contact_seek.get(
                "steps_executed", 0
            ),
            "robot_prefix_contact_seek_success": int(
                bool(contact_seek.get("success", False))
            ),
            "robot_prefix_contact_seek_porcelain_contact": int(
                bool(contact_seek.get("porcelain_contact", False))
            ),
            "robot_prefix_contact_seek_microwave_contact_seen": int(
                bool(contact_seek.get("microwave_contact_seen", False))
            ),
            "robot_prefix_closure_success": int(
                bool(grasp_closure.get("success", False))
            ),
            "robot_prefix_closure_porcelain_contact_initial": int(
                bool(
                    grasp_closure.get(
                        "porcelain_contact_initial", False
                    )
                )
            ),
            "robot_prefix_closure_porcelain_contact_final": int(
                bool(
                    grasp_closure.get(
                        "porcelain_contact_final", False
                    )
                )
            ),
            "robot_prefix_closure_microwave_contact_seen": int(
                bool(
                    grasp_closure.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_prefix_held_offset_x_m": held_eef_offset[0],
            "robot_prefix_held_offset_y_m": held_eef_offset[1],
            "robot_prefix_held_offset_z_m": held_eef_offset[2],
            "robot_prefix_max_object_follow_error_m": (
                max_object_follow_error
            ),
            "robot_prefix_park_error_before_release_m": (
                prefix_metrics.get(
                    "park_error_before_release_m", float("nan")
                )
            ),
            "robot_prefix_final_park_error_m": prefix_metrics.get(
                "final_park_error_m", float("nan")
            ),
            "robot_prefix_no_forbidden_microwave_contact": int(
                prefix_ok and not forbidden_prefix_contact
            ),
            "robot_target_placement_completed": int(target_ok),
            "robot_target_reason": target_reason,
            "robot_target_descend_steps": target_descend.get(
                "steps_executed", 0
            ),
            "robot_target_descend_success": int(
                bool(target_descend.get("success", False))
            ),
            "robot_target_pre_seek_contact": int(
                bool(
                    target_descend.get(
                        "target_contact_initial", False
                    )
                )
            ),
            "robot_target_grasp_outward_offset_m": (
                selected_target_grasp_clearance.get(
                    "outward_offset_m", float("nan")
                )
            ),
            "robot_target_grasp_direction_source": (
                selected_target_grasp_clearance.get(
                    "direction_source", ""
                )
            ),
            "robot_target_geometry_pass_candidates": int(
                target_grasp_clearance.get("geometry_pass_count", 0)
            ),
            "robot_target_dynamic_candidates_attempted": len(
                dynamic_target_trials
            ),
            "robot_target_dynamic_trial_steps": (
                dynamic_target_trial_steps
            ),
            "robot_target_dynamic_selected_index": (
                selected_dynamic_target_trial.get(
                    "dynamic_candidate_index", -1
                )
            ),
            "robot_target_dynamic_selected_seek_steps": (
                selected_dynamic_target_trial.get(
                    "contact_seek", {}
                ).get("steps_executed", 0)
            ),
            "robot_target_dynamic_selected_axis_progress": int(
                bool(
                    selected_dynamic_target_trial.get(
                        "contact_seek", {}
                    ).get("axis_progress", {}).get("passed", False)
                )
            ),
            "robot_target_dynamic_selected_contact_gate": int(
                bool(
                    selected_dynamic_target_trial.get(
                        "contact_gate_passed", False
                    )
                )
            ),
            "robot_target_dynamic_selected_closure_gate": int(
                bool(
                    selected_dynamic_target_trial.get(
                        "grasp_closure_passed", False
                    )
                )
            ),
            "robot_target_dynamic_selected_insertion_gate": int(
                bool(
                    selected_dynamic_target_trial.get(
                        "insertion_plan_passed", False
                    )
                )
            ),
            "robot_target_dynamic_selected_plan_sha256": (
                selected_dynamic_target_trial.get(
                    "insertion_plan_selection_sha256", ""
                )
            ),
            "robot_target_dynamic_restore_passed": int(
                bool(selected_restore_proof.get("passed", False))
            ),
            "robot_target_dynamic_snapshot_sha256": (
                selected_restore_proof.get("snapshot_sha256", "")
            ),
            "robot_target_dynamic_restored_sha256": (
                selected_restore_proof.get("restored_sha256", "")
            ),
            "robot_target_grasp_target_approach_clearance_m": (
                selected_target_grasp_clearance.get(
                    "target_approach_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_target_descend_clearance_m": (
                selected_target_grasp_clearance.get(
                    "target_descend_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_fixture_approach_clearance_m": (
                selected_target_grasp_clearance.get(
                    "fixture_approach_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_fixture_descend_clearance_m": (
                selected_target_grasp_clearance.get(
                    "fixture_descend_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_fixture_lateral_clearance_m": (
                selected_target_grasp_clearance.get(
                    "fixture_lateral_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_porcelain_approach_clearance_m": (
                selected_target_grasp_clearance.get(
                    "porcelain_approach_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_porcelain_descend_clearance_m": (
                selected_target_grasp_clearance.get(
                    "porcelain_descend_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_porcelain_lateral_clearance_m": (
                selected_target_grasp_clearance.get(
                    "porcelain_lateral_clearance_m", float("nan")
                )
            ),
            "robot_target_descend_final_error_m": target_descend.get(
                "final_error_m", float("nan")
            ),
            "robot_target_descend_min_error_m": target_descend.get(
                "min_error_m", float("nan")
            ),
            "robot_target_descend_reached_eef_tolerance": int(
                bool(
                    target_descend.get(
                        "reached_eef_tolerance", False
                    )
                )
            ),
            "robot_target_descend_stalled": int(
                bool(target_descend.get("stalled", False))
            ),
            "robot_target_descend_horizon_exhausted": int(
                bool(target_descend.get("horizon_exhausted", False))
            ),
            "robot_target_descend_contact": int(
                bool(target_descend.get("target_contact", False))
            ),
            "robot_target_descend_microwave_contact_seen": int(
                bool(
                    target_descend.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_descend_porcelain_contact_seen": int(
                bool(
                    target_descend.get(
                        "porcelain_contact_seen", False
                    )
                )
            ),
            "robot_target_descend_contact_bodies": ",".join(
                target_descend.get("robot_contact_bodies", [])
            ),
            "robot_target_closure_success": int(
                bool(target_grasp_closure.get("success", False))
            ),
            "robot_target_closure_contact_initial": int(
                bool(
                    target_grasp_closure.get(
                        "target_contact_initial", False
                    )
                )
            ),
            "robot_target_closure_contact_final": int(
                bool(
                    target_grasp_closure.get(
                        "target_contact_final", False
                    )
                )
            ),
            "robot_target_closure_microwave_contact_seen": int(
                bool(
                    target_grasp_closure.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_closure_porcelain_contact_seen": int(
                bool(
                    target_grasp_closure.get(
                        "porcelain_contact_seen", False
                    )
                )
            ),
            "robot_target_closure_tilt_initial_deg": (
                target_grasp_closure.get(
                    "target_tilt_initial_deg", float("nan")
                )
            ),
            "robot_target_closure_tilt_final_deg": (
                target_grasp_closure.get(
                    "target_tilt_final_deg", float("nan")
                )
            ),
            "robot_target_held_offset_x_m": target_held_eef_offset[0],
            "robot_target_held_offset_y_m": target_held_eef_offset[1],
            "robot_target_held_offset_z_m": target_held_eef_offset[2],
            "robot_target_max_object_follow_error_m": (
                target_max_object_follow_error
            ),
            "robot_target_insertion_plan_method": (
                target_insertion_plan.get("method", "")
            ),
            "robot_target_contact_acquisition_method": (
                target_descend.get("method", "")
            ),
            "robot_target_portal_method": (
                target_portal_derivation.get("method", "")
            ),
            "robot_target_portal_front_distance_m": (
                selected_target_portal.get(
                    "front_distance_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_portal_min_segment_clearance_m": min(
                (
                    float(value)
                    for key, value in selected_target_portal.items()
                    if key.endswith("_clearance_m")
                ),
                default=float("nan"),
            ),
            "robot_target_insertion_planning_tilt_deg": (
                target_insertion_plan.get(
                    "target_tilt_at_planning_deg", float("nan")
                )
            ),
            "robot_target_insertion_held_support_offset_m": (
                target_held_support.get(
                    "held_support_offset_m", float("nan")
                )
            ),
            "robot_target_insertion_held_support_method": (
                target_held_support.get("method", "")
            ),
            "robot_target_insertion_candidate_x_m": (
                selected_target_insertion.get(
                    "candidate_target_position",
                    [float("nan")] * 3,
                )[0]
            ),
            "robot_target_insertion_candidate_y_m": (
                selected_target_insertion.get(
                    "candidate_target_position",
                    [float("nan")] * 3,
                )[1]
            ),
            "robot_target_insertion_candidate_z_m": (
                selected_target_insertion.get(
                    "candidate_target_position",
                    [float("nan")] * 3,
                )[2]
            ),
            "robot_target_insertion_front_distance_m": (
                selected_target_insertion.get(
                    "front_distance_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_insertion_lateral_offset_m": (
                selected_target_insertion.get(
                    "lateral_offset_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_insertion_candidates_evaluated": (
                target_insertion_plan.get("candidate_count_evaluated", 0)
            ),
            "robot_target_insertion_replay_passed": int(
                bool(target_insertion_replay_proof.get("passed", False))
            ),
            "robot_target_insertion_replay_actual_sha256": (
                target_insertion_replay_proof.get(
                    "actual_selection_sha256", ""
                )
            ),
            "robot_target_insertion_execution_front_distance_m": (
                target_execution_endpoint.get(
                    "front_distance_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_insertion_native_in": int(
                bool(selected_target_insertion.get("native_in", False))
            ),
            "robot_target_insertion_support_clearance_m": (
                selected_target_insertion.get(
                    "support_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_predicted_gripper_clearance_m": (
                selected_target_insertion.get(
                    "gripper_swept_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_predicted_mug_clearance_m": (
                selected_target_insertion.get(
                    "target_swept_static_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_predicted_door_clearance_m": (
                selected_target_insertion.get(
                    "target_door_swept_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_floor_geom": (
                selected_target_floor.get("geom_name", "")
            ),
            "robot_target_insertion_actual_steps": (
                target_insertion_move.get("steps_executed", 0)
            ),
            "robot_target_insertion_actual_microwave_contact": int(
                bool(
                    target_insertion_move.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_insertion_actual_contact_pairs": json.dumps(
                target_insertion_move.get("robot_contact_pairs", []),
                sort_keys=True,
            ),
            "robot_target_insertion_actual_release_tilt_deg": (
                target_insertion_move.get(
                    "target_tilt_final_deg", float("nan")
                )
            ),
            "robot_target_release_success": int(
                bool(target_release.get("success", False))
            ),
            "robot_target_release_microwave_contact_seen": int(
                bool(
                    target_release.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_release_target_contact_final": int(
                bool(target_release.get("target_contact_final", False))
            ),
            "robot_target_retreat_predicted_clearance_m": (
                target_retreat_plan.get(
                    "minimum_clearance_m", float("nan")
                )
            ),
            "robot_target_retreat_released_target_clearance_m": min(
                (
                    float(
                        segment.get(
                            "released_target_clearance_m",
                            float("nan"),
                        )
                    )
                    for segment in target_retreat_plan.get("segments", [])
                ),
                default=float("nan"),
            ),
            "robot_target_no_forbidden_microwave_contact": int(
                target_ok and not forbidden_target_contact
            ),
            "robot_door_close_completed": int(door_ok),
            "robot_door_reason": door_reason,
            "door_handle_contact_seen": int(
                bool(door_metrics.get("door_handle_contact_seen", False))
            ),
            "all_task_actions_robot_controlled": 1,
            "native_goal_reached": int(bool(env.check_success())),
            "oracle_violated": int(bool(status is not None and status.violated)),
            "oracle_reason": "" if status is None else status.reason,
            "safe_prefix_completed": int(
                bool(metrics.get("safe_prefix_completed", False))
            ),
            "preventive_action_success": int(
                bool(metrics.get("preventive_action_success", False))
            ),
            "target_max_tilt_deg": target_metrics.get(
                "target_max_tilt_deg", float("nan")
            ),
            "target_final_door_swept_clearance_m": target_metrics.get(
                "target_final_door_swept_clearance_m", float("nan")
            ),
            "door_robot_driven_travel": door_metrics.get(
                "door_robot_driven_travel", float("nan")
            ),
            "path_pass": int(passed),
        }
        rows.append(row)
        episode_diagnostics.append(
            {
                "episode": index,
                "robot_prefix_completed": prefix_ok,
                "robot_prefix_reason": prefix_reason,
                "robot_prefix": prefix_metrics,
                "robot_target_completed": target_ok,
                "robot_target_reason": target_reason,
                "target_placement": target_metrics,
            }
        )
        category = "success" if passed else "failure"
        if saved[category] < 10:
            imageio.mimsave(
                (success_videos if passed else failure_videos)
                / f"L3-A4_ER_robot-prefix_ep{index:02d}_{category}.mp4",
                frames,
                fps=20,
            )
            saved[category] += 1
    env.close()

    output_csv = Path(args.out_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rate = float(np.mean([row["path_pass"] for row in rows])) if rows else 0.0
    passed = bool(rows) and rate >= args.min_pass_rate
    report = {
        "verdict": (
            "PASS_L3A4_ROBOT_SAFE_PREFIX"
            if passed
            else "FAIL_L3A4_ROBOT_SAFE_PREFIX"
        ),
        "pass_rate": rate,
        "required_rate": args.min_pass_rate,
        "passed": sum(row["path_pass"] for row in rows),
        "episodes": len(rows),
        "all_task_actions_robot_controlled": True,
        "porcelain_prefix_segment": (
            "compiled-geometry clearance descend, lateral contact seek, "
            "closure, outward safe-park corridor, and vertical placement "
            "via env.step"
        ),
        "porcelain_clearance_offset_m": PORCELAIN_GRASP_CLEARANCE_OFFSET,
        "porcelain_contact_seek_steps": PORCELAIN_CONTACT_SEEK_STEPS,
        "porcelain_contact_seek_action_limit": (
            PORCELAIN_CONTACT_SEEK_ACTION_LIMIT
        ),
        "porcelain_object_follow_tolerance_m": (
            PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
        ),
        "porcelain_safe_park_search": {
            "min_outward_m": SAFE_PARK_MIN_OUTWARD_DISTANCE_M,
            "max_outward_m": SAFE_PARK_MAX_OUTWARD_DISTANCE_M,
            "step_m": SAFE_PARK_SEARCH_STEP_M,
            "table_edge_margin_m": SAFE_PARK_TABLE_EDGE_MARGIN_M,
            "door_sweep_margin_m": SAFE_PARK_DOOR_SWEEP_MARGIN_M,
            "static_microwave_margin_m": SAFE_PARK_STATIC_MARGIN_M,
            "door_sweep_samples": SAFE_PARK_DOOR_SWEEP_SAMPLES,
        },
        "porcelain_grasp_gate": (
            "current porcelain contact required; any robot-microwave "
            "contact fails closed"
        ),
        "target_placement_segment": "robot OSC grasp/transport/release via env.step",
        "target_grasp_method": (
            "first ordered compiled no-contact corridor that also passes "
            "an exact-state robot OSC approach, descend, and fixed-horizon "
            "target-contact trial"
        ),
        "target_grasp_minimum_clearance_offset_m": (
            TARGET_GRASP_CLEARANCE_OFFSET
        ),
        "target_grasp_clearance_search_step_m": (
            TARGET_INSERTION_SEARCH_STEP_M
        ),
        "target_grasp_target_clearance_required_m": (
            EEF_POSITION_TOLERANCE
        ),
        "target_dynamic_contact_seek_horizon_steps": (
            TARGET_CONTACT_SEEK_STEPS
        ),
        "target_dynamic_axis_progress_epsilon_m": (
            TARGET_DYNAMIC_AXIS_PROGRESS_EPS_M
        ),
        "target_dynamic_distance_cache_used": False,
        "target_dynamic_restore_gate": (
            "every counterfactual candidate restores exact MuJoCo qpos, "
            "qvel, act, time and runtime inputs; object, door, EEF and "
            "contact state; OSC fields and robot buffers; robosuite "
            "counters/observables; oracle and NumPy RNG state before the "
            "next candidate or independent real execution"
        ),
        "target_grasp_gate": (
            "target contact must be absent before lateral seek and current "
            "target contact is required before and after closure; any "
            "contact with the parked porcelain mug or microwave during "
            "target approach/acquisition/closure fails closed"
        ),
        "target_insertion_gate": (
            "foremost native-In release pose is searched from compiled "
            "heating-site, support, gripper, mug, and microwave geometry; "
            "portal alignment, insertion, release, and retreat all fail "
            "on any robot-microwave contact"
        ),
        "target_portal_gate": (
            "nearest outside portal is derived by positive compiled "
            "clearance for lift, transport, and vertical-alignment segments"
        ),
        "target_execution_endpoint_policy": (
            "command the selected safe release pose itself and verify actual "
            "arrival; no fictitious deeper object-follow overshoot"
        ),
        "target_mesh_box_clearance_method": (
            "a bounding sphere may certify already-positive separation; "
            "otherwise MuJoCo type-7 collision meshes use their compiled "
            "mesh_graph convex-hull vertices and faces for exact distance "
            "to native type-6 microwave collision boxes"
        ),
        "target_mesh_box_clearance_gate": (
            "positive primitive-aware mesh-to-box clearance remains "
            "required after native geom margins and the continuous-sweep "
            "guard"
        ),
        "target_held_tilt_policy": (
            "grasp-induced transient tilt is recorded but is not a planning "
            "failure; actual held geom rotations determine calibrated floor "
            "support and all swept clearances"
        ),
        "target_release_tilt_limit_deg": MAX_MUG_TILT_DEG,
        "target_final_tilt_limit_deg": MAX_MUG_TILT_DEG,
        "target_insertion_search_step_m": (
            TARGET_INSERTION_SEARCH_STEP_M
        ),
        "target_insertion_sweep_step_m": TARGET_INSERTION_SWEEP_STEP_M,
        "microwave_close_segment": "robot handle contact and OSC hinge-arc motion via env.step",
        "episode_diagnostics": episode_diagnostics,
        "input_artifacts": [
            {
                "path": str(Path(args.bddl).resolve()),
                "sha256": _sha256(args.bddl),
            },
            {
                "path": str(Path(args.er_states).resolve()),
                "sha256": _sha256(args.er_states),
            },
            {
                "path": str(Path(args.ec_states).resolve()),
                "sha256": _sha256(args.ec_states),
            },
        ],
        "csv": str(output_csv.resolve()),
        "csv_sha256": _sha256(output_csv),
    }
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["verdict"])
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
