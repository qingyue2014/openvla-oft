"""Replay serialized native LIBERO states with fixed-fixture poses.

MuJoCo free-object state is present in flattened qpos/qvel arrays, but fixed
LIBERO fixtures are sampled into ``model.body_pos`` / ``model.body_quat`` and
are therefore absent from those arrays.  Native-only paired experiments must
restore both before producing the first policy observation.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.robot.libero.tasks.l3a1_native_replay import (
    materialize_l3a1_native_state,
)


def _decoded_json(value, *, field: str):
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON string")
    return json.loads(value)


def _restore_fixture_inventory(env, record: dict) -> None:
    names_value = record.get("fixture_replay_bodies_json")
    if names_value is None:
        return
    names = _decoded_json(names_value, field="fixture_replay_bodies_json")
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("fixture_replay_bodies_json must encode body names")

    positions = np.asarray(record.get("fixture_replay_positions"), dtype=float)
    quaternions = np.asarray(
        record.get("fixture_replay_quaternions"), dtype=float
    )
    if positions.shape != (len(names), 3):
        raise ValueError(
            "fixture_replay_positions shape mismatch: "
            f"{positions.shape} != {(len(names), 3)}"
        )
    if quaternions.shape != (len(names), 4):
        raise ValueError(
            "fixture_replay_quaternions shape mismatch: "
            f"{quaternions.shape} != {(len(names), 4)}"
        )
    if not np.all(np.isfinite(positions)) or not np.all(
        np.isfinite(quaternions)
    ):
        raise ValueError("fixture replay pose contains a non-finite value")

    for index, name in enumerate(names):
        body_id = env.sim.model.body_name2id(name)
        env.sim.model.body_pos[body_id] = positions[index]
        env.sim.model.body_quat[body_id] = quaternions[index]
    env.sim.forward()


def materialize_native_scene_state(env, record: dict) -> np.ndarray:
    """Restore fixed fixtures and return the exact flattened episode state."""
    if "initial_state" not in record:
        raise ValueError("serialized native scene record has no initial_state")
    _restore_fixture_inventory(env, record)

    # L3-A1 carries an additional support-relative bottle binding.  Keep its
    # established replay semantics after the generic fixture inventory has
    # been restored.
    if "support_relative_position" in record:
        return materialize_l3a1_native_state(env, record)

    # A single-fixture legacy record is also supported.  This is useful for a
    # microwave scene and remains backward compatible with L3-A1 artifacts.
    if "fixture_root_body" in record:
        return materialize_l3a1_native_state(env, record)
    return np.asarray(record["initial_state"]).copy()
