"""Native-fixture replay helpers for L3-A1 serialized bottle interventions."""

import numpy as np


def materialize_l3a1_native_state(env, record: dict) -> np.ndarray:
    """Bind a serialized bottle intervention to this native reset's drawer.

    LIBERO fixed fixtures are stored in ``model.body_pos`` rather than the
    flattened qpos/qvel state. The artifact therefore stores the intervened
    native bottle pose relative to the drawer and the exact cabinet pose
    sampled for that native reset. Replay restores that serialized native
    fixture pose, then translates only the bottle; it does not modify the
    task, prompt, geometry, or asset inventory.
    """
    state = np.asarray(record["initial_state"]).copy()
    fixture_root_body = record.get("fixture_root_body")
    if fixture_root_body is not None:
        required_fixture = (
            "fixture_root_position",
            "fixture_root_quaternion",
        )
        missing_fixture = [
            name for name in required_fixture if name not in record
        ]
        if missing_fixture:
            raise ValueError(
                "incomplete native fixture replay metadata: "
                + ", ".join(missing_fixture)
            )
        fixture_root_id = env.sim.model.body_name2id(str(fixture_root_body))
        env.sim.model.body_pos[fixture_root_id] = np.asarray(
            record["fixture_root_position"], dtype=float
        )
        env.sim.model.body_quat[fixture_root_id] = np.asarray(
            record["fixture_root_quaternion"], dtype=float
        )
        env.sim.forward()

    relative_position = record.get("support_relative_position")
    if relative_position is None:
        return state

    required = (
        "support_body",
        "bottle_qpos_flat_start",
        "bottle_qvel_flat_start",
        "bottle_world_quaternion",
        "bottle_world_qvel",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(
            "incomplete native L3-A1 replay metadata: " + ", ".join(missing)
        )
    # Apply the paired flattened state first so articulated drawer qpos is the
    # one associated with this demo before resolving the support world pose.
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    support_id = env.sim.model.body_name2id(str(record["support_body"]))
    support_position = env.sim.data.body_xpos[support_id].copy()
    qpos_start = int(record["bottle_qpos_flat_start"])
    qvel_start = int(record["bottle_qvel_flat_start"])
    state[qpos_start:qpos_start + 3] = (
        support_position + np.asarray(relative_position, dtype=float)
    )
    state[qpos_start + 3:qpos_start + 7] = np.asarray(
        record["bottle_world_quaternion"], dtype=float
    )
    state[qvel_start:qvel_start + 6] = np.asarray(
        record["bottle_world_qvel"], dtype=float
    )
    return state
