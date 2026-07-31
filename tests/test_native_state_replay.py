import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)


class _Model:
    names = ["world", "table", "microwave"]

    def __init__(self):
        self.body_pos = np.zeros((3, 3), dtype=float)
        self.body_quat = np.array([[1.0, 0.0, 0.0, 0.0]] * 3)

    def body_name2id(self, name):
        return self.names.index(name)


class _Sim:
    def __init__(self):
        self.model = _Model()
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


def test_replays_all_fixed_fixture_poses_before_returning_state():
    env = SimpleNamespace(sim=_Sim())
    state = np.arange(7, dtype=float)
    record = {
        "initial_state": state,
        "fixture_replay_bodies_json": json.dumps(["table", "microwave"]),
        "fixture_replay_positions": np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ),
        "fixture_replay_quaternions": np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    }

    replay = materialize_native_scene_state(env, record)

    np.testing.assert_array_equal(replay, state)
    np.testing.assert_allclose(env.sim.model.body_pos[1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(env.sim.model.body_pos[2], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(env.sim.model.body_quat[2], [0.0, 0.0, 0.0, 1.0])
    assert env.sim.forward_calls == 1


def test_fixture_replay_fails_closed_on_shape_mismatch():
    env = SimpleNamespace(sim=_Sim())
    record = {
        "initial_state": np.zeros(3),
        "fixture_replay_bodies_json": json.dumps(["table", "microwave"]),
        "fixture_replay_positions": np.zeros((1, 3)),
        "fixture_replay_quaternions": np.zeros((2, 4)),
    }

    with pytest.raises(ValueError, match="positions shape mismatch"):
        materialize_native_scene_state(env, record)
