from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks.l3a1_replay import clear_mujoco_replay_transients


def test_clear_mujoco_replay_transients_zeros_only_hidden_solver_buffers():
    data = SimpleNamespace(
        qacc_warmstart=np.ones(3),
        qfrc_applied=np.ones(4) * 2,
        xfrc_applied=np.ones((2, 6)) * 3,
        qpos=np.array([7.0, 8.0]),
        qvel=np.array([9.0]),
    )
    env = SimpleNamespace(sim=SimpleNamespace(data=data))

    clear_mujoco_replay_transients(env)

    assert not data.qacc_warmstart.any()
    assert not data.qfrc_applied.any()
    assert not data.xfrc_applied.any()
    assert np.array_equal(data.qpos, [7.0, 8.0])
    assert np.array_equal(data.qvel, [9.0])


def test_clear_mujoco_replay_transients_tolerates_optional_missing_buffers():
    env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(qacc_warmstart=np.ones(1)))
    )
    clear_mujoco_replay_transients(env)
    assert env.sim.data.qacc_warmstart[0] == 0
