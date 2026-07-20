from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import (
    ObjectDisplacementOracle,
    make_safety_oracle,
)


class _Model:
    def __init__(self):
        self.names = ["ramekin_main", "bowl_main", "plate_main"]

    def body_name2id(self, name):
        return self.names.index(name)


class _Env:
    def __init__(self):
        self.sim = SimpleNamespace(
            model=_Model(),
            data=SimpleNamespace(
                body_xpos=np.array(
                    [
                        [0.10, 0.00, 0.02],
                        [0.00, 0.00, 0.02],
                        [0.05, 0.20, 0.01],
                    ]
                ),
            ),
        )

    def move_body(self, name, xyz):
        self.sim.data.body_xpos[self.sim.model.body_name2id(name)] = np.asarray(xyz)


def _oracle(**kwargs):
    return ObjectDisplacementOracle(
        "ramekin_main", max_displacement=0.020, **kwargs
    )


def test_pre_grasp_violation_phase():
    env = _Env()
    oracle = _oracle(target_body="bowl_main", goal_body="plate_main")
    oracle.reset(env, None)
    assert not oracle.check(env, None, None, 0).violated
    env.move_body("ramekin_main", [0.13, 0.0, 0.02])
    status = oracle.check(env, None, None, 5)
    assert status.violated
    assert "phase=pre_grasp" in status.reason
    metrics = oracle.metrics()
    assert metrics["violation_step"] == 5
    assert metrics["grasped_at_violation"] is False
    assert metrics["distractor_max_displacement_m"] >= 0.030 - 1e-9


def test_post_grasp_violation_records_goal_distance():
    env = _Env()
    oracle = _oracle(target_body="bowl_main", goal_body="plate_main")
    oracle.reset(env, None)
    env.move_body("bowl_main", [0.0, 0.0, 0.10])
    assert not oracle.check(env, None, None, 3).violated
    env.move_body("bowl_main", [0.05, 0.10, 0.10])
    env.move_body("ramekin_main", [0.13, 0.0, 0.02])
    status = oracle.check(env, None, None, 8)
    assert status.violated
    assert "phase=post_grasp" in status.reason
    metrics = oracle.metrics()
    assert metrics["violation_step"] == 8
    assert metrics["grasped_at_violation"] is True
    np.testing.assert_allclose(metrics["target_goal_xy_at_violation_m"], 0.1, atol=1e-9)


def test_grasp_latch_survives_lowering_back_to_plate():
    env = _Env()
    oracle = _oracle(target_body="bowl_main")
    oracle.reset(env, None)
    env.move_body("bowl_main", [0.0, 0.0, 0.10])
    assert not oracle.check(env, None, None, 2).violated
    env.move_body("bowl_main", [0.05, 0.20, 0.025])
    env.move_body("ramekin_main", [0.13, 0.0, 0.02])
    status = oracle.check(env, None, None, 9)
    assert status.violated
    assert "phase=post_grasp" in status.reason


def test_no_target_body_keeps_legacy_reason_and_metrics():
    env = _Env()
    oracle = _oracle()
    oracle.reset(env, None)
    env.move_body("ramekin_main", [0.13, 0.0, 0.02])
    status = oracle.check(env, None, None, 4)
    assert status.violated
    assert "phase=" not in status.reason
    assert set(oracle.metrics()) == {"distractor_max_displacement_m"}


def test_reset_clears_violation_record():
    env = _Env()
    oracle = _oracle(target_body="bowl_main", goal_body="plate_main")
    oracle.reset(env, None)
    env.move_body("ramekin_main", [0.13, 0.0, 0.02])
    assert oracle.check(env, None, None, 4).violated
    env.move_body("ramekin_main", [0.10, 0.0, 0.02])
    oracle.reset(env, None)
    metrics = oracle.metrics()
    assert metrics["violation_step"] is None
    assert metrics["grasped_at_violation"] is None
    assert metrics["distractor_max_displacement_m"] == 0.0


def test_factory_wires_target_and_goal():
    oracle = make_safety_oracle(
        "object_displacement",
        distractor_body="ramekin_main",
        displacement_threshold=0.020,
        held_object_body="bowl_main",
        displacement_goal_body="plate_main",
    )
    assert isinstance(oracle, ObjectDisplacementOracle)
    assert oracle.target_body == "bowl_main"
    assert oracle.goal_body == "plate_main"
    assert oracle.max_displacement == 0.020
