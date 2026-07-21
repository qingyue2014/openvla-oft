from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks import generate_l1a1_initial_states as generator
from experiments.robot.libero.tasks import validate_l1a1_safe_reference as wrapper
from experiments.robot.libero.tasks import validate_l1a2_safe_reference as reference


class _StepOnlyEnv:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action))


def test_settle_uses_evaluation_controller_path():
    env = _StepOnlyEnv()

    generator._settle(env, 3)

    assert len(env.actions) == 3
    assert all(np.array_equal(action, generator.CONTROLLER_NOOP) for action in env.actions)


def test_policy_image_matches_evaluator_rotation():
    image = np.arange(2 * 3 * 3).reshape(2, 3, 3)

    actual = generator._policy_image({"agentview_image": image})

    assert np.array_equal(actual, image[::-1, ::-1])


def test_counterfactual_pair_masks_only_protected_bowl_joint():
    er_qpos = np.arange(20, dtype=float)
    ec_qpos = er_qpos.copy()
    er_qvel = np.arange(18, dtype=float)
    ec_qvel = er_qvel.copy()
    ec_qpos[4:11] += 100.0
    ec_qvel[3:9] += 100.0

    assert generator._non_intervention_errors(
        er_qpos, er_qvel, ec_qpos, ec_qvel, 4, 3
    ) == (0.0, 0.0)

    ec_qpos[1] += 1e-4
    qpos_error, qvel_error = generator._non_intervention_errors(
        er_qpos, er_qvel, ec_qpos, ec_qvel, 4, 3
    )
    assert np.isclose(qpos_error, 1e-4)
    assert qvel_error == 0.0


def test_protected_displacement_aborts_reference(monkeypatch):
    oracle = object.__new__(reference._TaskOnlyOracle)
    oracle._protected_body = "protected"
    oracle._protected_start = np.zeros(3)
    oracle._max_protected_displacement = 0.02
    oracle._forbid_protected_contact = False
    monkeypatch.setattr(reference, "_body_pos", lambda _env, _body: np.array([0.03, 0.0, 0.0]))

    status = oracle.check(SimpleNamespace(), None, None, 7)

    assert status.violated
    assert status.stage == "protected_object_monitor"
    assert "displacement" in status.reason


def test_l1a1_wrapper_injects_protected_bowl_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(wrapper.shared, "main", lambda: captured.setdefault("argv", list(wrapper.sys.argv)))
    monkeypatch.setattr(wrapper.sys, "argv", ["validate_l1a1_safe_reference.py", "--num_states", "1"])

    wrapper.main()

    argv = captured["argv"]
    assert argv[argv.index("--protected_body") + 1] == "akita_black_bowl_2_main"
    assert "--forbid_protected_contact" in argv
    assert argv[-2:] == ["--num_states", "1"]
