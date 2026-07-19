import numpy as np

from experiments.robot.libero.tasks.validate_l1c1_safe_reference import (
    _candidate_search_decision,
    _position_action,
)


def test_position_action_is_scaled_clipped_and_preserves_gripper_command():
    action = _position_action([0.0, 0.0, 0.0], [0.08, -0.02, 0.0], 1.0, 0.04)
    np.testing.assert_allclose(action[:3], [1.0, -0.5, 0.0])
    np.testing.assert_allclose(action[3:6], 0.0)
    assert action[-1] == 1.0


def test_candidate_search_accepts_only_complete_safe_success():
    assert _candidate_search_decision(
        {"safe_success": 1, "grasp_verified": 1, "failure_stage": ""}
    ) == "accept"
    assert _candidate_search_decision(
        {"safe_success": 0, "grasp_verified": 1, "failure_stage": ""}
    ) == "continue"


def test_candidate_search_stops_only_on_nonrecoverable_pregrasp_failure():
    assert _candidate_search_decision(
        {"safe_success": 0, "grasp_verified": 0, "failure_stage": "verify_grasp"}
    ) == "continue"
    assert _candidate_search_decision(
        {"safe_success": 0, "grasp_verified": 0, "failure_stage": "move_to_pregrasp"}
    ) == "stop"
