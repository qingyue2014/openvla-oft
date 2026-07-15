import numpy as np

from experiments.robot.libero.tasks.validate_l1c1_safe_reference import _position_action


def test_position_action_is_scaled_clipped_and_preserves_gripper_command():
    action = _position_action([0.0, 0.0, 0.0], [0.08, -0.02, 0.0], 1.0, 0.04)
    np.testing.assert_allclose(action[:3], [1.0, -0.5, 0.0])
    np.testing.assert_allclose(action[3:6], 0.0)
    assert action[-1] == 1.0
