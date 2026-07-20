import numpy as np

from experiments.robot.libero.tasks.validate_l1a34_reference import (
    _closing_axis_yaw_error_rad,
    _grasp_candidates,
    _quat_xyzw_to_mat,
)

# 180-degree rotation about (1,1,0)/sqrt(2): the default OSC hand orientation
# [[0,1,0],[1,0,0],[0,0,-1]], whose closing axis (hand x) is world +y.
DEFAULT_HAND_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])


def test_quat_to_mat_reproduces_default_hand_orientation():
    mat = _quat_xyzw_to_mat(DEFAULT_HAND_QUAT)
    np.testing.assert_allclose(
        mat, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), atol=1e-12
    )


def test_default_axis_has_zero_error_at_south_bearing():
    obs = {"robot0_eef_quat": DEFAULT_HAND_QUAT}
    assert abs(_closing_axis_yaw_error_rad(obs, -90.0)) < 1e-9
    assert abs(_closing_axis_yaw_error_rad(obs, 90.0)) < 1e-9  # modulo 180


def test_east_bearing_needs_quarter_turn():
    obs = {"robot0_eef_quat": DEFAULT_HAND_QUAT}
    error = _closing_axis_yaw_error_rad(obs, 0.0)
    assert abs(abs(error) - np.pi / 2.0) < 1e-9


def test_error_is_wrapped_to_half_turn():
    obs = {"robot0_eef_quat": DEFAULT_HAND_QUAT}
    for bearing in (-86.0, 94.0):
        error = _closing_axis_yaw_error_rad(obs, bearing)
        assert abs(error) < np.radians(5.0)


def test_l1a3_candidates_carry_yaw_only_for_safe(monkeypatch):
    import experiments.robot.libero.tasks.validate_l1a34_reference as mod

    monkeypatch.setattr(
        mod, "_world_aabb",
        lambda _env, _body: (np.array([-0.05, -0.05, 0.0]), np.array([0.05, 0.05, 0.05])),
    )
    candidates = mod._grasp_candidates(None, "l1a3", -86.0, "calibrate")
    unsafe = [c for c in candidates if c[0] == "unsafe"]
    safe = [c for c in candidates if c[0] == "safe"]
    assert all(c[3] is None for c in unsafe)
    assert all(c[3] is not None for c in safe)
    # first safe bearings are the reachable east/west rotations
    assert safe[0][3] == -86.0 + 90.0
    assert safe[2][3] == -86.0 - 90.0


def test_l1a4_candidates_prefer_robot_side_rim_grasp(monkeypatch):
    import experiments.robot.libero.tasks.validate_l1a34_reference as mod

    monkeypatch.setattr(
        mod, "_world_aabb",
        lambda _env, _body: (np.array([-0.05, -0.05, 0.0]), np.array([0.05, 0.05, 0.05])),
    )
    candidates = mod._grasp_candidates(None, "l1a4", 30.0, "calibrate")
    unsafe = [c for c in candidates if c[0] == "unsafe"]
    # both unsafe probes grasp at bearing 270 (toward the robot) and release at
    # the plate centroid — the placement prior under test
    for label, grasp, place, yaw in unsafe:
        assert grasp[1] < 0  # -y offset
        np.testing.assert_allclose(place, np.zeros(2), atol=1e-12)
        assert yaw is None
