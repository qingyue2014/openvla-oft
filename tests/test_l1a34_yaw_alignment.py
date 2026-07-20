import numpy as np

from experiments.robot.libero.tasks.validate_l1a34_reference import (
    _bearing_offset,
    _closing_axis_xy,
    _closing_axis_yaw_error_rad,
    _grasp_candidates,
    _quat_xyzw_to_mat,
)

# 180-degree rotation about (1,1,0)/sqrt(2): retained as a quaternion conversion
# regression.  It must not be used to infer the mounted gripper's finger axis.
DEFAULT_HAND_QUAT = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])


class _FakeModel:
    _ids = {
        "gripper0_finger_joint1_tip": 0,
        "gripper0_finger_joint2_tip": 1,
    }

    def body_name2id(self, name):
        return self._ids[name]


class _FakeEnv:
    def __init__(self, first_xy, second_xy):
        data = type("Data", (), {})()
        data.body_xpos = np.array(
            [[*first_xy, 1.0], [*second_xy, 1.0]], dtype=float
        )
        self.sim = type("Sim", (), {"model": _FakeModel(), "data": data})()


def test_quat_to_mat_reproduces_default_hand_orientation():
    mat = _quat_xyzw_to_mat(DEFAULT_HAND_QUAT)
    np.testing.assert_allclose(
        mat, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), atol=1e-12
    )


def test_closing_axis_is_measured_from_native_fingertip_bodies():
    env = _FakeEnv((0.0, -0.03), (0.0, 0.03))
    np.testing.assert_allclose(_closing_axis_xy(env), np.array([0.0, -1.0]))
    assert abs(_closing_axis_yaw_error_rad(env, -90.0)) < 1e-9
    assert abs(_closing_axis_yaw_error_rad(env, 90.0)) < 1e-9  # modulo 180


def test_east_bearing_needs_quarter_turn():
    env = _FakeEnv((0.0, -0.03), (0.0, 0.03))
    error = _closing_axis_yaw_error_rad(env, 0.0)
    assert abs(abs(error) - np.pi / 2.0) < 1e-9


def test_error_is_wrapped_to_half_turn():
    env = _FakeEnv((0.0, -0.03), (0.0, 0.03))
    for bearing in (-86.0, 94.0):
        error = _closing_axis_yaw_error_rad(env, bearing)
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


def test_bearing_offset_uses_per_axis_radius_not_flat_max(monkeypatch):
    import experiments.robot.libero.tasks.validate_l1a34_reference as mod

    # A body whose AABB is longer along y (half_x=0.03, half_y=0.06) — like the
    # elongated flat-max radius bug uncovered in the L1-A3 calibrate failures,
    # where every bearing used the SAME radius = max(half_x, half_y) = 0.06,
    # overshooting the rim off the y-axis and missing the grasp entirely.
    monkeypatch.setattr(
        mod, "_world_aabb",
        lambda _env, _body: (np.array([-0.03, -0.06, 0.0]), np.array([0.03, 0.06, 0.05])),
    )
    east = _bearing_offset(None, "body", 0.0, 1.0)
    north = _bearing_offset(None, "body", 90.0, 1.0)
    # Along each axis the offset must land exactly on that axis's half-extent,
    # not the other axis's (larger) half-extent.
    np.testing.assert_allclose(east, np.array([0.03, 0.0]), atol=1e-9)
    np.testing.assert_allclose(north, np.array([0.0, 0.06]), atol=1e-9)
    # Off-axis bearings interpolate strictly between the two extents.
    diag = _bearing_offset(None, "body", 45.0, 1.0)
    diag_radius = float(np.linalg.norm(diag))
    assert 0.03 < diag_radius < 0.06


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
