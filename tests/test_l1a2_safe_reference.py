from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks import generate_l1a2_initial_states as generator
from experiments.robot.libero.tasks import validate_l1a2_safe_reference as reference


class _NativeSuccessEnv:
    def __init__(self, success=True):
        self._success = success

    def check_success(self):
        return self._success


def test_native_goal_is_authoritative_over_asset_aabb_gap(monkeypatch):
    positions = {
        reference.TARGET: np.array([0.0, 0.0, 0.9]),
        reference.PLATE: np.array([0.0, 0.0, 0.9]),
    }
    monkeypatch.setattr(reference, "_body_pos", lambda _env, body: positions[body])
    monkeypatch.setattr(
        generator,
        "_world_aabb",
        lambda _env, body: (
            np.array([-0.05, -0.05, 0.80 if body == reference.TARGET else 0.85]),
            np.array([0.05, 0.05, 0.95 if body == reference.TARGET else 0.90]),
        ),
    )
    result = reference._bowl_on_plate(
        _NativeSuccessEnv(True),
        SimpleNamespace(max_place_xy_offset=0.06, max_place_height_gap=0.03),
    )
    assert result["task_success"]
    assert result["native_task_success"]
    assert np.isclose(result["place_bottom_gap_m"], -0.10)


def test_attempt_ranking_prefers_complete_safe_success():
    failed = {
        "safe_success": 0,
        "native_task_success": 1,
        "occluder_stable": 1,
        "grasp_verified": 1,
        "place_xy_offset_m": 0.001,
    }
    safe = dict(failed, safe_success=1, place_xy_offset_m=0.02)
    assert reference._reference_attempt_score(safe) > reference._reference_attempt_score(
        failed
    )


def test_transport_aborts_when_grasped_body_stops_following_eef(monkeypatch):
    obs = {"robot0_eef_pos": np.zeros(3)}
    args = SimpleNamespace(
        position_tolerance=0.001,
        max_waypoint_steps=10,
        max_position_command=0.25,
        position_scale=0.08,
        max_grasp_offset_drift=0.005,
    )

    def advance(_env, _obs, _oracle, _recorder, _action, step):
        return {"robot0_eef_pos": np.array([0.01, 0.0, 0.0])}, SimpleNamespace(
            violated=False
        )

    monkeypatch.setattr(reference, "_advance", advance)
    monkeypatch.setattr(reference, "_body_pos", lambda _env, _body: np.zeros(3))
    _, _, failure = reference._move_to(
        object(),
        obs,
        object(),
        object(),
        np.array([1.0, 0.0, 0.0]),
        1.0,
        0,
        args,
        "translate_above_plate",
        retained_body=reference.TARGET,
        retained_offset=np.zeros(3),
    )

    assert failure.reason == "grasp_slipped"
    assert failure.stage == "translate_above_plate"


def test_layout_displacement_gate_rejects_objects_that_settle_far_from_request(
    monkeypatch,
):
    requested = {"target": np.array([-0.075, 0.010])}
    monkeypatch.setattr(
        generator,
        "_body_pos",
        lambda _env, _body: np.array([-0.145, -0.004, 0.9]),
    )

    errors, displaced = generator._layout_xy_displacements(object(), requested)

    assert errors["target"] > generator.MAX_LAYOUT_XY_ERROR
    assert displaced == errors
