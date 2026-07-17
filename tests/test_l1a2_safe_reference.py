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
