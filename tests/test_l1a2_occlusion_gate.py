from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks import generate_l1a2_initial_states as l1a2


def _patch_measurements(monkeypatch, measurements):
    def fake_ratio(env, variant, camera, resolution=l1a2.OCCLUSION_GATE_RESOLUTION):
        return measurements[camera]

    monkeypatch.setattr(l1a2, "_occlusion_ratio", fake_ratio)


def test_er_accepts_partial_occlusion_in_any_policy_view(monkeypatch):
    _patch_measurements(
        monkeypatch,
        {
            "agentview": (0.08, 920, 1000),
            "robot0_eye_in_hand": (0.20, 800, 1000),
        },
    )
    ok, ratio, message = l1a2._occlusion_gate(object(), {})
    assert ok
    assert ratio == 0.20
    assert "robot0_eye_in_hand=0.200" in message


def test_er_uses_ten_percent_lower_bound(monkeypatch):
    _patch_measurements(
        monkeypatch,
        {
            "agentview": (0.10, 900, 1000),
            "robot0_eye_in_hand": (0.099, 901, 1000),
        },
    )
    ok, ratio, _ = l1a2._occlusion_gate(object(), {})
    assert ok
    assert ratio == 0.10


def test_ec_requires_all_visible_policy_views_to_be_clear(monkeypatch):
    _patch_measurements(
        monkeypatch,
        {
            "agentview": (0.01, 990, 1000),
            "robot0_eye_in_hand": (0.03, 970, 1000),
        },
    )
    ok, ratio, _ = l1a2._occlusion_gate(
        object(), {"is_matched_safe_control": True}
    )
    assert not ok
    assert ratio == 0.03


def test_tiny_baseline_view_does_not_create_false_occlusion(monkeypatch):
    _patch_measurements(
        monkeypatch,
        {
            "agentview": (0.16, 840, 1000),
            "robot0_eye_in_hand": (0.80, 10, 50),
        },
    )
    ok, ratio, message = l1a2._occlusion_gate(object(), {})
    assert ok
    assert ratio == 0.16
    assert "robot0_eye_in_hand" not in message


def test_main_upright_variant_does_not_gate_world_distance(monkeypatch):
    positions = {
        "target": np.array([0.0, 0.0, 0.90]),
        "cookie": np.array([0.30, 0.0, 0.94]),
        "plate": np.array([0.0, 0.30, 0.90]),
        "side": np.array([0.05, 0.0, 0.90]),
        "extra": np.array([0.30, 0.30, 0.90]),
    }
    monkeypatch.setattr(l1a2, "_body_pos", lambda _env, body: positions[body])
    variant = {
        "target_body": "target",
        "occluder_body": "cookie",
        "plate_body": "plate",
        "side_body": "side",
        "extra_side_body": "extra",
        "landmark_near_target": True,
        "use_upright_cookie_occlusion": True,
    }
    assert l1a2._layout_failure_reason(object(), variant) is None


def test_contact_distance_reports_deepest_matching_contact(monkeypatch):
    contacts = [
        SimpleNamespace(geom1=1, geom2=2, dist=-0.0004),
        SimpleNamespace(geom1=2, geom2=1, dist=-0.0030),
        SimpleNamespace(geom1=1, geom2=9, dist=-0.0100),
    ]
    env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(ncon=len(contacts), contact=contacts))
    )
    monkeypatch.setattr(
        l1a2,
        "_geom_ids_for_body",
        lambda _env, body: {1} if body == "bowl" else {2},
    )
    distance = l1a2._min_contact_distance_between_bodies(env, "bowl", "cookie")
    assert distance == -0.0030
    assert max(0.0, -distance) > l1a2.MAX_COOKIE_BOWL_PENETRATION
    assert 0.0004 <= l1a2.MAX_COOKIE_BOWL_PENETRATION
    assert l1a2._contact_between_bodies(env, "bowl", "cookie")
