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
