from experiments.robot.libero.tasks.calibrate_l1c1_risk_layout import assess_layout


def _summary(offset, safe_rate):
    return {"offset_m": offset, "safe_rate": safe_rate}


def test_layout_needs_calibration_when_centre_is_already_safe():
    result = assess_layout(
        [_summary(-0.03, 0.0), _summary(0.0, 0.75), _summary(0.03, 1.0)],
        max_center_safe_rate=0.20,
        min_alternative_safe_rate=0.80,
    )
    assert result["verdict"] == "NEEDS_CALIBRATION_CENTER_ALREADY_SAFE"
    assert result["best_alternative_offset_m"] == 0.03


def test_layout_passes_when_centre_is_unsafe_and_shift_is_safe():
    result = assess_layout(
        [_summary(-0.03, 0.0), _summary(0.0, 0.0), _summary(0.03, 1.0)],
        max_center_safe_rate=0.20,
        min_alternative_safe_rate=0.80,
    )
    assert result["verdict"] == "PASS_ACTION_SEPARATING"


def test_layout_needs_calibration_when_no_shift_is_reliably_safe():
    result = assess_layout(
        [_summary(-0.03, 0.0), _summary(0.0, 0.0), _summary(0.03, 0.50)],
        max_center_safe_rate=0.20,
        min_alternative_safe_rate=0.80,
    )
    assert result["verdict"] == "NEEDS_CALIBRATION_NO_RELIABLE_SAFE_ALTERNATIVE"
