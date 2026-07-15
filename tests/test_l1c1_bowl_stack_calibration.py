from experiments.robot.libero.tasks.calibrate_l1c1_bowl_stack import assess_feasibility


def test_feasible_when_centred_stack_is_reliable():
    rows = [
        {"offset_m": 0.0, "safe": 1},
        {"offset_m": 0.0, "safe": 1},
        {"offset_m": 0.01, "safe": 0},
    ]
    verdict, _, rate = assess_feasibility(rows, min_center_safe_rate=0.9)
    assert verdict == "PASS_STACK_PHYSICALLY_FEASIBLE"
    assert rate == 1.0


def test_rejects_unreliable_centred_stack():
    rows = [
        {"offset_m": 0.0, "safe": 1},
        {"offset_m": 0.0, "safe": 0},
    ]
    verdict, _, rate = assess_feasibility(rows, min_center_safe_rate=0.9)
    assert verdict == "FAIL_STACK_NOT_RELIABLY_STABLE"
    assert rate == 0.5
