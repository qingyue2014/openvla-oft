import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/robot/libero/tasks/validate_l3a4_goal_task4_dynamic.py"
)
SOURCE = SOURCE_PATH.read_text()


def test_source_parses_and_binds_exact_static_states():
    ast.parse(SOURCE)
    assert (
        '"4bc9d9b1e5c01f60b9f0b54a68450f95'
        'a71e958fd7aae01d7f0b1e738ff722c5"'
        in SOURCE
    )
    assert (
        '"be1d1ea777c5aef0a8891779acdadd9de'
        'f9d993533a9fe2633af65cb916e7385"'
        in SOURCE
    )
    assert "FROZEN_DIRECTION = \"+y\"" in SOURCE
    assert "SELECTED_B_CLEARANCE_M = 0.002" in SOURCE
    assert "WITNESS_B_CLEARANCE_M = 0.006" in SOURCE
    assert '"selected": _sha(selected)' in SOURCE
    assert '"witness": _sha(witness)' in SOURCE
    assert '"selected": SELECTED_STATE_SHA256' in SOURCE
    assert '"witness": WITNESS_STATE_SHA256' in SOURCE
    assert "if observed != expected:" in SOURCE


def test_witness_policy_gate_hard_stops_before_any_dynamic_execution():
    assert "WITNESS_MIN_PIXELS" in SOURCE
    assert "witness_wait0_policy_agentview_256.png" in SOURCE
    assert "witness_manual_review_request.json" in SOURCE
    assert "witness_manual_approval.json" in SOURCE
    assert '"PASS_MANUAL_POLICY_RGB_REVIEW"' in SOURCE
    approval = SOURCE.index("approval = _await_witness_approval(")
    dynamic_started = SOURCE.index(
        'report["dynamic_started"] = True', approval
    )
    preflight = SOURCE.index("preflights = {", dynamic_started)
    assert approval < dynamic_started < preflight


def test_fixed_risk_schedule_and_strict_chain_thresholds():
    for token in (
        "RISK_LOW_LIFT_M = 0.008",
        "RISK_LOW_LIFT_STEPS = 20",
        "RISK_TRANSLATION_M = 0.060",
        "RISK_TRANSLATION_STEPS = 12",
        "RISK_HOLD_STEPS = 80",
        "A_RELATIVE_MOTION_M = 0.005",
        "A_RELATIVE_TILT_DEG = 5.0",
        "B_RESPONSE_M = 0.015",
        "B_RESPONSE_TILT_DEG = 12.0",
        "CARRY_CONTACT_FRACTION = 0.95",
        "CARRY_RELATIVE_M = 0.003",
        "CARRY_RELATIVE_TILT_DEG = 3.0",
    ):
        assert token in SOURCE
    assert 'int(values[0]) < int(values[1]) < int(values[2])' in SOURCE
    assert '"A_B_integrated_normal_impulse_Ns"' in SOURCE
    assert "_relative_from_snapshot(" in SOURCE
    assert '"snapshot": _snapshot(env, bodies)' in SOURCE


def test_controls_replay_complete_trace_and_preserve_visible_a_geometry():
    assert 's_trace = [row["S_pose"] for row in carry_schedule]' in SOURCE
    assert "s_trace.append(pose.copy())" in SOURCE
    assert 'for name in ("S_fixed", "A_frozen", "A_disabled")' in SOURCE
    assert "env.sim.model.geom_contype[geom] = 0" in SOURCE
    assert "env.sim.model.geom_conaffinity[geom] = 0" in SOURCE
    assert "geom_group" in SOURCE
    assert "geom_rgba" not in SOURCE


def test_safe_reference_is_actual_osc_with_official_goal_and_fixed_path():
    for token in (
        "SAFE_B_VERTICAL_CLEARANCE_M = 0.100",
        "SAFE_GOAL_APPROACH_CLEARANCE_M = 0.120",
        "SAFE_Y_STEPS = 64",
        "SAFE_X_STEPS = 46",
        "SAFE_DESCEND_STEPS = 60",
        "SAFE_CONTACT_HOLD_STEPS = 5",
        "SAFE_RELEASE_STEPS = 12",
        "SAFE_RETREAT_M = 0.080",
        "SAFE_RETREAT_STEPS = 40",
        "SAFE_FINAL_SETTLE_STEPS = 120",
        "SAFE_HORIZONTAL_MAX_ACCEL_M_S2 = 0.150",
    ):
        assert token in SOURCE
    assert '"execution_mode": "actual_robot_OSC_reference"' in SOURCE
    assert "goal_success = bool(env.check_success())" in SOURCE
    assert '"safe_contact_hold"' in SOURCE
    assert '"safe_release"' in SOURCE
    assert '"safe_retreat"' in SOURCE
    assert '"safe_final_settle"' in SOURCE
    assert "retreat_schedule" in SOURCE


def test_scope_explicitly_excludes_formal_vla_hdf5_and_action_replay():
    assert '"no_hdf5": True' in SOURCE
    assert '"no_vla": True' in SOURCE
    assert '"no_formal": True' in SOURCE
    assert '"no_action_replay": True' in SOURCE
    assert '"vla_run": False' in SOURCE
    assert '"hdf5_written": False' in SOURCE
    assert '"formal_run": False' in SOURCE
    assert '"action_replay_run": False' in SOURCE
