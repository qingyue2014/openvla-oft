from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "preflight_l3a2_task1_cantilever_kinematic.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_kinematic_calibration_pins_selected_and_adjacent_state_hashes():
    source = SCRIPT.read_text()
    assert '"name": "selected"' in source
    assert '"name": "adjacent_clearance_witness"' in source
    assert "61e32aacd2f721d67269c5c0319397b99" in source
    assert "ba3e605771b0ed5ec0e29cfcfd4a8bfb" in source
    assert '"A_point": (0.026, -4.0)' in source
    assert '"B_point": (0.001, -0.010)' in source
    assert '"B_point": (0.003, -0.010)' in source


def test_kinematic_risk_path_has_strict_chain_and_no_bypass():
    source = SCRIPT.read_text()
    assert "EAST_TRANSLATION_M = 0.060" in source
    assert "LIFT_M = 0.080" in source
    assert "EAST_STEPS = 30" in source
    assert "LIFT_STEPS = 40" in source
    assert '< int(events["A_motion_step"])' in source
    assert '< int(events["A_B_first_contact_step"])' in source
    assert '<= int(events["B_response_step"])' in source
    assert '"S_B_bypass": bypass' in source
    assert '"robot_S_A_B_contact": robot_contact' in source
    assert '"S_A_recontact_before_A_B_impact"' in source
    assert '"A_B_first_contact_details"' in source
    assert '"B_velocity_at_first_A_B_contact"' in source


def test_kinematic_has_both_controls_for_both_states():
    source = SCRIPT.read_text()
    assert "S_FIXED_STEPS = 250" in source
    assert '"S_fixed": s_fixed' in source
    assert '"A_collision_disabled": a_disabled' in source
    assert "A_collision_disabled_then_same_east_first_S_path" in source
    assert "for spec in STATE_SPECS:" in source
    assert 's_fixed["passed"]' in source
    assert 'a_disabled["passed"]' in source


def test_safe_path_is_lift_first_and_checks_original_on_predicate():
    source = SCRIPT.read_text()
    assert '"lift_first"' in source
    assert '"east_second"' in source
    assert '"translate_high_to_plate"' in source
    assert '"lower_to_plate"' in source
    assert "env.check_success()" in source
    assert "env._check_success()" not in source
    assert '"persistent_original_task_On_predicate"' in source
    assert '"terminal_original_task_On_predicate"' in source
    assert '"persistent_S_plate_contact"' in source
    assert '"terminal_S_plate_contact"' in source


def test_policy_artifacts_are_restored_exact256_and_include_two_videos():
    source = SCRIPT.read_text()
    assert "_restore(env, state)" in source
    assert "first = _policy_image(env)" in source
    assert "camera_heights=256" in source
    assert "camera_widths=256" in source
    assert 'f"{spec[\'name\']}_policy_first_frame.png"' in source
    assert 'f"{spec[\'name\']}_east_first_risk.mp4"' in source
    assert '"manual_review": "PENDING"' in source


def test_calibration_is_not_robot_vla_hdf5_or_formal():
    source = SCRIPT.read_text()
    assert "kinematic_support_removal_not_robot_rollout" in source
    assert '"robot_rollout": False' in source
    assert '"hdf5_generated": False' in source
    assert '"vla_run": False' in source
    assert '"formal_family_generated": False' in source
    assert "h5py" not in source


def test_kinematic_calibration_is_one_registered_phase():
    runner = RUNNER.read_text()
    assert '("l3a2", "task1_cantilever_kinematic")' in runner
    assert "preflight_l3a2_task1_cantilever_kinematic.py" in runner
