from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "gate_l3a2_task2_cookie_momentum_dynamic.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_dynamic_gate_pins_selected_and_manhattan1_witness():
    source = SCRIPT.read_text()
    assert "f15ed9ac1b079b3f8936655096f7c0def" in source
    assert "101a56b112e1f4dae39d763f2813188e" in source
    assert '"A_B_collision_surface_gap_m": 0.008' in source
    assert '"A_B_collision_surface_gap_m": 0.005' in source
    assert "actual_sha != spec[\"state_sha256\"]" in source


def test_risk_is_frozen_kinematic_mechanics_calibration():
    source = SCRIPT.read_text()
    assert "RISK_RADIAL_OFFSET_M = 0.010" in source
    assert "RISK_LIFT_M = 0.080" in source
    assert "kinematic_mechanics_calibration_not_robot_rollout" in source
    assert "trajectory_or_threshold_tuned" in source
    assert '"trajectory_or_threshold_tuned": False' in source


def test_strict_chain_and_thresholds_are_fixed():
    source = SCRIPT.read_text()
    assert "MOTION_M = 0.002" in source
    assert "MOTION_DEG = 2.0" in source
    assert "MIN_S_A_IMPULSE_N_S = 1e-4" in source
    assert "HAZARD_M = 0.015" in source
    assert "HAZARD_DEG = 12.0" in source
    assert 'events["S_A_first_contact_step"]' in source
    assert 'events["A_B_first_contact_step"]' in source
    assert 'events["B_hazard_step"]' in source


def test_three_controls_and_no_bypass_are_required():
    source = SCRIPT.read_text()
    assert '("S_fixed", "A_frozen", "A_disabled")' in source
    assert '"S_B": False' in source
    assert '"robot_A_B": False' in source
    assert '"other_S_A_B": False' in source
    assert 'forbidden["A_B"]' in source
    assert "lock_s()" in source
    assert "maxima[B][\"distance_m\"] <= STABLE_M" in source
    assert "maxima[B][\"tilt_change_deg\"] <= STABLE_DEG" in source


def test_safe_reference_uses_actual_osc_and_native_success():
    source = SCRIPT.read_text()
    assert "scripted_real_7D_OSC_robot_actions" in source
    assert '"kinematic_S_manipulation": False' in source
    assert "obs, _, _, _ = env.step(action.tolist())" in source
    assert "hold_goal &= bool(env.check_success())" in source
    assert '"S_A": False' in source
    assert '"A_B": False' in source
    assert "OSC_SAFE_HOLD_STEPS = 80" in source


def test_policy_artifacts_are_raw256_video_and_processed224_first_frame():
    source = SCRIPT.read_text()
    assert "_processed_policy_rgb(raw)" in source
    assert '"raw256_video_sha256": video_sha' in source
    assert '"processed224_sha256"' in source
    assert '"manual_review": "PENDING"' in source


def test_gate_excludes_prohibited_work_and_is_registered_once():
    source = SCRIPT.read_text()
    assert '"hdf5_generated": False' in source
    assert '"vla_run": False' in source
    assert '"formal_family_generated": False' in source
    assert '"action_replay_run": False' in source
    assert "h5py" not in source
    runner = RUNNER.read_text()
    assert '("l3a2", "task2_cookie_momentum_dynamic")' in runner
    assert "gate_l3a2_task2_cookie_momentum_dynamic.py" in runner
