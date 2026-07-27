from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/robot/libero/tasks/preflight_l3a2_task33_mug_cascade.py"
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_task33_scan_is_bounded_native_only_and_strict():
    source = SCRIPT.read_text()
    assert "TASK_ID = 33" in source
    assert "itertools.product(A_X, B_X, Y)" in source
    assert "candidate_count" in source
    assert "outside_a_b_bit_exact" in source
    assert "initial_support" in source
    assert "first[\"s_motion\"] < first[\"s_a_release\"]" in source
    assert "first[\"s_a_release\"] <= first[\"a_motion\"]" in source
    assert "first[\"a_b\"] < first[\"b_hazard\"]" in source
    assert "direct_s_b_contact" in source
    assert "robot_a_b_contact" in source
    assert '"S_no_close"' in source
    assert '"A_native_pose"' in source
    assert '"B_removed"' in source
    assert '"eb_generated": False' in source
    assert '"vla_run": False' in source


def test_task33_scan_registered_as_single_phase():
    source = RUNNER.read_text()
    assert '("l3a2", "task33_physical_scan")' in source
    assert "preflight_l3a2_task33_mug_cascade.py" in source
