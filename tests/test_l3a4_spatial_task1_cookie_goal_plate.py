from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/audit_l3a4_spatial_task1_cookie_goal_plate.py"
)


def test_exact_native_contract_and_goal_semantics():
    text = SCRIPT.read_text()
    assert '"S": "akita_black_bowl_1"' in text
    assert '"A": "cookies_1"' in text
    assert '"B": "plate_1"' in text
    assert "(:goal (And (On akita_black_bowl_1 plate_1)) )" in text
    assert '"prompt_override": False' in text
    assert "EXPECTED_BASE_SHA256" in text
    assert '"future_evaluator_num_steps_wait": 0' in text


def test_frozen_36_points_and_40_240_80_protocol():
    text = SCRIPT.read_text()
    assert "A_OFFSETS_M = (0.012, 0.020, 0.028)" in text
    assert "B_CLEARANCES_M = (0.002, 0.006, 0.010)" in text
    assert 'default=40' in text
    assert 'default=240' in text
    assert 'default=80' in text
    assert '"candidate_count": len(rows)' in text


def test_static_support_visibility_and_grasp_gates():
    text = SCRIPT.read_text()
    for gate in (
        "S_A", "A_table", "B_table", "A_B", "S_B",
        "robot_A", "robot_B", "A_landmark", "A_other_bowl",
    ):
        assert gate in text
    assert 'occupancy["S_A"] >= 0.95' in text
    assert "max_force > 1e-6" in text
    assert 'pixels["S"] >= 40' in text
    assert 'pixels["A"] >= 30' in text
    assert 'pixels["B"] >= 30' in text
    assert "free_margin >= 0.005" in text


def test_no_release_dynamic_vla_or_hdf5():
    text = SCRIPT.read_text()
    assert '"release_run": False' in text
    assert '"dynamic_run": False' in text
    assert '"vla_run": False' in text
    assert '"hdf5_written": False' in text
    assert '"custom_assets": False' in text
