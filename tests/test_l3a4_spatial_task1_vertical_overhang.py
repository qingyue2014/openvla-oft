from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/audit_l3a4_spatial_task1_vertical_overhang.py"
)


def test_fixed_36_native_only_points():
    text = SCRIPT.read_text()
    assert "A_OFFSETS_M = (0.012, 0.020, 0.028)" in text
    assert "B_CLEARANCES_M = (0.002, 0.006, 0.010)" in text
    assert "VERTICAL_OVERLAP_M = -0.002" in text
    assert '"candidate_count": len(rows)' in text
    assert '"custom_assets": False' in text


def test_hold_support_visibility_and_grasp_gates():
    text = SCRIPT.read_text()
    for gate in (
        "S_A", "B_table", "A_table", "A_B",
        "S_B", "robot_A", "robot_B",
    ):
        assert gate in text
    assert 'occupancy["S_A"] >= 0.95' in text
    assert "max_force > 1e-6" in text
    assert "pixels >= 40" in text
    assert "pixels / baseline_pixels >= 0.30" in text
    assert "free_margin >= 0.005" in text


def test_no_release_dynamic_or_vla():
    text = SCRIPT.read_text()
    assert '"dynamic_run": False' in text
    assert '"vla_run": False' in text
    assert "release" not in text.lower()


def test_candidate_preserves_policy_entry_qvel_bytes():
    text = SCRIPT.read_text()
    assert text.count("zero_velocity=False") == 4
    assert "candidate changed outside A/B qpos" in text
