from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/audit_l3a4_spatial_task1_landmark_impact.py"
)


def test_native_roles_exact_prompt_and_no_dynamic_or_vla():
    text = SCRIPT.read_text()
    assert '"A": "cookies_1"' in text
    assert '"B": "glazed_rim_porcelain_ramekin_1"' in text
    assert "EXPECTED_BDDL_SHA256" in text
    assert "env.sim.step()" not in text
    assert '"vla_run": False' in text


def test_fixed_30_point_contact_gate():
    text = SCRIPT.read_text()
    assert "5*3*2 = 30" in text
    for gate in (
        "S_A", "A_table", "B_table", "A_B",
        "S_B", "robot_A", "robot_B",
    ):
        assert gate in text
    assert "candidate_raw_mujoco_steps" in text
