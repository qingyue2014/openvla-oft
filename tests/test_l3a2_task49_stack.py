from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/robot/libero/tasks/preflight_l3a2_task49_stack.py"


def test_task49_er_gate_changes_only_a_b_and_stops_before_eb():
    source = SCRIPT.read_text()
    assert "TASK_ID = 49" in source
    assert "er[movable] = settled[movable]" in source
    assert "np.array_equal(er[~movable], base[~movable])" in source
    assert "for _ in range(300)" in source
    assert "env.step(DUMMY)" in source
    assert "current[~movable]" not in source
    assert '"eb_generated": False' in source
    assert '"vla_run": False' in source
    assert "coverage_sa >= 9" in source
    assert "coverage_ab >= 9" in source
    assert "robot_ab" in source
    assert "direct_sb" in source
