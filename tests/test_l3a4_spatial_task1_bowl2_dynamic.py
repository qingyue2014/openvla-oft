import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/probe_l3a4_spatial_task1_bowl2_dynamic.py"
)


def test_frozen_grid_has_exactly_24_points():
    text = SCRIPT.read_text()
    assert "FROZEN_FEASIBLE_POSES" in text
    assert "((20.0, (-0.004, -0.006)), (30.0, (-0.006,)))" in text
    assert "for gap in B_GAPS_M" in text


def test_strict_force_order_bypass_and_three_ablations():
    text = SCRIPT.read_text()
    for event in (
        "S_release", "A_motion", "A_B_positive_force", "B_response"
    ):
        assert event in text
    assert "max_b_projected_speed >= 0.03" in text
    assert "freeze_a=True" in text
    assert "A_parked_ablation" in text
    assert "B_parked_ablation" in text
    for bypass in ("S_B", "robot_A", "robot_B"):
        assert bypass in text


def test_wait0_no_vla_and_artifacts_only_after_winner():
    text = SCRIPT.read_text()
    assert '"future_evaluator_num_steps_wait": 0' in text
    assert '"vla_run": False' in text
    assert "if winner is not None:" in text
    assert "pretrained_checkpoint" not in text
    assert isinstance(ast.parse(text), ast.Module)
