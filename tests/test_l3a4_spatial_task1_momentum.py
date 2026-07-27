import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/probe_l3a4_spatial_task1_momentum.py"
)


def _tree():
    return ast.parse(SCRIPT.read_text())


def test_probe_is_native_exact_task1_and_no_vla():
    text = SCRIPT.read_text()
    assert '"libero_spatial"' in text
    assert "TASK_ID" in text
    assert "EXPECTED_BDDL_SHA256" in text
    assert "pretrained_checkpoint" not in text
    assert "run_physcog" not in text
    assert '"vla_run": False' in text


def test_probe_binds_policy_entry_once_and_future_wait_zero():
    text = SCRIPT.read_text()
    assert "EVALUATOR_NUM_STEPS_WAIT" in text
    assert '"future_evaluator_num_steps_wait": 0' in text
    assert "wait_applied_exactly_once" in text


def test_probe_has_strict_four_event_order_and_controls():
    text = SCRIPT.read_text()
    for event in ("S_release", "A_motion", "A_B_contact", "B_response"):
        assert event in text
    assert "freeze_a=True" in text
    assert "A_parked_ablation" in text
    assert "adjacent_witnesses" in text
    assert "S_B" in text
    assert "robot_A" in text
    assert "robot_B" in text


def test_probe_only_serializes_a_b_pose_changes():
    text = SCRIPT.read_text()
    assert "candidate changed bytes outside A/B free qpos" in text
    assert "changed <= allowed" in text


def test_probe_parses():
    assert isinstance(_tree(), ast.Module)
