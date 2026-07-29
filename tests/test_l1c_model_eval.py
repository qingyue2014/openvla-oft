import json
from pathlib import Path

import pytest

from experiments.robot.libero.tasks.summarize_l1c_model_eval import (
    _condition_metrics,
    _load_index,
)


def test_l1c_summary_counts_safe_success_separately_from_task_success():
    rows = [
        {"episode_idx": 0, "success": True, "violated": False},
        {"episode_idx": 1, "success": True, "violated": True},
        {"episode_idx": 2, "success": False, "violated": False},
    ]
    metrics = _condition_metrics(rows)
    assert metrics["task_successes"] == 2
    assert metrics["safety_violations"] == 1
    assert metrics["safe_successes"] == 1
    assert metrics["safe_success_rate"] == pytest.approx(1 / 3)


def test_l1c_summary_rejects_missing_or_reordered_episodes(tmp_path):
    path = tmp_path / "index.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"episode_idx": 1, "success": True, "violated": False},
                {"episode_idx": 0, "success": True, "violated": False},
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="exact ordered range"):
        _load_index(path, 2)


def test_l1c_model_runner_enforces_gates_and_outcome_labeled_videos():
    script = Path(
        "experiments/robot/libero/tasks/run_model_l1c_eval.sh"
    ).read_text()
    assert "PASS_STACK_PHYSICALLY_FEASIBLE" in script
    assert "PASS_STATIC_OCCUPANCY_LAYOUT" in script
    assert "PASS_DYNAMIC_SAFE_REFERENCE" in script
    assert '"${VIDEO_DIR}/${condition}_safe-success.mp4"' in script
    assert '"${VIDEO_DIR}/${condition}_violation.mp4"' in script
    assert '"${VIDEO_DIR}/${condition}_task-failure.mp4"' in script


def test_l1c1_uses_one_stack_drop_bound_for_reference_policy_and_replay():
    script = Path(
        "experiments/robot/libero/tasks/run_l1c1_task2.sh"
    ).read_text()
    assert 'MAX_UPPER_DROP="${MAX_UPPER_DROP:-0.030}"' in script
    assert '--max_upper_drop "${MAX_UPPER_DROP}"' in script
    assert '--native_stack_max_upper_drop "${MAX_UPPER_DROP}"' in script


def test_l1c_replay_capture_preserves_selected_model_preprocessing():
    evaluator = Path(
        "experiments/robot/libero/run_physcog_libero_l1_eval.py"
    ).read_text()
    capture_start = evaluator.index("def capture_replay_observation")
    capture_end = evaluator.index("\n    def check_safety", capture_start)
    capture_body = evaluator[capture_start:capture_end]
    assert "cfg.model_family" in capture_body


@pytest.mark.parametrize(
    "runner",
    (
        "experiments/robot/libero/tasks/run_l1c1_task2.sh",
        "experiments/robot/libero/tasks/run_l1c_occupied.sh",
    ),
)
def test_l1c_runners_forward_selected_model_contract(runner):
    script = Path(runner).read_text()
    assert "--model_family" in script
    assert "--pi05_host" in script
    assert "--pi05_port" in script
    assert "--pi05_replan_steps" in script
    assert "--gr00t_n16_host" in script
    assert "--gr00t_n16_port" in script
    assert "--gr00t_n16_request_timeout_s" in script
    assert "--num_open_loop_steps" in script
    assert "--render_gpu_device_id" in script
