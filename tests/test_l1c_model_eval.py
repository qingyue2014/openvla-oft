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


def test_l1c4_model_runner_is_hash_bound_and_never_recalibrates_frozen_scene():
    script = Path(
        "experiments/robot/libero/tasks/run_model_l1c_eval.sh"
    ).read_text()
    assert "l1c4) ;;" in script
    assert "cdbc127944eefd6db20ee085e4d3778a2da0b45cffafcc03d3a8b3db066581c7" in script
    assert "4abd7874cae7232c3c8d1275617f89b88450c4ada2e017f7a1ca10a74e9141a0" in script
    assert "0a2073f67bc57012c645bc7cdb3f91d0bde55c3826b3b838b6ce4a800a801e4b" in script
    l1c4_block = script.split('elif [[ "${SCENARIO}" == "l1c4" ]]', 1)[1].split("else", 1)[0]
    assert " calibrate" not in l1c4_block
    assert " safe_reference" not in l1c4_block
    assert "verify_l1c4_frozen_inputs" in l1c4_block


def test_l1c4_preview_uses_model_specific_policy_camera_contract():
    pipeline = Path(
        "experiments/robot/libero/tasks/l1c_occupied_pipeline.py"
    ).read_text()
    assert '--policy_model_family' in pipeline
    assert "agentview/pi05-rotate180-resize-with-pad/224" in pipeline
    assert "agentview/cosmos-vertical-flip/native-256" in pipeline
    assert '"image_sha256"' in pipeline


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
    assert "--num_open_loop_steps" in script
    assert "--render_gpu_device_id" in script
