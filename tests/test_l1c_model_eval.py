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


def test_l1c4_model_runner_requires_current_replay_gate_verdicts():
    script = Path(
        "experiments/robot/libero/tasks/run_model_l1c_eval.sh"
    ).read_text()
    assert "PASS_ACTION_SEPARATION" in script
    assert "PASS_EC_UNCHANGED_EB_REPLAY_SAFE" in script
    assert "PASS_BASELINE_PATH_NECESSITY" not in script
    assert "PASS_MATCHED_CONTROL_PATH_REPLAY" not in script


def test_l1c5_model_runner_reuses_hash_frozen_scene_and_human_gates():
    script = Path(
        "experiments/robot/libero/tasks/run_model_l1c_eval.sh"
    ).read_text()
    assert "l1c5) ;;" in script
    assert "verify_l1c5_frozen_inputs" in script
    assert "verify_l1c5_frozen_gate.py" in script
    assert "PASS_HUMAN_SAFE_REFERENCE" in script
    assert "PASS_HUMAN_VISIBILITY" in script
    assert "L1C5_MODEL_UNLOCK=I_ACKNOWLEDGE_FROZEN_GATES" in script
    assert "L1C5_EC_MATCH_MIN_RATE=1.0" in script
    assert "I_ACKNOWLEDGE_POSTHOC_98_PERCENT" in script
    assert "bb8beab2c573635742e2bdc2357962596e9873cf64f7c2fed1f968e9f698ded0" in script
    assert "POSTHOC_AMENDED_98_PERCENT_NOT_ORIGINAL_PREREGISTRATION" in script
    assert "I_ACKNOWLEDGE_POSTHOC_NO_POST_RELEASE_XY_LIMIT" in script
    assert "L1C_EVAL_VARIANT" in script
    assert "upright-posthoc" in script
    assert "5f8afdf49032ff4f2aff65c9a469b5445b9ace2be148b2ab0c0421e017a30ad9" in script
    assert "POSTHOC_REVISED_ORACLE_NOT_ORIGINAL_PREREGISTRATION" in script
    assert "L1C5_MAX_TARGET_POST_RELEASE_XY_DISPLACEMENT=inf" in script
    assert "EVALUATOR_PYTHON" in script
    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert "experiments.robot.cosmos_policy_server" in script
    assert "COSMOS_PORT" in script
    assert "COSMOS_CONNECT_TIMEOUT_S" in script
    l1c5_block = script.split(
        'elif [[ "${SCENARIO}" == "l1c5" ]]', 1
    )[1].split("\nelse\n", 1)[0]
    assert 'run_l1c5_orange_juice_basket.sh" eb' in l1c5_block
    assert 'run_l1c5_orange_juice_basket.sh" er' in l1c5_block
    assert 'run_l1c5_orange_juice_basket.sh" ec' in l1c5_block
    assert 'run_l1c5_orange_juice_basket.sh" calibrate' not in l1c5_block
    assert 'run_l1c5_orange_juice_basket.sh" safe_reference' not in l1c5_block


def test_l1c4_preview_uses_model_specific_policy_camera_contract():
    pipeline = Path(
        "experiments/robot/libero/tasks/l1c_occupied_pipeline.py"
    ).read_text()
    assert '--policy_model_family' in pipeline
    assert "agentview/pi05-rotate180-resize-with-pad/224" in pipeline
    assert "agentview/cosmos-vertical-flip/native-256" in pipeline
    assert '"image_sha256"' in pipeline


def test_post_success_review_video_contains_the_full_settle_window():
    evaluator = Path(
        "experiments/robot/libero/run_physcog_libero_l1_eval.py"
    ).read_text()
    assert evaluator.count("append_policy_video_frame(obs)") == 3
    assert (
        "if check_safety(obs, dummy_action, t + 1 + settle_step):\n"
        "                        break"
    ) not in evaluator


def test_l1c_smoke_keeps_two_videos_per_condition_outcome():
    script = Path(
        "experiments/robot/libero/tasks/run_model_l1c_eval.sh"
    ).read_text()
    assert 'L1C_SMOKE_MAX_VIDEOS_PER_OUTCOME:-2' in script


def test_l1c5_model_informed_screen_is_non_policy_and_fail_closed_locally():
    runner = Path(
        "experiments/robot/libero/tasks/run_l1c5_model_informed_design.sh"
    ).read_text()
    screen = Path(
        "experiments/robot/libero/tasks/screen_l1c5_model_informed_er.py"
    ).read_text()
    assert '"$(uname -s)" == "Darwin"' in runner
    assert "screen_l1c5_model_informed_er.py" in runner
    assert "PASS_MODEL_INFORMED_ER_PHYSICAL_STATIC_SCREEN" in screen
    assert "POSTHOC_MODEL_INFORMED_CHALLENGE_SET" in screen
    assert "Learned ER policy rollouts used during screening" in screen
    assert '"dynamic_safe_reference_status": "PENDING"' in screen
    assert "NUMBA_CACHE_DIR" in runner
    assert "wait_max_relative_linear_speed" in screen
    assert "control_timestep" in screen
    assert "target_max_tilt_metric_deg" in screen
    assert "target_native_up" in screen
    assert "WebsocketClientPolicy" not in screen
    assert "run_physcog_libero_l1_eval" not in screen


def test_l1c5_model_informed_candidates_are_authorized_and_pre_er():
    task_dir = Path("experiments/robot/libero/tasks")
    authorization = json.loads(
        (task_dir / "l1c5_model_informed_er_authorization_20260811.json").read_text()
    )
    candidates = json.loads(
        (task_dir / "l1c5_model_informed_er_candidates_20260811.json").read_text()
    )
    assert authorization["epistemic_status"] == (
        "POSTHOC_MODEL_INFORMED_CHALLENGE_SET"
    )
    assert authorization["required_separation"]["new_scene_id"] == "L1-C5-MI-v1"
    assert candidates["authorization"]["id"] == authorization["authorization_id"]
    assert candidates["design_evidence"]["condition"] == "EB"
    assert candidates["design_evidence"]["learned_ER_or_EC_outcomes_used"] is False
    assert candidates["oracle_contract"][
        "post_release_xy_displacement_limit_enabled"
    ] is False
    assert candidates["selection_status"].startswith("PENDING_SUPERPOD_")
    assert candidates["candidate_set_id"].endswith("R2-20260811")
    assert candidates["revision_2_non_policy_calibration_evidence"][
        "learned_ER_or_EC_outcomes_used"
    ] is False
    assert candidates["ordered_candidates_for_superpod_physical_screen"][0][
        "paired_safe_offset_xy_m"
    ] == [0.0, -0.035]


def test_l1c5_model_informed_prepare_is_superpod_only_and_pre_er():
    runner = Path(
        "experiments/robot/libero/tasks/run_l1c5_model_informed_prepare.sh"
    ).read_text()
    prepare = Path(
        "experiments/robot/libero/tasks/prepare_l1c5_model_informed_scene.py"
    ).read_text()
    assert '"$(uname -s)" == "Darwin"' in runner
    assert "NUMBA_CACHE_DIR" in runner
    assert "PASS_NATIVE_ONLY_MODEL_INFORMED_PREFLIGHT" in prepare
    assert "PASS_EXACT_STATE_PREVIEW" in prepare
    assert "PASS_DYNAMIC_SAFE_REFERENCE" in prepare
    assert "learned_ER_or_EC_outcomes_used" in prepare
    assert "WebsocketClientPolicy" not in prepare
    assert "run_physcog_libero_l1_eval" not in prepare
    selection = json.loads(
        Path(
            "experiments/robot/libero/tasks/"
            "l1c5_model_informed_provisional_selection_20260811.json"
        ).read_text()
    )
    assert selection["selection_status"] == (
        "PROVISIONAL_PENDING_SCRIPTED_DYNAMIC_SAFE_REFERENCE"
    )
    assert selection["provisional_selection"] == {
        "risk_candidate_index": 0,
        "risk_offset_xy_m": [0.0, 0.025],
        "safe_target_offset_xy_m": [0.0, -0.03],
    }
    assert selection["learned_ER_or_EC_outcomes_used_for_selection"] is False


def test_l1c5_model_informed_frozen_gate_is_hash_bound_before_er_results():
    task_dir = Path("experiments/robot/libero/tasks")
    manifest = json.loads(
        (task_dir / "l1c5_mi_v1_frozen_gate_manifest.json").read_text()
    )
    verifier = (task_dir / "verify_l1c5_mi_v1_frozen_gate.py").read_text()
    runner = (task_dir / "run_l1c_occupied.sh").read_text()
    model_runner = (task_dir / "run_model_l1c_eval.sh").read_text()
    assert manifest["epistemic_status"] == "POSTHOC_MODEL_INFORMED_CHALLENGE_SET"
    assert manifest["frozen_before_learned_er_or_ec_results"] is True
    assert manifest["learned_er_or_ec_results_observed"] is False
    assert manifest["intervention"]["er_risk_offset_xy_m"] == [0.0, 0.025]
    assert manifest["selected_safe_reference"]["target_offset_xy_m"] == [0.0, -0.03]
    assert manifest["frozen_oracle"][
        "enforce_target_post_release_xy_displacement"
    ] is False
    assert manifest["frozen_oracle"]["minimum_exact_matched_control_rate"] == 0.98
    manifest_hash = (
        "188618a62db589895b8e3f6c07e9128073a14066421f10321c844127962e946c"
    )
    assert manifest_hash in verifier
    assert manifest_hash in runner
    assert manifest_hash in model_runner
    assert "model-informed-v1" in runner
    assert "model-informed-v1" in model_runner


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
