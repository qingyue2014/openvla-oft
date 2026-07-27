from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "experiments/robot/libero/tasks/task49_v2_eb_competence.py"
SHELL = ROOT / "experiments/robot/libero/tasks/run_l3a2_task49_v2_eb_competence.sh"
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_task49_v2_eb_is_exact_native_single_episode():
    source = SHELL.read_text()
    assert "--task_suite_name libero_90" in source
    assert "--task_ids 49" in source
    assert "--safety_oracle none" in source
    assert "--num_trials_per_task 1" in source
    assert "--num_steps_wait 0" in source
    assert "--save_video_mode all" in source
    assert "--save_trajectory True" in source
    assert "--task_description_override" not in source
    assert "task49_v2_eb_competence.py" in source


def test_task49_v2_eb_binds_contract_actions_video_and_failure_stop():
    source = PYTHON.read_text()
    assert "BASE_SHA256" in source
    assert "runtime_contract_sha256" in source
    assert "relevant_world_poses" in source
    assert "cameras" in source
    assert "model_summary" in source
    assert "actions.shape[1] != 7" in source
    assert "expected exactly one rollout video" in source
    assert '"hard_stop_on_failure": True' in source
    assert '"additional_episode_run": False' in source
    assert '"task_description_override": None' in source
    assert '"FAIL_BASE_TASK_COMPETENCE"' in source
    assert "if not success:" in source
    assert "raise SystemExit(2)" in source


def test_task49_v2_eb_phase_is_registered():
    source = RUNNER.read_text()
    assert '("l3a2", "task49_v2_eb_competence")' in source
    assert "run_l3a2_task49_v2_eb_competence.sh" in source
