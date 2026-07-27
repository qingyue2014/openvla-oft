from experiments.robot.libero.tasks.physcog_remote_agent import (
    L3A4_CHECKPOINT,
    PHASES,
    PhaseSpec,
    RemoteConfig,
    build_batch_script,
    build_isolated_sync_script,
    build_sync_script,
    classify_result,
    extract_verdicts,
    parse_markers,
)


def _config():
    return RemoteConfig(
        host="superpod.ust.hk",
        user="researcher",
        control_socket="/tmp/control socket",
        remote_repo="/home/researcher/repo with space",
        remote_python_bin="/home/researcher/conda/bin",
        branch="physcog-libero-l1",
        account="trllmout",
        partition="normal",
        nodes=1,
        gpus=2,
        time_limit="02:00:00",
    )


def test_l1a2_registry_exposes_validation_phases_without_arbitrary_shell():
    assert set(phase for scenario, phase in PHASES if scenario == "l1a2") == {
        "check",
        "formal",
        "preview",
        "safe_reference",
        "smoke",
    }
    assert PHASES[("l1a2", "safe_reference")].count_env == "SAFE_REF_STATES"
    assert "SAVE_VIDEO_MODE=all" in PHASES[("l1a2", "smoke")].command
    assert PHASES[("l1a2", "smoke")].count_env == "SMOKE_TRIALS"
    assert "experiments/logs/l1a2_smoke_videos" in PHASES[("l1a2", "smoke")].artifacts
    formal = PHASES[("l1a2", "formal")]
    assert formal.count_env == "NUM_TRIALS"
    assert "FAMILIES=l1a2" in formal.command
    assert "SEEDS=42" in formal.command


def test_l3a1_registry_exposes_only_gated_pipeline_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l3a1") == {
        "check", "geometry_sweep", "corner_sweep", "edge_sweep", "edge_preview",
        "corridor_sweep", "preview",
        "safe_reference", "smoke", "formal",
    }
    assert PHASES[("l3a1", "check")].count_env == "NUM_TRIALS"
    assert PHASES[("l3a1", "preview")].count_env == "PREVIEW_STATES"
    assert "experiments/logs/l3a1_init_evidence" in PHASES[("l3a1", "preview")].artifacts
    assert PHASES[("l3a1", "safe_reference")].count_env == "SAFE_REF_STATES"
    safe_reference = PHASES[("l3a1", "safe_reference")]
    assert "experiments/logs/l3a1_safe_reference_trajectories" in safe_reference.artifacts
    assert "experiments/logs/l3a1_safe_reference_videos" in safe_reference.artifacts
    assert "experiments/logs/l3a1_causal_reference.md" in safe_reference.artifacts
    assert PHASES[("l3a1", "smoke")].count_env == "SMOKE_TRIALS"
    corridor = PHASES[("l3a1", "corridor_sweep")]
    assert corridor.command[-1].endswith("sweep_l3a1_corridor.sh")
    assert "experiments/logs/l3a1_corridor_sweep.md" in corridor.artifacts
    corner = PHASES[("l3a1", "corner_sweep")]
    assert corner.command[-1].endswith("sweep_l3a1_corner_geometry.py")
    assert "experiments/logs/l3a1_corner_sweep.md" in corner.artifacts
    assert "experiments/logs/l3a1_corner_sweep.csv" in corner.artifacts

    edge = PHASES[("l3a1", "edge_sweep")]
    assert edge.command[-1].endswith("sweep_l3a1_edge_geometry.py")
    assert "experiments/logs/l3a1_edge_sweep.md" in edge.artifacts
    assert "experiments/logs/l3a1_edge_sweep.csv" in edge.artifacts
    assert "experiments/logs/l3a1_edge_contacts.csv" in edge.artifacts
    edge_preview = PHASES[("l3a1", "edge_preview")]
    assert edge_preview.command[-1].endswith("export_l3a1_edge_preview.py")
    assert edge_preview.artifacts == ("experiments/logs/l3a1_edge_preview",)
    formal = PHASES[("l3a1", "formal")]
    assert "FAMILIES=l3a1" in formal.command
    assert "SEEDS=42" in formal.command
    assert "SAVE_VIDEO_MODE=none" in formal.command


def test_l3a4_registry_separates_geometry_scene_and_policy_phases():
    phases = {phase for scenario, phase in PHASES if scenario == "l3a4"}
    assert {
        "geometry_sweep",
        "geometry_trace",
        "check",
        "preview",
        "safe_reference",
        "eb_replay",
        "eb_source_pilot",
        "native_task6_pilot",
        "task64_contract_audit",
        "task64_competence",
        "task55_contract_audit",
        "task55_static_probe",
        "task55_tower_v2",
        "task55_vertical_mirror_v3",
        "spatial_task1_native_audit",
        "goal_task4_fixed_dynamic",
        "smoke",
    } <= phases
    assert PHASES[("l3a4", "geometry_sweep")].count_env == "L3A4_SWEEP_TRIALS"
    assert PHASES[("l3a4", "check")].count_env == "NUM_TRIALS"
    assert "experiments/logs/l3a4_scene" in PHASES[("l3a4", "check")].artifacts
    assert not PHASES[("l3a4", "preview")].clean_artifacts_before_run
    assert PHASES[("l3a4", "safe_reference")].count_env == "SAFE_REFERENCE_STATES"
    assert PHASES[("l3a4", "safe_reference")].additional_count_envs == (
        "EB_REPLAY_EPISODES",
    )
    assert PHASES[("l3a4", "safe_reference")].environment == (
        ("CHECKPOINT", L3A4_CHECKPOINT),
    )
    assert PHASES[("l3a4", "eb_replay")].environment == (
        ("CHECKPOINT", L3A4_CHECKPOINT),
    )
    assert PHASES[("l3a4", "smoke")].environment == (
        ("CHECKPOINT", L3A4_CHECKPOINT),
    )
    source_pilot = PHASES[("l3a4", "eb_source_pilot")]
    assert source_pilot.count_env == "NUM_TRIALS"
    assert ("L3A4_EB_SOURCE_MAX_STEPS", "600") in source_pilot.environment
    assert "evaluation.eval_physcog_libero_l1()" in source_pilot.command[-1]
    native_pilot = PHASES[("l3a4", "native_task6_pilot")]
    assert native_pilot.count_env == "NUM_TRIALS"
    assert '"--task_ids", "6"' in native_pilot.command[-1]
    assert "--bddl_file" not in native_pilot.command[-1]
    assert "--initial_states_path" not in native_pilot.command[-1]
    assert "L3-A4-native-task6" not in native_pilot.command[-1]
    contract = PHASES[("l3a4", "task64_contract_audit")]
    assert "PASS_L3A4_TASK64_NATIVE_CONTRACT" in contract.command[-1]
    assert "prompt_override" in contract.command[-1]
    competence = PHASES[("l3a4", "task64_competence")]
    assert "64" in competence.command
    assert "--task_description_override" not in competence.command
    assert "--bddl_file" not in competence.command
    assert "--initial_states_path" not in competence.command
    task55_contract = PHASES[("l3a4", "task55_contract_audit")]
    assert "PASS_L3A4_TASK55_NATIVE_ONLY_CONTRACT" in task55_contract.command[-1]
    assert '"custom_assets": False' in task55_contract.command[-1]
    assert '"serialized_pose_changes_only": True' in task55_contract.command[-1]
    task55_probe = PHASES[("l3a4", "task55_static_probe")]
    assert any(
        item.endswith("probe_l3a4_task55_native_chain.py")
        for item in task55_probe.command
    )
    assert "--fail_on_invalid" in task55_probe.command
    task55_tower = PHASES[("l3a4", "task55_tower_v2")]
    assert any(
        item.endswith("probe_l3a4_task55_native_tower.py")
        for item in task55_tower.command
    )
    assert "--fail_on_invalid" in task55_tower.command
    task55_mirror = PHASES[("l3a4", "task55_vertical_mirror_v3")]
    assert any(
        item.endswith("probe_l3a4_task55_vertical_mirror.py")
        for item in task55_mirror.command
    )
    assert "--fail_on_invalid" in task55_mirror.command
    goal_task4 = PHASES[("l3a4", "goal_task4_native_static")]
    assert any(
        item.endswith("audit_l3a4_goal_task4_native_static.py")
        for item in goal_task4.command
    )
    assert goal_task4.artifacts == (
        "experiments/logs/l3a4_goal_task4_native_static",
    )
    goal_task4_dynamic = PHASES[
        ("l3a4", "goal_task4_fixed_dynamic")
    ]
    assert any(
        item.endswith("validate_l3a4_goal_task4_dynamic.py")
        for item in goal_task4_dynamic.command
    )
    assert goal_task4_dynamic.artifacts == (
        "experiments/logs/l3a4_goal_task4_dynamic",
    )


def test_batch_script_has_required_slurm_header_modules_and_fresh_artifacts():
    spec = PhaseSpec(command=("bash", "path with space/runner.sh", "phase"), count_env="N")
    script = build_batch_script(
        _config(), spec, count=8, scenario="l1a2", phase="check", remote_log="/tmp/job.out"
    )
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --partition=normal" in script
    assert "#SBATCH --account=trllmout" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --output=/tmp/job.out" in script
    assert "--pty" not in script
    assert "source /etc/profile.d/modules.sh" in script
    assert "module avail" in script
    assert 'module load slurm "nvhpc-hpcx-cuda12/23.11"' in script
    assert script.index("source /etc/profile.d/modules.sh") < script.index("set -uo pipefail")
    assert "export N=8" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert "'/home/researcher/repo with space'" in script
    assert "'path with space/runner.sh'" in script
    assert "__PHYSCOG_COMPUTE_NODE__" in script
    assert "__PHYSCOG_EXIT_CODE__" in script


def test_batch_script_exports_explicit_libero_dependency_root():
    cfg = _config()
    cfg = RemoteConfig(**{**cfg.__dict__, "libero_root": "/home/researcher/LIBERO src"})
    script = build_batch_script(
        cfg, PhaseSpec(command=("true",)), count=1,
        scenario="l3a1", phase="check", remote_log="/tmp/job.out",
    )
    assert "export PYTHONPATH='/home/researcher/LIBERO src':${PYTHONPATH:-}" in script


def test_smoke_batch_requests_five_fresh_all_video_trials():
    script = build_batch_script(
        _config(), PHASES[("l1a2", "smoke")], count=5,
        scenario="l1a2", phase="smoke", remote_log="/tmp/smoke.out"
    )
    assert "export SMOKE_TRIALS=5" in script
    assert "SAVE_VIDEO_MODE=all" in script
    assert "rm -rf experiments/logs/l1a2_smoke_videos" in script


def test_no_sync_omits_remote_checkout_mutation():
    script = build_sync_script(_config(), "/tmp/jobs", sync=False)
    assert "git fetch" not in script
    assert "git checkout" not in script
    assert "git pull" not in script
    assert "mkdir -p /tmp/jobs" in script


def test_sync_script_fast_forwards_configured_branch():
    script = build_sync_script(_config(), "/tmp/jobs", sync=True)
    assert "git fetch origin physcog-libero-l1" in script
    assert "git pull --ff-only origin physcog-libero-l1" in script


def test_isolated_sync_uses_commit_worktree_without_mutating_shared_checkout():
    script = build_isolated_sync_script(
        _config(),
        "/home/researcher/repo with space/.physcog-agent/worktrees/abc1234",
        "/home/researcher/repo with space/.physcog-agent/worktrees/abc1234/jobs",
        "abc1234",
    )
    assert "git fetch origin physcog-libero-l1" in script
    assert "git worktree add --detach" in script
    assert "abc1234" in script
    assert "git checkout" not in script
    assert "git pull" not in script


def test_verdict_extraction_understands_reports_stdout_and_pairing_manifest():
    text = """
    - Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**
    verdict=FAIL_EXAMPLE
    {"occlusion_gate": "PASS"}
    """
    assert extract_verdicts(text) == [
        "PASS_DYNAMIC_SAFE_REFERENCE",
        "FAIL_EXAMPLE",
        "PASS",
    ]


def test_classification_prioritizes_crashes_over_stale_pass_reports():
    text = "Verdict: PASS_DYNAMIC_SAFE_REFERENCE\nTraceback (most recent call last)"
    assert classify_result(1, text, extract_verdicts(text)) == "validator_bug"
    assert classify_result(0, "Verdict: FAIL_LAYOUT", ["FAIL_LAYOUT"]) == "gate_failure"
    assert classify_result(0, "Verdict: PASS_LAYOUT", ["PASS_LAYOUT"]) == "pass"
    assert classify_result(1, "srun: error: allocation failed", []) == "infrastructure_failure"


def test_classification_ignores_egl_destructor_traceback_after_success():
    text = """verdict=PASS_L1A2_SMOKE
Exception ignored in: <function EGLGLContext.__del__ at 0x123>
Traceback (most recent call last):
  File \"egl_context.py\", line 155, in __del__
OpenGL.raw.EGL._errors.EGLError: EGL_NOT_INITIALIZED
__PHYSCOG_EXIT_CODE__=0
"""
    verdicts = extract_verdicts(text)
    assert classify_result(0, text, verdicts) == "pass"


def test_remote_markers_are_parsed_for_ledger():
    output = """noise
__PHYSCOG_LOGIN_NODE__=slogin-01
__PHYSCOG_COMPUTE_NODE__=dgx-27
__PHYSCOG_COMMIT__=abc123
__PHYSCOG_EXIT_CODE__=0
"""
    assert parse_markers(output) == {
        "login_node": "slogin-01",
        "compute_node": "dgx-27",
        "commit": "abc123",
        "exit_code": "0",
    }
