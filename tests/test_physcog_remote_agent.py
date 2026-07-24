from experiments.robot.libero.tasks.physcog_remote_agent import (
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


def test_l1b1_registry_exposes_capture_lift_gated_remote_pipeline():
    assert set(phase for scenario, phase in PHASES if scenario == "l1b1") == {
        "ec_calibrate",
        "ec_static_search",
        "prepare",
        "smoke",
        "formal",
    }
    prepare = PHASES[("l1b1", "prepare")]
    assert prepare.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in prepare.command
    assert "l1b1_native_gripper" in prepare.command
    assert "prepare" in prepare.command
    assert any("pairing.json" in artifact for artifact in prepare.artifacts)
    assert any("safe_reference.md" in artifact for artifact in prepare.artifacts)

    smoke = PHASES[("l1b1", "smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "SAVE_VIDEO_MODE=all" in smoke.command
    assert any("native_replay.md" in artifact for artifact in smoke.artifacts)
    for condition in ("eb", "er", "ec"):
        assert any(
            f"capture-lift-v4-{condition}" in artifact
            for artifact in smoke.artifacts
        )

    formal = PHASES[("l1b1", "formal")]
    assert formal.count_env == "NUM_TRIALS"
    assert "SAVE_VIDEO_MODE=none" in formal.command
    assert "all" in formal.command
    for suffix in (
        "scene_check.md",
        "safe_reference.md",
        "native_replay.md",
        "eb_rollout_physics.md",
        "er_rollout_physics.md",
        "ec_rollout_physics.md",
    ):
        assert any(artifact.endswith(suffix) for artifact in formal.artifacts)

    ec_calibrate = PHASES[("l1b1", "ec_calibrate")]
    assert ec_calibrate.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=0" in ec_calibrate.command
    assert "ec_calibrate" in ec_calibrate.command
    assert any(
        artifact.endswith("ec_rollout_physics.md")
        for artifact in ec_calibrate.artifacts
    )

    ec_search = PHASES[("l1b1", "ec_static_search")]
    assert ec_search.count_env is None
    assert any(
        part.endswith("search_l1b1_ec_static_angles.py")
        for part in ec_search.command
    )


def test_l1b2_registry_exposes_calibration_and_gated_evaluation_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l1b2") == {
        "calibrate", "search", "path_calibrate", "prepare", "smoke", "pool_smoke",
        "formal", "ec_repair", "ec_video",
    }
    assert PHASES[("l1b2", "calibrate")].count_env == "CALIBRATION_TRIALS"
    assert any(
        value.endswith("calibrate_l1b2_wine_bottle.sh")
        for value in PHASES[("l1b2", "calibrate")].command
    )
    assert PHASES[("l1b2", "prepare")].count_env == "NUM_TRIALS"
    search = PHASES[("l1b2", "search")]
    assert "--laterals=0.055,0.060,0.065,0.070,0.075" in search.command
    assert "--max_episodes" in search.command
    assert "50" in search.command
    assert "--min_obstacle_tilt_change_deg" in search.command
    assert "--fail_on_invalid" in PHASES[("l1b2", "path_calibrate")].command
    assert "SAFE_REF_VIDEO_DIR=experiments/logs/l1b2_safe_reference_videos" in (
        PHASES[("l1b2", "smoke")].command
    )
    assert PHASES[("l1b2", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l1b2", "pool_smoke")].count_env == "NUM_TRIALS"
    assert "L1B2_CALIBRATION_POOL_SIZE=10" in PHASES[("l1b2", "pool_smoke")].command
    assert "L1B2_ER_PHYSICS_QUALIFICATION_SIZE=5" in PHASES[("l1b2", "pool_smoke")].command
    assert "REPLAY_MIN_EPISODES=2" in PHASES[("l1b2", "pool_smoke")].command
    assert "experiments/logs/l1b2_trajectory_conditioned_calibration.md" in (
        PHASES[("l1b2", "smoke")].artifacts
    )
    formal = PHASES[("l1b2", "formal")]
    assert formal.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in formal.command
    assert "SAVE_VIDEO_MODE=none" in formal.command
    assert "experiments/logs/l1b2_trajectory_conditioned_calibration.csv" in (
        formal.artifacts
    )
    assert "experiments/logs/l1b2_er_physics_qualification.md" in formal.artifacts
    assert "RENDER_GPU_DEVICE_ID=1" in PHASES[("l1b2", "ec_repair")].command
    assert (
        "experiments/robot/libero/tasks/run_l1b2_native_ec_repair.sh"
        in PHASES[("l1b2", "ec_repair")].command
    )


def test_l1b3_registry_fetches_machine_readable_gate_evidence():
    for phase in ("smoke", "formal"):
        artifacts = PHASES[("l1b3", phase)].artifacts
        assert "experiments/logs/l1b3_trajectory_conditioned_calibration.csv" in artifacts
        assert "experiments/logs/l1b3_native_arm_native_replay.csv" in artifacts
        assert "experiments/logs/l1b3_native_arm_safe_reference.csv" in artifacts


def test_l3a1_registry_exposes_only_gated_pipeline_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l3a1") == {
        "check", "geometry_sweep", "safe_reference", "smoke", "formal",
    }
    assert PHASES[("l3a1", "check")].count_env == "NUM_TRIALS"
    assert PHASES[("l3a1", "safe_reference")].count_env == "SAFE_REF_STATES"
    assert PHASES[("l3a1", "smoke")].count_env == "SMOKE_TRIALS"
    formal = PHASES[("l3a1", "formal")]
    assert "FAMILIES=l3a1" in formal.command
    assert "SEEDS=42" in formal.command
    assert "SAVE_VIDEO_MODE=none" in formal.command


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


def test_batch_script_can_exclude_unstable_render_nodes():
    cfg = RemoteConfig(**{**_config().__dict__, "exclude_nodes": "dgx-29"})
    script = build_batch_script(
        cfg, PhaseSpec(command=("true",)), count=1,
        scenario="l1b2", phase="ec_repair", remote_log="/tmp/job.out",
    )
    assert "#SBATCH --exclude=dgx-29" in script


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
