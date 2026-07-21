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


def test_l1b5_registry_exposes_strict_gated_remote_pipeline():
    assert set(phase for scenario, phase in PHASES if scenario == "l1b5") == {
        "prepare", "smoke", "calibration_eb", "geometry_sweep",
        "geometry_refine", "replay_calibration", "formal",
    }
    prepare = PHASES[("l1b5", "prepare")]
    assert prepare.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in prepare.command
    assert "l1b5_native_gripper" in prepare.command
    assert "prepare" in prepare.command
    assert any("pairing.json" in artifact for artifact in prepare.artifacts)
    assert any("safe_reference.md" in artifact for artifact in prepare.artifacts)

    smoke = PHASES[("l1b5", "smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in smoke.command
    assert "SAVE_VIDEO_MODE=all" in smoke.command
    for condition in ("eb", "er", "ec"):
        assert any(
            f"-{condition}-smoke-seed42" in artifact
            for artifact in smoke.artifacts
        )

    calibration_eb = PHASES[("l1b5", "calibration_eb")]
    assert calibration_eb.count_env == "NUM_TRIALS"
    assert "RUN_ID_SUFFIX=calibration-seed42" in calibration_eb.command
    assert "SAVE_VIDEO_MODE=none" in calibration_eb.command

    replay_calibration = PHASES[("l1b5", "replay_calibration")]
    assert replay_calibration.count_env is None
    assert "RUN_ID_SUFFIX=calibration-seed42" in replay_calibration.command
    assert "replay_calibration" in replay_calibration.command
    assert any("native_replay.md" in artifact for artifact in replay_calibration.artifacts)
    assert any("control_replay.md" in artifact for artifact in replay_calibration.artifacts)
    assert not any("native_replay" in artifact for artifact in smoke.artifacts)

    geometry_sweep = PHASES[("l1b5", "geometry_sweep")]
    assert geometry_sweep.count_env is None
    assert any(
        value.endswith("search_l1b_native_replay_positions.py")
        for value in geometry_sweep.command
    )
    assert "0.40,0.45,0.50,0.55,0.60" in geometry_sweep.command
    assert "0.0" in geometry_sweep.command

    geometry_refine = PHASES[("l1b5", "geometry_refine")]
    assert geometry_refine.count_env is None
    assert "0.46,0.48,0.50,0.52,0.54,0.56" in geometry_refine.command
    assert "0.0" in geometry_refine.command

    formal = PHASES[("l1b5", "formal")]
    assert formal.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in formal.command
    assert "FAMILIES=l1b5" in formal.command
    assert "SEEDS=42" in formal.command
    assert any("result_tables.md" in artifact for artifact in formal.artifacts)
    assert any("control_replay.md" in artifact for artifact in formal.artifacts)
    assert any("l1b5_attribution.md" in artifact for artifact in formal.artifacts)


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
