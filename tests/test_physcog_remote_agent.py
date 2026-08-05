from pathlib import Path

from experiments.robot.libero.tasks.physcog_remote_agent import (
    PHASES,
    PhaseSpec,
    RemoteConfig,
    build_batch_script,
    build_isolated_sync_script,
    build_sync_script,
    classify_result,
    extract_registered_review_videos,
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
        "check", "geometry_sweep", "safe_reference", "smoke", "formal",
    }
    assert PHASES[("l3a1", "check")].count_env == "NUM_TRIALS"
    assert PHASES[("l3a1", "safe_reference")].count_env == "SAFE_REF_STATES"
    assert PHASES[("l3a1", "smoke")].count_env == "SMOKE_TRIALS"
    formal = PHASES[("l3a1", "formal")]
    assert "FAMILIES=l3a1" in formal.command
    assert "SEEDS=42" in formal.command
    assert "SAVE_VIDEO_MODE=none" in formal.command


def test_l1c1_registry_exposes_gated_formal_pipeline_and_calibration_tools():
    assert set(phase for scenario, phase in PHASES if scenario == "l1c1") == {
        "init", "preview", "validate_layout", "first_policy_gate", "repair_eb", "safe_reference", "smoke", "formal",
        "recalibrate", "recalibrate15", "direction_sweep",
        "angle0", "angle45", "angle90", "angle135",
        "angle225", "angle270", "angle315",
    }
    assert PHASES[("l1c1", "init")].count_env == "NUM_TRIALS"
    assert "RISK_DEPENDENT_XY_ANGLE_DEG=135" in PHASES[("l1c1", "init")].command
    assert PHASES[("l1c1", "preview")].count_env == "PREVIEW_NUM_STATES"
    assert "bowl_stack_validate" in PHASES[("l1c1", "validate_layout")].command
    first_policy = PHASES[("l1c1", "first_policy_gate")]
    assert first_policy.count_env == "NUM_TRIALS"
    assert "bowl_stack_first_policy_gate" in first_policy.command
    assert len(first_policy.inputs) == 3
    assert "review/L1-C1_task/first_policy_gate" in first_policy.artifacts
    repair_eb = PHASES[("l1c1", "repair_eb")]
    assert repair_eb.count_env == "NUM_TRIALS"
    assert "bowl_stack_repair_eb" in repair_eb.command
    assert len(repair_eb.inputs) == 3
    assert "experiments/logs/l1c1_eb_repair_build.json" in repair_eb.artifacts
    safe_reference = PHASES[("l1c1", "safe_reference")]
    assert safe_reference.count_env == "CALIBRATION_NUM_STATES"
    assert "RENDER_GPU_DEVICE_ID=1" in safe_reference.command
    assert "bowl_stack_safe_reference" in safe_reference.command
    assert "experiments/logs/l1c1_safe_reference_videos" in safe_reference.artifacts
    smoke = PHASES[("l1c1", "smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "SAVE_VIDEO_MODE=all" in smoke.command
    assert "bowl_stack_smoke" in smoke.command
    assert len(smoke.inputs) == 5
    assert any("smoke-repaired" in artifact for artifact in smoke.artifacts)
    formal = PHASES[("l1c1", "formal")]
    assert formal.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in formal.command
    assert "SAVE_VIDEO_MODE=all" in formal.command
    assert any("bowl_stack_eb_repaired_states.hdf5" in item for item in formal.command)
    assert len(formal.inputs) == 7
    assert any(
        target.endswith("l1c1_task2_bowl_stack_eb_repaired_states.hdf5")
        for _, target in formal.inputs
    )
    assert any(
        target.endswith("repaired_eb_smoke/HUMAN_REVIEW.json")
        for _, target in formal.inputs
    )
    assert "bowl_stack_eval" in formal.command
    phase = PHASES[("l1c1", "recalibrate")]
    assert phase.count_env == "NUM_TRIALS"
    assert "RISK_DEPENDENT_XY_OFFSET=0.0125" in phase.command
    assert "bowl_stack_recalibrate" in phase.command
    assert "experiments/logs/l1c1_bowl_stack_eb_replay.md" in phase.artifacts
    assert "RISK_DEPENDENT_XY_OFFSET=0.015" in PHASES[
        ("l1c1", "recalibrate15")
    ].command
    assert "bowl_stack_direction_sweep" in PHASES[
        ("l1c1", "direction_sweep")
    ].command
    assert "RISK_DEPENDENT_XY_ANGLE_DEG=90" in PHASES[
        ("l1c1", "angle90")
    ].command


def test_l1c1_preview_and_formal_reuse_the_generated_state_bundle():
    runner = Path("experiments/robot/libero/tasks/run_l1c1_task2.sh").read_text()
    preview_body = runner.split("run_bowl_stack_preview() {", 1)[1].split("\n}", 1)[0]
    formal_body = runner.split("  bowl_stack_eval)\n", 1)[1].split("\n    ;;", 1)[0]
    assert "generate_bowl_stack_candidate" not in preview_body
    assert "require_states" in preview_body
    assert "generate_bowl_stack_candidate" not in formal_body
    assert "require_repaired_eb_formal_gate" in formal_body
    assert "run_bowl_stack_validation" in formal_body
    assert "prepare_bowl_stack_formal_outputs" in formal_body


def test_l1c1_smoke_reuses_frozen_repaired_bundle():
    runner = Path("experiments/robot/libero/tasks/run_l1c1_task2.sh").read_text()
    smoke_body = runner.split("  bowl_stack_smoke)\n", 1)[1].split("\n    ;;", 1)[0]
    assert "generate_bowl_stack_candidate" not in smoke_body
    assert "require_repaired_eb_static_gate" in smoke_body
    assert "run_bowl_stack_native_preflight" in smoke_body
    assert "require_bowl_stack_bundle" in smoke_body
    assert "summarize_l1c1_smoke.py" in smoke_body


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
    assert 'export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/physcog-numba-' in script
    assert 'export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/physcog-triton-' in script
    assert 'mkdir -p "$NUMBA_CACHE_DIR"' in script
    assert 'mkdir -p "$TRITON_CACHE_DIR"' in script
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


def test_batch_script_exports_shared_repo_for_read_only_paired_inputs():
    script = build_batch_script(
        _config(), PhaseSpec(command=("true",)), count=1,
        scenario="l1c1", phase="recalibrate", remote_log="/tmp/job.out",
        shared_repo="/home/researcher/shared repo",
    )
    assert "export PHYSCOG_SHARED_REPO='/home/researcher/shared repo'" in script


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


def test_review_video_fetching_is_limited_to_registered_formal_prefix():
    text = """
    `rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb/episode=1.mp4`
    `rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk/episode=1.mp4`
    `rollouts/libero_spatial/L1-C1-hidden-bowl-stack-ec/episode=1.mp4`
    `rollouts/libero_spatial/other/video.mp4`
    `rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk/../secret.mp4`
    `rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk/episode=1.mp4`
    """
    assert extract_registered_review_videos(text, "l1c1", "formal") == [
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb/episode=1.mp4",
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk/episode=1.mp4",
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-ec/episode=1.mp4",
    ]
    assert extract_registered_review_videos(text, "l1c1", "preview") == []
