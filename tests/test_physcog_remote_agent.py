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


def test_l1a34_registry_exposes_boundary_pipeline_phases():
    for scenario in ("l1a3", "l1a4"):
        assert set(phase for s, phase in PHASES if s == scenario) == {
            "check", "preview", "calibrate", "safe_reference", "smoke",
        }
        assert PHASES[(scenario, "check")].count_env == "NUM_TRIALS"
        assert PHASES[(scenario, "calibrate")].count_env == "CALIBRATION_NUM_STATES"
        assert PHASES[(scenario, "safe_reference")].count_env == "SAFE_REF_STATES"
        assert PHASES[(scenario, "smoke")].count_env == "SMOKE_TRIALS"
        assert PHASES[(scenario, "check")].command[-1] == f"{scenario}_check"
        assert f"experiments/logs/{scenario}_calibration.md" in PHASES[
            (scenario, "calibrate")
        ].artifacts
    assert (
        "experiments/robot/libero/tasks/l1a3_task1_ramekin_rim_gap_pairing.json"
        in PHASES[("l1a3", "check")].artifacts
    )
    assert (
        "experiments/robot/libero/tasks/l1a4_task1_plate_crowding_pairing.json"
        in PHASES[("l1a4", "check")].artifacts
    )


def test_l3c2_registry_exposes_shared_space_pipeline_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l3c2") == {
        "check", "calibrate", "safe_reference", "smoke", "probe", "formal",
    }
    calibration_dir = "experiments/robot/libero/tasks/l3c2_calibration_v2"
    output_root = "rollouts/l3c2_frypan_shared_space_v2"
    safe_reference_dir = "experiments/logs/l3c2_safe_reference_v2"

    check = PHASES[("l3c2", "check")]
    assert check.command == ("python", "-m", "pytest", "-q", "tests/test_l3c_libero_integration.py")

    calibrate = PHASES[("l3c2", "calibrate")]
    assert calibrate.count_env == "NUM_STATES"
    assert f"CALIBRATION_DIR={calibration_dir}" in calibrate.command
    assert f"{calibration_dir}/calibration_report.json" in calibrate.artifacts
    assert f"{calibration_dir}/paired_initial_states.npz" in calibrate.artifacts

    safe_reference = PHASES[("l3c2", "safe_reference")]
    assert safe_reference.count_env is None
    assert f"{safe_reference_dir}/result.json" in safe_reference.artifacts

    for phase in ("smoke", "probe", "formal"):
        spec = PHASES[("l3c2", phase)]
        assert "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora" in spec.command
        assert f"CALIBRATION_DIR={calibration_dir}" in spec.command
        assert f"OUTPUT_ROOT={output_root}" in spec.command
        assert f"SAFE_REFERENCE_DIR={safe_reference_dir}" in spec.command
    assert PHASES[("l3c2", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l3c2", "probe")].count_env == "PROBE_TRIALS"
    assert PHASES[("l3c2", "formal")].count_env == "TRIALS"
    assert f"{output_root}/action_separation_report.json" in PHASES[("l3c2", "probe")].artifacts
    assert "experiments/logs/l3c2_frypan_shared_space_v2.json" in PHASES[("l3c2", "formal")].artifacts


def test_l1a2r_registry_exposes_gated_four_way_pipeline_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l1a2r") == {
        "check",
        "preview",
        "calibrate",
        "safe_reference",
        "smoke",
        "eb",
        "replay_gate",
        "eval",
        "attribution",
    }
    runner = "experiments/robot/libero/tasks/run_l1a2r_corridor_hazard.sh"
    for phase in ("check", "preview", "calibrate", "safe_reference", "eb",
                  "replay_gate", "attribution"):
        assert PHASES[("l1a2r", phase)].command[-2:] == (runner, phase)
    assert PHASES[("l1a2r", "check")].count_env == "NUM_TRIALS"
    assert PHASES[("l1a2r", "calibrate")].count_env == "CALIBRATION_NUM_STATES"
    assert PHASES[("l1a2r", "safe_reference")].count_env == "SAFE_REF_STATES"
    assert PHASES[("l1a2r", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l1a2r", "eb")].count_env == "NUM_TRIALS"
    assert PHASES[("l1a2r", "eval")].count_env == "NUM_TRIALS"
    assert (
        "experiments/robot/libero/tasks/l1a2r_task1_corridor_hazard_pairing.json"
        in PHASES[("l1a2r", "check")].artifacts
    )
    assert (
        "experiments/logs/l1a2r_eb_replay_er_occ.md"
        in PHASES[("l1a2r", "replay_gate")].artifacts
    )
    assert (
        "experiments/logs/l1a2r_eb_replay_er_vis.md"
        in PHASES[("l1a2r", "replay_gate")].artifacts
    )
    assert "SAVE_VIDEO_MODE=all" in PHASES[("l1a2r", "smoke")].command
    assert "SAVE_VIDEO_MODE=violation" in PHASES[("l1a2r", "eval")].command


def test_verdict_extraction_understands_boundary_gate_manifest():
    assert extract_verdicts('{"boundary_gate": "PASS"}') == ["PASS"]


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
