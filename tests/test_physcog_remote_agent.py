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


def test_robocasa_l2a1_registry_stops_before_policy_or_formal_evaluation():
    phases = {
        phase
        for scenario, phase in PHASES
        if scenario == "robocasa_l2a1"
    }
    assert phases == {
        "probe",
        "static_live",
        "geometry",
        "layout_scan",
        "initial_unreviewed",
        "initial_reviewed",
    }
    runner = "experiments/robot/robocasa/scripts/run_superpod.sh"
    for phase in phases:
        spec = PHASES[("robocasa_l2a1", phase)]
        assert spec.command == ("bash", runner, phase)
        assert spec.count_env is None
        assert spec.artifacts == (
            f"experiments/logs/robocasa_superpod/{phase}",
        )


def test_robocasa_l2a2_registry_stops_before_policy_or_formal_evaluation():
    phases = {
        phase
        for scenario, phase in PHASES
        if scenario == "robocasa_l2a2"
    }
    assert phases == {
        "static_live",
        "initial_unreviewed",
        "initial_reviewed",
    }
    runner = "experiments/robot/robocasa/scripts/run_superpod.sh"
    for phase in phases:
        spec = PHASES[("robocasa_l2a2", phase)]
        assert spec.command == (
            "env",
            "ROBOCASA_SCENE=L2-A2",
            "ROBOCASA_LOG_NAMESPACE=l2a2",
            "bash",
            runner,
            phase,
        )
        assert spec.count_env is None
        assert spec.artifacts == (
            f"experiments/logs/robocasa_superpod/l2a2/{phase}",
        )


def test_robocasa_l2a3_registry_stops_before_policy_or_formal_evaluation():
    phases = {
        phase
        for scenario, phase in PHASES
        if scenario == "robocasa_l2a3"
    }
    assert phases == {
        "static_live",
        "initial_unreviewed",
        "initial_reviewed",
    }
    runner = "experiments/robot/robocasa/scripts/run_superpod.sh"
    for phase in phases:
        spec = PHASES[("robocasa_l2a3", phase)]
        assert spec.command == (
            "env",
            "ROBOCASA_SCENE=L2-A3",
            "ROBOCASA_LOG_NAMESPACE=l2a3",
            "bash",
            runner,
            phase,
        )
        assert spec.count_env is None
        assert spec.artifacts == (
            f"experiments/logs/robocasa_superpod/l2a3/{phase}",
        )


def test_l1c4_registry_separates_native_gates_from_model_smoke():
    phases = {phase for scenario, phase in PHASES if scenario == "l1c4"}
    assert phases == {
        "check",
        "preview",
        "calibrate",
        "safe_reference",
        "smoke",
    }
    runner = "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh"
    for phase in phases:
        spec = PHASES[("l1c4", phase)]
        assert runner in spec.command
        assert "RENDER_GPU_DEVICE_ID=1" in spec.command
        assert not any("libero_90" in value for value in spec.command)
    assert PHASES[("l1c4", "check")].count_env == "NUM_TRIALS"
    assert PHASES[("l1c4", "calibrate")].count_env == "CALIBRATION_NUM_STATES"
    assert (
        PHASES[("l1c4", "safe_reference")].count_env
        == "CALIBRATION_NUM_STATES"
    )
    smoke = PHASES[("l1c4", "smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "SAVE_VIDEO_MODE=all" in smoke.command
    assert "MAX_VIDEOS_PER_OUTCOME=10" in smoke.command
    assert "review/L1-C4_task" in smoke.artifacts
    assert (
        "rollouts/libero_object/L1-C4-occupied-basket-risk"
        in smoke.artifacts
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


def test_l1a3_registry_exposes_native_gated_pipeline():
    assert set(phase for scenario, phase in PHASES if scenario == "l1a3") == {
        "attribution",
        "check",
        "formal",
        "preview",
        "smoke",
    }
    runner = "experiments/robot/libero/tasks/run_l1a3.sh"
    for phase in ("check", "preview", "smoke", "formal", "attribution"):
        assert runner in PHASES[("l1a3", phase)].command
    assert PHASES[("l1a3", "check")].count_env == "NUM_TRIALS"
    assert PHASES[("l1a3", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l1a3", "formal")].count_env == "NUM_TRIALS"
    formal = PHASES[("l1a3", "formal")]
    assert "RENDER_GPU_DEVICE_ID=1" in formal.command
    assert "SAVE_VIDEO_MODE=all" in formal.command
    assert "MAX_VIOLATION_VIDEOS=10" in formal.command
    assert "MAX_SUCCESS_VIDEOS=10" in formal.command
    assert "MAX_FAILURE_VIDEOS=10" in formal.command
    assert (
        "experiments/robot/libero/tasks/l1a3_native_preflight.json"
        in PHASES[("l1a3", "check")].artifacts
    )
    assert (
        "experiments/logs/l1a3_eb_to_er_replay.md"
        in PHASES[("l1a3", "formal")].artifacts
    )


def test_l1a4_spatial_replacement_phases_are_registered():
    assert not any(scenario == "l1a4" for scenario, _ in PHASES)
    check = PHASES[("l1a4s", "check")]
    assert check.count_env == "NUM_STATES"
    assert "experiments/robot/libero/tasks/run_l1a4_spatial.sh" in check.command
    assert (
        "experiments/robot/libero/tasks/l1a4_spatial_pairing.json"
        in check.artifacts
    )

    eb = PHASES[("l1a4s", "eb_capability_pi05")]
    assert eb.count_env == "EB_CAPABILITY_TRIALS"
    assert (
        "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh"
        in eb.command
    )
    assert (
        "rollouts/libero_spatial/L1-A4-between-eb-native-pi05"
        in eb.artifacts
    )


def test_l1b6_registry_exposes_calibration_and_gated_evaluation_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l1b6") == {
        "calibrate", "search", "path_calibrate", "prepare", "smoke", "pool_smoke",
        "formal", "ec_repair", "ec_video",
    }
    assert PHASES[("l1b6", "calibrate")].count_env == "CALIBRATION_TRIALS"
    assert any(
        value.endswith("calibrate_l1b6_wine_bottle.sh")
        for value in PHASES[("l1b6", "calibrate")].command
    )
    assert PHASES[("l1b6", "prepare")].count_env == "NUM_TRIALS"
    search = PHASES[("l1b6", "search")]
    assert "--laterals=0.055,0.060,0.065,0.070,0.075" in search.command
    assert "--max_episodes" in search.command
    assert "50" in search.command
    assert "--min_obstacle_tilt_change_deg" in search.command
    assert "--fail_on_invalid" in PHASES[("l1b6", "path_calibrate")].command
    assert "SAFE_REF_VIDEO_DIR=experiments/logs/l1b6_safe_reference_videos" in (
        PHASES[("l1b6", "smoke")].command
    )
    assert PHASES[("l1b6", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l1b6", "pool_smoke")].count_env == "NUM_TRIALS"
    assert "L1B6_CALIBRATION_POOL_SIZE=10" in PHASES[("l1b6", "pool_smoke")].command
    assert "L1B6_ER_PHYSICS_QUALIFICATION_SIZE=5" in PHASES[("l1b6", "pool_smoke")].command
    assert "REPLAY_MIN_EPISODES=2" in PHASES[("l1b6", "pool_smoke")].command
    assert "experiments/logs/l1b6_trajectory_conditioned_calibration.md" in (
        PHASES[("l1b6", "smoke")].artifacts
    )
    formal = PHASES[("l1b6", "formal")]
    assert formal.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in formal.command
    assert "SAVE_VIDEO_MODE=all" in formal.command
    assert "experiments/logs/l1b6_trajectory_conditioned_calibration.csv" in (
        formal.artifacts
    )
    assert "experiments/logs/l1b6_er_physics_qualification.md" in formal.artifacts
    assert "RENDER_GPU_DEVICE_ID=1" in PHASES[("l1b6", "ec_repair")].command


def test_l1a2r_registry_exposes_occluded_arm_pipeline_phases():
    assert set(phase for scenario, phase in PHASES if scenario == "l1a2r") == {
        "prepare", "smoke", "formal", "attribution",
    }
    runner = "experiments/robot/libero/tasks/run_l1b_swept.sh"
    for phase, mode in (
        ("prepare", "prepare"), ("smoke", "smoke"),
        ("formal", "all"), ("attribution", "attribution"),
    ):
        spec = PHASES[("l1a2r", phase)]
        assert spec.command[-3:] == (runner, "l1a2r_occluded_arm", mode)
    assert PHASES[("l1a2r", "prepare")].count_env == "NUM_TRIALS"
    assert PHASES[("l1a2r", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l1a2r", "formal")].count_env == "NUM_TRIALS"
    assert (
        "experiments/logs/l1a2r_occluded_arm_trajectory_conditioned_calibration.md"
        in PHASES[("l1a2r", "prepare")].artifacts
    )
    assert (
        "experiments/robot/libero/tasks/l1a2r_occluded_arm_pairing.json"
        in PHASES[("l1a2r", "prepare")].artifacts
    )
    assert (
        "experiments/logs/l1a2r_occluded_arm_native_replay.md"
        in PHASES[("l1a2r", "formal")].artifacts
    )
    assert "SAVE_VIDEO_MODE=all" in PHASES[("l1a2r", "formal")].command
    assert (
        "experiments/logs/l1a2r_visibility_attribution.md"
        in PHASES[("l1a2r", "attribution")].artifacts
    )


def test_l1a2rh_registry_carries_visibility_family_on_l1b6():
    assert set(phase for scenario, phase in PHASES if scenario == "l1a2rh") == {
        "prepare", "smoke", "formal", "attribution",
    }
    runner = "experiments/robot/libero/tasks/run_l1b_swept.sh"
    for phase, mode in (
        ("prepare", "prepare"), ("smoke", "smoke"),
        ("formal", "all"), ("attribution", "attribution"),
    ):
        spec = PHASES[("l1a2rh", phase)]
        assert spec.command[-3:] == (runner, "l1a2r_occluded_held", mode)
    assert PHASES[("l1a2rh", "prepare")].count_env == "NUM_TRIALS"
    assert PHASES[("l1a2rh", "smoke")].count_env == "SMOKE_TRIALS"
    assert PHASES[("l1a2rh", "formal")].count_env == "NUM_TRIALS"
    assert (
        "experiments/logs/l1a2r_occluded_held_trajectory_conditioned_calibration.csv"
        in PHASES[("l1a2rh", "formal")].artifacts
    )
    assert (
        "experiments/logs/l1a2r_held_visibility_attribution.md"
        in PHASES[("l1a2rh", "attribution")].artifacts
    )


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
    assert "SAVE_VIDEO_MODE=all" in formal.command
    assert "MAX_VIOLATION_VIDEOS=10" in formal.command
    assert "MAX_SUCCESS_VIDEOS=10" in formal.command
    assert "MAX_FAILURE_VIDEOS=10" in formal.command
    assert (
        "rollouts/libero_10/L3-A1-drawer-bottle-er-support-removal-seed42"
        in formal.artifacts
    )


def test_all_formal_phases_cap_or_disable_inline_video_retention():
    formal_specs = [
        spec for (_, phase), spec in PHASES.items() if phase == "formal"
    ]
    assert formal_specs
    for spec in formal_specs:
        if "SAVE_VIDEO_MODE=none" in spec.command:
            continue
        assert "SAVE_VIDEO_MODE=all" in spec.command
        assert "MAX_VIOLATION_VIDEOS=10" in spec.command
        assert "MAX_SUCCESS_VIDEOS=10" in spec.command
        assert "MAX_FAILURE_VIDEOS=10" in spec.command


def test_l3b1_policy_phases_pin_grpo_checkpoint_and_sampling_protocol():
    for phase in (
        "probe",
        "risk",
        "smoke",
        "formal",
        "native_cap_smoke",
        "native_cap_formal",
    ):
        command = PHASES[("l3b1", phase)].command
        assert "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90" in command
        assert "DO_SAMPLE=True" in command
        assert "TEMPERATURE=1.6" in command
        assert "TOP_P=1.0" in command
    assert PHASES[("l3b1", "native_cap_prepare")].count_env == "NUM_STATES"
    assert PHASES[("l3b1", "native_cap_smoke")].count_env == "SMOKE_TRIALS"
    native_formal = PHASES[("l3b1", "native_cap_formal")]
    assert native_formal.count_env == "NUM_TRIALS"
    assert "RENDER_GPU_DEVICE_ID=1" in PHASES[("l3b1", "native_cap_smoke")].command
    assert "RENDER_GPU_DEVICE_ID=1" in native_formal.command
    assert "SAVE_VIDEO_MODE=all" in native_formal.command
    assert "MAX_VIOLATION_VIDEOS=10" in native_formal.command
    assert "MAX_SUCCESS_VIDEOS=10" in native_formal.command
    assert "MAX_FAILURE_VIDEOS=10" in native_formal.command


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
    assert "export LIBERO_ROOT='/home/researcher/LIBERO src'" in script
    assert "export PYTHONPATH='/home/researcher/LIBERO src':${PYTHONPATH:-}" in script


def test_batch_script_can_exclude_unstable_render_nodes():
    cfg = RemoteConfig(**{**_config().__dict__, "exclude_nodes": "dgx-29"})
    script = build_batch_script(
        cfg, PhaseSpec(command=("true",)), count=1,
        scenario="l1b6", phase="ec_repair", remote_log="/tmp/job.out",
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


def test_l3b1_registry_exposes_capability_and_risk_arms():
    phases = {phase for scenario, phase in PHASES if scenario == "l3b1"}
    assert phases == {
        "bodies",
        "prepare",
        "check",
        "preview",
        "probe",
        "native_cap_prepare",
        "native_cap_smoke",
        "native_cap_formal",
        "risk_check",
        "risk_preview",
        "risk",
        "summarize",
        "reference",
        "smoke",
        "formal",
    }

    assert PHASES[("l3b1", "check")].count_env == "NUM_STATES"
    assert PHASES[("l3b1", "risk_check")].count_env == "NUM_STATES"
    assert PHASES[("l3b1", "probe")].count_env == "NUM_TRIALS"
    assert PHASES[("l3b1", "risk")].count_env == "NUM_TRIALS"
    assert PHASES[("l3b1", "prepare")].count_env == "NUM_STATES"
    assert PHASES[("l3b1", "formal")].count_env == "NUM_TRIALS"

    for phase in phases:
        command = PHASES[("l3b1", phase)].command
        assert command[0] in {"bash", "env"}
        assert "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh" in command

    # Both arms are video-gated evidence, so all rollouts must be kept.
    assert "SAVE_VIDEO_MODE=all" in PHASES[("l3b1", "probe")].command
    assert "SAVE_VIDEO_MODE=all" in PHASES[("l3b1", "risk")].command
    assert "MAX_VIDEOS_PER_OUTCOME=10" in PHASES[("l3b1", "formal")].command

    # Capability and risk arms must not share a rollout directory, otherwise the
    # summarize phase would mix prompts.
    cap_rollouts = "rollouts/libero_90/L3-B1-bottle-in-drawer-capability"
    risk_rollouts = "rollouts/libero_90/L3-B1-bottle-in-drawer-risk"
    assert cap_rollouts in PHASES[("l3b1", "probe")].artifacts
    assert risk_rollouts in PHASES[("l3b1", "risk")].artifacts
    assert cap_rollouts not in PHASES[("l3b1", "risk")].artifacts

    assert (
        "experiments/robot/libero/tasks/l3b1_capability_states.hdf5"
        in PHASES[("l3b1", "check")].artifacts
    )
    assert (
        "experiments/robot/libero/tasks/l3b1_risk_states.hdf5"
        in PHASES[("l3b1", "risk_check")].artifacts
    )
    # Safe SR is zeroed by construction in this scene, so the outcome breakdown
    # is the reportable artifact and must be registered.
    assert (
        "experiments/logs/l3b1_risk_outcomes.md"
        in PHASES[("l3b1", "summarize")].artifacts
    )
