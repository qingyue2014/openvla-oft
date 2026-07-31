from experiments.robot.libero.tasks.physcog_remote_agent import (
    PHASES,
    RemoteConfig,
    build_batch_script,
)
from experiments.robot.libero.tasks.record_experiment_results import (
    ATTRIBUTION_FILE_METADATA,
    RUN_METADATA,
)


def test_replacement_l3a_run_ids_are_registered_without_legacy_a2():
    expected = {
        "L3-A2-eb-milk-butter": ("L3", "L3-A2"),
        "L3-A2-er-milk-butter": ("L3", "L3-A2"),
        "L3-A2-ec-milk-butter": ("L3", "L3-A2"),
        "L3-A3-plate-bottle-eb": ("L3", "L3-A3"),
        "L3-A3-plate-bottle-er": ("L3", "L3-A3"),
        "L3-A3-plate-bottle-ec": ("L3", "L3-A3"),
        "L3-A4-microwave-mug-eb": ("L3", "L3-A4"),
        "L3-A4-microwave-mug-er": ("L3", "L3-A4"),
        "L3-A4-microwave-mug-ec": ("L3", "L3-A4"),
    }
    for run_id, identity in expected.items():
        assert RUN_METADATA[run_id][:2] == identity
    assert "L3-A2-bowl-drawer" not in RUN_METADATA
    assert ATTRIBUTION_FILE_METADATA["l3a2_attribution"] == ("L3", "L3-A2")
    assert ATTRIBUTION_FILE_METADATA["l3a3_attribution"] == ("L3", "L3-A3")
    assert ATTRIBUTION_FILE_METADATA["l3a4_attribution"] == ("L3", "L3-A4")


def test_remote_l3a_phases_use_native_suite_runners_and_review_storage():
    for scenario in ("l3a2", "l3a3", "l3a4"):
        for phase in ("check", "smoke", "formal"):
            spec = PHASES[(scenario, phase)]
            command = " ".join(spec.command)
            assert "libero_90" not in command
            assert f"run_{scenario}_" in command
            assert any(
                artifact.startswith(
                    f"review/L3-A{int(scenario[-1])}_task"
                )
                for artifact in spec.artifacts
            )


def test_l3a3_remote_phases_bind_controller_reference_and_formal_tables():
    check = PHASES[("l3a3", "check")]
    assert "prepare" in check.command
    assert "review/L3-A3_task/L3-A3_controller_safe_reference.npz" in check.artifacts
    assert "review/L3-A3_task/L3-A3_controller_safe_reference.mp4" in check.artifacts
    assert "review/L3-A3_task/L3-A3_safe_reference.json" in check.artifacts

    formal = PHASES[("l3a3", "formal")]
    assert "experiments/logs/l3a3_results.csv" in formal.artifacts
    assert "experiments/logs/l3a3_results.md" in formal.artifacts
    assert "experiments/logs/l3a3_result_tables.md" in formal.artifacts


def test_staged_l3a_remote_phases_preserve_prerequisite_review_evidence():
    for scenario, review_root in (
        ("l3a2", "review/L3-A2_task"),
        ("l3a3", "review/L3-A3_task"),
        ("l3a4", "review/L3-A4_task"),
    ):
        smoke = PHASES[(scenario, "smoke")]
        formal = PHASES[(scenario, "formal")]
        assert smoke.cleanup_artifacts is not None
        assert formal.cleanup_artifacts is not None
        assert review_root not in smoke.cleanup_artifacts
        assert review_root not in formal.cleanup_artifacts


def test_remote_batch_redirects_runtime_caches_off_full_home_volume():
    script = build_batch_script(
        RemoteConfig(
            host="superpod.example",
            user="tester",
            control_socket="/tmp/test.sock",
            remote_repo="/project/team/l3a",
            remote_python_bin="/opt/conda/bin",
            branch="test",
            account="team",
            partition="normal",
            nodes=1,
            gpus=1,
            time_limit="00:30:00",
        ),
        PHASES[("l3a3", "check")],
        1,
        "l3a3",
        "check",
        "/project/team/l3a/check.out",
    )
    assert "NUMBA_CACHE_DIR=/project/team/l3a/.physcog-agent/cache/numba" in script
    assert "XDG_CACHE_HOME=/project/team/l3a/.physcog-agent/cache/xdg" in script
    assert "MPLCONFIGDIR=/project/team/l3a/.physcog-agent/cache/matplotlib" in script
