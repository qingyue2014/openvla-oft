from experiments.robot.libero.tasks.physcog_remote_agent import PHASES
from experiments.robot.libero.tasks.record_experiment_results import (
    ATTRIBUTION_FILE_METADATA,
    RUN_METADATA,
)


def test_replacement_l3a_run_ids_are_registered_without_legacy_a2():
    expected = {
        "L3-A2-eb-milk-butter": ("L3", "L3-A2"),
        "L3-A2-er-milk-butter": ("L3", "L3-A2"),
        "L3-A2-ec-milk-butter": ("L3", "L3-A2"),
        "L3-A4-microwave-mug-eb": ("L3", "L3-A4"),
        "L3-A4-microwave-mug-er": ("L3", "L3-A4"),
        "L3-A4-microwave-mug-ec": ("L3", "L3-A4"),
    }
    for run_id, identity in expected.items():
        assert RUN_METADATA[run_id][:2] == identity
    assert "L3-A2-bowl-drawer" not in RUN_METADATA
    assert ATTRIBUTION_FILE_METADATA["l3a2_attribution"] == ("L3", "L3-A2")
    assert ATTRIBUTION_FILE_METADATA["l3a4_attribution"] == ("L3", "L3-A4")


def test_remote_l3a_phases_use_native_suite_runners_and_review_storage():
    for scenario in ("l3a2", "l3a4"):
        for phase in ("check", "smoke", "formal"):
            spec = PHASES[(scenario, phase)]
            command = " ".join(spec.command)
            assert "libero_90" not in command
            assert f"run_{scenario}_" in command
            assert any(
                artifact.startswith(
                    f"review/L3-A{2 if scenario == 'l3a2' else 4}_task"
                )
                for artifact in spec.artifacts
            )
