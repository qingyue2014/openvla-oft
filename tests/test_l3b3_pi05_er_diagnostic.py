import json

from experiments.robot.libero.tasks.summarize_l3b3_pi05_er_checkpoint_diagnostic import (
    RUN_NOTE,
    summarize,
)


def _write_row(tmp_path, *, success, settle):
    trajectory = tmp_path / "trajectories"
    trajectory.mkdir()
    row = {
        "model_family": "pi05",
        "run_id_note": RUN_NOTE,
        "task_suite_name": "libero_10",
        "task_id": 9,
        "success": success,
        "runtime_initial_gate": {
            "scenario": "L3-B3-MICROWAVE-PRECONDITION",
            "condition": "closed_microwave",
            "physical_gate_pass": True,
        },
        "l3b3_microwave_sequence": {
            "scenario": "L3-B3-MICROWAVE-PRECONDITION",
            "condition": "closed_microwave",
            "door_opened": True,
            "insertion_after_open": success,
            "reclose_after_insertion": success,
            "failure_stage": (
                "full_open_insert_reclose" if success else "opened_without_insertion"
            ),
            "post_success_settle": settle,
        },
    }
    (trajectory / "index.jsonl").write_text(json.dumps(row) + "\n")
    return trajectory


def test_pi05_er_diagnostic_is_inconclusive_without_placement(tmp_path):
    trajectory = _write_row(
        tmp_path,
        success=False,
        settle={"sample_count": 0, "strict_target_stability_pass": False},
    )
    result = summarize(trajectory, expected_count=1)
    assert result["checkpoint_cause_test"] == (
        "INCONCLUSIVE_PI05_DID_NOT_COMPLETE_PLACEMENT"
    )
    assert result["formal_authorized"] is False


def test_pi05_er_diagnostic_reports_persistent_instability(tmp_path):
    trajectory = _write_row(
        tmp_path,
        success=True,
        settle={"sample_count": 100, "strict_target_stability_pass": False},
    )
    result = summarize(trajectory, expected_count=1)
    assert result["checkpoint_cause_test"] == (
        "MICROBOUNCE_OR_INSTABILITY_PERSISTS_UNDER_PI05"
    )
    assert result["do_not_pool_as_formal_model_evidence"] is True
