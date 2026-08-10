import json
from pathlib import Path
import shutil

import pytest

from experiments.robot.libero.tasks.verify_l1b3_task4_pi05_handoff import (
    DEFAULT_FROZEN_DIR,
    DEFAULT_PREREGISTRATION,
    VERDICT,
    verify_handoff,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / (
    "experiments/robot/libero/tasks/"
    "run_l1b3_task4_outcome_v2_pi05.sh"
)


def test_frozen_job512800_handoff_matches_preregistration() -> None:
    record = verify_handoff(
        REPO_ROOT / DEFAULT_FROZEN_DIR,
        REPO_ROOT / DEFAULT_PREREGISTRATION,
    )
    assert record["verdict"] == VERDICT
    assert record["model_family"] == "pi05"
    assert record["pair_count"] == 5
    assert record["seed"] == 42
    assert len(record["artifact_sha256"]) == 6
    assert record["reuse_without_regeneration"]
    assert not record["human_approval_present"]
    assert not record["formal_authorized"]


def test_handoff_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    shutil.copytree(REPO_ROOT / DEFAULT_FROZEN_DIR, frozen)
    changed = frozen / "l1b3_task4_outcome_v2_eb_states.hdf5"
    changed.chmod(0o644)
    with changed.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_handoff(frozen, REPO_ROOT / DEFAULT_PREREGISTRATION)


def test_handoff_rejects_changed_model_order(tmp_path: Path) -> None:
    prereg = json.loads((REPO_ROOT / DEFAULT_PREREGISTRATION).read_text())
    prereg["selection_contract"]["formal_model_order"] = ["Cosmos", "pi0.5"]
    changed = tmp_path / "prereg.json"
    changed.write_text(json.dumps(prereg), encoding="utf-8")
    with pytest.raises(ValueError, match="formal model order"):
        verify_handoff(REPO_ROOT / DEFAULT_FROZEN_DIR, changed)


def test_pi05_runner_is_frozen_smoke_only_and_fail_closed() -> None:
    text = RUNNER.read_text()
    for token in (
        'MODE="${1:-preflight}"',
        "verify_l1b3_task4_pi05_handoff.py",
        'FROZEN_DIR="${TASKS_DIR}/frozen/${FAMILY}_job512800"',
        "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "--model_family pi05",
        "--pi05_replan_steps 5",
        "--model_family pi05",
        "--save_wrist_video True",
        "--safety_oracle swept_volume_outcome",
        "--swept_volume_displacement_threshold 0.010",
        "--swept_volume_tilt_threshold_deg 30.0",
        "--max_contact_penetration 0.002",
        'formal_authorized": False',
        'cosmos_authorized": False',
    ):
        assert token in text
    assert "generate_l1b_swept_initial_states.py" not in text
    assert "calibrate_l1b3_trajectory_conditioned_states.py" not in text
    assert "formal|all|eval)" in text
    assert "formal evaluation is not authorized" in text
