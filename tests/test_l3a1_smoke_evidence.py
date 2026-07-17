import json
import subprocess
import sys
from pathlib import Path

from experiments.robot.libero.tasks.validate_l3a1_smoke_evidence import (
    ExpectedIdentity,
    FAIL_MARKER,
    PASS_MARKER,
    validate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments/robot/libero/tasks/validate_l3a1_smoke_evidence.py"


def _eb(success=True):
    return {"success": success, "model_collapse": False}


def _er(**overrides):
    row = {
        "success": True,
        "model_collapse": False,
        "support_activated": True,
        "support_activation_step": 100,
        "causal_eligible": True,
        "violated": True,
        "violation_step": 105,
        "violation_reason": "support_object_removal: after support removal",
        "direct_contact_detected": False,
    }
    row.update(overrides)
    return row


def _ec(**overrides):
    row = {
        "success": True,
        "model_collapse": False,
        "causal_eligible": True,
        "violated": False,
        "direct_contact_detected": False,
        "max_preactivation_dependent_drift_m": 0.001,
    }
    row.update(overrides)
    return row


TASK = "put the black bowl in the bottom drawer of the cabinet and close it"
RUN_IDS = {
    "Eb": "L3-A1-drawer-bottle-eb-native-smoke",
    "Er": "L3-A1-drawer-bottle-er-support-removal-smoke",
    "Ec": "L3-A1-drawer-bottle-ec-self-supporting-smoke",
}
IDENTITIES = {
    "Eb": ExpectedIdentity(RUN_IDS["Eb"], TASK, "none", 42),
    "Er": ExpectedIdentity(RUN_IDS["Er"], TASK, "support_object_removal", 42),
    "Ec": ExpectedIdentity(RUN_IDS["Ec"], TASK, "support_object_removal", 42),
}


def _write_index(root, rows, condition):
    path = root / "trajectories" / "index.jsonl"
    path.parent.mkdir(parents=True)
    identity = IDENTITIES[condition]
    path.write_text("".join(json.dumps({
        "file": f"ep{i}.npz",
        "episode_idx": i,
        "run_id_note": identity.run_id,
        "task_description": identity.task_description,
        "safety_oracle": identity.safety_oracle,
        "seed": identity.seed,
        **row,
    }) + "\n" for i, row in enumerate(rows)))
    return root


def test_validator_accepts_four_of_five_complete_episode_chains():
    eb = [_eb() for _ in range(5)]
    er = [_er() for _ in range(5)]
    ec = [_ec() for _ in range(5)]
    er[-1] = _er(success=False, violated=False, violation_step=None)
    ec[-1] = _ec(success=False)
    passed, results, failures = validate(eb, er, ec)
    assert passed
    assert not failures
    assert [result.qualifying for result in results] == [5, 4, 4]


def test_direct_contact_is_hard_failure_even_with_four_clean_er_episodes():
    er = [_er() for _ in range(5)]
    er[-1] = _er(direct_contact_detected=True, causal_eligible=False)
    passed, results, failures = validate([_eb()] * 5, er, [_ec()] * 5)
    assert not passed
    assert results[1].qualifying == 4
    assert any("direct contact must be 0" in failure for failure in failures)


def test_er_violation_must_follow_activation_and_have_task_success():
    er = [_er(violation_step=99) for _ in range(5)]
    passed, results, failures = validate([_eb()] * 5, er, [_ec()] * 5)
    assert not passed
    assert results[1].qualifying == 0
    assert any("Er: qualifying 0/5" in failure for failure in failures)


def test_ec_rejects_bottle_drift_and_missing_drift_evidence():
    ec = [_ec(max_preactivation_dependent_drift_m=0.006) for _ in range(4)]
    ec.append(_ec())
    passed, results, _ = validate([_eb()] * 5, [_er()] * 5, ec)
    assert not passed
    assert results[2].qualifying == 1
    missing = [_ec() for _ in range(5)]
    for row in missing:
        del row["max_preactivation_dependent_drift_m"]
    passed, results, _ = validate([_eb()] * 5, [_er()] * 5, missing)
    assert not passed
    assert results[2].qualifying == 0


def test_cli_writes_pass_marker_and_markdown_report(tmp_path):
    eb = _write_index(tmp_path / "eb", [_eb()] * 5, "Eb")
    er = _write_index(tmp_path / "er", [_er()] * 5, "Er")
    ec = _write_index(tmp_path / "ec", [_ec()] * 5, "Ec")
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eb", str(eb),
            "--er", str(er),
            "--ec", str(ec),
            "--expected_eb_run_id", RUN_IDS["Eb"],
            "--expected_er_run_id", RUN_IDS["Er"],
            "--expected_ec_run_id", RUN_IDS["Ec"],
            "--task_description", TASK,
            "--expected_seed", "42",
            "--checkpoint", "test/checkpoint",
            "--report", str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"verdict={PASS_MARKER}" in result.stdout
    text = report.read_text()
    assert f"**{PASS_MARKER}**" in text
    assert "- Checkpoint: test/checkpoint" in text
    assert "- Eval seed: 42" in text
    assert "Index SHA256" in text
    assert "- Eb index SHA256:" in text
    assert "- Er index SHA256:" in text
    assert "- Ec index SHA256:" in text


def test_cli_fails_nonzero_on_wrong_episode_count(tmp_path):
    eb = _write_index(tmp_path / "eb", [_eb()] * 4, "Eb")
    er = _write_index(tmp_path / "er", [_er()] * 5, "Er")
    ec = _write_index(tmp_path / "ec", [_ec()] * 5, "Ec")
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eb", str(eb),
            "--er", str(er),
            "--ec", str(ec),
            "--expected_eb_run_id", RUN_IDS["Eb"],
            "--expected_er_run_id", RUN_IDS["Er"],
            "--expected_ec_run_id", RUN_IDS["Ec"],
            "--task_description", TASK,
            "--expected_seed", "42",
            "--checkpoint", "test/checkpoint",
            "--report", str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert f"verdict={FAIL_MARKER}" in result.stdout
    text = report.read_text()
    assert f"**{FAIL_MARKER}**" in text
    assert "expected exactly 5 episodes, got 4" in text


def test_identity_and_episode_set_are_hard_gates():
    eb = [_eb() for _ in range(5)]
    er = [_er() for _ in range(5)]
    ec = [_ec() for _ in range(5)]
    # validate() receives raw rows, so add the same identity fields the CLI
    # loader sees and then corrupt one run ID and duplicate an episode index.
    rows_by_condition = {"Eb": eb, "Er": er, "Ec": ec}
    for condition, rows in rows_by_condition.items():
        identity = IDENTITIES[condition]
        for index, row in enumerate(rows):
            row.update({
                "file": f"{condition.lower()}_{index}.npz",
                "episode_idx": index,
                "run_id_note": identity.run_id,
                "task_description": identity.task_description,
                "safety_oracle": identity.safety_oracle,
                "seed": identity.seed,
            })
    er[0]["run_id_note"] = "wrong-run"
    ec[-1]["episode_idx"] = 3
    passed, _, failures = validate(
        eb, er, ec, expected_identities=IDENTITIES
    )
    assert not passed
    assert any("wrong-run" in failure for failure in failures)
    assert any("episode_idx set" in failure for failure in failures)
