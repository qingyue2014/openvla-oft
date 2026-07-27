import json
import subprocess
import sys
from pathlib import Path

from experiments.robot.libero.tasks.validate_l3a1_smoke_evidence import (
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
        "direct_contact_before_causal_violation_detected": False,
        "post_violation_direct_contact_detected": False,
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


def _write_index(root, rows):
    path = root / "trajectories" / "index.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps({"file": f"ep{i}.npz", **row}) + "\n" for i, row in enumerate(rows)))
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
    er[-1] = _er(
        direct_contact_detected=True,
        direct_contact_before_causal_violation_detected=True,
        causal_eligible=False,
    )
    passed, results, failures = validate([_eb()] * 5, er, [_ec()] * 5)
    assert not passed
    assert results[1].qualifying == 4
    assert any(
        "disqualifying direct contact must be 0" in failure
        for failure in failures
    )


def test_er_contact_after_established_violation_is_reported_but_not_disqualified():
    er = [_er() for _ in range(5)]
    er[0] = _er(
        direct_contact_detected=True,
        post_violation_direct_contact_detected=True,
    )
    passed, results, failures = validate([_eb()] * 5, er, [_ec()] * 5)
    assert passed
    assert not failures
    assert results[1].qualifying == 5
    assert results[1].direct_contacts == 0
    assert results[1].downstream_contacts == 1


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
    eb = _write_index(tmp_path / "eb", [_eb()] * 5)
    er = _write_index(tmp_path / "er", [_er()] * 5)
    ec = _write_index(tmp_path / "ec", [_ec()] * 5)
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eb", str(eb),
            "--er", str(er),
            "--ec", str(ec),
            "--report", str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"verdict={PASS_MARKER}" in result.stdout
    assert f"**{PASS_MARKER}**" in report.read_text()


def test_cli_fails_nonzero_on_wrong_episode_count(tmp_path):
    eb = _write_index(tmp_path / "eb", [_eb()] * 4)
    er = _write_index(tmp_path / "er", [_er()] * 5)
    ec = _write_index(tmp_path / "ec", [_ec()] * 5)
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eb", str(eb),
            "--er", str(er),
            "--ec", str(ec),
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
