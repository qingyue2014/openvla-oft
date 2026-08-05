import hashlib
import json

import pytest

from experiments.robot.libero.tasks.check_l1c1_formal_gate import (
    FIRST_POLICY_VERDICT,
    REPAIR_VERDICT,
    validate_formal_gate,
    validate_static_gate,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _passing_bundle(tmp_path):
    state = tmp_path / "eb.hdf5"
    state.write_bytes(b"repaired-eb")
    state_sha = hashlib.sha256(state.read_bytes()).hexdigest()
    repair = tmp_path / "repair.json"
    _write_json(
        repair,
        {
            "verdict": REPAIR_VERDICT,
            "episode_count": 50,
            "output_state_sha256": state_sha,
        },
    )
    first_policy = tmp_path / "first_policy.json"
    _write_json(
        first_policy,
        {
            "verdict": FIRST_POLICY_VERDICT,
            "episode_count": 50,
            "condition_counts": {"eb": 50, "er": 50, "ec": 50},
            "cross_condition_state_failures": [],
            "state_files": {"eb": {"sha256": state_sha}},
        },
    )
    first_policy_sha = hashlib.sha256(first_policy.read_bytes()).hexdigest()
    frame_review = tmp_path / "frame_review.json"
    _write_json(
        frame_review,
        {
            "scenario": "L1-C1",
            "scope": "all exact first-policy agent/wrist frames",
            "approved": True,
            "reviewer": "reviewer",
            "reviewed_at": "2026-08-04T12:00:00+08:00",
            "physical_gate_verdict": FIRST_POLICY_VERDICT,
            "physical_gate_manifest_sha256": first_policy_sha,
        },
    )
    smoke_review = tmp_path / "smoke_review.json"
    _write_json(
        smoke_review,
        {
            "scenario": "L1-C1",
            "scope": "all repaired-bundle smoke videos",
            "approved": True,
            "reviewer": "reviewer",
            "reviewed_at": "2026-08-04T13:00:00+08:00",
            "reviewed_videos": ["review/L1-C1_task/repaired_eb_smoke/eb_pass.mp4"],
        },
    )
    return state, repair, first_policy, frame_review, smoke_review, state_sha


def test_formal_gate_binds_repaired_state_and_both_human_reviews(tmp_path):
    state, repair, first_policy, frame_review, smoke_review, state_sha = (
        _passing_bundle(tmp_path)
    )
    assert validate_formal_gate(
        eb_state=state,
        repair_manifest_path=repair,
        first_policy_manifest_path=first_policy,
        first_policy_review_path=frame_review,
        smoke_review_path=smoke_review,
        expected_episodes=50,
    ) == state_sha


def test_static_gate_allows_smoke_without_pretending_human_review_is_complete(tmp_path):
    state, repair, first_policy, _, _, state_sha = _passing_bundle(tmp_path)
    assert validate_static_gate(
        eb_state=state,
        repair_manifest_path=repair,
        first_policy_manifest_path=first_policy,
        expected_episodes=50,
    ) == state_sha


@pytest.mark.parametrize("blocked_review", ["frame", "smoke"])
def test_formal_gate_fails_closed_without_explicit_human_approval(
    tmp_path, blocked_review
):
    state, repair, first_policy, frame_review, smoke_review, _ = _passing_bundle(
        tmp_path
    )
    path = frame_review if blocked_review == "frame" else smoke_review
    review = json.loads(path.read_text(encoding="utf-8"))
    review["approved"] = False
    _write_json(path, review)
    with pytest.raises(ValueError, match="approved must be true"):
        validate_formal_gate(
            eb_state=state,
            repair_manifest_path=repair,
            first_policy_manifest_path=first_policy,
            first_policy_review_path=frame_review,
            smoke_review_path=smoke_review,
            expected_episodes=50,
        )


def test_formal_gate_rejects_state_hash_drift(tmp_path):
    state, repair, first_policy, frame_review, smoke_review, _ = _passing_bundle(
        tmp_path
    )
    state.write_bytes(b"different-state")
    with pytest.raises(ValueError, match="construction manifest"):
        validate_formal_gate(
            eb_state=state,
            repair_manifest_path=repair,
            first_policy_manifest_path=first_policy,
            first_policy_review_path=frame_review,
            smoke_review_path=smoke_review,
            expected_episodes=50,
        )
