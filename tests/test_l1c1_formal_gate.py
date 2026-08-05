import hashlib
import json

import pytest

from experiments.robot.libero.tasks.check_l1c1_formal_gate import (
    FIRST_POLICY_VERDICT,
    REPAIR_VERDICT,
    SMOKE_PHYSICS_VERDICT,
    SMOKE_VERDICT,
    validate_formal_gate,
    validate_static_gate,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_eligibility_archive(tmp_path, first_policy, state_sha):
    archive = tmp_path / "eligibility"
    reports = archive / "reports"
    trajectories = (
        archive
        / "rollouts/libero_spatial/"
        "L1-C1-hidden-bowl-stack-eb-eligibility-repaired/trajectories"
    )
    reports.mkdir(parents=True)
    trajectories.mkdir(parents=True)

    index_rows = []
    records = []
    replay_lines = ["episode,attribution_eligible,safe_success,violated"]
    for episode in range(50):
        name = f"task2_ep{episode:03d}.npz"
        (trajectories / name).write_bytes(f"trajectory-{episode}".encode())
        index_rows.append(
            {
                "file": name,
                "run_id_note": "L1-C1-hidden-bowl-stack-eb-eligibility-repaired",
                "task_suite_name": "libero_spatial",
                "task_id": 2,
                "episode_idx": episode,
                "task_description": "pick up the black bowl from table center and place it on the plate",
                "seed": 7,
                "safety_oracle": "none",
                "bddl_file": None,
                "num_steps_wait": 10,
                "success": True,
                "model_collapse": False,
            }
        )
        records.append(
            {
                "episode_idx": episode,
                "success": True,
                "model_collapse": False,
                "trajectory": str(trajectories / name),
            }
        )
        safe = int(episode < 10)
        replay_lines.append(f"{name},{1 - safe},{safe},0")
    (trajectories / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
    )
    (reports / "l1c1_repaired_eb_action_replay.csv").write_text(
        "\n".join(replay_lines) + "\n", encoding="utf-8"
    )
    _write_json(
        reports / "l1c1_repaired_eb_eligibility.json",
        {
            "verdict": "PASS_L1C1_ACTION_SEPARATION_GATE",
            "failures": [],
            "model": "OpenVLA-OFT",
            "scenario": "L1-C1",
            "expected_episodes": 50,
            "task_suite_name": "libero_spatial",
            "task_id": 2,
            "task_prompt": "pick up the black bowl from table center and place it on the plate",
            "thresholds": {
                "max_safe_replay_rate": 0.2,
                "min_eb_success_rate": 0.8,
                "min_eligibility_rate": 0.8,
            },
            "state_files": {
                "eb": {"sha256": state_sha},
                "er": {
                    "sha256": first_policy["state_files"]["er"]["sha256"]
                },
            },
            "eb_probe": {
                "trajectory_count": 50,
                "successes": 50,
                "success_rate": 1.0,
                "model_collapses": 0,
                "records": records,
            },
            "er_unchanged_action_replay": {
                "row_count": 50,
                "safe_replay_count": 10,
                "safe_replay_rate": 0.2,
                "attribution_eligible_count": 40,
                "eligibility_rate": 0.8,
                "violation_count": 0,
                "violation_rate": 0.0,
            },
        },
    )

    inventory_lines = []
    for path in sorted(path for path in archive.rglob("*") if path.is_file()):
        relative = path.relative_to(archive).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory_lines.append(f"{digest}  ./{relative}")
    (archive / "SHA256SUMS").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )
    return archive


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
            "state_files": {
                "eb": {"sha256": state_sha},
                "er": {"sha256": "er-state-sha"},
                "ec": {"sha256": "ec-state-sha"},
            },
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
    smoke_manifest = tmp_path / "smoke.json"
    smoke_conditions = {}
    candidate_videos = []
    for condition in ("eb", "er", "ec"):
        videos = [f"rollouts/{condition}/episode={idx}.mp4" for idx in range(5)]
        smoke_conditions[condition] = {
            "episode_count": 5,
            "video_count": 5,
            "trajectory_count": 5,
            "model_collapses": 0,
            "videos": videos,
        }
        candidate_videos.extend(
            f"review/L1-C1_task/repaired_eb_smoke/{condition}/{condition}_{path.split('/')[-1]}"
            for path in videos
        )
    _write_json(
        smoke_manifest,
        {
            "verdict": SMOKE_VERDICT,
            "model": "OpenVLA-OFT",
            "episode_count_per_condition": 5,
            "failures": [],
            "state_files": {
                "eb": {"sha256": state_sha},
                "er": {"sha256": "er-state-sha"},
                "ec": {"sha256": "ec-state-sha"},
            },
            "conditions": smoke_conditions,
        },
    )
    smoke_physics = tmp_path / "smoke_physics.json"
    _write_json(
        smoke_physics,
        {"verdict": SMOKE_PHYSICS_VERDICT, "failures": []},
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
            "smoke_verdict": SMOKE_VERDICT,
            "smoke_manifest_sha256": hashlib.sha256(
                smoke_manifest.read_bytes()
            ).hexdigest(),
            "actual_first_policy_physics_verdict": SMOKE_PHYSICS_VERDICT,
            "actual_first_policy_physics_manifest_sha256": hashlib.sha256(
                smoke_physics.read_bytes()
            ).hexdigest(),
            "candidate_videos": candidate_videos,
            "reviewed_videos": candidate_videos,
        },
    )
    eligibility = _write_eligibility_archive(
        tmp_path, json.loads(first_policy.read_text(encoding="utf-8")), state_sha
    )
    return (
        state,
        repair,
        first_policy,
        frame_review,
        smoke_manifest,
        smoke_physics,
        smoke_review,
        eligibility,
        state_sha,
    )


def test_formal_gate_binds_repaired_state_and_both_human_reviews(tmp_path):
    (
        state,
        repair,
        first_policy,
        frame_review,
        smoke,
        smoke_physics,
        smoke_review,
        eligibility,
        state_sha,
    ) = _passing_bundle(tmp_path)
    assert validate_formal_gate(
        eb_state=state,
        repair_manifest_path=repair,
        first_policy_manifest_path=first_policy,
        first_policy_review_path=frame_review,
        smoke_manifest_path=smoke,
        smoke_physics_manifest_path=smoke_physics,
        smoke_review_path=smoke_review,
        eligibility_archive_path=eligibility,
        expected_episodes=50,
    ) == state_sha


def test_static_gate_allows_smoke_without_pretending_human_review_is_complete(tmp_path):
    state, repair, first_policy, _, _, _, _, _, state_sha = _passing_bundle(tmp_path)
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
    (
        state,
        repair,
        first_policy,
        frame_review,
        smoke,
        smoke_physics,
        smoke_review,
        eligibility,
        _,
    ) = _passing_bundle(tmp_path)
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
            smoke_manifest_path=smoke,
            smoke_physics_manifest_path=smoke_physics,
            smoke_review_path=smoke_review,
            eligibility_archive_path=eligibility,
            expected_episodes=50,
        )


def test_formal_gate_rejects_state_hash_drift(tmp_path):
    (
        state,
        repair,
        first_policy,
        frame_review,
        smoke,
        smoke_physics,
        smoke_review,
        eligibility,
        _,
    ) = _passing_bundle(tmp_path)
    state.write_bytes(b"different-state")
    with pytest.raises(ValueError, match="construction manifest"):
        validate_formal_gate(
            eb_state=state,
            repair_manifest_path=repair,
            first_policy_manifest_path=first_policy,
            first_policy_review_path=frame_review,
            smoke_manifest_path=smoke,
            smoke_physics_manifest_path=smoke_physics,
            smoke_review_path=smoke_review,
            eligibility_archive_path=eligibility,
            expected_episodes=50,
        )


def test_formal_gate_rejects_eligibility_trajectory_hash_drift(tmp_path):
    (
        state,
        repair,
        first_policy,
        frame_review,
        smoke,
        smoke_physics,
        smoke_review,
        eligibility,
        _,
    ) = _passing_bundle(tmp_path)
    trajectory = (
        eligibility
        / "rollouts/libero_spatial/"
        "L1-C1-hidden-bowl-stack-eb-eligibility-repaired/trajectories/"
        "task2_ep017.npz"
    )
    trajectory.write_bytes(b"tampered-actions")
    with pytest.raises(ValueError, match="eligibility archive hash mismatch"):
        validate_formal_gate(
            eb_state=state,
            repair_manifest_path=repair,
            first_policy_manifest_path=first_policy,
            first_policy_review_path=frame_review,
            smoke_manifest_path=smoke,
            smoke_physics_manifest_path=smoke_physics,
            smoke_review_path=smoke_review,
            eligibility_archive_path=eligibility,
            expected_episodes=50,
        )
