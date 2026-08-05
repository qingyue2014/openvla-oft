import csv
import hashlib
import json
from collections import Counter

import pytest

from experiments.robot.libero.tasks.check_l1c1_cascade_gate import (
    CONDITIONS,
    EXPECTED_REVIEW_SCOPE,
    REQUIRED_REPORT_VERDICTS,
    TASK_PROMPT,
    validate_cascade_gate,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_inventory(root, inventory_path):
    lines = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path == inventory_path:
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{_sha(path)}  ./{relative}")
    inventory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _passing_bundle(tmp_path):
    archive = tmp_path / "formal-openvla"
    archive.mkdir()
    commit = "a" * 40

    state_dir = archive / "initial_layouts"
    state_dir.mkdir()
    state_names = {
        "eb": "l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
        "er": "l1c1_task2_bowl_stack_candidate_states.hdf5",
        "ec": "l1c1_task2_bowl_stack_ec_states.hdf5",
    }
    state_hashes = {}
    destinations = {}
    for condition, name in state_names.items():
        path = state_dir / name
        path.write_bytes(f"frozen-{condition}".encode())
        state_hashes[condition] = _sha(path)
        destinations[condition] = f"experiments/robot/libero/tasks/{name}"

    _write_json(
        archive / "run.json",
        {
            "scenario": "l1c1",
            "phase": "formal",
            "classification": "pass",
            "returncode": 0,
            "count": 50,
            "count_env": "NUM_TRIALS",
            "local_commit": commit,
            "job_id": "12345",
            "remote_markers": {
                "commit": commit,
                "exit_code": "0",
                "compute_node": "dgx-test",
            },
            "missing_artifacts": [],
            "missing_review_videos": [],
            "uploaded_inputs": [
                {
                    "destination": destinations[condition],
                    "sha256": state_hashes[condition],
                }
                for condition in ("eb", "er", "ec")
            ],
        },
    )

    outcome_counts = {
        "eb": Counter(episodes=50, successes=48, violations=0, safe_successes=48),
        "er": Counter(episodes=50, successes=2, violations=24, safe_successes=1),
        "ec": Counter(episodes=50, successes=49, violations=0, safe_successes=49),
    }
    for condition, (run_note, oracle) in CONDITIONS.items():
        trajectory_dir = archive / "rollouts/libero_spatial" / run_note / "trajectories"
        trajectory_dir.mkdir(parents=True)
        rows = []
        for episode in range(50):
            name = f"task2_ep{episode:03d}.npz"
            (trajectory_dir / name).write_bytes(f"{condition}-{episode}".encode())
            success_limit = outcome_counts[condition]["successes"]
            success = episode < success_limit
            violated = condition == "er" and 1 <= episode < 25
            rows.append(
                {
                    "file": name,
                    "run_id_note": run_note,
                    "task_suite_name": "libero_spatial",
                    "task_id": 2,
                    "episode_idx": episode,
                    "task_description": TASK_PROMPT,
                    "seed": 7,
                    "safety_oracle": oracle,
                    "bddl_file": None,
                    "num_steps_wait": 10,
                    "success": success,
                    "violated": violated,
                    "model_collapse": False,
                }
            )
        (trajectory_dir / "index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        calculated = Counter(
            episodes=len(rows),
            successes=sum(row["success"] for row in rows),
            violations=sum(row["violated"] for row in rows),
            safe_successes=sum(row["success"] and not row["violated"] for row in rows),
        )
        outcome_counts[condition] = calculated

    reports = archive / "reports"
    reports.mkdir()
    for filename, verdict in REQUIRED_REPORT_VERDICTS.items():
        (reports / filename).write_text(f"Verdict: {verdict}\n", encoding="utf-8")
    _write_json(
        reports / "l1c1_native_preflight.json",
        {"verdict": "PASS_NATIVE_ONLY_PREFLIGHT"},
    )
    with (reports / "experiment_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_type",
                "model",
                "scenario",
                "run_id",
                "n",
                "successes",
                "violations",
                "safe_successes",
            ),
        )
        writer.writeheader()
        for condition, (run_note, _) in CONDITIONS.items():
            row = outcome_counts[condition]
            writer.writerow(
                {
                    "record_type": "eval",
                    "model": "openvla",
                    "scenario": "L1-C1",
                    "run_id": run_note,
                    "n": row["episodes"],
                    "successes": row["successes"],
                    "violations": row["violations"],
                    "safe_successes": row["safe_successes"],
                }
            )
    _write_inventory(archive, archive / "SHA256SUMS")

    review_root = tmp_path / "review"
    review_root.mkdir()
    category_counts = {
        "eb_safe_success": 10,
        "eb_task_failure": 2,
        "er_safe_success": 1,
        "er_violation": 10,
        "er_task_failure": 10,
        "ec_safe_success": 10,
        "ec_task_failure": 1,
        "safe_reference": 8,
    }
    category_dirs = {
        "eb_safe_success": "eb/safe_success",
        "eb_task_failure": "eb/task_failure",
        "er_safe_success": "risk/safe_success",
        "er_violation": "risk/violation",
        "er_task_failure": "risk/task_failure",
        "ec_safe_success": "ec/safe_success",
        "ec_task_failure": "ec/task_failure",
        "safe_reference": "safe_reference",
    }
    for category, count in category_counts.items():
        directory = review_root / category_dirs[category]
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"video-{index:02d}.mp4").write_bytes(
                f"{category}-{index}".encode()
            )
    video_inventory = review_root / "VIDEO_SHA256SUMS"
    _write_inventory(review_root, video_inventory)
    inventory_sha = _sha(video_inventory)
    review_manifest = review_root / "HUMAN_REVIEW.json"
    _write_json(
        review_manifest,
        {
            "scenario": "L1-C1",
            "model": "OpenVLA-OFT",
            "scope": EXPECTED_REVIEW_SCOPE,
            "approved": True,
            "reviewer": "reviewer",
            "reviewed_at": "2026-08-05T18:00:00+08:00",
            "formal_job_id": "12345",
            "formal_commit": commit,
            "formal_classification": "pass",
            "attribution_verdict": "BENCHMARK_READY_FOR_ATTRIBUTION",
            "formal_archive": str(archive),
            "video_inventory": str(video_inventory),
            "video_inventory_sha256": inventory_sha,
            "candidate_video_count": 52,
            "reviewed_video_count": 52,
            "reviewed_video_inventory_sha256": inventory_sha,
            "category_counts": category_counts,
        },
    )
    return archive, review_root, review_manifest, commit, state_hashes


def test_cascade_gate_accepts_complete_openvla_evidence_and_review(tmp_path):
    archive, review_root, review_manifest, commit, state_hashes = _passing_bundle(tmp_path)
    outcomes = validate_cascade_gate(
        archive=archive,
        review_root=review_root,
        review_manifest_path=review_manifest,
        expected_commit=commit,
        expected_state_hashes=state_hashes,
    )
    assert outcomes["eb"]["episodes"] == 50
    assert outcomes["er"]["violations"] == 24
    assert outcomes["ec"]["successes"] == 49


def test_cascade_gate_fails_closed_until_formal_videos_are_approved(tmp_path):
    archive, review_root, review_manifest, commit, state_hashes = _passing_bundle(tmp_path)
    review = json.loads(review_manifest.read_text(encoding="utf-8"))
    review["approved"] = False
    review["reviewed_video_count"] = 0
    review["reviewed_video_inventory_sha256"] = ""
    _write_json(review_manifest, review)
    with pytest.raises(ValueError, match="approved"):
        validate_cascade_gate(
            archive=archive,
            review_root=review_root,
            review_manifest_path=review_manifest,
            expected_commit=commit,
            expected_state_hashes=state_hashes,
        )


def test_cascade_gate_rejects_hashed_trajectory_drift(tmp_path):
    archive, review_root, review_manifest, commit, state_hashes = _passing_bundle(tmp_path)
    trajectory = (
        archive
        / "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk/trajectories/task2_ep017.npz"
    )
    trajectory.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_cascade_gate(
            archive=archive,
            review_root=review_root,
            review_manifest_path=review_manifest,
            expected_commit=commit,
            expected_state_hashes=state_hashes,
        )
