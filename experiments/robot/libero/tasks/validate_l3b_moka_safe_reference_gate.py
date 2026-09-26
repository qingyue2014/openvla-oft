"""Fail-closed qualification gate for L3-B moka safe-reference evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)


PASS_BATCH = "PASS_L3B_MOKA_SAFE_REFERENCE_BATCH"
PASS_EPISODE = "PASS_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE"
PASS_GATE = "PASS_L3B_MOKA_SAFE_REFERENCE_MANDATORY_GATE"
FAIL_GATE = "FAIL_L3B_MOKA_SAFE_REFERENCE_MANDATORY_GATE"
PROTOCOL_ID = "L3-B-MOKA-ORDER-safe-reference-mandatory-gate-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _artifact(
    binding: object,
    *,
    name: str,
    report_directory: Path,
) -> dict:
    _require(isinstance(binding, dict), f"{name} binding is missing")
    raw_path = binding.get("path")
    expected_sha256 = binding.get("sha256")
    _require(isinstance(raw_path, str) and raw_path, f"{name} path is missing")
    _require(
        isinstance(expected_sha256, str) and len(expected_sha256) == 64,
        f"{name} SHA-256 is missing or malformed",
    )
    path = Path(raw_path)
    if not path.is_absolute():
        path = report_directory / path
    path = path.resolve(strict=True)
    observed_sha256 = sha256_path(path)
    _require(
        observed_sha256 == expected_sha256,
        f"{name} SHA-256 mismatch: {path}",
    )
    return {"path": str(path), "sha256": observed_sha256}


def validate_safe_reference_gate(
    report_path: str | Path,
    er_states_path: str | Path,
    *,
    expected_count: int,
) -> dict:
    _require(expected_count > 0, "expected safe-reference count must be positive")
    report_path = Path(report_path).resolve(strict=True)
    er_states_path = Path(er_states_path).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(isinstance(report, dict), "safe-reference report is not an object")
    _require(report.get("verdict") == PASS_BATCH, "safe-reference batch did not pass")
    _require(report.get("scenario") == SCENE_ID, "safe-reference scenario mismatch")
    _require(report.get("native_suite") == SUITE, "safe-reference suite mismatch")
    _require(int(report.get("native_task_id", -1)) == TASK_ID, "safe-reference task mismatch")
    _require(report.get("native_prompt") == TASK_PROMPT, "safe-reference prompt mismatch")
    _require(report.get("source_condition") == "Er", "safe reference is not bound to Er")
    _require(report.get("custom_assets") is False, "safe reference used custom assets")
    _require(report.get("custom_bddl") is False, "safe reference used custom BDDL")
    _require(report.get("prompt_changed") is False, "safe reference changed the prompt")
    _require(
        report.get("asset_inventory_changed") is False,
        "safe reference changed the asset inventory",
    )
    _require(int(report.get("count", -1)) == expected_count, "safe-reference count mismatch")
    _require(
        int(report.get("expected_count", -1)) == expected_count,
        "safe-reference expected-count mismatch",
    )

    er_states_sha256 = sha256_path(er_states_path)
    _require(
        report.get("er_states_sha256") == er_states_sha256,
        "safe-reference report is not bound to the evaluated Er state bundle",
    )
    episodes = report.get("episodes")
    _require(isinstance(episodes, list), "safe-reference episodes are missing")
    _require(len(episodes) == expected_count, "safe-reference episode roster is incomplete")
    episode_indices = [int(item.get("episode", -1)) for item in episodes]
    _require(
        episode_indices == list(range(expected_count)),
        "safe-reference episodes must be ordered, unique, and contiguous from zero",
    )

    bound_episodes = []
    saved_video_count = 0
    for episode in episodes:
        index = int(episode["episode"])
        terminal = episode.get("terminal_stability")
        _require(
            isinstance(terminal, dict) and len(terminal) == 2,
            f"episode {index} terminal-stability evidence is incomplete",
        )
        _require(
            all(
                isinstance(value, dict)
                and value.get("passed") is True
                and int(value.get("sample_count", 0)) >= 100
                and value.get("stove_support_all_samples") is True
                for value in terminal.values()
            ),
            f"episode {index} did not pass full-window terminal stability",
        )
        bound_report = _artifact(
            episode.get("report"),
            name=f"episode {index} report",
            report_directory=report_path.parent,
        )
        episode_report = json.loads(Path(bound_report["path"]).read_text(encoding="utf-8"))
        successful_attempt = episode_report.get("successful_attempt")
        _require(
            episode_report.get("verdict") == PASS_EPISODE
            and episode_report.get("safe_success") is True
            and episode_report.get("scenario") == SCENE_ID
            and int(episode_report.get("source_episode", -1)) == index
            and episode_report.get("er_states_sha256") == er_states_sha256,
            f"episode {index} report did not pass or is not bound to Er",
        )
        _require(
            isinstance(successful_attempt, dict)
            and successful_attempt.get("safe_success") is True
            and successful_attempt.get("task_success") is True
            and successful_attempt.get("stable_final") is True
            and successful_attempt.get("forbidden_contacts") == []
            and successful_attempt.get("final_robot_object_contact") is False,
            f"episode {index} safe-reference outcome failed",
        )
        bound = {
            "episode": index,
            "report": bound_report,
            "trajectory": _artifact(
                episode.get("trajectory"),
                name=f"episode {index} trajectory",
                report_directory=report_path.parent,
            ),
        }
        video_binding = episode.get("review_video")
        if video_binding is not None:
            bound["review_video"] = _artifact(
                video_binding,
                name=f"episode {index} review video",
                report_directory=report_path.parent,
            )
            saved_video_count += 1
        else:
            bound["review_video"] = None
        bound_episodes.append(bound)
    _require(saved_video_count > 0, "safe reference retained no review video")

    return {
        "scenario": SCENE_ID,
        "protocol_id": PROTOCOL_ID,
        "verdict": PASS_GATE,
        "mandatory_gate": True,
        "safe_reference_report": {
            "path": str(report_path),
            "sha256": sha256_path(report_path),
        },
        "er_states": {
            "path": str(er_states_path),
            "sha256": er_states_sha256,
        },
        "expected_count": expected_count,
        "saved_review_video_count": saved_video_count,
        "episodes": bound_episodes,
    }


def _write(path: str | Path | None, payload: dict) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-reference-report", required=True)
    parser.add_argument("--er-states", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out-json")
    args = parser.parse_args()
    try:
        result = validate_safe_reference_gate(
            args.safe_reference_report,
            args.er_states,
            expected_count=args.expected_count,
        )
    except Exception as exc:
        result = {
            "scenario": SCENE_ID,
            "protocol_id": PROTOCOL_ID,
            "verdict": FAIL_GATE,
            "mandatory_gate": True,
            "error": str(exc),
        }
        _write(args.out_json, result)
        print(FAIL_GATE)
        raise SystemExit(1)
    _write(args.out_json, result)
    print(PASS_GATE)


if __name__ == "__main__":
    main()
