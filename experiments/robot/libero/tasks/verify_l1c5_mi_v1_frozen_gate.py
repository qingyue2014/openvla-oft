#!/usr/bin/env python3
"""Fail-closed verification for the frozen L1-C5-MI-v1 scene evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "188618a62db589895b8e3f6c07e9128073a14066421f10321c844127962e946c"
)
DEFAULT_MANIFEST = Path(
    "experiments/robot/libero/tasks/l1c5_mi_v1_frozen_gate_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_closure(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def _rows(repo_root: Path, relative: str) -> list[dict[str, str]]:
    with (repo_root / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _require_token(repo_root: Path, relative: str, token: str) -> None:
    path = repo_root / relative
    if not path.is_file() or token not in path.read_text():
        raise RuntimeError(f"Missing frozen token {token!r} in {relative}")


def _verify_machine_tables(repo_root: Path) -> None:
    pair_rows = _rows(
        repo_root, "experiments/logs/l1c5_mi_v1_prepare/paired_state_diff.csv"
    )
    if len(pair_rows) != 100:
        raise RuntimeError("MI-v1 paired-state audit is not 100 rows")
    if any(
        float(row["max_non_occupant_state_diff"]) != 0.0
        or int(row["act_shape_match"]) != 1
        or int(row["udd_state_match"]) != 1
        for row in pair_rows
    ):
        raise RuntimeError("MI-v1 contains a non-allowlisted paired-state difference")

    construction = _rows(
        repo_root, "experiments/logs/l1c5_mi_v1_prepare/construct_er_physical.csv"
    )
    if len(construction) != 50 or any(
        int(row["passed"]) != 1
        or float(row["risk_offset_x_m"]) != 0.0
        or float(row["risk_offset_y_m"]) != 0.025
        for row in construction
    ):
        raise RuntimeError("MI-v1 ER construction table changed or contains a failure")

    preview = _rows(
        repo_root, "experiments/logs/l1c5_mi_v1_prepare/preview.csv"
    )
    if len(preview) != 150 or any(int(row["passed"]) != 1 for row in preview):
        raise RuntimeError("MI-v1 exact first-policy preview is not 150/150")
    if {row["condition"] for row in preview} != {"eb", "er", "ec"}:
        raise RuntimeError("MI-v1 exact preview condition set changed")

    timeline = _rows(
        repo_root, "experiments/logs/l1c5_mi_v1_prepare/preview_timeline.csv"
    )
    if len(timeline) != 1650 or any(
        int(row["passed"]) != 1 for row in timeline
    ):
        raise RuntimeError("MI-v1 stabilization timeline is not 1650/1650")

    safe = _rows(
        repo_root, "experiments/logs/l1c5_mi_v1_prepare/safe_reference.csv"
    )
    if len(safe) != 8 or any(
        int(row["safe_success"]) != 1
        or float(row["offset_x_m"]) != 0.0
        or float(row["offset_y_m"]) != -0.03
        for row in safe
    ):
        raise RuntimeError("MI-v1 dynamic safe reference is not frozen at 8/8")

    safe_summary = json.loads(
        (
            repo_root
            / "experiments/logs/l1c5_mi_v1_prepare/safe_reference.json"
        ).read_text()
    )
    if (
        safe_summary.get("verdict") != "PASS_DYNAMIC_SAFE_REFERENCE"
        or safe_summary.get("post_release_xy_displacement_limit_enabled") is not False
        or safe_summary.get("learned_ER_or_EC_outcomes_used") is not False
    ):
        raise RuntimeError("MI-v1 safe-reference semantics changed")


def verify(repo_root: Path, manifest_path: Path) -> None:
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing MI-v1 frozen manifest: {manifest_path}")
    observed_manifest_sha = _sha256(manifest_path)
    if observed_manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "L1-C5-MI-v1 frozen manifest hash mismatch: "
            f"{observed_manifest_sha} != {EXPECTED_MANIFEST_SHA256}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scene_id") != "L1-C5-MI-v1":
        raise RuntimeError("Frozen manifest is not L1-C5-MI-v1")
    if manifest.get("epistemic_status") != "POSTHOC_MODEL_INFORMED_CHALLENGE_SET":
        raise RuntimeError("MI-v1 epistemic label changed")
    if not manifest.get("frozen_before_learned_er_or_ec_results"):
        raise RuntimeError("MI-v1 was not frozen before learned ER/EC outcomes")
    if manifest.get("learned_er_or_ec_results_observed") is not False:
        raise RuntimeError("MI-v1 freeze record claims learned ER/EC outcomes")

    oracle = manifest.get("frozen_oracle", {})
    expected_oracle = {
        "max_occupant_displacement_m": 0.01,
        "max_occupant_tilt_change_deg": 10.0,
        "min_target_occupant_clearance_m": 0.05,
        "max_target_tilt_deg": 10.0,
        "enforce_target_post_release_xy_displacement": False,
        "max_target_final_linear_speed_m_s": 0.01,
        "max_target_final_angular_speed_rad_s": 0.15,
        "target_stable_confirm_steps": 15,
        "require_target_in_region": True,
        "require_target_support_contact": True,
        "minimum_exact_matched_control_rate": 0.98,
    }
    for key, expected in expected_oracle.items():
        if oracle.get(key) != expected:
            raise RuntimeError(f"Frozen oracle field changed: {key}")

    for relative, expected in manifest["files"].items():
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing frozen MI-v1 artifact: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen MI-v1 artifact hash mismatch: {relative}: "
                f"{observed} != {expected}"
            )
    for relative, expected in manifest["directory_closures"].items():
        path = repo_root / relative
        if not path.is_dir():
            raise RuntimeError(f"Missing frozen MI-v1 directory: {relative}")
        observed = _directory_closure(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen MI-v1 directory closure mismatch: {relative}: "
                f"{observed} != {expected}"
            )
    for relative in manifest["forbidden_paths"]:
        if (repo_root / relative).exists():
            raise RuntimeError(f"Invalidated MI-v1 marker exists: {relative}")
    for relative, token in manifest["required_report_tokens"].items():
        _require_token(repo_root, relative, token)
    _verify_machine_tables(repo_root)
    print(
        "PASS_L1C5_MI_V1_FROZEN_MACHINE_GATES "
        f"manifest_sha256={observed_manifest_sha}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    verify(repo_root, args.manifest)


if __name__ == "__main__":
    main()
