#!/usr/bin/env python3
"""Fail-closed verification for the frozen L1-C5 machine evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "14eeb148208f536eca7920ddde28b285502007d6b719739f6742712264cd5937"
)
DEFAULT_MANIFEST = Path(
    "experiments/robot/libero/tasks/l1c5_frozen_gate_manifest.json"
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


def _require_report_token(repo_root: Path, path: str, token: str) -> None:
    report = repo_root / path
    if not report.is_file() or token not in report.read_text():
        raise RuntimeError(f"Missing frozen gate token {token!r} in {path}")


def _verify_calibration(repo_root: Path, manifest: dict[str, object]) -> None:
    evidence = manifest["selected_safe_reference"]
    selected = tuple(float(value) for value in evidence["offset_xy_m"])
    rows = list(
        csv.DictReader(
            (repo_root / "experiments/logs/l1c5_calibration.csv").open()
        )
    )
    center = [
        row
        for row in rows
        if float(row["offset_x_m"]) == 0.0
        and float(row["offset_y_m"]) == 0.0
    ]
    selected_rows = [
        row
        for row in rows
        if (
            float(row["offset_x_m"]),
            float(row["offset_y_m"]),
        )
        == selected
    ]
    expected_n = int(evidence["num_states"])
    if len(center) != expected_n or any(int(row["safe_success"]) for row in center):
        raise RuntimeError("Frozen direct-center unsafe calibration changed")
    if len(selected_rows) != expected_n or not all(
        int(row["safe_success"]) for row in selected_rows
    ):
        raise RuntimeError("Frozen selected-offset calibration changed")


def _verify_safe_reference(repo_root: Path, manifest: dict[str, object]) -> None:
    evidence = manifest["selected_safe_reference"]
    selected = tuple(float(value) for value in evidence["offset_xy_m"])
    rows = list(
        csv.DictReader(
            (repo_root / "experiments/logs/l1c5_safe_reference.csv").open()
        )
    )
    expected_n = int(evidence["num_states"])
    if len(rows) != expected_n or not all(
        int(row["safe_success"]) for row in rows
    ):
        raise RuntimeError("Frozen dynamic safe-reference rate is not 8/8")
    if any(
        (float(row["offset_x_m"]), float(row["offset_y_m"])) != selected
        for row in rows
    ):
        raise RuntimeError("Dynamic safe reference used a non-frozen offset")


def verify(repo_root: Path, manifest_path: Path) -> None:
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing L1-C5 frozen gate manifest: {manifest_path}")
    observed_manifest_sha = _sha256(manifest_path)
    if observed_manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "L1-C5 frozen gate manifest hash mismatch: "
            f"{observed_manifest_sha} != {EXPECTED_MANIFEST_SHA256}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("scenario") != "L1-C5":
        raise RuntimeError("Frozen gate manifest scenario is not L1-C5")
    if not manifest.get("frozen_before_learned_policy_results"):
        raise RuntimeError("L1-C5 gate was not frozen before learned-policy results")

    for relative, expected in manifest["files"].items():
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing frozen L1-C5 artifact: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen L1-C5 artifact hash mismatch: {relative}: "
                f"{observed} != {expected}"
            )

    for relative, expected in manifest["directory_closures"].items():
        path = repo_root / relative
        if not path.is_dir():
            raise RuntimeError(f"Missing frozen L1-C5 directory: {relative}")
        observed = _directory_closure(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen L1-C5 directory closure mismatch: {relative}: "
                f"{observed} != {expected}"
            )

    for relative in manifest["forbidden_paths"]:
        if (repo_root / relative).exists():
            raise RuntimeError(f"Invalidated L1-C5 artifact marker exists: {relative}")
    for relative, token in manifest["required_report_tokens"].items():
        _require_report_token(repo_root, relative, token)

    _verify_calibration(repo_root, manifest)
    _verify_safe_reference(repo_root, manifest)
    print(
        "PASS_L1C5_FROZEN_MACHINE_GATES "
        f"manifest_sha256={observed_manifest_sha}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--print-directory-closure", type=Path)
    args = parser.parse_args()
    repo_root = (
        args.repo_root.resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    if args.print_directory_closure is not None:
        path = args.print_directory_closure
        if not path.is_absolute():
            path = repo_root / path
        print(json.dumps(_directory_closure(path), sort_keys=True))
        return
    verify(repo_root, args.manifest)


if __name__ == "__main__":
    main()
