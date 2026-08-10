"""Fail-closed runtime validation for four-suite native occupied-goal tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.robot.libero.tasks.l1c_occupied_pipeline import (
    STANDARD_LIBERO_SUITES,
    _json_sha256,
    _runtime_asset_inventory,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path: str | Path) -> dict:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        _mark_invalid(path, f"cannot read native-only manifest: {exc}")
        raise ValueError(f"Invalid native-only manifest {path}: {exc}") from exc
    if manifest.get("verdict") != "PASS_NATIVE_ONLY_PREFLIGHT":
        _fail(path, "native-only preflight verdict is not passing")
    scenario = manifest.get("scenario")
    if scenario not in {"L1-C4", "L1-C5"}:
        _fail(path, f"unexpected scenario {manifest.get('scenario')!r}")
    if manifest.get("native_suite") not in STANDARD_LIBERO_SUITES:
        _fail(
            path,
            f"{scenario} task is not from one of the four standard LIBERO suites",
        )
    if manifest.get("native_suite") == "libero_90":
        _fail(path, f"libero_90 is forbidden for {scenario}")
    if manifest.get("custom_assets") != []:
        _fail(path, "custom assets were declared")
    if scenario == "L1-C5":
        if not manifest.get("native_goal_canonical") or not manifest.get(
            "native_goal_sha256"
        ):
            _fail(path, "native parsed-goal signature is missing")
        audits = manifest.get("paired_observed_diff_audit", [])
        if not audits or any(
            row.get("verdict") != "PASS_ALLOWLISTED_OCCUPANT_JOINT_ONLY"
            for row in audits
        ):
            _fail(path, "paired observed-diff allowlist audit is missing or failed")
        if not manifest.get("design_prereg", {}).get("sha256"):
            _fail(path, "L1-C5 design preregistration is not hash-bound")
    return manifest


def _mark_invalid(manifest_path: Path, reason: str) -> None:
    marker = manifest_path.with_suffix(".invalid.json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "verdict": "INVALID_NATIVE_ONLY_PREFLIGHT",
                "reason": reason,
                "invalidates": [
                    "scene",
                    "jobs",
                    "metrics",
                    "videos",
                    "tables",
                    "html",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _fail(manifest_path: Path, reason: str) -> None:
    _mark_invalid(manifest_path, reason)
    raise ValueError(f"occupied-goal native-only hard stop: {reason}")


def verify_evaluation_request(
    manifest_path: str,
    *,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    task_bddl: str,
    policy_prompt: str,
    initial_states_path: str,
) -> None:
    """Bind the evaluator request to the exact preflighted native task."""
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    expected = {
        "native_suite": task_suite_name,
        "native_task_id": int(task_id),
        "native_prompt": task_language,
    }
    for key, actual in expected.items():
        if manifest.get(key) != actual:
            _fail(
                path,
                f"{key} mismatch: {manifest.get(key)!r} != {actual!r}",
            )
    if policy_prompt != task_language:
        _fail(
            path,
            f"policy prompt differs from native prompt: "
            f"{policy_prompt!r} != {task_language!r}",
        )
    task_bddl_path = Path(task_bddl).resolve()
    if (
        Path(manifest.get("native_bddl_resolved_path", "")).resolve()
        != task_bddl_path
    ):
        _fail(path, "evaluated BDDL path differs from the native BDDL")
    if _sha256(task_bddl_path) != manifest.get("native_bddl_sha256"):
        _fail(path, "evaluated native BDDL content changed after preflight")

    state_path = Path(initial_states_path).resolve()
    conditions = manifest.get("evaluated_conditions", {})
    matches = [
        record
        for record in conditions.values()
        if Path(record.get("state_file", "")).resolve() == state_path
    ]
    if len(matches) != 1:
        _fail(path, "evaluated initial-state file was not preflighted")
    if _sha256(state_path) != matches[0].get("state_sha256"):
        _fail(path, "evaluated initial-state file changed after preflight")
    if matches[0].get("prompt") != task_language:
        _fail(path, "evaluated state prompt differs from native prompt")
    if matches[0].get("bddl") != manifest.get("native_bddl"):
        _fail(path, "evaluated state BDDL metadata mismatch")


def verify_runtime_asset_inventory(
    manifest_path: str, model
) -> None:
    """Compare the evaluator's compiled model to the preflight inventory."""
    path = Path(manifest_path)
    manifest = _load_manifest(path)
    actual_hash = _json_sha256(_runtime_asset_inventory(model))
    expected_hash = manifest.get("native_asset_inventory_sha256")
    if actual_hash != expected_hash:
        _fail(
            path,
            "compiled evaluator asset inventory differs from the selected "
            "native task",
        )
