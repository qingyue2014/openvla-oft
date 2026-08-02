"""Fail-closed native-task and asset-closure contract for L1-A1 v4.

The evaluated task is the unmodified ``libero_spatial`` task whose benchmark
prompt is ``pick up the black bowl next to the ramekin and place it on the
plate``.  The contract hashes the native BDDL, its exact goal section, and the
transitive XML / mesh / material / texture closure used by the native scene.
It also binds the generated Eb/Er/Ec state artifacts before evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from experiments.robot.libero.tasks.validate_l1a3_native_preflight import (
    _inventory,
    _prompt,
    _read_attr,
    _resolve_libero_root,
    _section,
    _sha256,
)


SCENE_ID = "L1-A1-V4"
TASK_SUITE = "libero_spatial"
TASK_ID = 1
TASK_FILE = (
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl"
)
TASK_PROMPT = "pick up the black bowl next to the ramekin and place it on the plate"
BDDL_PROMPT = "Pick the akita black bowl next to the ramekin and place it on the plate"
EXPECTED_GOAL_PREDICATES = ("On akita_black_bowl_1 plate_1",)
EXPECTED_FIXTURES = {
    "main_table": "table",
    "wooden_cabinet_1": "wooden_cabinet",
    "flat_stove_1": "flat_stove",
}
EXPECTED_OBJECTS = {
    "akita_black_bowl_1": "akita_black_bowl",
    "akita_black_bowl_2": "akita_black_bowl",
    "cookies_1": "cookies",
    "glazed_rim_porcelain_ramekin_1": "glazed_rim_porcelain_ramekin",
    "plate_1": "plate",
}
EXPECTED_RUNTIME_BODIES = (
    "table",
    "wooden_cabinet_1_main",
    "flat_stove_1_main",
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
)

VERDICT = "PASS_L1A1_V4_NATIVE_ONLY_PREFLIGHT"
INTERVENTION_ID = "l1a1_native_ramekin_relation_postwait_v4_c02_settle500"
PHYSICAL_GATE_VERDICT = "PASS_L1A1_V4_POSTWAIT_PHYSICAL_GATE"
FORMAL_WAIT_STEPS = 10
MAX_RECEPTACLE_TILT_DEG = 1.0


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return _sha256(Path(path))


def _goal_predicates(text: str) -> tuple[str, ...]:
    goal = _section(text, "goal")
    predicates = tuple(
        " ".join(match.group(1).split())
        for match in re.finditer(r"\((On\s+[A-Za-z0-9_]+\s+[A-Za-z0-9_]+)\)", goal)
    )
    if not predicates:
        raise ValueError("native BDDL goal contains no supported predicates")
    return predicates


def resolve_native_bddl() -> Path:
    return (
        _resolve_libero_root()
        / "libero"
        / "libero"
        / "bddl_files"
        / TASK_SUITE
        / TASK_FILE
    ).resolve(strict=True)


def _native_assets_root() -> Path:
    return (
        _resolve_libero_root() / "libero" / "libero" / "assets"
    ).resolve(strict=True)


def _native_asset_roots() -> tuple[Path, ...]:
    assets = _native_assets_root()
    return tuple(
        (assets / relative).resolve(strict=True)
        for relative in (
            "scenes/libero_kitchen_tabletop_base_style.xml",
            "articulated_objects/wooden_cabinet.xml",
            "articulated_objects/flat_stove.xml",
            "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
            "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml",
            "stable_scanned_objects/plate/plate.xml",
            "stable_hope_objects/cookies/cookies.xml",
        )
    )


def _referenced_paths(path: Path) -> tuple[Path, ...]:
    suffix = path.suffix.lower()
    references: list[str] = []
    if suffix == ".xml":
        root = ET.parse(path).getroot()
        for element in root.iter():
            reference = element.attrib.get("file")
            if reference:
                references.append(reference)
    elif suffix == ".obj":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("mtllib "):
                references.extend(stripped.split()[1:])
    elif suffix == ".mtl":
        material_map = re.compile(
            r"^(?:map_[A-Za-z0-9_]+|bump|disp|decal|refl)\s+(.+)$",
            re.IGNORECASE,
        )
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = material_map.match(line.strip())
            if match:
                # MuJoCo-compatible material files in LIBERO place the path at
                # the end even when map options precede it.
                references.append(match.group(1).split()[-1])
    return tuple((path.parent / value).resolve(strict=True) for value in references)


def referenced_native_asset_paths() -> tuple[Path, ...]:
    """Return the complete task asset closure, rejecting path escapes."""

    assets = _native_assets_root()
    pending = list(_native_asset_roots())
    resolved: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if path != assets and assets not in path.parents:
            raise ValueError(f"asset resolves outside native LIBERO assets: {path}")
        resolved.add(path)
        pending.extend(_referenced_paths(path))
    return tuple(sorted(resolved))


def native_asset_manifest() -> dict[str, dict[str, object]]:
    root = _resolve_libero_root()
    return {
        str(path.relative_to(root)): {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in referenced_native_asset_paths()
    }


def verify_native_asset_provenance() -> dict[str, object]:
    """Verify every referenced asset is tracked and unmodified in LIBERO."""

    root = _resolve_libero_root()
    relative_paths = [
        str(path.relative_to(root)) for path in referenced_native_asset_paths()
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *relative_paths,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot verify native LIBERO asset provenance") from exc
    if dirty:
        raise ValueError(f"selected LIBERO assets are modified or untracked: {dirty}")
    manifest = native_asset_manifest()
    return {
        "libero_commit": commit,
        "native_asset_files": manifest,
        "native_asset_manifest_sha256": _json_sha256(manifest),
        "all_assets_unmodified": True,
    }


def validate_native_task(
    native_bddl: Path,
    evaluated_bddl: Path,
    evaluated_prompt: str,
) -> dict[str, object]:
    native = native_bddl.resolve(strict=True)
    evaluated = evaluated_bddl.resolve(strict=True)
    if native.name != TASK_FILE or native.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native task source: {native}")
    if "bddl_files" not in native.parts or not native.samefile(evaluated):
        raise ValueError(f"evaluated BDDL is not the selected native task: {evaluated}")
    text = native.read_text(encoding="utf-8")
    bddl_prompt = _prompt(text)
    if bddl_prompt != BDDL_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            "prompt mismatch: "
            f"native_bddl={bddl_prompt!r}, native_benchmark={TASK_PROMPT!r}, "
            f"evaluated={evaluated_prompt!r}"
        )
    fixtures = _inventory(text, "fixtures")
    objects = _inventory(text, "objects")
    goals = _goal_predicates(text)
    if fixtures != EXPECTED_FIXTURES or objects != EXPECTED_OBJECTS:
        raise ValueError(
            f"native asset inventory mismatch: fixtures={fixtures}, objects={objects}"
        )
    if goals != EXPECTED_GOAL_PREDICATES:
        raise ValueError(f"native goal mismatch: {goals}")
    goal_source = _section(text, "goal")
    provenance = verify_native_asset_provenance()
    inventory = {"fixtures": fixtures, "objects": objects}
    return {
        "schema_version": 2,
        "scenario": SCENE_ID,
        "verdict": VERDICT,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": TASK_FILE,
        "prompt": TASK_PROMPT,
        "bddl_prompt": bddl_prompt,
        "native_bddl": str(native),
        "evaluated_bddl": str(evaluated),
        "native_bddl_source": text,
        "bddl_sha256": _file_sha256(native),
        "goal_source": goal_source,
        "goal_predicates": list(goals),
        "goal_signature_sha256": hashlib.sha256(goal_source.encode("utf-8")).hexdigest(),
        "fixtures": fixtures,
        "objects": objects,
        "asset_inventory_sha256": _json_sha256(inventory),
        "source_to_project_inventory_delta": {"fixtures": [], "objects": []},
        "source_to_project_bddl_delta": "none; evaluated BDDL is the native source",
        "custom_assets": [],
        "custom_bddl": False,
        **provenance,
    }


def write_preflight(manifest_path: Path, report_path: Path) -> dict[str, object]:
    native = resolve_native_bddl()
    record = validate_native_task(native, native, TASK_PROMPT)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            (
                "# L1-A1 v4 Native-Only Preflight",
                "",
                f"- Verdict: **{VERDICT}**",
                f"- Selected native task: `{TASK_SUITE}/{TASK_FILE}` (id `{TASK_ID}`)",
                f"- Exact benchmark prompt: `{TASK_PROMPT}`",
                f"- Exact goal predicates: `{list(EXPECTED_GOAL_PREDICATES)}`",
                f"- Native BDDL SHA-256: `{record['bddl_sha256']}`",
                f"- Goal signature SHA-256: `{record['goal_signature_sha256']}`",
                f"- Inventory signature SHA-256: `{record['asset_inventory_sha256']}`",
                f"- Referenced native asset files: `{len(record['native_asset_files'])}`",
                f"- Native asset closure SHA-256: `{record['native_asset_manifest_sha256']}`",
                f"- LIBERO commit: `{record['libero_commit']}`",
                "- Source-to-project inventory delta: `empty`",
                "- Source-to-project BDDL delta: `none`",
                "- Custom assets / BDDL / prompt override: `none`",
                "- State-file hashes and the complete layout allowlist are bound after paired generation.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Verdict: {VERDICT}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    return record


def verify_state_file(state_path: Path, record: Mapping[str, object]) -> None:
    import h5py

    state_path = state_path.resolve(strict=True)
    with h5py.File(state_path, "r") as handle:
        expected = {
            "native_only": True,
            "task_suite_name": TASK_SUITE,
            "task_id": TASK_ID,
            "task_file": TASK_FILE,
            "native_prompt": TASK_PROMPT,
            "native_bddl_sha256": record["bddl_sha256"],
            "goal_signature_sha256": record["goal_signature_sha256"],
            "asset_inventory_sha256": record["asset_inventory_sha256"],
            "native_asset_manifest_sha256": record["native_asset_manifest_sha256"],
            "libero_commit": record["libero_commit"],
            "intervention_id": INTERVENTION_ID,
            "physical_gate_verdict": PHYSICAL_GATE_VERDICT,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "max_receptacle_tilt_deg": MAX_RECEPTACLE_TILT_DEG,
        }
        missing = sorted(set(expected) - set(handle.attrs))
        if missing:
            raise ValueError(f"{state_path} missing native-only attrs: {missing}")
        mismatches = {
            name: (_read_attr(handle.attrs, name), value)
            for name, value in expected.items()
            if _read_attr(handle.attrs, name) != value
        }
        if mismatches:
            raise ValueError(f"{state_path} native-only metadata mismatch: {mismatches}")


def bind_generated_artifacts(
    manifest_path: Path,
    pairing_path: Path,
    condition_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Bind exact paired state bytes and pairing manifest before rollout."""

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("verdict") != VERDICT:
        raise ValueError("cannot bind artifacts to an invalid L1-A1-v4 preflight")
    evaluated: dict[str, dict[str, object]] = {}
    signatures = set()
    for condition, path in condition_paths.items():
        resolved = Path(path).resolve(strict=True)
        verify_state_file(resolved, record)
        evaluated[condition.lower()] = {
            "path": str(resolved),
            "sha256": _file_sha256(resolved),
        }
        signatures.add(record["asset_inventory_sha256"])
    if set(evaluated) != {"eb", "er", "ec"} or len(signatures) != 1:
        raise ValueError("Eb/Er/Ec state artifacts do not share one inventory signature")
    pairing = pairing_path.resolve(strict=True)
    record["evaluated_conditions"] = evaluated
    record["pairing_manifest"] = str(pairing)
    record["pairing_manifest_sha256"] = _file_sha256(pairing)
    record["condition_inventory_signatures_identical"] = True
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def verify_evaluation_request(
    manifest_path: str,
    *,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    task_bddl: str,
    policy_prompt: str,
    initial_states_path: str,
) -> dict[str, object]:
    """Recheck native identity, asset bytes, and the exact state artifact."""

    path = Path(manifest_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != VERDICT:
        raise ValueError(f"invalid L1-A1-v4 preflight verdict in {path}")
    fresh = validate_native_task(resolve_native_bddl(), Path(task_bddl), task_language)
    for key in (
        "task_suite_name",
        "task_id",
        "task_file",
        "prompt",
        "bddl_sha256",
        "goal_signature_sha256",
        "asset_inventory_sha256",
        "native_asset_manifest_sha256",
        "libero_commit",
    ):
        if record.get(key) != fresh.get(key):
            raise ValueError(f"stale or mismatched L1-A1-v4 preflight field: {key}")
    if (task_suite_name, int(task_id), task_language, policy_prompt) != (
        TASK_SUITE,
        TASK_ID,
        TASK_PROMPT,
        TASK_PROMPT,
    ):
        raise ValueError("L1-A1-v4 runtime task or prompt mismatch")
    states = Path(initial_states_path).resolve(strict=True)
    matched = [
        evidence
        for evidence in (record.get("evaluated_conditions") or {}).values()
        if Path(evidence["path"]) == states
    ]
    if len(matched) != 1 or matched[0].get("sha256") != _file_sha256(states):
        raise ValueError("L1-A1-v4 runtime state artifact is not bound to the preflight")
    pairing = Path(record["pairing_manifest"]).resolve(strict=True)
    if _file_sha256(pairing) != record.get("pairing_manifest_sha256"):
        raise ValueError("L1-A1-v4 pairing manifest changed after preflight binding")
    verify_state_file(states, fresh)
    print(f"Verdict: {VERDICT} (runtime recheck)")
    return record


def verify_runtime_asset_inventory(manifest_path: str | Path, model) -> dict[str, int]:
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if record.get("asset_inventory_sha256") != _json_sha256(
        {"fixtures": EXPECTED_FIXTURES, "objects": EXPECTED_OBJECTS}
    ):
        raise ValueError("L1-A1-v4 native asset inventory signature mismatch")
    resolved: dict[str, int] = {}
    for body in EXPECTED_RUNTIME_BODIES:
        try:
            resolved[body] = int(model.body_name2id(body))
        except Exception as exc:
            raise ValueError(f"L1-A1-v4 native body missing at runtime: {body}") from exc
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/robot/libero/tasks/l1a1_v4_native_preflight.json",
    )
    parser.add_argument(
        "--report", default="experiments/logs/l1a1_v4_native_preflight.md"
    )
    args = parser.parse_args()
    write_preflight(Path(args.manifest), Path(args.report))


if __name__ == "__main__":
    main()
