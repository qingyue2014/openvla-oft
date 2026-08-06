"""Fail-closed native-task and asset preflight for L1-B3 Task-4 Outcome V2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENE_ID = "L1-B3-Task4-Outcome-V2"
FAMILY = "l1b3_task4_outcome_v2"
TASK_SUITE = "libero_goal"
TASK_ID = 4
TASK_PROMPT = "put the bowl on top of the cabinet"
TASK_GOALS = ["On akita_black_bowl_1 wooden_cabinet_1_top_side"]
VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_NATIVE_PREFLIGHT"
INTERVENTION_ALLOWLIST = {
    "body": "wine_bottle_1_main",
    "pose_fields": ["free_joint.qpos.x", "free_joint.qpos.y"],
    "velocity_fields": [
        "free_joint.qvel.x",
        "free_joint.qvel.y",
        "free_joint.qvel.z",
        "free_joint.qvel.rx",
        "free_joint.qvel.ry",
        "free_joint.qvel.rz",
    ],
    "all_other_state_fields": "must_be_byte_identical",
}


def _registered_family_spec() -> dict[str, object]:
    """Read the selected literal family without importing simulator modules."""
    generator = REPO_ROOT / (
        "experiments/robot/libero/tasks/"
        "generate_l1b_swept_initial_states.py"
    )
    module = ast.parse(generator.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}

    class ResolveNames(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if node.id not in constants:
                raise ValueError(
                    f"non-literal name {node.id!r} in registered v2 family"
                )
            return ast.copy_location(ast.Constant(constants[node.id]), node)

    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id != "FAMILIES"
        ):
            try:
                constants[statement.targets[0].id] = ast.literal_eval(
                    statement.value
                )
            except ValueError:
                pass
        if not any(
            isinstance(target, ast.Name) and target.id == "FAMILIES"
            for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.Dict):
            break
        for key, value in zip(statement.value.keys, statement.value.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == FAMILY
            ):
                spec = ast.literal_eval(ResolveNames().visit(value))
                if not isinstance(spec, dict):
                    raise ValueError("registered v2 family is not a dictionary")
                return spec
    raise ValueError(f"family {FAMILY!r} is not registered in {generator}")


def _resolve_libero_repository() -> Path:
    candidates = []
    if os.environ.get("LIBERO_ROOT"):
        candidates.append(Path(os.environ["LIBERO_ROOT"]))
    candidates.extend(
        (
            REPO_ROOT / "_deps/LIBERO",
            REPO_ROOT.parent / "LIBERO",
            REPO_ROOT.parent / "libero",
        )
    )
    for raw in candidates:
        for candidate in (raw, raw.parent):
            root = candidate.expanduser().resolve()
            if (
                root.joinpath(
                    "libero/libero/benchmark/libero_suite_task_map.py"
                ).is_file()
                and root.joinpath("libero/libero/bddl_files").is_dir()
                and root.joinpath("libero/libero/assets").is_dir()
            ):
                return root
    raise FileNotFoundError(
        "could not locate a native LIBERO repository; set LIBERO_ROOT to the "
        "repository that supplies formal evaluation"
    )


def _benchmark_task_contract(
    libero_root: Path,
) -> tuple[str, str, Path]:
    task_map_path = libero_root / (
        "libero/libero/benchmark/libero_suite_task_map.py"
    )
    module = ast.parse(task_map_path.read_text(encoding="utf-8"))
    task_map = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in statement.targets
        ):
            task_map = ast.literal_eval(statement.value)
            break
    if not isinstance(task_map, dict) or TASK_SUITE not in task_map:
        raise ValueError("native LIBERO task map does not define libero_goal")
    suite_tasks = task_map[TASK_SUITE]
    if not isinstance(suite_tasks, list) or TASK_ID >= len(suite_tasks):
        raise ValueError("native LIBERO task map has no selected task id")
    task_name = str(suite_tasks[TASK_ID])
    benchmark_prompt = " ".join(task_name.split("_"))
    return task_name, benchmark_prompt, task_map_path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        raise ValueError(f"native BDDL is missing :{name}")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"native BDDL has unterminated :{name}")


def _prompt(text: str) -> str:
    match = re.search(r"\(:language\s+([^)]+)\)", text)
    if match is None:
        raise ValueError("native BDDL has no :language prompt")
    return " ".join(match.group(1).split())


def _inventory(text: str, section: str) -> dict[str, str]:
    payload = _section(text, section)
    inventory: dict[str, str] = {}
    for match in re.finditer(
        r"(?m)^\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)*)\s*-\s*"
        r"([A-Za-z0-9_]+)\s*$",
        payload,
    ):
        names, asset_type = match.groups()
        for name in names.split():
            inventory[name] = asset_type
    if not inventory:
        raise ValueError(f"native BDDL :{section} inventory is empty")
    return inventory


def _goal_predicates(goal_source: str) -> list[str]:
    predicates = []
    for match in re.finditer(
        r"\(([A-Z][A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)+)\)",
        goal_source,
    ):
        predicate = " ".join(match.group(1).split())
        if not predicate.startswith("And "):
            predicates.append(predicate)
    if not predicates:
        raise ValueError("native BDDL goal contains no predicates")
    return predicates


def _libero_root(native_bddl: Path) -> Path:
    parts = native_bddl.resolve().parts
    try:
        index = parts.index("libero", max(0, len(parts) - 8))
    except ValueError as exc:
        raise ValueError(f"cannot resolve LIBERO root from {native_bddl}") from exc
    # Native layout is <root>/libero/libero/bddl_files/...
    for parent in native_bddl.parents:
        if (parent / "libero" / "libero" / "assets").is_dir():
            return parent.resolve()
    raise ValueError(f"cannot resolve LIBERO repository from {native_bddl}")


def _referenced_paths(path: Path) -> tuple[Path, ...]:
    if path.suffix.lower() != ".xml":
        return ()
    root = ET.parse(path).getroot()
    return tuple(
        (path.parent / element.attrib["file"]).resolve(strict=True)
        for element in root.iter()
        if element.attrib.get("file")
    )


def _asset_root_candidates(
    assets_root: Path, inventory: dict[str, str]
) -> tuple[Path, ...]:
    roots = {
        path.resolve()
        for asset_type in sorted(set(inventory.values()))
        for path in assets_root.rglob(f"{asset_type}.xml")
    }
    # The main table is supplied by the native scene XML rather than a
    # standalone table.xml asset.
    scene_root = assets_root / "scenes/libero_kitchen_tabletop_base_style.xml"
    if scene_root.is_file():
        roots.add(scene_root.resolve())
    unresolved = sorted(
        asset_type
        for asset_type in set(inventory.values())
        if asset_type != "table"
        and not any(path.stem == asset_type for path in roots)
    )
    if unresolved:
        raise ValueError(f"could not resolve native asset XML roots: {unresolved}")
    return tuple(sorted(roots))


def _asset_closure(
    assets_root: Path, inventory: dict[str, str]
) -> tuple[Path, ...]:
    pending = list(_asset_root_candidates(assets_root, inventory))
    resolved: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if path != assets_root and assets_root not in path.parents:
            raise ValueError(f"asset resolves outside native LIBERO: {path}")
        resolved.add(path)
        pending.extend(_referenced_paths(path))
    if not resolved:
        raise ValueError("native asset closure is empty")
    return tuple(sorted(resolved))


def _verify_git_clean(libero_root: Path, paths: list[Path]) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=libero_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative = [str(path.relative_to(libero_root)) for path in paths]
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *relative,
        ],
        cwd=libero_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"selected native files are modified/untracked: {dirty}")
    return commit


def build_record() -> dict[str, object]:
    spec = _registered_family_spec()
    required_spec = {
        "bddl_file": None,
        "native_assets_only": True,
        "preserve_native_layout": True,
        "outcome_based": True,
        "scene_contract": "l1b3_task4_swept_outcome_v2",
    }
    mismatches = {
        key: (spec.get(key), value)
        for key, value in required_spec.items()
        if spec.get(key) != value
    }
    if mismatches:
        raise ValueError(f"registered v2 family contract changed: {mismatches}")

    selected_libero_root = _resolve_libero_repository()
    task_name, benchmark_prompt, task_map_path = _benchmark_task_contract(
        selected_libero_root
    )
    if benchmark_prompt != TASK_PROMPT:
        raise ValueError(
            f"native benchmark prompt mismatch: {benchmark_prompt!r}"
        )
    native_bddl = Path(
        selected_libero_root,
        "libero/libero/bddl_files",
        TASK_SUITE,
        f"{task_name}.bddl",
    ).resolve(strict=True)
    text = native_bddl.read_text(encoding="utf-8")
    bddl_prompt = _prompt(text)
    fixtures = _inventory(text, "fixtures")
    objects = _inventory(text, "objects")
    goal_source = _section(text, "goal")
    goals = _goal_predicates(goal_source)
    if goals != TASK_GOALS:
        raise ValueError(
            f"native task goal mismatch: observed={goals} expected={TASK_GOALS}"
        )
    libero_root = _libero_root(native_bddl)
    if libero_root != selected_libero_root:
        raise ValueError("task map and selected BDDL resolve to different LIBERO roots")
    assets_root = (libero_root / "libero" / "libero" / "assets").resolve(
        strict=True
    )
    inventory = {"fixtures": fixtures, "objects": objects}
    closure = _asset_closure(
        assets_root, {**fixtures, **objects}
    )
    commit = _verify_git_clean(
        libero_root, [task_map_path, native_bddl, *closure]
    )
    asset_files = {
        str(path.relative_to(libero_root)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in closure
    }
    prereg = REPO_ROOT / (
        "experiments/robot/libero/tasks/"
        "l1b3_task4_outcome_v2_design_prereg.json"
    )
    preregistration = json.loads(prereg.read_text(encoding="utf-8"))
    native_contract = preregistration.get("native_task", {})
    if (
        preregistration.get("family") != FAMILY
        or native_contract.get("benchmark_prompt") != TASK_PROMPT
        or native_contract.get("native_bddl_embedded_language") != bddl_prompt
        or native_contract.get("evaluated_policy_prompt_source")
        != "native benchmark task language, with no override"
    ):
        raise ValueError(
            "preregistration does not match the selected native prompt contract"
        )
    project_relative_paths = (
        "experiments/robot/libero/physcog_oracles.py",
        "experiments/robot/libero/libero_utils.py",
        "experiments/robot/libero/run_physcog_libero_l1_eval.py",
        "experiments/robot/libero/tasks/generate_l1b_swept_initial_states.py",
        "experiments/robot/libero/tasks/calibrate_l1b3_trajectory_conditioned_states.py",
        "experiments/robot/libero/tasks/replay_l1b_outcome_eb_actions.py",
        "experiments/robot/libero/tasks/validate_l1b_swept_states.py",
        "experiments/robot/libero/tasks/validate_l1b_rollout_physics.py",
        "experiments/robot/libero/tasks/validate_l1b_safe_reference.py",
        "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_initial_gate.py",
        "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_preflight.py",
        "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
        "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
    )
    project_paths = (prereg,) + tuple(
        REPO_ROOT / relative for relative in project_relative_paths
    )
    libero_utils_text = (
        REPO_ROOT / "experiments/robot/libero/libero_utils.py"
    ).read_text(encoding="utf-8")
    base_runner_text = (
        REPO_ROOT
        / "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh"
    ).read_text(encoding="utf-8")
    policy_prompt_path_verified = bool(
        "task_description = task.language" in libero_utils_text
        and "--task_description_override" not in base_runner_text
    )
    if not policy_prompt_path_verified:
        raise ValueError(
            "cannot verify that evaluation passes the native benchmark prompt "
            "to the policy without an override"
        )
    project_files = {
        str(path.relative_to(REPO_ROOT)): _sha256(path)
        for path in project_paths
    }
    inventory_signature = _json_sha256(inventory)
    return {
        "schema_version": 1,
        "scenario": SCENE_ID,
        "family": FAMILY,
        "verdict": VERDICT,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": f"{task_name}.bddl",
        "benchmark_prompt": TASK_PROMPT,
        "benchmark_task_map": str(task_map_path),
        "benchmark_task_map_sha256": _sha256(task_map_path),
        "selected_policy_prompt_source": (
            "native benchmark/libero_suite_task_map.py filename-derived language"
        ),
        "evaluated_policy_prompt_matches_selected_native_task": True,
        "evaluated_policy_prompt_path_verified": policy_prompt_path_verified,
        "bddl_prompt": bddl_prompt,
        "native_bddl_embedded_prompt_matches_benchmark_prompt": (
            bddl_prompt == TASK_PROMPT
        ),
        "native_bddl": str(native_bddl),
        "evaluated_bddl": str(native_bddl),
        "native_bddl_sha256": _sha256(native_bddl),
        "evaluated_bddl_sha256": _sha256(native_bddl),
        "goal_source": goal_source,
        "goal_predicates": goals,
        "goal_signature_sha256": hashlib.sha256(
            goal_source.encode("utf-8")
        ).hexdigest(),
        "fixtures": fixtures,
        "objects": objects,
        "inventory_signature": inventory_signature,
        "condition_inventory_signatures": {
            condition: inventory_signature
            for condition in ("eb", "er", "ec")
        },
        "cross_condition_inventory_signatures_identical": True,
        "source_to_project_inventory_delta": {
            "fixtures": [],
            "objects": [],
        },
        "source_to_project_bddl_delta": (
            "none; evaluated BDDL is the selected native source"
        ),
        "source_to_project_layout_delta": {
            "eb": "native task-4 source state followed by common settling",
            "er_ec": (
                "only native wine-bottle free-joint x/y pose and zeroed "
                "free-joint velocity"
            ),
        },
        "intervention_id": "l1b3_task4_native_wine_pose_outcome_v2",
        "intervention_allowlist": INTERVENTION_ALLOWLIST,
        "custom_assets": [],
        "custom_bddl": False,
        "libero_commit": commit,
        "all_selected_native_files_unmodified": True,
        "native_asset_files": asset_files,
        "native_asset_manifest_sha256": _json_sha256(asset_files),
        "project_file_hashes": project_files,
    }


def write_preflight(manifest: Path, report: Path) -> dict[str, object]:
    record = build_record()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# L1-B3 Task-4 Outcome V2 native preflight",
                "",
                f"Verdict: **{VERDICT}**",
                "",
                f"- Native task: `{TASK_SUITE}/{record['task_file']}` "
                f"(id `{TASK_ID}`)",
                f"- Benchmark prompt: `{TASK_PROMPT}`",
                f"- Evaluated policy prompt path matches the native benchmark: "
                f"`{record['evaluated_policy_prompt_path_verified']}`",
                f"- Unmodified native BDDL embedded `:language`: "
                f"`{record['bddl_prompt']}`",
                f"- Embedded BDDL language matches benchmark language: "
                f"`{record['native_bddl_embedded_prompt_matches_benchmark_prompt']}`",
                f"- Goal predicates: `{record['goal_predicates']}`",
                f"- Native/evaluated BDDL SHA-256: "
                f"`{record['native_bddl_sha256']}`",
                f"- Inventory signature: `{record['inventory_signature']}`",
                f"- Native asset files hashed: "
                f"`{len(record['native_asset_files'])}`",
                f"- Native asset closure: "
                f"`{record['native_asset_manifest_sha256']}`",
                f"- LIBERO commit: `{record['libero_commit']}`",
                "- Source-to-project inventory/BDDL delta: `none`",
                "- Cross-condition allowlist: native wine-bottle x/y pose and "
                "free-joint velocity only.",
                "- Custom assets, BDDL, prompt, or goal changes: `none`",
                "- Note: the official task-4 BDDL contains upstream stale drawer "
                "wording in its embedded `:language`; evaluation uses the "
                "unchanged native benchmark task prompt above, and the native "
                "goal predicate targets the cabinet.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Verdict: {VERDICT}")
    print(f"Manifest: {manifest}")
    print(f"Report: {report}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_native_preflight.json"
        ),
    )
    parser.add_argument(
        "--report",
        default="experiments/logs/l1b3_task4_outcome_v2_native_preflight.md",
    )
    args = parser.parse_args()
    write_preflight(Path(args.manifest), Path(args.report))


if __name__ == "__main__":
    main()
