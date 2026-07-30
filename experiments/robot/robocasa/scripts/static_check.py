#!/usr/bin/env python3
"""Static policy checker for RoboCasa PhysCog scenes.

Default mode is pure source analysis (AST), so it runs without robocasa or
mujoco installed. ``--live`` additionally constructs each scene in all three
conditions and compares prompts and asset inventories.

    python experiments/robot/robocasa/scripts/static_check.py
    python experiments/robot/robocasa/scripts/static_check.py --live --scene L1-A1
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[4]
ROBOCASA = ROOT / "experiments" / "robot" / "robocasa"
ENVS = ROOT / "experiments" / "robot" / "robocasa" / "envs"
TASKS = ROOT / "experiments" / "robot" / "robocasa" / "tasks"

REQUIRED_ATTRS = (
    "physcog_scene_id",
    "physcog_factor",
    "physcog_variable",
    "physcog_intervention",
    "physcog_hazard_objs",
    "physcog_detour_metric",
    "physcog_detour_threshold",
)

CONDITIONS = ("Eb", "Er", "Ec")
ASSET_SUFFIXES = {
    ".dae",
    ".jpeg",
    ".jpg",
    ".mesh",
    ".mjcf",
    ".mtl",
    ".obj",
    ".png",
    ".stl",
    ".svg",
    ".tif",
    ".tiff",
    ".xml",
}
CUSTOM_ASSET_SYMBOLS = {
    "MujocoObject",
    "MujocoXMLObject",
    "MJCFObject",
    "register_object",
    "register_fixture",
    "register_object_class",
}


class Failure(Exception):
    pass


def sublevel_of(scene_id):
    """``"L1-B5" -> "L1-B"``: the sub-level is the id minus its scene number."""
    return (scene_id or "").rstrip("0123456789")


def _class_attrs(node: ast.ClassDef):
    out = {}
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            out[stmt.target.id] = stmt.value
    return out


def _methods(node: ast.ClassDef):
    return {
        s.name: s
        for s in node.body
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _native_imports(tree):
    native = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.ImportFrom):
            continue
        module = stmt.module or ""
        if not module.startswith("robocasa.environments.kitchen."):
            continue
        for alias in stmt.names:
            native[alias.asname or alias.name] = f"{module}.{alias.name}"
    return native


def _local_classes(tree):
    return {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }


def _resolve_native_bases(node, classes, native_imports, seen=None):
    seen = set() if seen is None else seen
    if node.name in seen:
        return set()
    seen.add(node.name)
    out = set()
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else None
        if name in native_imports:
            out.add(native_imports[name])
        elif name in classes:
            out.update(
                _resolve_native_bases(classes[name], classes, native_imports, seen)
            )
    return out


def _local_lineage(node, classes, seen=None):
    seen = set() if seen is None else seen
    if node.name in seen:
        return []
    seen.add(node.name)
    out = [node]
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else None
        if name in classes:
            out.extend(_local_lineage(classes[name], classes, seen))
    return out


def _lineage_methods(lineage):
    methods = {}
    # Base first, derived last, matching normal method override semantics.
    for node in reversed(lineage):
        methods.update(_methods(node))
    return methods


def check_project_local_assets():
    problems = []
    for path in ROBOCASA.rglob("*"):
        if path.is_file() and path.suffix.lower() in ASSET_SUFFIXES:
            problems.append(
                f"{path.relative_to(ROOT)}: project-local asset files are forbidden"
            )
    return problems


def _literal_strings(node):
    return {
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }


def check_source(path: pathlib.Path):
    """Return (scene_ids, problems) for one sub-level module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    problems, scene_ids = [], []
    native_imports = _native_imports(tree)
    classes = _local_classes(tree)

    for sub in ast.walk(tree):
        if isinstance(sub, ast.Call) and _call_name(sub.func) in CUSTOM_ASSET_SYMBOLS:
            problems.append(
                f"{path.name}:{getattr(sub, 'lineno', '?')}: constructs or "
                f"registers a custom asset via {_call_name(sub.func)}"
            )
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            value = sub.value.strip()
            suffix = pathlib.PurePath(value).suffix.lower()
            if suffix in ASSET_SUFFIXES and (
                "experiments/robot/robocasa" in value
                or (not pathlib.PurePath(value).is_absolute() and (path.parent / value).exists())
            ):
                problems.append(
                    f"{path.name}:{getattr(sub, 'lineno', '?')}: references "
                    f"project-local asset {value!r}"
                )
    for helper in classes.values():
        helper_methods = _methods(helper)
        if "get_ep_meta" in helper_methods:
            problems.append(
                f"{path.name}:{helper.name}: overrides get_ep_meta; native "
                "prompt interception is forbidden in scene modules"
            )
        if "_check_success" in helper_methods:
            problems.append(
                f"{path.name}:{helper.name}: overrides native _check_success"
            )

    scene_list = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == "SCENES":
                    scene_list = stmt.value
    if scene_list is None:
        problems.append(f"{path.name}: module defines no SCENES tuple")

    listed = set()
    if isinstance(scene_list, (ast.Tuple, ast.List)):
        listed = {e.id for e in scene_list.elts if isinstance(e, ast.Name)}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs = _class_attrs(node)
        if "physcog_scene_id" not in attrs:
            continue  # helper/base class, not a scene

        sid = getattr(attrs["physcog_scene_id"], "value", None)
        scene_ids.append(sid)
        tag = f"{path.name}:{node.name}"

        if node.name not in listed:
            problems.append(f"{tag}: not listed in SCENES")

        native_bases = _resolve_native_bases(node, classes, native_imports)
        if len(native_bases) != 1:
            problems.append(
                f"{tag}: must resolve to exactly one native RoboCasa task class; "
                f"got {sorted(native_bases)}"
            )

        for attr in REQUIRED_ATTRS:
            if attr not in attrs:
                problems.append(f"{tag}: missing required attribute {attr}")

        direct_meths = _methods(node)
        lineage = _local_lineage(node, classes)
        meths = _lineage_methods(lineage)

        # Prompt and success guards: scene code may never interpose on the
        # native task's exact language or success predicate.
        for meth in direct_meths.values():
            for sub in ast.walk(meth):
                if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = (
                        sub.targets
                        if isinstance(sub, ast.Assign)
                        else [sub.target]
                    )
                    for t in targets:
                        if (
                            isinstance(t, ast.Subscript)
                            and isinstance(t.slice, ast.Constant)
                            and t.slice.value == "lang"
                        ):
                            problems.append(
                                f"{tag}.{meth.name}: assigns ep_meta['lang']; "
                                "the native prompt may not be rewritten"
                            )

        # intervention must be declared and consistent with what is used
        interv = attrs.get("physcog_intervention")
        interv_name = ""
        if isinstance(interv, ast.Attribute):
            interv_name = interv.attr
        src = " ".join(ast.dump(base) for base in lineage)
        if "override_category" in src:
            problems.append(
                f"{tag}: condition-dependent category override is forbidden; "
                "pin categories identically across Eb/Er/Ec"
            )
        if interv_name not in ("POSE", "FIXTURE_STATE", "DYNAMIC"):
            problems.append(
                f"{tag}: physcog_intervention must be one of POSE, FIXTURE_STATE, "
                f"DYNAMIC; CATEGORY is forbidden; got {interv_name or '?'}"
            )
        if "_physcog_step_intervention" in direct_meths and interv_name != "DYNAMIC":
            problems.append(
                f"{tag}: schedules a step intervention but declares "
                f"Intervention.{interv_name or '?'}"
            )
        if "_physcog_setup_scene" in direct_meths and interv_name not in (
            "FIXTURE_STATE",
            "DYNAMIC",
        ):
            problems.append(
                f"{tag}: changes fixture state but declares "
                f"Intervention.{interv_name or '?'}"
            )
        if interv_name == "FIXTURE_STATE" and "_physcog_setup_scene" not in meths:
            problems.append(
                f"{tag}: declares FIXTURE_STATE but has no _physcog_setup_scene"
            )
        if interv_name == "DYNAMIC" and "_physcog_step_intervention" not in meths:
            problems.append(
                f"{tag}: declares DYNAMIC but has no _physcog_step_intervention"
            )
        # every condition must be covered by the override table
        ov = meths.get("_physcog_obj_overrides")
        if ov is not None:
            keys = {
                k.value
                for sub in ast.walk(ov)
                if isinstance(sub, ast.Dict)
                for k in sub.keys
                if isinstance(k, ast.Constant)
            }
            missing = [c for c in CONDITIONS if c not in keys]
            if missing:
                problems.append(
                    f"{tag}._physcog_obj_overrides: no entry for {missing}"
                )
        elif "_physcog_apply_cfgs" not in meths:
            problems.append(
                f"{tag}: defines neither _physcog_obj_overrides nor "
                "_physcog_apply_cfgs; the three conditions would be identical"
            )

        # the safety oracle must exist -- without it Er can never violate
        if "_physcog_check_safety" not in meths:
            problems.append(f"{tag}: no _physcog_check_safety oracle")

        # Every concrete scene must pin its native object sampling categories;
        # `"all"` makes cross-condition asset identity unverifiable.
        pin = meths.get("_physcog_pin_categories")
        if pin is None:
            problems.append(
                f"{tag}: no _physcog_pin_categories; native asset inventory "
                "cannot be fixed across conditions"
            )
        elif "all" in _literal_strings(pin):
            problems.append(
                f"{tag}._physcog_pin_categories: obj_groups='all' is forbidden"
            )

        # detour threshold must be a real number, not the 0.0 default
        thr = attrs.get("physcog_detour_threshold")
        if isinstance(thr, ast.Constant) and not thr.value:
            problems.append(
                f"{tag}: physcog_detour_threshold is 0; G3 (detour is real) "
                "cannot be evaluated"
            )

    return scene_ids, problems


def check_live(scene_id=None, manifest_path=None):
    sys.path.insert(0, str(ROOT))
    from experiments.robot.robocasa.physcog.registry import all_scenes
    from experiments.robot.robocasa.physcog.preflight import (
        make_condition_record,
        validate_condition_records,
    )

    problems = []
    manifests = []
    scenes = all_scenes()
    targets = [scene_id] if scene_id else sorted(scenes)
    for sid in targets:
        if sid not in scenes:
            problems.append(f"unknown scene {sid!r}; known: {sorted(scenes)}")
            continue
        cls = scenes[sid]
        records = []
        try:
            for cond in CONDITIONS:
                env = cls(
                    condition=cond,
                    robots="PandaOmron",
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    seed=0,
                    robot_spawn_deviation_pos_x=0.0,
                    robot_spawn_deviation_pos_y=0.0,
                    robot_spawn_deviation_rot=0.0,
                )
                try:
                    env.reset()
                    records.append(make_condition_record(env))
                finally:
                    env.close()
            manifest = validate_condition_records(
                records, hazard_objs=cls.physcog_hazard_objs
            )
            manifests.append(manifest)
            print(
                f"{sid}: native_task={manifest['native_task']['class']} "
                f"lang={manifest['native_prompt']!r} "
                f"preflight={manifest['preflight_sha256']}"
            )
        except Exception as exc:
            if manifest_path:
                destination = pathlib.Path(manifest_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        {
                            "valid": False,
                            "scene_id": sid,
                            "error": str(exc),
                            "conditions_constructed": records,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            problems.append(
                f"{sid}: live native-only preflight failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
    if manifest_path and not problems:
        destination = pathlib.Path(manifest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"valid": True, "scenes": manifests}, indent=2, sort_keys=True)
            + "\n"
        )
    return problems


def check_repository():
    """Run all simulator-free source checks and return ``(ids, problems)``."""

    problems, all_ids = check_project_local_assets(), []
    for path in sorted(ENVS.glob("l[123]_[abc].py")):
        ids, source_problems = check_source(path)
        all_ids += ids
        problems += source_problems

    for sid in all_ids:
        sub = sublevel_of(sid)
        spec = TASKS / f"{sub}_SPEC.md"
        if not spec.exists():
            problems.append(f"{sid}: missing spec {spec.relative_to(ROOT)}")

    dupes = {scene_id for scene_id in all_ids if all_ids.count(scene_id) > 1}
    if dupes:
        problems.append(f"duplicate scene ids: {sorted(dupes)}")
    return all_ids, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also construct the scenes")
    ap.add_argument("--scene", default=None)
    ap.add_argument(
        "--manifest",
        default=None,
        help="write the live native task/prompt/asset manifest after PASS",
    )
    args = ap.parse_args()

    all_ids, problems = check_repository()

    if args.live:
        problems += check_live(args.scene, args.manifest)
    elif args.manifest:
        problems.append("--manifest requires --live")

    n_sub = len({sublevel_of(i) for i in all_ids if i})
    print(f"checked {len(all_ids)} scenes across {n_sub} sub-levels")
    for p in problems:
        print(f"  FAIL {p}")
    if problems:
        sys.exit(1)
    print("  OK")


if __name__ == "__main__":
    main()
