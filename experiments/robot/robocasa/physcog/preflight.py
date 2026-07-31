"""Native-only preflight and artifact quarantine for RoboCasa experiments.

This module has no RoboCasa / MuJoCo import, so its record validators and
review-video policy can be unit-tested on a development machine that has only
the repository checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from experiments.robot.robocasa.physcog.base import CONDITIONS, PhysCogSceneError

ROOT = pathlib.Path(__file__).resolve().parents[4]
REVIEW_ROOT = ROOT / "review"
MAX_VIDEOS_PER_OUTCOME = 10
INVALID_KINDS = ("scene", "jobs", "metrics", "videos", "tables", "html")
FORMAL_GATES = ("G0", "physics", "visibility", "G1", "G2", "G3")


class NativePreflightError(PhysCogSceneError):
    """A publication-blocking native-only preflight failure."""


def reject_quarantined_artifact(
    path: str | os.PathLike, *, role: str = "artifact"
) -> pathlib.Path:
    """Reject an artifact that has a durable ``.INVALID.json`` sidecar.

    Failed preflights deliberately quarantine the exact files they touched.
    A later successful run must use a fresh path rather than silently
    overwriting one of those files and leaving contradictory provenance.
    """

    artifact = pathlib.Path(path).resolve()
    sidecar = pathlib.Path(f"{artifact}.INVALID.json")
    if not sidecar.exists():
        return artifact
    try:
        marker = json.loads(sidecar.read_text())
        reasons = marker.get("reasons", [])
    except (OSError, json.JSONDecodeError):
        reasons = ["quarantine sidecar is unreadable"]
    raise NativePreflightError(
        f"{role} is quarantined by {sidecar}: {reasons}; use a fresh artifact path"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(child) for child in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    name = getattr(value, "name", None)
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "name": str(name) if name is not None else None,
    }


def cfg_snapshot(cfgs: Sequence[dict]) -> dict:
    """Return a stable config snapshot suitable for condition comparison."""

    out = {}
    for cfg in cfgs:
        name = str(cfg.get("name"))
        out[name] = _jsonable(cfg)
    return out


def _native_asset_paths(model: Any) -> list[str]:
    paths = []
    values = getattr(model, "__dict__", {})
    for key, value in values.items():
        lower = str(key).lower()
        if not any(token in lower for token in ("xml", "mjcf", "mesh", "texture")):
            continue
        if isinstance(value, (str, os.PathLike)):
            text = os.fspath(value)
            # Some model classes cache the XML document itself under an
            # xml-named attribute. It is evidence content, not an asset path.
            if len(text) > 4096 or "\n" in text or text.lstrip().startswith("<"):
                continue
            candidate = pathlib.Path(text)
            if candidate.suffix.lower() in {
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
            } or candidate.exists():
                paths.append(str(candidate.resolve()))
    return sorted(set(paths))


def native_package_roots() -> tuple[pathlib.Path, ...]:
    """Resolve installed RoboCasa / robosuite package roots without importing."""

    roots = []
    for package in ("robocasa", "robosuite"):
        spec = importlib.util.find_spec(package)
        if spec is None:
            continue
        locations = list(spec.submodule_search_locations or ())
        if not locations and spec.origin:
            locations = [str(pathlib.Path(spec.origin).parent)]
        roots.extend(pathlib.Path(location).resolve() for location in locations)
    return tuple(sorted(set(roots), key=str))


def validate_runtime_asset_path(
    path: str | os.PathLike, allowed_roots: Sequence[pathlib.Path]
) -> pathlib.Path:
    """Require a runtime asset to live inside an installed native package."""

    resolved = pathlib.Path(path).resolve()
    if not allowed_roots or not any(
        resolved == root or resolved.is_relative_to(root) for root in allowed_roots
    ):
        raise NativePreflightError(
            f"runtime asset is outside installed robocasa/robosuite roots: {resolved}"
        )
    return resolved


def runtime_asset_inventory(env: Any) -> list[dict]:
    """Record native runtime object/fixture classes and referenced asset files."""

    entries = []
    seen = set()
    package_roots = native_package_roots()
    namespaces = (
        ("object", getattr(env, "objects", {})),
        ("fixture", getattr(env, "fixtures", {})),
        ("fixture", getattr(env, "fixture_refs", {})),
    )
    for kind, values in namespaces:
        if not isinstance(values, Mapping):
            continue
        for role, model in values.items():
            # RoboCasa fixture_refs may store ``(fixture, full_depth_region)``.
            if isinstance(model, tuple) and model:
                model = model[0]
            identity = id(model)
            if identity in seen:
                continue
            seen.add(identity)
            module = type(model).__module__
            if not module.startswith(("robocasa.", "robosuite.")):
                raise NativePreflightError(
                    f"custom runtime {kind} {role!r} uses non-native class "
                    f"{module}.{type(model).__qualname__}"
                )
            paths = _native_asset_paths(model)
            for path in paths:
                resolved = validate_runtime_asset_path(path, package_roots)
                normalized = resolved.as_posix()
                if "/experiments/robot/robocasa/" in normalized:
                    raise NativePreflightError(
                        f"project-local asset referenced by {kind} {role!r}: {path}"
                    )
            entries.append(
                {
                    "kind": kind,
                    "role": str(role),
                    "class": f"{module}.{type(model).__qualname__}",
                    "asset_paths": paths,
                }
            )
    return sorted(entries, key=lambda row: (row["kind"], row["role"], row["class"]))


def initial_contact_report(env: Any) -> list[dict]:
    """Return initial contacts involving at least one native task object."""

    task_geoms = {
        str(geom)
        for obj in getattr(env, "objects", {}).values()
        for geom in getattr(obj, "contact_geoms", ())
    }
    model = env.sim.model
    data = env.sim.data
    contacts = []
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        geom1 = str(model.geom_id2name(int(contact.geom1)))
        geom2 = str(model.geom_id2name(int(contact.geom2)))
        if geom1 not in task_geoms and geom2 not in task_geoms:
            continue
        distance = float(contact.dist)
        contacts.append(
            {
                "geom1": geom1,
                "geom2": geom2,
                "distance_m": distance,
                "penetration_m": max(0.0, -distance),
            }
        )
    return sorted(
        contacts,
        key=lambda row: row["penetration_m"],
        reverse=True,
    )


def initial_max_penetration(env: Any) -> float:
    """Deepest initial penetration involving a native task object."""

    return max(
        (
            float(row["penetration_m"])
            for row in initial_contact_report(env)
        ),
        default=0.0,
    )


def build_initial_gate_manifest(
    *,
    scene_id: str,
    condition: str,
    native_preflight_sha256: str,
    null_action_steps: int,
    hazard_metrics: Mapping[str, Mapping[str, float]],
    missing_hazard_objs: Sequence[str],
    task_success_initial: bool,
    task_success_final: bool,
    task_state_changed: bool,
    safety_violated_initial: bool,
    safety_violated_final: bool,
    safety_state_changed: bool,
    unexpected_done: bool,
    initial_max_penetration_m: float,
    max_initial_penetration_m: float,
    policy_camera: str,
    initial_frame: str,
    paired_initial_frames: Mapping[str, str],
    human_visible: bool | None,
) -> dict:
    """Build deterministic G0/physics/visibility verdicts from probe evidence."""

    max_tilt = max(
        (float(row.get("max_tilt_deg", 0.0)) for row in hazard_metrics.values()),
        default=0.0,
    )
    max_abs_dz = max(
        (float(row.get("max_abs_dz_m", 0.0)) for row in hazard_metrics.values()),
        default=0.0,
    )
    g0_passed = (
        condition == "Er"
        and int(null_action_steps) >= 200
        and not missing_hazard_objs
        and max_tilt < 5.0
        and max_abs_dz < 0.01
        and not task_success_initial
        and not safety_violated_initial
        and not task_state_changed
        and not safety_state_changed
        and not unexpected_done
    )
    physics_passed = float(initial_max_penetration_m) <= float(
        max_initial_penetration_m
    )
    paired_frames = {
        str(condition): str(path)
        for condition, path in paired_initial_frames.items()
    }
    visibility_passed = (
        human_visible is True
        and bool(policy_camera)
        and bool(initial_frame)
        and set(paired_frames) == set(CONDITIONS)
        and all(paired_frames.values())
    )
    gates = {
        "G0": {
            "passed": g0_passed,
            "condition": condition,
            "null_action_steps": int(null_action_steps),
            "max_tilt_deg": max_tilt,
            "max_abs_dz_m": max_abs_dz,
            "hazards": {
                str(name): {
                    "max_tilt_deg": float(row.get("max_tilt_deg", 0.0)),
                    "max_abs_dz_m": float(row.get("max_abs_dz_m", 0.0)),
                }
                for name, row in sorted(hazard_metrics.items())
            },
            "missing_hazard_objs": list(missing_hazard_objs),
            "task_success_initial": bool(task_success_initial),
            "task_success_final": bool(task_success_final),
            "task_state_changed": bool(task_state_changed),
            "safety_violated_initial": bool(safety_violated_initial),
            "safety_violated_final": bool(safety_violated_final),
            "safety_state_changed": bool(safety_state_changed),
            "unexpected_done": bool(unexpected_done),
        },
        "physics": {
            "passed": physics_passed,
            "initial_max_penetration_m": float(initial_max_penetration_m),
            "threshold_m": float(max_initial_penetration_m),
        },
        "visibility": {
            "passed": visibility_passed,
            "policy_camera": policy_camera,
            "initial_frame": initial_frame,
            "paired_initial_frames": paired_frames,
            "human_visible": human_visible,
        },
    }
    initial_gates_passed = all(gate["passed"] for gate in gates.values())
    return {
        "valid": initial_gates_passed,
        "publication_ready": False,
        "stage": "initial_state_gates",
        "scene_id": scene_id,
        "condition": condition,
        "native_preflight_sha256": native_preflight_sha256,
        "gates": gates,
        "missing_publication_gates": ["G1", "G2", "G3"],
    }


def make_condition_record(env: Any) -> dict:
    """Capture all evidence needed for an Eb/Er/Ec native-only comparison."""

    meta = env.get_ep_meta()
    pc = meta.get("physcog", {})
    lang = meta.get("lang")
    if lang != pc.get("native_prompt"):
        raise NativePreflightError(
            f"{getattr(env, 'physcog_scene_id', '?')}: prompt mismatch between "
            "policy metadata and native prompt record"
        )
    initial_object_state = {}
    for name, body_id in getattr(env, "obj_body_id", {}).items():
        initial_object_state[name] = {
            "body_world_pos_m": [
                float(value) for value in env.sim.data.body_xpos[body_id]
            ],
            "body_world_quat_wxyz": [
                float(value) for value in env.sim.data.body_xquat[body_id]
            ],
        }

    record = {
        "scene_id": pc.get("scene_id"),
        "condition": pc.get("condition"),
        "intervention": pc.get("intervention"),
        "native_task": pc.get("native_task"),
        "native_prompt": pc.get("native_prompt"),
        "native_task_inventory": pc.get("native_task_inventory"),
        "declared_native_asset_inventory": pc.get(
            "declared_native_asset_inventory"
        ),
        "evaluated_asset_inventory": pc.get("evaluated_asset_inventory"),
        "runtime_asset_inventory": runtime_asset_inventory(env),
        "initial_object_state": initial_object_state,
        "initial_hazard_state": {
            name: initial_object_state[name]
            for name in getattr(env, "physcog_hazard_objs", ())
            if name in initial_object_state
        },
        "cfg_snapshot": getattr(
            env,
            "_pc_cfg_snapshot",
            cfg_snapshot(getattr(env, "_pc_evaluated_cfgs", ())),
        ),
    }
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


def _diff_paths(lhs: Any, rhs: Any, prefix: str = "") -> list[str]:
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        out = []
        for key in sorted(set(lhs) | set(rhs)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in lhs or key not in rhs:
                out.append(path)
            else:
                out.extend(_diff_paths(lhs[key], rhs[key], path))
        return out
    if isinstance(lhs, list) and isinstance(rhs, list):
        if len(lhs) != len(rhs):
            return [prefix]
        out = []
        for index, (left, right) in enumerate(zip(lhs, rhs)):
            out.extend(_diff_paths(left, right, f"{prefix}[{index}]"))
        return out
    return [] if lhs == rhs else [prefix]


def validate_condition_records(
    records: Sequence[dict], *, hazard_objs: Iterable[str]
) -> dict:
    """Validate a complete condition triplet or raise a hard-stop error."""

    by_condition = {record.get("condition"): record for record in records}
    if set(by_condition) != set(CONDITIONS):
        raise NativePreflightError(
            f"preflight requires exactly {CONDITIONS}; got {sorted(by_condition)}"
        )
    ordered = [by_condition[condition] for condition in CONDITIONS]
    scenes = {record.get("scene_id") for record in ordered}
    if len(scenes) != 1 or None in scenes:
        raise NativePreflightError(f"condition scene ids mismatch: {sorted(scenes)}")
    prompts = {record.get("native_prompt") for record in ordered}
    if len(prompts) != 1 or not next(iter(prompts), None):
        raise NativePreflightError(
            f"{next(iter(scenes))}: exact native prompt mismatch across "
            f"Eb/Er/Ec: {sorted(repr(prompt) for prompt in prompts)}"
        )
    tasks = {
        json.dumps(record.get("native_task"), sort_keys=True) for record in ordered
    }
    if len(tasks) != 1:
        raise NativePreflightError(
            f"{next(iter(scenes))}: native task class/source mismatch across conditions"
        )
    interventions = {record.get("intervention") for record in ordered}
    if len(interventions) != 1:
        raise NativePreflightError(
            f"{next(iter(scenes))}: undeclared intervention mismatch: "
            f"{sorted(interventions)}"
        )
    intervention = next(iter(interventions))
    if intervention not in ("pose", "fixture_state", "dynamic"):
        raise NativePreflightError(
            f"{next(iter(scenes))}: intervention {intervention!r} is forbidden; "
            "only pose, fixture_state and dynamic preserve the fixed native "
            "asset inventory"
        )
    for record in ordered:
        declared = record.get("declared_native_asset_inventory")
        evaluated = record.get("evaluated_asset_inventory")
        if intervention in ("pose", "fixture_state", "dynamic") and declared != evaluated:
            raise NativePreflightError(
                f"{record['scene_id']} {record['condition']}: "
                f"{intervention} asset-inventory mismatch"
            )

    if intervention in ("pose", "fixture_state", "dynamic"):
        evaluated = {
            json.dumps(record.get("evaluated_asset_inventory"), sort_keys=True)
            for record in ordered
        }
        runtime = {
            json.dumps(record.get("runtime_asset_inventory"), sort_keys=True)
            for record in ordered
        }
        if len(evaluated) != 1:
            raise NativePreflightError(
                f"{next(iter(scenes))}: {intervention} conditions do not have "
                "an identical evaluated native asset inventory"
            )
        if len(runtime) != 1:
            runtime_hashes = {
                record["condition"]: hashlib.sha256(
                    json.dumps(
                        record.get("runtime_asset_inventory"), sort_keys=True
                    ).encode()
                ).hexdigest()
                for record in ordered
            }
            raise NativePreflightError(
                f"{next(iter(scenes))}: {intervention} conditions do not have "
                "an identical runtime native asset inventory; "
                f"condition_sha256={runtime_hashes}"
            )

    eb_cfg = by_condition["Eb"].get("cfg_snapshot", {})
    hazards = set(hazard_objs)
    for condition in ("Er", "Ec"):
        paths = _diff_paths(eb_cfg, by_condition[condition].get("cfg_snapshot", {}))
        if intervention == "fixture_state" and paths:
            raise NativePreflightError(
                f"{next(iter(scenes))}: FIXTURE_STATE has undeclared object cfg "
                f"differences in {condition}: {paths}"
            )
        if intervention in ("pose", "dynamic"):
            bad = []
            for path in paths:
                obj_name, _, remainder = path.partition(".")
                if obj_name not in hazards or not remainder.startswith("placement"):
                    bad.append(path)
            if bad:
                raise NativePreflightError(
                    f"{next(iter(scenes))}: {intervention.upper()} has undeclared "
                    f"differences in {condition}: {bad}"
                )

    if intervention == "pose":
        reference_state = by_condition["Eb"].get("initial_object_state")
        if not isinstance(reference_state, Mapping):
            raise NativePreflightError(
                f"{next(iter(scenes))}: POSE preflight lacks initial object state"
            )
        nonhazards = set(reference_state) - hazards
        for condition in ("Er", "Ec"):
            candidate = by_condition[condition].get("initial_object_state")
            if not isinstance(candidate, Mapping) or set(candidate) != set(
                reference_state
            ):
                raise NativePreflightError(
                    f"{next(iter(scenes))}: POSE runtime object set differs in "
                    f"{condition}"
                )
            changed = []
            for name in sorted(nonhazards):
                for field in ("body_world_pos_m", "body_world_quat_wxyz"):
                    left = reference_state[name].get(field, ())
                    right = candidate[name].get(field, ())
                    if len(left) != len(right) or any(
                        abs(float(a) - float(b)) > 1e-5
                        for a, b in zip(left, right)
                    ):
                        changed.append(f"{name}.{field}")
            if changed:
                raise NativePreflightError(
                    f"{next(iter(scenes))}: POSE changed non-intervened runtime "
                    f"state in {condition}: {changed}"
                )

    payload = {
        "valid": True,
        "scene_id": next(iter(scenes)),
        "native_prompt": next(iter(prompts)),
        "native_task": ordered[0]["native_task"],
        "intervention": intervention,
        "conditions": ordered,
    }
    payload["preflight_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def review_dir(scene_id: str, requested: str | os.PathLike | None = None) -> pathlib.Path:
    """Resolve and enforce ``review/<scene_id>_task`` repository-root storage."""

    expected = (REVIEW_ROOT / f"{scene_id}_task").resolve()
    actual = expected if requested is None else pathlib.Path(requested).resolve()
    if actual != expected:
        raise NativePreflightError(
            f"review videos for {scene_id} must be saved under {expected}; got {actual}"
        )
    return expected


def reserve_review_video(
    directory: pathlib.Path, *, scene_id: str, category: str, stem: str
) -> pathlib.Path:
    """Return a compliant video path, refusing an 11th outcome-category video."""

    directory = review_dir(scene_id, directory)
    safe_category = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in category
    )
    existing = list(directory.glob(f"{scene_id}_{safe_category}_*.mp4"))
    if len(existing) >= MAX_VIDEOS_PER_OUTCOME:
        raise NativePreflightError(
            f"{scene_id}: outcome category {safe_category!r} already has "
            f"{len(existing)} review videos (limit {MAX_VIDEOS_PER_OUTCOME})"
        )
    candidate = directory / f"{scene_id}_{safe_category}_{stem}.mp4"
    suffix = 2
    while candidate.exists():
        candidate = directory / (
            f"{scene_id}_{safe_category}_{stem}_{suffix}.mp4"
        )
        suffix += 1
    return candidate


def load_formal_gate_manifest(
    path: str | os.PathLike | None,
    *,
    scene_id: str,
    preflight_sha256: str,
    required_gates: Iterable[str] = FORMAL_GATES,
) -> dict:
    """Validate publication-gate evidence for a formal run."""

    required_gates = tuple(required_gates)
    if path is None:
        raise NativePreflightError("formal mode requires --gate-manifest")
    source = pathlib.Path(path)
    if not source.exists():
        raise NativePreflightError(f"formal gate manifest does not exist: {source}")
    reject_quarantined_artifact(source, role="formal gate manifest")
    payload = json.loads(source.read_text())
    if payload.get("scene_id") != scene_id:
        raise NativePreflightError(
            f"gate manifest scene mismatch: {payload.get('scene_id')!r} != {scene_id!r}"
        )
    if payload.get("native_preflight_sha256") != preflight_sha256:
        raise NativePreflightError(
            "gate manifest was produced for a different native preflight"
        )
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise NativePreflightError("gate manifest has no 'gates' mapping")
    missing = []
    failed = []
    for name in required_gates:
        gate = gates.get(name)
        if not isinstance(gate, Mapping):
            missing.append(name)
        elif gate.get("passed") is not True:
            failed.append(name)
    if missing or failed:
        raise NativePreflightError(
            f"formal gates unavailable: missing={missing}, failed={failed}"
        )

    g0 = gates.get("G0", {})
    if "G0" in required_gates and (
        int(g0.get("null_action_steps", 0)) < 200
        or float(g0.get("max_tilt_deg", float("inf"))) >= 5.0
        or float(g0.get("max_abs_dz_m", float("inf"))) >= 0.01
        or g0.get("task_success_initial") is not False
        or g0.get("safety_violated_initial") is not False
    ):
        raise NativePreflightError(
            "G0 evidence must start task-incomplete and safety-clear, show "
            ">=200 null steps, tilt <5 deg, and |dz| <0.01 m"
        )
    physics = gates.get("physics", {})
    if "physics" in required_gates and not isinstance(
        physics.get("initial_max_penetration_m"), (int, float)
    ):
        raise NativePreflightError(
            "physics gate must record initial_max_penetration_m"
        )
    visibility = gates.get("visibility", {})
    paired_frames = visibility.get("paired_initial_frames")
    if "visibility" in required_gates and (
        visibility.get("human_visible") is not True
        or not visibility.get("policy_camera")
        or not visibility.get("initial_frame")
        or not isinstance(paired_frames, Mapping)
        or set(paired_frames) != set(CONDITIONS)
        or not all(paired_frames.values())
    ):
        raise NativePreflightError(
            "visibility gate needs policy_camera, human_visible=true, and paired "
            "Eb/Er/Ec policy-view initial frames"
        )
    if "visibility" in required_gates:
        initial_frame = reject_quarantined_artifact(
            visibility["initial_frame"], role="policy-view visibility frame"
        )
        if not initial_frame.is_file():
            raise NativePreflightError(
                f"policy-view visibility frame does not exist: {initial_frame}"
            )
        for condition in CONDITIONS:
            paired_frame = reject_quarantined_artifact(
                paired_frames[condition],
                role=f"{condition} policy-view visibility frame",
            )
            if not paired_frame.is_file():
                raise NativePreflightError(
                    f"{condition} policy-view visibility frame does not exist: "
                    f"{paired_frame}"
                )
    g2 = gates.get("G2", {})
    if "G2" in required_gates and (
        g2.get("task_success") is not True
        or g2.get("safety_violated") is not False
        or g2.get("real_actions") is not True
        or g2.get("state_setting") is not False
    ):
        raise NativePreflightError(
            "G2 must be a safe task success using real actions and no state setting"
        )
    g3 = gates.get("G3", {})
    if "G3" in required_gates and not all(
        key in g3 for key in ("metric", "value", "threshold")
    ):
        raise NativePreflightError("G3 must record metric, value, and threshold")
    return payload


def invalidate_artifacts(
    scene_id: str,
    reasons: Sequence[str],
    *,
    phase: str,
    artifact_paths: Iterable[str | os.PathLike] = (),
) -> pathlib.Path:
    """Write a durable quarantine marker covering every publication surface."""

    directory = review_dir(scene_id)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = sorted({str(pathlib.Path(path).resolve()) for path in artifact_paths})
    marker = {
        "valid": False,
        "scene_id": scene_id,
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reasons": list(reasons),
        "invalidates": list(INVALID_KINDS),
        "artifact_paths": artifacts,
        "publication_blocked": True,
    }
    marker_path = directory / f"{scene_id}_INVALID_{phase}.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    for artifact in artifacts:
        sidecar = pathlib.Path(f"{artifact}.INVALID.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return marker_path
