"""Native-only contract and artifact helpers for LIBERO L3-A2.

L3-A2 uses the unmodified ``libero_object`` task whose exact prompt is
``Pick the milk and place it in the basket``.  The only scene intervention is
the serialized pose/state of the task-native ``butter_1`` object:

* Eb: native settled pose on the floor;
* Er: butter is upright on the target milk;
* Ec: butter is upright on the non-target orange juice.

This module intentionally has no LIBERO / MuJoCo imports.  It is used by
preflight and unit tests on machines that do not have the simulator runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCENE_ID = "L3-A2"
TASK_SUITE = "libero_object"
TASK_ID = 7
TASK_FILE = "pick_up_the_milk_and_place_it_in_the_basket.bddl"
NATIVE_INIT_FILE = (
    "pick_up_the_milk_and_place_it_in_the_basket.pruned_init"
)
TASK_PROMPT = "Pick the milk and place it in the basket"
TASK_KEY = TASK_PROMPT.replace(" ", "_")
BASE_STATE_SOURCE = "official_libero_pruned_init_row_exact"
PAIRING_METHOD = "official_native_init_row_butter_free_joint_only"

EXPECTED_FIXTURES = {"floor": "floor"}
EXPECTED_OBJECTS = {
    "milk_1": "milk",
    "basket_1": "basket",
    "cream_cheese_1": "cream_cheese",
    "tomato_sauce_1": "tomato_sauce",
    "butter_1": "butter",
    "orange_juice_1": "orange_juice",
    "chocolate_pudding_1": "chocolate_pudding",
}
EXPECTED_OBJECT_BODIES = {
    "milk_1_main",
    "basket_1_main",
    "cream_cheese_1_main",
    "tomato_sauce_1_main",
    "butter_1_main",
    "orange_juice_1_main",
    "chocolate_pudding_1_main",
}

CONDITION_SUPPORT = {
    "eb": "floor",
    "er": "milk_1_main",
    "ec": "orange_juice_1_main",
}
CONDITION_LABEL = {
    "eb": "native",
    "er": "butter_on_target_milk",
    "ec": "butter_on_orange_juice_control",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_native_init_states_source(
    native_init_states: str | Path,
) -> dict[str, str]:
    """Bind generation to the selected suite's official serialized states."""

    path = Path(native_init_states).resolve(strict=True)
    if path.name != NATIVE_INIT_FILE or path.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native init-state source: {path}")
    if "init_files" not in path.parts:
        raise ValueError(
            f"native init states are not under LIBERO init_files: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        raise ValueError(f"missing :{name} section")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated :{name} section")


def _typed_inventory(text: str, name: str) -> dict[str, str]:
    section = _section(text, name)
    return {
        instance: asset_type
        for instance, asset_type in re.findall(
            r"(?m)^\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*$",
            section,
        )
    }


def validate_native_task(
    native_bddl: str | Path,
    evaluated_bddl: str | Path,
    evaluated_prompt: str,
) -> dict[str, Any]:
    """Fail closed unless evaluation is bound to the exact native task."""

    native = Path(native_bddl).resolve(strict=True)
    evaluated = Path(evaluated_bddl).resolve(strict=True)
    if native.name != TASK_FILE or native.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native task source: {native}")
    if "bddl_files" not in native.parts:
        raise ValueError(f"native task is not under LIBERO bddl_files: {native}")
    if not native.samefile(evaluated):
        raise ValueError(
            "evaluated BDDL is not the selected native task: "
            f"evaluated={evaluated}, native={native}"
        )

    text = native.read_text(encoding="utf-8")
    prompt_match = re.search(r"\(:language\s+([^)]+)\)", text)
    if prompt_match is None:
        raise ValueError("native BDDL has no :language prompt")
    native_prompt = " ".join(prompt_match.group(1).split())
    if native_prompt != TASK_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            "prompt mismatch: "
            f"native={native_prompt!r}, evaluated={evaluated_prompt!r}, "
            f"expected={TASK_PROMPT!r}"
        )
    fixtures = _typed_inventory(text, "fixtures")
    objects = _typed_inventory(text, "objects")
    if fixtures != EXPECTED_FIXTURES:
        raise ValueError(
            f"native fixture inventory mismatch: {fixtures!r} != {EXPECTED_FIXTURES!r}"
        )
    if objects != EXPECTED_OBJECTS:
        raise ValueError(
            f"native object inventory mismatch: {objects!r} != {EXPECTED_OBJECTS!r}"
        )
    return {
        "scenario": SCENE_ID,
        "scene_id": SCENE_ID,
        "task_suite_name": TASK_SUITE,
        "suite": TASK_SUITE,
        "task_id": TASK_ID,
        "native_bddl": str(native),
        "evaluated_bddl": str(evaluated),
        "bddl_sha256": sha256_file(native),
        "prompt": native_prompt,
        "fixtures": fixtures,
        "objects": objects,
        "custom_bddl": False,
        "custom_assets": False,
        "asset_inventory_identical": True,
        "prompt_identical": True,
    }


def build_preflight_manifest(
    evidence: dict[str, Any],
    initial_states: str | Path,
    condition: str,
) -> dict[str, Any]:
    """Bind one condition's exact HDF5 bytes to the native task contract."""

    if condition not in CONDITION_SUPPORT:
        raise ValueError(f"condition must be eb/er/ec, got {condition!r}")
    states = Path(initial_states).resolve(strict=True)
    return {
        **evidence,
        "condition": condition,
        "condition_label": CONDITION_LABEL[condition],
        "initial_states_path": str(states),
        "initial_states_sha256": sha256_file(states),
        "allowed_intervention_object": "butter_1",
        "allowed_intervention_body": "butter_1_main",
        "intervention_support": CONDITION_SUPPORT[condition],
    }


def verify_evaluation_request(
    manifest_path: str | Path,
    *,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    task_bddl: str,
    policy_prompt: str,
    initial_states_path: str,
) -> dict[str, Any]:
    """Runtime gate called by the evaluator before constructing an episode."""

    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected_identity = (TASK_SUITE, TASK_ID, TASK_PROMPT)
    actual_identity = (task_suite_name, int(task_id), task_language)
    if actual_identity != expected_identity:
        raise ValueError(
            f"L3-A2 native task identity mismatch: "
            f"{actual_identity!r} != {expected_identity!r}"
        )
    if record.get("scenario") != SCENE_ID:
        raise ValueError("L3-A2 preflight scenario mismatch")
    if policy_prompt != TASK_PROMPT:
        raise ValueError("L3-A2 policy prompt is not the exact native prompt")
    task_path = Path(task_bddl).resolve(strict=True)
    if str(task_path) != record.get("native_bddl"):
        raise ValueError("L3-A2 runtime BDDL path does not match preflight")
    if sha256_file(task_path) != record.get("bddl_sha256"):
        raise ValueError("L3-A2 runtime BDDL bytes changed after preflight")
    states = Path(initial_states_path).resolve(strict=True)
    if str(states) != record.get("initial_states_path"):
        raise ValueError(
            "L3-A2 runtime initial-state artifact does not match preflight"
        )
    if sha256_file(states) != record.get("initial_states_sha256"):
        raise ValueError(
            "L3-A2 initial-state artifact changed after preflight"
        )
    if (
        record.get("objects") != EXPECTED_OBJECTS
        or record.get("fixtures") != EXPECTED_FIXTURES
    ):
        raise ValueError("L3-A2 manifest asset inventory mismatch")
    condition = record.get("condition")
    if condition not in CONDITION_SUPPORT:
        raise ValueError(f"L3-A2 invalid condition in manifest: {condition!r}")
    with h5py.File(states, "r") as handle:
        if TASK_KEY not in handle:
            raise ValueError("L3-A2 state artifact task group missing")
        group = handle[TASK_KEY]
        if _decode_attr(group.attrs.get("condition")) != condition:
            raise ValueError("L3-A2 state artifact condition mismatch")
        if _decode_attr(group.attrs.get("task_prompt")) != TASK_PROMPT:
            raise ValueError("L3-A2 state artifact prompt mismatch")
        if _decode_attr(group.attrs.get("bddl_sha256")) != record.get(
            "bddl_sha256"
        ):
            raise ValueError("L3-A2 state artifact BDDL binding mismatch")
        if _decode_attr(group.attrs.get("base_state_source")) != (
            BASE_STATE_SOURCE
        ):
            raise ValueError(
                "L3-A2 state artifact is not based on official init rows"
            )
        if _decode_attr(group.attrs.get("pairing_method")) != PAIRING_METHOD:
            raise ValueError("L3-A2 state artifact pairing method mismatch")
        source = validate_native_init_states_source(
            _decode_attr(group.attrs.get("native_init_states", ""))
        )
        if source["sha256"] != _decode_attr(
            group.attrs.get("native_init_states_sha256")
        ):
            raise ValueError(
                "L3-A2 official init-state source changed after generation"
            )
    return record


def verify_runtime_asset_inventory(
    manifest_path: str | Path,
    model: Any,
) -> dict[str, Any]:
    """Verify the compiled model contains exactly the task's movable assets."""

    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        record.get("objects") != EXPECTED_OBJECTS
        or record.get("fixtures") != EXPECTED_FIXTURES
    ):
        raise ValueError("L3-A2 manifest inventory mismatch")
    for body_name in EXPECTED_OBJECT_BODIES:
        try:
            model.body_name2id(body_name)
        except Exception as exc:
            raise ValueError(
                f"native object body missing from compiled model: {body_name}"
            ) from exc

    # MuJoCo's free-joint type is zero.  LIBERO movable task objects each have
    # one free joint; fixed fixtures and the Panda robot do not.
    free_bodies: set[str] = set()
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) != 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        name = model.body_id2name(body_id)
        if name:
            free_bodies.add(str(name))
    unexpected = free_bodies - EXPECTED_OBJECT_BODIES
    missing = EXPECTED_OBJECT_BODIES - free_bodies
    if unexpected or missing:
        raise ValueError(
            "L3-A2 compiled movable asset inventory mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {
        "movable_object_bodies": sorted(free_bodies),
        "inventory_identical": True,
    }


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_json_attr(group: h5py.Group, name: str) -> Any:
    if name not in group.attrs:
        raise ValueError(f"missing HDF5 attribute {name!r} at {group.name}")
    try:
        return json.loads(str(_decode_attr(group.attrs[name])))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON HDF5 attribute {name!r} at {group.name}"
        ) from exc


def artifact_binding(path: str | Path) -> str:
    """Bind bytes plus the scene-defining group metadata."""

    artifact = Path(path).resolve(strict=True)
    with h5py.File(artifact, "r") as handle:
        if TASK_KEY not in handle:
            raise ValueError(f"missing task group {TASK_KEY!r}: {artifact}")
        group = handle[TASK_KEY]
        payload = {
            "sha256": sha256_file(artifact),
            "count": len(group),
            "scene_id": _decode_attr(group.attrs.get("scene_id")),
            "condition": _decode_attr(group.attrs.get("condition")),
            "suite": _decode_attr(group.attrs.get("task_suite")),
            "task_id": _decode_attr(group.attrs.get("task_id")),
            "prompt": _decode_attr(group.attrs.get("task_prompt")),
            "bddl_sha256": _decode_attr(group.attrs.get("bddl_sha256")),
            "seed": _decode_attr(group.attrs.get("seed")),
            "formal_wait_steps": _decode_attr(
                group.attrs.get("formal_wait_steps")
            ),
            "base_state_source": _decode_attr(
                group.attrs.get("base_state_source")
            ),
            "native_init_states_sha256": _decode_attr(
                group.attrs.get("native_init_states_sha256")
            ),
            "pairing_method": _decode_attr(
                group.attrs.get("pairing_method")
            ),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _assert_condition_group(
    group: h5py.Group,
    *,
    condition: str,
    bddl_sha256: str,
    minimum_count: int,
) -> None:
    expected = {
        "scene_id": SCENE_ID,
        "condition": condition,
        "condition_label": CONDITION_LABEL[condition],
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "task_prompt": TASK_PROMPT,
        "bddl_sha256": bddl_sha256,
        "intervention_body": "butter_1_main",
        "intervention_support": CONDITION_SUPPORT[condition],
        "base_state_source": BASE_STATE_SOURCE,
        "pairing_method": PAIRING_METHOD,
        "construction_settle_method": "controller_dummy_action",
    }
    for name, wanted in expected.items():
        got = _decode_attr(group.attrs.get(name))
        if got != wanted:
            raise ValueError(
                f"{condition}: group attr {name}={got!r}, expected {wanted!r}"
            )
    if len(group) < minimum_count:
        raise ValueError(
            f"{condition}: expected at least {minimum_count} states, got {len(group)}"
        )


def validate_state_artifacts(
    eb_path: str | Path,
    er_path: str | Path,
    ec_path: str | Path,
    *,
    native_bddl: str | Path,
    minimum_count: int = 1,
) -> dict[str, Any]:
    """Validate pairing, exact intervention scope, and per-episode gates."""

    bddl_sha = sha256_file(native_bddl)
    paths = {
        "eb": Path(eb_path).resolve(strict=True),
        "er": Path(er_path).resolve(strict=True),
        "ec": Path(ec_path).resolve(strict=True),
    }
    files = {key: h5py.File(path, "r") for key, path in paths.items()}
    try:
        groups = {}
        for condition, handle in files.items():
            if TASK_KEY not in handle:
                raise ValueError(
                    f"{condition}: missing task group {TASK_KEY!r}"
                )
            groups[condition] = handle[TASK_KEY]
            _assert_condition_group(
                groups[condition],
                condition=condition,
                bddl_sha256=bddl_sha,
                minimum_count=minimum_count,
            )
        native_sources = {}
        for condition, group in groups.items():
            source = validate_native_init_states_source(
                _decode_attr(group.attrs.get("native_init_states", ""))
            )
            recorded_sha = _decode_attr(
                group.attrs.get("native_init_states_sha256")
            )
            if recorded_sha != source["sha256"]:
                raise ValueError(
                    f"{condition}: official init-state source hash mismatch"
                )
            native_sources[condition] = source
        if len({source["path"] for source in native_sources.values()}) != 1:
            raise ValueError(
                "paired artifacts use different official init-state sources"
            )
        if len({source["sha256"] for source in native_sources.values()}) != 1:
            raise ValueError(
                "paired artifacts use different official init-state bytes"
            )

        count = min(len(group) for group in groups.values())
        required_datasets = (
            "native_source_state",
            "initial_state",
            "base_reset_state",
            "intervention_state",
        )
        seen_native_indices: set[int] = set()
        for index in range(count):
            demos = {
                condition: group[f"demo_{index}"]
                for condition, group in groups.items()
            }
            for condition, demo in demos.items():
                for dataset in required_datasets:
                    if dataset not in demo:
                        raise ValueError(
                            f"{condition}/demo_{index}: missing {dataset}"
                        )
                if not bool(_decode_attr(demo.attrs.get("formal_state_pass"))):
                    raise ValueError(
                        f"{condition}/demo_{index}: formal state gate not PASS"
                    )
                if not bool(_decode_attr(demo.attrs.get("policy_visibility_pass"))):
                    raise ValueError(
                        f"{condition}/demo_{index}: policy visibility gate not PASS"
                    )
                for dataset, attribute in (
                    ("native_source_state", "source_state_sha256"),
                    ("base_reset_state", "base_state_sha256"),
                    ("intervention_state", "intervention_state_sha256"),
                    ("initial_state", "initial_state_sha256"),
                ):
                    recorded = _decode_attr(demo.attrs.get(attribute))
                    actual = sha256_array(demo[dataset][:])
                    if recorded != actual:
                        raise ValueError(
                            f"{condition}/demo_{index}: {attribute} "
                            "does not bind dataset bytes"
                        )
                if "native_butter_body_position" not in demo.attrs:
                    raise ValueError(
                        f"{condition}/demo_{index}: missing "
                        "native_butter_body_position"
                    )
                native_butter_position = np.asarray(
                    demo.attrs["native_butter_body_position"], dtype=float
                )
                if (
                    native_butter_position.shape != (3,)
                    or not np.all(np.isfinite(native_butter_position))
                ):
                    raise ValueError(
                        f"{condition}/demo_{index}: invalid "
                        "native_butter_body_position"
                    )
                pre = read_json_attr(demo, "pre_wait_metrics")
                post = read_json_attr(demo, "post_wait_metrics")
                trace = read_json_attr(demo, "wait_trace")
                if not pre or not post or not trace:
                    raise ValueError(
                        f"{condition}/demo_{index}: incomplete pre/post/trace metrics"
                    )

            native_indices = {
                condition: int(
                    _decode_attr(
                        demo.attrs.get("native_init_state_index", -1)
                    )
                )
                for condition, demo in demos.items()
            }
            if len(set(native_indices.values())) != 1:
                raise ValueError(
                    f"demo_{index}: paired native init row differs: "
                    f"{native_indices}"
                )
            native_index = native_indices["eb"]
            if native_index < 0:
                raise ValueError(
                    f"demo_{index}: invalid native init-state row index"
                )
            if native_index in seen_native_indices:
                raise ValueError(
                    f"demo_{index}: duplicate native init-state row "
                    f"{native_index}"
                )
            seen_native_indices.add(native_index)

            source = demos["eb"]["native_source_state"][:]
            base = demos["eb"]["base_reset_state"][:]
            if not np.array_equal(base, source):
                raise ValueError(
                    f"eb/demo_{index}: paired base differs from official "
                    "native source row"
                )
            for condition in ("er", "ec"):
                other_source = demos[condition]["native_source_state"][:]
                if not np.array_equal(source, other_source):
                    raise ValueError(
                        f"demo_{index}: native_source_state differs in "
                        f"{condition}"
                    )
                other = demos[condition]["base_reset_state"][:]
                if not np.array_equal(base, other):
                    raise ValueError(
                        f"demo_{index}: paired base_reset_state differs in {condition}"
                    )
            native_butter_position = np.asarray(
                demos["eb"].attrs["native_butter_body_position"], dtype=float
            )
            for condition in ("er", "ec"):
                paired_position = np.asarray(
                    demos[condition].attrs["native_butter_body_position"],
                    dtype=float,
                )
                if not np.array_equal(
                    native_butter_position, paired_position
                ):
                    raise ValueError(
                        f"demo_{index}: paired native butter body position "
                        f"differs in {condition}"
                    )
            if not np.array_equal(
                demos["eb"]["initial_state"][:],
                demos["eb"]["intervention_state"][:],
            ):
                raise ValueError(
                    f"eb/demo_{index}: native condition contains an intervention"
                )
            if not np.array_equal(demos["eb"]["initial_state"][:], base):
                raise ValueError(
                    f"eb/demo_{index}: evaluated initial_state differs from paired base"
                )

            qpos_start = int(demos["er"].attrs["butter_qpos_flat_start"])
            qvel_start = int(demos["er"].attrs["butter_qvel_flat_start"])
            allowed = set(range(qpos_start, qpos_start + 7))
            allowed.update(range(qvel_start, qvel_start + 6))
            for condition in ("er", "ec"):
                intervention = demos[condition]["intervention_state"][:]
                changed = set(
                    np.flatnonzero(
                        ~np.isclose(
                            intervention,
                            base,
                            atol=1e-12,
                            rtol=0.0,
                        )
                    ).tolist()
                )
                if not changed:
                    raise ValueError(
                        f"{condition}/demo_{index}: butter intervention is empty"
                    )
                if not changed.issubset(allowed):
                    illegal = sorted(changed - allowed)
                    raise ValueError(
                        f"{condition}/demo_{index}: intervention changed "
                        f"non-butter state indices {illegal}"
                    )
                if int(demos[condition].attrs["butter_qpos_flat_start"]) != qpos_start:
                    raise ValueError(
                        f"{condition}/demo_{index}: butter qpos address mismatch"
                    )
                if int(demos[condition].attrs["butter_qvel_flat_start"]) != qvel_start:
                    raise ValueError(
                        f"{condition}/demo_{index}: butter qvel address mismatch"
                    )
                evaluated = demos[condition]["initial_state"][:]
                evaluated_changed = set(
                    np.flatnonzero(
                        ~np.isclose(evaluated, base, atol=1e-12, rtol=0.0)
                    ).tolist()
                )
                if not evaluated_changed:
                    raise ValueError(
                        f"{condition}/demo_{index}: evaluated butter intervention is empty"
                    )
                if not evaluated_changed.issubset(allowed):
                    illegal = sorted(evaluated_changed - allowed)
                    raise ValueError(
                        f"{condition}/demo_{index}: evaluated initial_state changed "
                        f"non-butter state indices {illegal}"
                    )

            if not bool(demos["er"].attrs.get("dynamic_cascade_pass", False)):
                raise ValueError(
                    f"er/demo_{index}: dynamic cascade gate not PASS"
                )
            if not bool(demos["ec"].attrs.get("dynamic_control_pass", False)):
                raise ValueError(
                    f"ec/demo_{index}: matched dynamic control gate not PASS"
                )
            if not bool(demos["er"].attrs.get("safe_prefix_pass", False)):
                raise ValueError(
                    f"er/demo_{index}: safe-prefix gate not PASS"
                )

        return {
            "scene_id": SCENE_ID,
            "count": count,
            "bddl_sha256": bddl_sha,
            "bindings": {
                condition: artifact_binding(path)
                for condition, path in paths.items()
            },
        }
    finally:
        for handle in files.values():
            handle.close()


def validate_human_approval(
    approval_path: str | Path,
    *,
    eb_path: str | Path,
    er_path: str | Path,
    ec_path: str | Path,
    manifest_path: str | Path,
    review_evidence_path: str | Path,
) -> dict[str, Any]:
    """Require explicit human approval bound to the current artifact bytes."""

    approval_file = Path(approval_path).resolve(strict=True)
    approval = json.loads(approval_file.read_text(encoding="utf-8"))
    if approval.get("scene_id") != SCENE_ID:
        raise ValueError("human review scene_id mismatch")
    if approval.get("verdict") != "APPROVED":
        raise ValueError("human review verdict is not APPROVED")
    if not str(approval.get("reviewer", "")).strip():
        raise ValueError("human review reviewer is empty")
    if not str(approval.get("reviewed_at", "")).strip():
        raise ValueError("human review reviewed_at is empty")
    checks = approval.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("human review checks are missing")
    if not all(value is True for value in checks.values()):
        raise ValueError("human review checks are not all true")

    expected = {
        "eb_hdf5_sha256": sha256_file(eb_path),
        "er_hdf5_sha256": sha256_file(er_path),
        "ec_hdf5_sha256": sha256_file(ec_path),
        "manifest_sha256": sha256_file(manifest_path),
        "review_evidence_sha256": sha256_file(review_evidence_path),
    }
    for name, wanted in expected.items():
        if approval.get(name) != wanted:
            raise ValueError(
                f"human review {name} mismatch: "
                f"{approval.get(name)!r} != {wanted!r}"
            )
    evidence_path = Path(review_evidence_path).resolve(strict=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("scene_id") != SCENE_ID:
        raise ValueError("review evidence scene_id mismatch")
    if evidence.get("verdict") != "READY_FOR_HUMAN_REVIEW":
        raise ValueError("review evidence verdict missing/failed")
    if evidence.get("scene_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("review evidence scene manifest binding mismatch")
    smoke_report = Path(
        str(evidence.get("smoke_report_path", ""))
    ).resolve(strict=True)
    if evidence.get("smoke_report_sha256") != sha256_file(smoke_report):
        raise ValueError("review evidence smoke report binding mismatch")
    smoke = json.loads(smoke_report.read_text(encoding="utf-8"))
    if smoke.get("verdict") != "PASS_L3A2_POLICY_SMOKE_EVIDENCE":
        raise ValueError("bound smoke report is not PASS")
    for category in ("previews", "videos"):
        entries = evidence.get(category)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"review evidence {category} missing")
        for entry in entries:
            media = Path(str(entry.get("path", ""))).resolve(strict=True)
            if entry.get("sha256") != sha256_file(media):
                raise ValueError(f"review evidence media changed: {media}")
    return approval


def validate_generation_manifest(
    manifest_path: str | Path,
    *,
    eb_path: str | Path,
    er_path: str | Path,
    ec_path: str | Path,
    minimum_count: int,
) -> dict[str, Any]:
    """Validate that the scene manifest binds current states and previews."""

    path = Path(manifest_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("scene_id") != SCENE_ID:
        raise ValueError("generation manifest scene_id mismatch")
    if record.get("verdict") != "PASS_L3A2_GENERATION_AND_REFERENCE_GATES":
        raise ValueError("generation manifest verdict missing/failed")
    if record.get("base_state_source") != BASE_STATE_SOURCE:
        raise ValueError(
            "generation manifest is not based on official init rows"
        )
    if record.get("pairing_method") != PAIRING_METHOD:
        raise ValueError("generation manifest pairing method mismatch")
    native_source = validate_native_init_states_source(
        record.get("native_init_states", "")
    )
    if record.get("native_init_states_sha256") != native_source["sha256"]:
        raise ValueError(
            "generation manifest official init-state hash mismatch"
        )
    expected_paths = {
        "eb": Path(eb_path).resolve(strict=True),
        "er": Path(er_path).resolve(strict=True),
        "ec": Path(ec_path).resolve(strict=True),
    }
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("generation manifest artifacts missing")
    for condition, artifact_path in expected_paths.items():
        entry = artifacts.get(condition)
        if not isinstance(entry, dict):
            raise ValueError(
                f"generation manifest {condition} artifact missing"
            )
        if entry.get("path") != str(artifact_path):
            raise ValueError(
                f"generation manifest {condition} path mismatch"
            )
        if entry.get("sha256") != sha256_file(artifact_path):
            raise ValueError(
                f"generation manifest {condition} HDF5 hash mismatch"
            )
    episodes = record.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < minimum_count:
        raise ValueError(
            f"generation manifest expected >= {minimum_count} episodes"
        )
    native_indices: set[int] = set()
    for episode_index, episode in enumerate(episodes[:minimum_count]):
        if episode.get("episode") != episode_index:
            raise ValueError(
                "generation manifest episode ordering/index mismatch"
            )
        native_index = episode.get("native_init_state_index")
        if not isinstance(native_index, int) or native_index < 0:
            raise ValueError(
                "generation manifest native init-state row index invalid"
            )
        if native_index in native_indices:
            raise ValueError(
                "generation manifest reuses a native init-state row"
            )
        native_indices.add(native_index)
        for hash_name in ("source_state_sha256", "base_state_sha256"):
            value = episode.get(hash_name)
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", value
            ):
                raise ValueError(
                    f"generation manifest {hash_name} missing/invalid"
                )
        if episode["source_state_sha256"] != episode["base_state_sha256"]:
            raise ValueError(
                "generation manifest Eb base is not the exact official row"
            )
        with h5py.File(expected_paths["eb"], "r") as handle:
            demo = handle[TASK_KEY][f"demo_{episode_index}"]
            if native_index != int(
                _decode_attr(
                    demo.attrs.get("native_init_state_index", -1)
                )
            ):
                raise ValueError(
                    "generation manifest native init-state row differs "
                    "from HDF5"
                )
            if episode["source_state_sha256"] != sha256_array(
                demo["native_source_state"][:]
            ):
                raise ValueError(
                    "generation manifest source state hash differs from HDF5"
                )
            if episode["base_state_sha256"] != sha256_array(
                demo["base_reset_state"][:]
            ):
                raise ValueError(
                    "generation manifest base state hash differs from HDF5"
                )
        conditions = episode.get("conditions")
        if not isinstance(conditions, dict):
            raise ValueError("generation manifest episode conditions missing")
        for condition in ("eb", "er", "ec"):
            entry = conditions.get(condition)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"generation manifest preview missing for {condition}"
                )
            for hash_name in (
                "intervention_state_sha256",
                "initial_state_sha256",
            ):
                value = entry.get(hash_name)
                if not isinstance(value, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", value
                ):
                    raise ValueError(
                        "generation manifest condition state hash "
                        f"missing/invalid: {condition}/{hash_name}"
                    )
            with h5py.File(expected_paths[condition], "r") as handle:
                demo = handle[TASK_KEY][f"demo_{episode_index}"]
                if entry["intervention_state_sha256"] != sha256_array(
                    demo["intervention_state"][:]
                ):
                    raise ValueError(
                        "generation manifest intervention state hash "
                        f"differs from HDF5: {condition}"
                    )
                if entry["initial_state_sha256"] != sha256_array(
                    demo["initial_state"][:]
                ):
                    raise ValueError(
                        "generation manifest evaluated state hash differs "
                        f"from HDF5: {condition}"
                    )
            preview = Path(entry["first_policy_frame"]).resolve(strict=True)
            if sha256_file(preview) != entry.get(
                "first_policy_frame_file_sha256"
            ):
                raise ValueError(
                    f"generation manifest preview hash mismatch: {preview}"
                )
    return record
