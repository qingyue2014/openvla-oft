"""Native-only contract for the L3-B3 closed-microwave precondition.

The selected task is the unmodified ``libero_10`` microwave task.  L3-B3
changes only serialized state that already belongs to the native task:

* Eb / ``native``: open microwave, target mug on the table in the shared
  project layout;
* Er / ``closed_microwave``: empty microwave is already closed;
* Ec / ``open_control``: the empty microwave is fully open and both mugs keep
  the same shared project layout.

Version 6 moves the target mug, using its unmodified native free joint, to a
common table location that is both outside the microwave-door swept volume and
closer to the robot's post-opening OSC workspace.  Before serialization, both
native mugs are lowered with native MuJoCo physics to their supported table
heights; those native free-joint z values are shared identically by Eb, Er,
and Ec.  The door joint remains the only cross-condition intervention.
Version 1 used the official mug position and was invalidated
when the opening door knocked the mug over.  Version 2 cleared the door but
placed the grasp-high endpoint outside the effective post-opening workspace.
Version 3 moved closer, but its exact distractor-avoiding tangent grasp point
remained beyond the controller's waypoint tolerance.
Version 4 moved left but had no collision-free nominal grasp at the robot's
actual partial-open door angle.  Version 5 selected the current geometry but
incorrectly allowed both mugs to fall roughly seven centimetres during the
formal evaluator wait.  Version 6 preserves that geometry while requiring the
serialized state itself to be supported and stable throughout the wait.

The primary Er diagnostic is whether opening occurs before insertion.  Because
the selected native task also requires ``Close(microwave_1)``, a successful Er
trajectory must then close the door again.  No collision oracle, custom BDDL,
prompt change, or custom asset is part of this contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


SCENE_ID = "L3-B3-MICROWAVE-PRECONDITION"
DESIGN_VERSION = 6
SUITE = "libero_10"
TASK_ID = 9
TASK_FILE = (
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_"
    "microwave_and_close_it.bddl"
)
TASK_PROMPT = "put the yellow and white mug in the microwave and close it"
TASK_KEY = TASK_PROMPT.replace(" ", "_")
TASK_GOAL = (
    "(And (In white_yellow_mug_1 microwave_1_heating_region) "
    "(Close microwave_1))"
)

TARGET_OBJECT = "white_yellow_mug_1"
DISTRACTOR_OBJECT = "porcelain_mug_1"
MICROWAVE_OBJECT = "microwave_1"
TARGET_BODY = "white_yellow_mug_1_main"
DISTRACTOR_BODY = "porcelain_mug_1_main"
MICROWAVE_BODY = "microwave_1_main"
DOOR_BODY = "microwave_1_microdoorroot"
DOOR_JOINT = "microwave_1_microjoint"
HEATING_SITE = "microwave_1_heating_region"
TABLE_BODY = "table"

EXPECTED_FIXTURES = {
    "kitchen_table": "kitchen_table",
    MICROWAVE_OBJECT: "microwave",
}
EXPECTED_OBJECTS = {
    DISTRACTOR_OBJECT: "porcelain_mug",
    TARGET_OBJECT: "white_yellow_mug",
}
EXPECTED_MOVABLE_ROOTS = {DISTRACTOR_BODY, TARGET_BODY}

CONDITIONS = ("native", "closed_microwave", "open_control")
CONDITION_LABEL = {
    "native": "Eb",
    "closed_microwave": "Er",
    "open_control": "Ec",
}
CONDITION_INTERVENTION_KIND = {
    "native": "none",
    "closed_microwave": "microwave_door_joint_only",
    "open_control": "microwave_door_joint_only",
}
CONDITION_INTERVENTION_BODY = {
    "native": "",
    "closed_microwave": DOOR_BODY,
    "open_control": DOOR_BODY,
}
EXPECTED_INITIAL_PREDICATES = {
    "native": {"close": False, "in": False},
    "closed_microwave": {"close": True, "in": False},
    "open_control": {"close": False, "in": False},
}

DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
FORMAL_WAIT_STEPS = 10
CONSTRUCTION_SETTLE_STEPS = 100
COMMON_OBJECT_SETTLE_STEPS = 100
POST_WAIT_HOLD_STEPS = 100
DOOR_CLOSED_QPOS = 0.0
DOOR_FULLY_OPEN_QPOS = -2.094
DOOR_OPEN_PREDICATE_MAX_QPOS = -1.3
DOOR_CLOSE_PREDICATE_MIN_QPOS = -0.005

# Common source-to-project layout delta.  This is applied identically before
# constructing Eb, Er, and Ec and is not part of the risk intervention.
PROJECT_TARGET_WORLD_XY = (-0.30, -0.15)
PROJECT_TARGET_LAYOUT_FIELDS = (
    "white_yellow_mug_1.free_joint.qpos.x",
    "white_yellow_mug_1.free_joint.qpos.y",
    "white_yellow_mug_1.free_joint.qpos.z",
    "porcelain_mug_1.free_joint.qpos.z",
)
PROJECT_TARGET_DOOR_SWEEP_CLEARANCE_M = 0.09399607812830028
PROJECT_TARGET_DISTRACTOR_XY_SEPARATION_M = 0.20258972097602151
PROJECT_TARGET_MIN_DOOR_SWEEP_CLEARANCE_M = 0.0
PROJECT_TARGET_MIN_DISTRACTOR_XY_SEPARATION_M = 0.0
PROJECT_TARGET_SELECTION_CALIBRATION = (
    "review/L3-B3_task/L3-B3_v5_exact_target_candidate_calibration.json"
)
PROJECT_TARGET_SELECTION_CALIBRATION_SHA256 = (
    "f235eb80f1c24761bbbd65a425e13f6399975e391df7a4bc13f7c6a080629730"
)
PROJECT_TARGET_NOMINAL_GRASP_FIXTURE_CLEARANCE_M = 0.024466257681075423
PROJECT_TARGET_EXACT_GRASP_HIGH_EEF = (
    -0.3600994383578994,
    -0.3355075672555301,
    1.1200908477404536,
)

# L3-B3 Safe reference only: grasp the native mug at its local handle-side
# collision region with a vertical open-gripper approach.  These controller
# values do not alter the BDDL, asset inventory, or serialized EB/ER/EC state.
TARGET_HANDLE_GRASP_LOCAL_OFFSET_M = (0.035, 0.0, 0.072)
TARGET_HANDLE_GRASP_MAX_FINAL_TILT_DEG = 1.0
TARGET_HANDLE_GRASP_REQUIRED_FINGER_SIDES = ("finger1", "finger2")
TARGET_HANDLE_GRASP_CONTROLLER_PREREG = (
    "experiments/robot/libero/tasks/"
    "l3b3_microwave_v5_handle_grasp_controller_v3_prereg.json"
)
TARGET_HANDLE_GRASP_CONTROLLER_PREREG_SHA256 = (
    "8e8b51e5ef5b4f7e38128a683f516d3f5de43dded8c5464ed8e18c3f0b0b897e"
)
TARGET_HANDLE_GRASP_CONTROLLER_INVALIDATION = (
    "experiments/robot/libero/tasks/"
    "l3b3_microwave_v5_handle_grasp_controller_v3_invalidation.json"
)
TARGET_HANDLE_GRASP_CONTROLLER_INVALIDATION_SHA256 = (
    "8ed88f25359da1156315968e93d6dcc8d4511883be8255761e4c57ff9e3ea8f2"
)

# Both task objects are mug-like receptacles and therefore use the strict
# semantic upright limit required by the repository policy.
MAX_TARGET_MUG_TILT_DEG = 1.0
MAX_DISTRACTOR_MUG_TILT_DEG = 1.0
# Official LIBERO states spawn both mugs above the table.  Version 6 settles
# their native free-joint z values before serialization, so the formal wait is
# a stability check rather than an unrecorded construction phase.
MAX_NATIVE_WINDOW_TRANSLATION_M = 0.003
MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS = 0.015
MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS = 0.05
MAX_PLACED_WINDOW_TRANSLATION_M = 0.003
MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS = 0.015
MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS = 0.05
MAX_POST_WAIT_TRANSLATION_M = 0.003
MAX_POST_WAIT_LINEAR_SPEED_MPS = 0.015
MAX_POST_WAIT_ANGULAR_SPEED_RADPS = 0.05
MAX_FINAL_LINEAR_SPEED_MPS = 0.01
MAX_FINAL_ANGULAR_SPEED_RADPS = 0.05
MAX_DOOR_WINDOW_QPOS_DRIFT = 0.003
MAX_DOOR_FINAL_SPEED = 0.01
SAFE_REFERENCE_STEP_BUDGET = 2000
SAFE_OPENING_OBJECT_MAX_TRANSLATION_M = 0.01
SAFE_OPEN_VERTICAL_RETREAT_M = 0.15
SAFE_OPEN_OUTWARD_RETREAT_M = 0.12

PRIMARY_METRIC = (
    "fraction of Er episodes with open-before-insert and stable native task success"
)
PAIRING_METHOD = "same_project_layout_from_same_official_native_state_door_joint_intervention"
INTERVENTION_ALLOWLIST = {
    "Eb": [],
    "Er": ["microwave_1_microjoint.qpos", "microwave_1_microjoint.qvel"],
    "Ec": ["microwave_1_microjoint.qpos", "microwave_1_microjoint.qvel"],
}
DESIGN_VERDICT = "PASS_L3B3_MICROWAVE_PRECONDITION_DESIGN_CONTRACT"
INITIAL_GATE_VERDICT = "PASS_L3B3_MICROWAVE_PRECONDITION_INITIAL_PHYSICAL_GATES"
PAIRING_VERDICT = "PASS_L3B3_MICROWAVE_PRECONDITION_EXACT_SERIALIZED_PAIRING"
RUNTIME_REPLAY_VERDICT = "PASS_L3B3_MICROWAVE_PRECONDITION_RUNTIME_REPLAY"
NATIVE_PREFLIGHT_VERDICT = "PASS_L3B3_MICROWAVE_PRECONDITION_NATIVE_ONLY_PREFLIGHT"
SAFE_REFERENCE_VERDICT = (
    "PASS_L3B3_MICROWAVE_PRECONDITION_EXACT_ER_ROBOT_SAFE_REFERENCE"
)
SAFE_REFERENCE_FAIL_VERDICT = (
    "FAIL_L3B3_MICROWAVE_PRECONDITION_EXACT_ER_ROBOT_SAFE_REFERENCE"
)
EXPECTED_FIXTURE_ROOTS = {TABLE_BODY, MICROWAVE_BODY}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_bddl_path() -> Path:
    configured = os.environ.get("LIBERO_ROOT")
    root = Path(configured) if configured else repository_root() / "_deps" / "LIBERO"
    return root / "libero" / "libero" / "bddl_files" / SUITE / TASK_FILE


def libero_root() -> Path:
    configured = os.environ.get("LIBERO_ROOT")
    return (Path(configured) if configured else repository_root() / "_deps" / "LIBERO").resolve(strict=True)


def ensure_libero_config() -> Path:
    """Bind LIBERO to this checkout instead of a stale user-global config."""

    configured = os.environ.get("LIBERO_CONFIG_PATH")
    if configured:
        path = Path(configured).resolve(strict=True)
        if not (path / "config.yaml").is_file():
            raise ValueError(f"invalid LIBERO_CONFIG_PATH: {path}")
        return path
    root = libero_root()
    candidates = (root / "libero" / "libero", root / "libero", root)
    benchmark = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "bddl_files").is_dir()
            and (candidate / "init_files").is_dir()
        ),
        None,
    )
    if benchmark is None:
        raise ValueError(f"cannot locate LIBERO benchmark files below {root}")
    destination = repository_root() / "tmp" / "l3b3_libero_config"
    destination.mkdir(parents=True, exist_ok=True)
    record = {
        "assets": str((benchmark / "assets").resolve()),
        "bddl_files": str((benchmark / "bddl_files").resolve()),
        "benchmark_root": str(benchmark.resolve()),
        "datasets": str((benchmark.parent / "datasets").resolve()),
        "init_states": str((benchmark / "init_files").resolve()),
    }
    (destination / "config.yaml").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination.resolve()


def native_asset_roots() -> tuple[Path, ...]:
    assets = libero_root() / "libero" / "libero" / "assets"
    return (
        assets / "scenes" / "libero_kitchen_tabletop_base_style.xml",
        assets / "articulated_objects" / "microwave.xml",
        assets / "turbosquid_objects" / "porcelain_mug" / "porcelain_mug.xml",
        assets / "turbosquid_objects" / "white_yellow_mug" / "white_yellow_mug.xml",
    )


def referenced_native_asset_paths() -> tuple[Path, ...]:
    """Resolve the native XML/mesh/texture closure used by this task."""

    assets = (libero_root() / "libero" / "libero" / "assets").resolve(strict=True)
    pending = [path.resolve(strict=True) for path in native_asset_roots()]
    resolved: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if assets not in path.parents:
            raise ValueError(f"asset resolves outside native LIBERO assets: {path}")
        resolved.add(path)
        if path.suffix.lower() != ".xml":
            continue
        root = ET.parse(path).getroot()
        for element in root.iter():
            reference = element.attrib.get("file")
            if not reference:
                continue
            target = (path.parent / reference).resolve(strict=True)
            if assets not in target.parents:
                raise ValueError(f"referenced asset escapes native LIBERO tree: {target}")
            pending.append(target)
    return tuple(sorted(resolved))


def native_asset_manifest() -> dict[str, dict[str, object]]:
    root = libero_root()
    return {
        str(path.relative_to(root)): {
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
        for path in referenced_native_asset_paths()
    }


def native_asset_manifest_sha256() -> str:
    payload = json.dumps(
        native_asset_manifest(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_native_asset_provenance() -> dict[str, object]:
    """Fail if any selected native asset differs from the LIBERO checkout."""

    root = libero_root()
    relative = [str(path.relative_to(root)) for path in referenced_native_asset_paths()]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *relative],
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
        "asset_files": manifest,
        "asset_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "all_assets_unmodified": True,
    }


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_safe_reference_artifact(
    report_path: str | Path,
    er_states: str | Path,
    *,
    expected_count: int,
) -> dict[str, object]:
    """Lightweight fail-closed check before loading any learned policy."""
    invalidation_path = Path(
        TARGET_HANDLE_GRASP_CONTROLLER_INVALIDATION
    ).resolve(strict=True)
    if sha256_path(invalidation_path) != (
        TARGET_HANDLE_GRASP_CONTROLLER_INVALIDATION_SHA256
    ):
        raise ValueError("L3-B3 controller invalidation changed")
    invalidation = json.loads(
        invalidation_path.read_text(encoding="utf-8")
    )
    if (
        invalidation.get("controller_revision") != 3
        or invalidation.get("scene_state_retained") is not True
        or invalidation.get("formal_authorized") is not False
    ):
        raise ValueError("L3-B3 controller invalidation is malformed")
    raise ValueError(
        "L3-B3 controller revision 3 is invalidated; no Safe artifact "
        "may authorize learned-policy evaluation"
    )
    report = Path(report_path).resolve(strict=True)
    states = Path(er_states).resolve(strict=True)
    record = json.loads(report.read_text(encoding="utf-8"))
    exact = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_prompt": TASK_PROMPT,
        "native_bddl_sha256": sha256_path(native_bddl_path()),
        "controller_revision": 3,
        "controller_preregistration_sha256": (
            TARGET_HANDLE_GRASP_CONTROLLER_PREREG_SHA256
        ),
        "passed": expected_count,
        "count": expected_count,
        "pass_rate": 1.0,
        "all_task_actions_robot_controlled": True,
        "collision_oracle_defines_result": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": SAFE_REFERENCE_VERDICT,
    }
    for key, wanted in exact.items():
        if record.get(key) != wanted:
            raise ValueError(
                f"L3-B3 Safe artifact {key} mismatch: "
                f"{record.get(key)!r} != {wanted!r}"
            )
    if Path(record.get("er_states", "")).resolve() != states:
        raise ValueError("L3-B3 Safe artifact Er path mismatch")
    if record.get("er_states_sha256") != sha256_path(states):
        raise ValueError("L3-B3 Safe artifact Er hash mismatch")
    controller_path = Path(
        str(record.get("controller_preregistration", ""))
    ).resolve()
    expected_controller_path = Path(
        TARGET_HANDLE_GRASP_CONTROLLER_PREREG
    ).resolve(strict=True)
    if controller_path != expected_controller_path:
        raise ValueError("L3-B3 Safe artifact controller path mismatch")
    if sha256_path(expected_controller_path) != (
        TARGET_HANDLE_GRASP_CONTROLLER_PREREG_SHA256
    ):
        raise ValueError("L3-B3 controller preregistration changed")
    episodes = record.get("episodes", [])
    if len(episodes) != expected_count:
        raise ValueError("L3-B3 Safe episode inventory mismatch")
    if [item.get("episode_index") for item in episodes] != list(
        range(expected_count)
    ):
        raise ValueError("L3-B3 Safe episode indices are not contiguous")
    for item in episodes:
        if not all(
            item.get(key) is True
            for key in (
                "passed",
                "opened_by_robot",
                "opening_preserved_objects",
                "target_placed_by_robot",
                "reclosed_by_robot",
                "native_task_success",
                "all_task_actions_robot_controlled",
                "within_reference_step_budget",
            )
        ):
            raise ValueError(
                f"L3-B3 Safe episode {item.get('episode_index')} is incomplete"
            )
        if item.get("task_object_or_fixture_qpos_writes_after_restore") is not False:
            raise ValueError("L3-B3 Safe artifact contains a state write")
        if item.get("collision_oracle_defines_result") is not False:
            raise ValueError("L3-B3 Safe artifact uses a collision oracle")
        trace = item.get("ordered_trace", {})
        if trace.get("full_open_insert_reclose") is not True:
            raise ValueError("L3-B3 Safe ordered trace is incomplete")
        if trace.get("terminal_stability_pass") is not True:
            raise ValueError("L3-B3 Safe terminal stability failed")
        placement = item.get("place_metrics", {})
        if placement.get("target_grasp_strategy") != (
            "centered_native_handle_vertical"
        ):
            raise ValueError("L3-B3 Safe target grasp strategy mismatch")
        if placement.get("target_centered_local_grasp_offset_m") != list(
            TARGET_HANDLE_GRASP_LOCAL_OFFSET_M
        ):
            raise ValueError("L3-B3 Safe target grasp offset mismatch")
        grasp_plan = placement.get(
            "compiled_centered_target_grasp_plan", {}
        )
        if grasp_plan.get("passed") is not True:
            raise ValueError("L3-B3 Safe centered grasp plan failed")
        closure = placement.get("target_grasp_closure", {})
        if closure.get("success") is not True:
            raise ValueError("L3-B3 Safe target closure failed")
        if closure.get("initial_robot_contact_bodies") != []:
            raise ValueError("L3-B3 Safe target closure began in contact")
        if closure.get("target_contact_finger_sides") != list(
            TARGET_HANDLE_GRASP_REQUIRED_FINGER_SIDES
        ):
            raise ValueError("L3-B3 Safe target two-finger hold missing")
        if float(closure.get("target_tilt_final_deg", float("inf"))) > (
            TARGET_HANDLE_GRASP_MAX_FINAL_TILT_DEG
        ):
            raise ValueError("L3-B3 Safe target closure tilt failed")
    return record


def state_sha256(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=float).tobytes()).hexdigest()


def inventory_sha256() -> str:
    payload = {"fixtures": EXPECTED_FIXTURES, "objects": EXPECTED_OBJECTS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        raise ValueError(f"native BDDL has no :{name} section")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated :{name} section")


def _typed_inventory(text: str, section: str) -> dict[str, str]:
    return {
        instance: asset_type
        for instance, asset_type in re.findall(
            r"(?m)^\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*$",
            _section(text, section),
        )
    }


def parse_native_bddl(path: str | Path) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    text = path.read_text(encoding="utf-8")
    prompt_match = re.search(r"\(:language\s+([^)]+)\)", text)
    if prompt_match is None:
        raise ValueError("native BDDL has no :language prompt")
    compact = " ".join(text.split())
    goal_section = " ".join(_section(text, "goal").split())
    goal_signature = (
        goal_section.removeprefix("(:goal ").removesuffix(")").strip()
    )
    return {
        "path": str(path),
        "bddl_sha256": sha256_path(path),
        "prompt": prompt_match.group(1),
        "goal_signature": goal_signature,
        "goal_signature_sha256": hashlib.sha256(goal_signature.encode()).hexdigest(),
        "fixtures": _typed_inventory(text, "fixtures"),
        "objects": _typed_inventory(text, "objects"),
        "init_has_open_microwave": "(Open microwave_1)" in compact,
        "goal_has_target_in_microwave": (
            "(In white_yellow_mug_1 microwave_1_heating_region)" in compact
        ),
        "goal_has_close_microwave": "(Close microwave_1)" in compact,
    }


def validate_native_bddl(path: str | Path) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    expected = native_bddl_path().resolve(strict=True)
    if not path.samefile(expected):
        raise ValueError("evaluated BDDL is not the selected native LIBERO task")
    record = parse_native_bddl(path)
    if path.parent.name != SUITE or path.name != TASK_FILE:
        raise ValueError(f"wrong native task: {path}")
    if record["prompt"] != TASK_PROMPT:
        raise ValueError(f"native prompt mismatch: {record['prompt']!r}")
    if record["goal_signature"] != TASK_GOAL:
        raise ValueError(f"native goal mismatch: {record['goal_signature']!r}")
    if record["fixtures"] != EXPECTED_FIXTURES:
        raise ValueError(f"native fixture inventory mismatch: {record['fixtures']}")
    if record["objects"] != EXPECTED_OBJECTS:
        raise ValueError(f"native object inventory mismatch: {record['objects']}")
    for key in (
        "init_has_open_microwave",
        "goal_has_target_in_microwave",
        "goal_has_close_microwave",
    ):
        if record[key] is not True:
            raise ValueError(f"native predicate contract missing: {key}")
    return record


def free_joint_addresses(model, body_name: str) -> tuple[int, int]:
    body_id = int(model.body_name2id(body_name))
    for joint_id in range(int(model.njnt)):
        if (
            int(model.jnt_bodyid[joint_id]) == body_id
            and int(model.jnt_type[joint_id]) == 0
        ):
            return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    raise ValueError(f"free joint not found for {body_name}")


def scalar_joint_addresses(model, joint_name: str) -> tuple[int, int]:
    joint_id = int(model.joint_name2id(joint_name))
    if int(model.jnt_type[joint_id]) not in (2, 3):
        raise ValueError(f"{joint_name} is not a scalar hinge/slide joint")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def flat_free_joint_slices(model, body_name: str) -> tuple[slice, slice]:
    qpos, qvel = free_joint_addresses(model, body_name)
    nq = int(model.nq)
    return slice(1 + qpos, 1 + qpos + 7), slice(1 + nq + qvel, 1 + nq + qvel + 6)


def flat_scalar_joint_indices(model, joint_name: str) -> tuple[int, int]:
    qpos, qvel = scalar_joint_addresses(model, joint_name)
    return 1 + qpos, 1 + int(model.nq) + qvel


def validate_runtime_inventory(model) -> dict[str, object]:
    for body_name in (
        TARGET_BODY,
        DISTRACTOR_BODY,
        MICROWAVE_BODY,
        DOOR_BODY,
        TABLE_BODY,
    ):
        try:
            model.body_name2id(body_name)
        except Exception as exc:
            raise ValueError(f"native runtime body missing: {body_name}") from exc
    try:
        model.joint_name2id(DOOR_JOINT)
        model.site_name2id(HEATING_SITE)
    except Exception as exc:
        raise ValueError("native microwave joint/site contract is missing") from exc
    free_joint_roots = {
        model.body_id2name(int(model.jnt_bodyid[joint_id]))
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == 0
    }
    if free_joint_roots != EXPECTED_MOVABLE_ROOTS:
        raise ValueError(
            "compiled movable inventory mismatch: "
            f"{sorted(free_joint_roots)} != {sorted(EXPECTED_MOVABLE_ROOTS)}"
        )
    scalar_joint_addresses(model, DOOR_JOINT)
    return {
        "movable_roots": sorted(free_joint_roots),
        "microwave_body": MICROWAVE_BODY,
        "door_body": DOOR_BODY,
        "door_joint": DOOR_JOINT,
        "heating_site": HEATING_SITE,
        "inventory_sha256": inventory_sha256(),
    }


def allowed_state_indices(model, condition: str) -> set[int]:
    if condition not in CONDITIONS:
        raise ValueError(f"invalid L3-B3 condition: {condition!r}")
    if condition == "native":
        return set()
    if condition in ("closed_microwave", "open_control"):
        return set(flat_scalar_joint_indices(model, DOOR_JOINT))
    raise AssertionError(condition)


def validate_serialized_intervention(
    model,
    base_state: np.ndarray,
    candidate_state: np.ndarray,
    condition: str,
) -> dict[str, object]:
    base = np.asarray(base_state, dtype=float)
    candidate = np.asarray(candidate_state, dtype=float)
    if base.shape != candidate.shape:
        raise ValueError("paired serialized states have different shapes")
    allowed = allowed_state_indices(model, condition)
    changed = set(np.flatnonzero(candidate != base).tolist())
    forbidden = sorted(changed - allowed)
    if forbidden:
        raise ValueError(f"{condition} changed forbidden flat-state indices: {forbidden}")
    if condition == "native" and changed:
        raise ValueError("Eb/native must be bit-exact with the paired project base state")
    if condition != "native" and not changed:
        raise ValueError(f"{condition} has no serialized intervention")
    return {
        "condition": condition,
        "allowed_indices": sorted(allowed),
        "changed_indices": sorted(changed),
        "outside_allowed_bit_exact": True,
    }


def _descendant_body_ids(model, root_id: int) -> set[int]:
    result = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if (
                body_id not in result
                and int(model.body_parentid[body_id]) in result
            ):
                result.add(body_id)
                changed = True
    return result


def contact_body_names(env, body_name: str) -> list[str]:
    model, data = env.sim.model, env.sim.data
    root_id = int(model.body_name2id(body_name))
    descendants = _descendant_body_ids(model, root_id)
    geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendants
    }
    contacts: set[str] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in geom_ids:
            other = second
        elif second in geom_ids:
            other = first
        else:
            continue
        name = model.body_id2name(int(model.geom_bodyid[other]))
        if name:
            contacts.add(str(name))
    return sorted(contacts)


def body_measurement(env, body_name: str) -> dict[str, object]:
    model, data = env.sim.model, env.sim.data
    body_id = int(model.body_name2id(body_name))
    rotation = np.asarray(data.body_xmat[body_id], dtype=float).reshape(3, 3)
    try:
        _, velocity_address = free_joint_addresses(model, body_name)
    except ValueError:
        linear_speed = angular_speed = None
    else:
        velocity = np.asarray(
            data.qvel[velocity_address : velocity_address + 6], dtype=float
        )
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
    return {
        "position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(
            data.body_xquat[body_id], dtype=float
        ).tolist(),
        "tilt_deg": float(
            np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        ),
        "linear_speed_mps": linear_speed,
        "angular_speed_radps": angular_speed,
        "contacts": contact_body_names(env, body_name),
    }


def _predicate_env(env):
    return getattr(env, "env", env)


def predicate_state(env) -> dict[str, bool]:
    evaluator = _predicate_env(env)
    return {
        "close": bool(evaluator._eval_predicate(("close", MICROWAVE_OBJECT))),
        "in": bool(
            evaluator._eval_predicate(("in", TARGET_OBJECT, HEATING_SITE))
        ),
    }


def scene_measurement(env) -> dict[str, object]:
    qpos_address, qvel_address = scalar_joint_addresses(
        env.sim.model, DOOR_JOINT
    )
    return {
        TARGET_BODY: body_measurement(env, TARGET_BODY),
        DISTRACTOR_BODY: body_measurement(env, DISTRACTOR_BODY),
        DOOR_BODY: body_measurement(env, DOOR_BODY),
        "door_joint": {
            "qpos": float(env.sim.data.qpos[qpos_address]),
            "speed": float(abs(env.sim.data.qvel[qvel_address])),
        },
        "predicates": predicate_state(env),
    }


def body_window_stats(
    samples: list[dict[str, object]], body_name: str
) -> dict[str, float | None]:
    origin = np.asarray(samples[0][body_name]["position"], dtype=float)
    linear = [
        float(sample[body_name]["linear_speed_mps"])
        for sample in samples
        if sample[body_name]["linear_speed_mps"] is not None
    ]
    angular = [
        float(sample[body_name]["angular_speed_radps"])
        for sample in samples
        if sample[body_name]["angular_speed_radps"] is not None
    ]
    return {
        "max_translation_drift_m": max(
            float(
                np.linalg.norm(
                    np.asarray(sample[body_name]["position"], dtype=float)
                    - origin
                )
            )
            for sample in samples
        ),
        "max_tilt_deg": max(
            float(sample[body_name]["tilt_deg"]) for sample in samples
        ),
        "max_linear_speed_mps": max(linear, default=None),
        "max_angular_speed_radps": max(angular, default=None),
    }


def door_window_stats(samples: list[dict[str, object]]) -> dict[str, float]:
    initial = float(samples[0]["door_joint"]["qpos"])
    return {
        "max_qpos_drift": max(
            abs(float(sample["door_joint"]["qpos"]) - initial)
            for sample in samples
        ),
        "max_speed": max(
            float(sample["door_joint"]["speed"]) for sample in samples
        ),
        "min_qpos": min(
            float(sample["door_joint"]["qpos"]) for sample in samples
        ),
        "max_qpos": max(
            float(sample["door_joint"]["qpos"]) for sample in samples
        ),
    }


def save_state_bundle(
    path: str | Path, *, condition: str, records: list[dict[str, object]]
) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        group.attrs.update(
            {
                "scenario": SCENE_ID,
                "design_version": DESIGN_VERSION,
                "condition": condition,
                "condition_label": CONDITION_LABEL[condition],
                "task_suite_name": SUITE,
                "task_id": TASK_ID,
                "task_prompt": TASK_PROMPT,
                "task_file": TASK_FILE,
                "count": len(records),
                "pairing_method": PAIRING_METHOD,
                "custom_bddl": False,
                "custom_assets": False,
                "prompt_override": False,
                "native_bddl_sha256": sha256_path(native_bddl_path()),
                "native_goal_signature": TASK_GOAL,
                "native_asset_manifest_sha256": native_asset_manifest_sha256(),
                "intervention_allowlist": json.dumps(
                    INTERVENTION_ALLOWLIST[CONDITION_LABEL[condition]],
                    sort_keys=True,
                ),
            }
        )
        for index, record in enumerate(records):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset(
                "initial_state",
                data=np.asarray(record["initial_state"], dtype=float),
            )
            demo.create_dataset(
                "base_reset_state",
                data=np.asarray(record["base_reset_state"], dtype=float),
            )
            demo.attrs["success"] = True
            for key, value in record["attrs"].items():
                if isinstance(value, (dict, list, tuple)):
                    demo.attrs[key] = json.dumps(value, sort_keys=True)
                else:
                    demo.attrs[key] = value
    return path


class MicrowavePreconditionSequenceTracker:
    """Track the strict Er open-before-insert event order.

    The trace is diagnostic metadata.  It never replaces LIBERO's native task
    success predicate and it is not a collision or safety oracle.
    """

    def __init__(self, env, condition: str, *, policy_start_step: int):
        if condition not in CONDITIONS:
            raise ValueError(f"invalid L3-B3 condition: {condition!r}")
        self.env = env
        self.condition = condition
        self.policy_start_step = int(policy_start_step)
        self.initial = predicate_state(env)
        expected = EXPECTED_INITIAL_PREDICATES[condition]
        if self.initial != expected:
            raise ValueError(
                f"{condition} initial predicates {self.initial} != {expected}"
            )
        self.previous = dict(self.initial)
        self.open_step: int | None = None
        self.insertion_step: int | None = None
        self.reclose_step: int | None = None

    def observe(self, step: int) -> dict[str, bool]:
        current = predicate_state(self.env)
        step = int(step)
        if self.condition == "closed_microwave":
            if (
                self.open_step is None
                and self.previous["close"]
                and not current["close"]
            ):
                self.open_step = step
            if (
                self.open_step is not None
                and self.insertion_step is None
                and step > self.open_step
                and not self.previous["in"]
                and current["in"]
            ):
                self.insertion_step = step
            if (
                self.insertion_step is not None
                and self.reclose_step is None
                and step > self.insertion_step
                and not self.previous["close"]
                and current["close"]
            ):
                self.reclose_step = step
        self.previous = current
        return current

    def finalize(self, *, task_success: bool, final_step: int) -> dict[str, object]:
        final = predicate_state(self.env)
        terminal = scene_measurement(self.env)
        terminal_failures: list[str] = []
        for body, limit in (
            (TARGET_BODY, MAX_TARGET_MUG_TILT_DEG),
            (DISTRACTOR_BODY, MAX_DISTRACTOR_MUG_TILT_DEG),
        ):
            measure = terminal[body]
            if float(measure["tilt_deg"]) > limit:
                terminal_failures.append(f"{body}:tilt")
            if float(measure["linear_speed_mps"]) > MAX_POST_WAIT_LINEAR_SPEED_MPS:
                terminal_failures.append(f"{body}:linear_speed")
            if float(measure["angular_speed_radps"]) > MAX_POST_WAIT_ANGULAR_SPEED_RADPS:
                terminal_failures.append(f"{body}:angular_speed")
            if any(str(name).startswith("robot0_") for name in measure["contacts"]):
                terminal_failures.append(f"{body}:robot_contact")
        if task_success and not any(
            str(name).startswith("microwave_1")
            for name in terminal[TARGET_BODY]["contacts"]
        ):
            terminal_failures.append("target:missing_microwave_support")
        if TABLE_BODY not in terminal[DISTRACTOR_BODY]["contacts"]:
            terminal_failures.append("distractor:missing_table_support")
        if float(terminal["door_joint"]["speed"]) > MAX_DOOR_FINAL_SPEED:
            terminal_failures.append("door:terminal_speed")
        if task_success and final != {"close": True, "in": True}:
            terminal_failures.append("native_goal_not_stable")
        terminal_stable = not terminal_failures
        full_order = bool(
            self.condition == "closed_microwave"
            and self.open_step is not None
            and self.insertion_step is not None
            and self.reclose_step is not None
            and self.open_step < self.insertion_step < self.reclose_step
            and task_success
            and terminal_stable
        )
        if self.condition != "closed_microwave":
            failure_stage = "not_applicable"
        elif self.open_step is None:
            failure_stage = "door_not_opened"
        elif self.insertion_step is None:
            failure_stage = "opened_without_insertion"
        elif self.reclose_step is None:
            failure_stage = "insertion_without_reclose"
        elif not task_success:
            failure_stage = "ordered_trace_without_native_success"
        elif not terminal_stable:
            failure_stage = "native_success_without_terminal_stability"
        else:
            failure_stage = "full_open_insert_reclose"
        return {
            "scenario": SCENE_ID,
            "condition": self.condition,
            "policy_start_step": self.policy_start_step,
            "final_step": int(final_step),
            "initial_predicates": self.initial,
            "final_predicates": final,
            "door_opened": self.open_step is not None,
            "open_step": self.open_step,
            "opened_before_insertion": bool(
                self.open_step is not None
                and self.insertion_step is not None
                and self.open_step < self.insertion_step
            ),
            "insertion_after_open": self.insertion_step is not None,
            "insertion_step": self.insertion_step,
            "reclose_after_insertion": self.reclose_step is not None,
            "reclose_step": self.reclose_step,
            "full_open_insert_reclose": full_order,
            "failure_stage": failure_stage,
            "native_task_success": bool(task_success),
            "stable_native_task_success": bool(task_success and terminal_stable),
            "terminal_stability_pass": terminal_stable,
            "terminal_stability_failures": sorted(set(terminal_failures)),
            "terminal_measurement": terminal,
            "defines_task_success": False,
            "collision_oracle_used": False,
        }
