"""Fail-closed structural, physical, visual, and pairing validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    BOTTLE_BODY,
    BOWL_BODY,
    CONDITIONS,
    CONDITION_INTERVENTION_BODY,
    CONDITION_INTERVENTION_KIND,
    CONDITION_LABEL,
    DESIGN_VERSION,
    DRAWER_BODY,
    EXPECTED_FIXTURE_ROOTS,
    EXPECTED_INITIAL_PREDICATES,
    INITIAL_GATE_VERDICT,
    MAX_BOTTLE_TILT_DEG,
    MAX_BOWL_TILT_DEG,
    PAIRING_METHOD,
    PAIRING_VERDICT,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    sha256_path,
    state_sha256,
)
from experiments.robot.libero.tasks.validate_l3b_bowl_design import (
    validate_spec as validate_design_preregistration,
)


POLICY_IMAGE_SPECS = {
    "agentview_raw_256": (256, 256),
    "wrist_raw_256": (256, 256),
    "agentview_pi05_224": (224, 224),
    "wrist_pi05_224": (224, 224),
}
DESIGN_PATH = Path(__file__).with_name("l3b_bowl_v1_design_prereg.json")


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _json_attr(attrs, name: str):
    if name not in attrs:
        raise ValueError(f"missing required attribute {name!r}")
    value = _decode(attrs[name])
    return json.loads(value) if isinstance(value, str) else value


def _validate_images(paths: dict, context: str) -> dict:
    if set(paths) != set(POLICY_IMAGE_SPECS):
        raise ValueError(f"{context} policy image inventory mismatch")
    result = {}
    for label, shape in POLICY_IMAGE_SPECS.items():
        path = Path(paths[label]).resolve(strict=True)
        image = np.asarray(imageio.imread(path))
        if tuple(image.shape[:2]) != shape or image.shape[2:] != (3,):
            raise ValueError(f"{context} {label} has invalid shape {image.shape}")
        if float(np.std(image)) < 1.0:
            raise ValueError(f"{context} {label} is blank or near-constant")
        result[label] = sha256_path(path)
    return result


def _validate_physical(attrs, condition: str, context: str) -> dict:
    if bool(attrs.get("physical_gate_pass", False)) is not True:
        raise ValueError(f"{context} lacks physical gate PASS")
    pre = _json_attr(attrs, "formal_pre_wait_json")
    first = _json_attr(attrs, "formal_first_policy_json")
    formal = _json_attr(attrs, "formal_window_stats_json")
    hold = _json_attr(attrs, "post_wait_hold_stats_json")
    for state_name, record in (("pre", pre), ("first", first)):
        if set(record) != {
            BOWL_BODY,
            BOTTLE_BODY,
            DRAWER_BODY,
            "drawer_joint",
            "predicates",
        }:
            raise ValueError(f"{context} {state_name} physical inventory mismatch")
    if first["predicates"] != EXPECTED_INITIAL_PREDICATES[condition]:
        raise ValueError(f"{context} first-policy predicate mismatch")
    for label, stats in (("formal", formal), ("hold", hold)):
        if float(stats[BOWL_BODY]["max_tilt_deg"]) > MAX_BOWL_TILT_DEG:
            raise ValueError(f"{context} {label} bowl tilt exceeds 1 degree")
        if float(stats[BOTTLE_BODY]["max_tilt_deg"]) > MAX_BOTTLE_TILT_DEG:
            raise ValueError(f"{context} {label} bottle tilt exceeds limit")
    bowl_contacts = [str(value) for value in first[BOWL_BODY]["contacts"]]
    bottle_contacts = [str(value) for value in first[BOTTLE_BODY]["contacts"]]
    if condition == "prerequisite_done":
        if not any(value.startswith("white_cabinet_1_cabinet_bottom") for value in bowl_contacts):
            raise ValueError(f"{context} Ec bowl lacks drawer support")
    elif "table" not in bowl_contacts:
        raise ValueError(f"{context} bowl lacks table support")
    if "table" not in bottle_contacts:
        raise ValueError(f"{context} bottle lacks table support")
    if any(value.startswith(("robot0_", "wine_bottle_1")) for value in bowl_contacts):
        raise ValueError(f"{context} bowl has forbidden first-policy contact")
    return {
        "pre_wait": pre,
        "first_policy": first,
        "formal_window_stats": formal,
        "post_wait_hold_stats": hold,
    }


def _expected_group(condition: str) -> dict:
    return {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "condition": condition,
        "condition_label": CONDITION_LABEL[condition],
        "task_suite_name": SUITE,
        "task_id": TASK_ID,
        "task_prompt": TASK_PROMPT,
        "task_file": TASK_FILE,
        "pairing_method": PAIRING_METHOD,
        "custom_bddl": False,
        "custom_assets": False,
        "prompt_override": False,
    }


def validate_one(
    path: str | Path,
    condition: str,
    *,
    design_preregistration: str | Path = DESIGN_PATH,
) -> list[dict]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    path = Path(path).resolve(strict=True)
    design = validate_design_preregistration(design_preregistration)
    official_state_indices = design["official_state_indices"]
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError(f"{path} has unexpected task keys")
        group = handle[TASK_KEY]
        mismatches = {
            key: (_decode(group.attrs.get(key)), expected)
            for key, expected in _expected_group(condition).items()
            if _decode(group.attrs.get(key)) != expected
        }
        if mismatches:
            raise ValueError(f"{path} group metadata mismatch: {mismatches}")
        count = int(group.attrs.get("count", -1))
        if count != len(official_state_indices) or count != len(group):
            raise ValueError(f"{path} count mismatch")
        records = []
        for index in range(count):
            name = f"demo_{index}"
            if name not in group:
                raise ValueError(f"{path} missing {name}")
            demo = group[name]
            context = f"{path}:{name}"
            if set(demo) != {"initial_state", "base_reset_state"}:
                raise ValueError(f"{context} state dataset inventory mismatch")
            initial = np.asarray(demo["initial_state"][:], dtype=float)
            base = np.asarray(demo["base_reset_state"][:], dtype=float)
            if initial.ndim != 1 or initial.shape != base.shape:
                raise ValueError(f"{context} state shape mismatch")
            expected_attrs = {
                "condition": condition,
                "condition_label": CONDITION_LABEL[condition],
                "design_version": DESIGN_VERSION,
                "episode_index": index,
                "native_init_state_index": official_state_indices[index],
                "base_state_sha256": state_sha256(base),
                "initial_state_sha256": state_sha256(initial),
                "intervention_body": CONDITION_INTERVENTION_BODY[condition],
                "intervention_kind": CONDITION_INTERVENTION_KIND[condition],
                "asset_inventory_changed": False,
                "prompt_changed": False,
                "bddl_changed": False,
            }
            mismatches = {
                key: (_decode(demo.attrs.get(key)), value)
                for key, value in expected_attrs.items()
                if _decode(demo.attrs.get(key)) != value
            }
            if mismatches:
                raise ValueError(f"{context} metadata mismatch: {mismatches}")
            intervention = _json_attr(demo.attrs, "intervention_json")
            if intervention.get("intervention_kind") != CONDITION_INTERVENTION_KIND[condition]:
                raise ValueError(f"{context} intervention metadata mismatch")
            allowed = np.zeros(initial.shape, dtype=bool)
            if condition == "native":
                if not np.array_equal(initial, base):
                    raise ValueError(f"{context} Eb is not bit-exact native")
            elif condition == "premature_close":
                qpos_index = int(intervention.get("qpos_flat_index", -1))
                qvel_index = int(intervention.get("qvel_flat_index", -1))
                if min(qpos_index, qvel_index) < 0:
                    raise ValueError(f"{context} Er joint indices missing")
                allowed[[qpos_index, qvel_index]] = True
            else:
                qpos_start = int(intervention.get("qpos_flat_start", -1))
                qvel_start = int(intervention.get("qvel_flat_start", -1))
                if min(qpos_start, qvel_start) < 0:
                    raise ValueError(f"{context} Ec free-joint slices missing")
                allowed[qpos_start : qpos_start + 7] = True
                allowed[qvel_start : qvel_start + 6] = True
            if condition != "native":
                if not np.array_equal(initial[~allowed], base[~allowed]):
                    raise ValueError(f"{context} changes state outside allowed intervention")
                if np.array_equal(initial[allowed], base[allowed]):
                    raise ValueError(f"{context} intervention changed no serialized value")
            names = _json_attr(demo.attrs, "fixture_replay_bodies_json")
            positions = np.asarray(demo.attrs["fixture_replay_positions"], dtype=float)
            quaternions = np.asarray(demo.attrs["fixture_replay_quaternions"], dtype=float)
            if set(names) != EXPECTED_FIXTURE_ROOTS:
                raise ValueError(f"{context} fixture inventory mismatch")
            if positions.shape != (len(names), 3) or quaternions.shape != (len(names), 4):
                raise ValueError(f"{context} fixture replay pose shape mismatch")
            physical = _validate_physical(demo.attrs, condition, context)
            image_paths = _json_attr(demo.attrs, "policy_images_json")
            records.append(
                {
                    "initial": initial,
                    "base": base,
                    "native_index": official_state_indices[index],
                    "fixture_names": names,
                    "fixture_positions": positions,
                    "fixture_quaternions": quaternions,
                    "intervention": intervention,
                    "physical": physical,
                    "image_paths": image_paths,
                    "image_hashes": _validate_images(image_paths, context),
                }
            )
    return records


def _validate_manifest(
    path: str | Path,
    bindings: dict[str, Path],
    *,
    design_preregistration: str | Path,
) -> dict:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    design = validate_design_preregistration(design_preregistration)
    official_state_indices = design["official_state_indices"]
    if (
        record.get("verdict") != INITIAL_GATE_VERDICT
        or record.get("scenario") != SCENE_ID
        or int(record.get("design_version", -1)) != DESIGN_VERSION
        or record.get("native_suite") != SUITE
        or int(record.get("native_task_id", -1)) != TASK_ID
        or record.get("native_prompt") != TASK_PROMPT
        or record.get("official_native_state_indices") != official_state_indices
        or record.get("design_preregistration") != design
    ):
        raise ValueError("initial manifest identity/design mismatch")
    for forbidden in ("custom_assets", "custom_bddl", "prompt_changed", "asset_inventory_changed"):
        if record.get(forbidden) is not False:
            raise ValueError(f"initial manifest prohibited field: {forbidden}")
    for condition, state_path in bindings.items():
        binding = record.get("state_bundles", {}).get(condition, {})
        if (
            Path(binding.get("path", "")).resolve() != state_path
            or binding.get("sha256") != sha256_path(state_path)
        ):
            raise ValueError(f"initial manifest state binding mismatch: {condition}")
    for episode in record.get("episodes", []):
        for condition in ("premature_close", "prerequisite_done"):
            if episode.get("visibility", {}).get(condition, {}).get("passed") is not True:
                raise ValueError(f"manifest visibility gate missing for {condition}")
    return record


def validate_pairing(
    eb_path,
    er_path,
    ec_path,
    *,
    initial_manifest,
    design_preregistration: str | Path = DESIGN_PATH,
) -> dict:
    paths = {
        "native": Path(eb_path).resolve(strict=True),
        "premature_close": Path(er_path).resolve(strict=True),
        "prerequisite_done": Path(ec_path).resolve(strict=True),
    }
    design = validate_design_preregistration(design_preregistration)
    official_state_indices = design["official_state_indices"]
    bundles = {
        condition: validate_one(
            path,
            condition,
            design_preregistration=design_preregistration,
        )
        for condition, path in paths.items()
    }
    count = len(bundles["native"])
    if any(len(records) != count for records in bundles.values()):
        raise ValueError("paired condition counts differ")
    for index in range(count):
        eb, er, ec = (bundles[name][index] for name in CONDITIONS)
        if not np.array_equal(eb["base"], er["base"]) or not np.array_equal(eb["base"], ec["base"]):
            raise ValueError(f"paired base mismatch at demo_{index}")
        if not np.array_equal(eb["initial"], eb["base"]):
            raise ValueError(f"Eb/base mismatch at demo_{index}")
        if len({eb["native_index"], er["native_index"], ec["native_index"]}) != 1:
            raise ValueError(f"native index mismatch at demo_{index}")
        if eb["fixture_names"] != er["fixture_names"] or eb["fixture_names"] != ec["fixture_names"]:
            raise ValueError(f"fixture name mismatch at demo_{index}")
        for field in ("fixture_positions", "fixture_quaternions"):
            if not np.array_equal(eb[field], er[field]) or not np.array_equal(eb[field], ec[field]):
                raise ValueError(f"fixture pose mismatch at demo_{index}")
    _validate_manifest(
        initial_manifest,
        paths,
        design_preregistration=design_preregistration,
    )
    return {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "verdict": PAIRING_VERDICT,
        "count": count,
        "official_native_state_indices": official_state_indices,
        "design_preregistration": design,
        "initial_manifest": {
            "path": str(Path(initial_manifest).resolve(strict=True)),
            "sha256": sha256_path(initial_manifest),
        },
        "bindings": {
            condition: {"path": str(path), "sha256": sha256_path(path)}
            for condition, path in paths.items()
        },
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "formal_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--initial-manifest", required=True)
    parser.add_argument(
        "--design-preregistration",
        default=str(DESIGN_PATH),
    )
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = validate_pairing(
        args.eb,
        args.er,
        args.ec,
        initial_manifest=args.initial_manifest,
        design_preregistration=args.design_preregistration,
    )
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{PAIRING_VERDICT} count={result['count']}")


if __name__ == "__main__":
    main()
