"""Fail-closed validation of the provisional L3-B moka state bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITIONS,
    CONDITION_LABEL,
    CONDITION_INTERVENTION_BODY,
    CONDITION_SLOT,
    DESIGN_VERSION,
    EXPECTED_FIXTURE_ROOTS,
    MAX_RECEPTACLE_TILT_DEG,
    POT_1,
    POT_2,
    POT_BODIES,
    SCENE_ID,
    STOVE_BODY,
    SUITE,
    TABLE_BODY,
    TASK_FILE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    sha256_path,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v7_design import (
    validate_spec as validate_design_preregistration,
)


VERDICT = "PASS_L3B_MOKA_EXACT_SERIALIZED_PAIRING"
INITIAL_GATE_VERDICT = "PASS_L3B_MOKA_INITIAL_PHYSICAL_GATES"
PAIRING_METHOD = "same_official_native_state_same_moka_pot_alternate_slot"
POLICY_IMAGE_SPECS = {
    "agentview_raw_256": (256, 256),
    "wrist_raw_256": (256, 256),
    "agentview_pi05_224": (224, 224),
    "wrist_pi05_224": (224, 224),
}
DESIGN_PREREGISTRATION = Path(__file__).with_name(
    "l3b_moka_v7_design_prereg.json"
)


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _json_attr(attrs, name: str):
    if name not in attrs:
        raise ValueError(f"missing required attribute {name!r}")
    value = _decode(attrs[name])
    return json.loads(value) if isinstance(value, str) else value


def _state_sha256(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=float).tobytes()).hexdigest()


def _support_ok(measure: dict, expected: str) -> bool:
    contacts = [str(value) for value in measure.get("contacts", [])]
    if expected == "stove":
        return any(value.startswith("flat_stove_1_") for value in contacts)
    return TABLE_BODY in contacts


def _validate_policy_images(paths: dict, context: str) -> dict[str, str]:
    if set(paths) != set(POLICY_IMAGE_SPECS):
        raise ValueError(
            f"{context} policy image inventory mismatch: {sorted(paths)}"
        )
    hashes = {}
    for label, expected_shape in POLICY_IMAGE_SPECS.items():
        path = Path(paths[label]).resolve(strict=True)
        image = np.asarray(imageio.imread(path))
        if tuple(image.shape[:2]) != expected_shape:
            raise ValueError(
                f"{context} {label} shape {image.shape[:2]} != {expected_shape}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{context} {label} is not RGB")
        if float(np.std(image)) < 1.0:
            raise ValueError(f"{context} {label} is blank or near-constant")
        hashes[label] = sha256_path(path)
    return hashes


def _validate_physical_record(attrs, condition: str, context: str) -> dict:
    if not bool(attrs.get("physical_gate_pass", False)):
        raise ValueError(f"{context} lacks a physical gate PASS")
    pre = _json_attr(attrs, "formal_pre_wait_json")
    first = _json_attr(attrs, "formal_first_policy_json")
    formal_stats = _json_attr(attrs, "formal_window_stats_json")
    hold_stats = _json_attr(attrs, "post_wait_hold_stats_json")
    for body in (*POT_BODIES, STOVE_BODY):
        for label, record in (
            ("pre-wait", pre),
            ("first-policy", first),
            ("formal-window", formal_stats),
            ("post-wait-hold", hold_stats),
        ):
            if body not in record:
                raise ValueError(f"{context} {label} omits {body}")
        if float(formal_stats[body]["max_tilt_deg"]) > MAX_RECEPTACLE_TILT_DEG:
            raise ValueError(f"{context} {body} exceeds formal tilt limit")
        if float(hold_stats[body]["max_tilt_deg"]) > MAX_RECEPTACLE_TILT_DEG:
            raise ValueError(f"{context} {body} exceeds hold tilt limit")

    placed_body = CONDITION_INTERVENTION_BODY[condition]
    for body in POT_BODIES:
        expected_support = "stove" if body == placed_body else "table"
        if not _support_ok(first[body], expected_support):
            raise ValueError(
                f"{context} {body} lacks {expected_support} support "
                "at the first policy frame"
            )
        other = POT_2 if body == POT_1 else POT_1
        other_prefix = other.removesuffix("_main")
        if any(
            str(contact).startswith(other_prefix)
            for contact in first[body].get("contacts", [])
        ):
            raise ValueError(f"{context} begins with moka-pot contact")
    return {
        "pre_wait": pre,
        "first_policy": first,
        "formal_window_stats": formal_stats,
        "post_wait_hold_stats": hold_stats,
    }


def _validate_one(path: str | Path, condition: str) -> list[dict]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    path = Path(path).resolve(strict=True)
    pool = validate_design_preregistration(DESIGN_PREREGISTRATION)
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError(f"{path} has unexpected task keys: {sorted(handle)}")
        group = handle[TASK_KEY]
        expected_group = {
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
        mismatches = {
            key: (_decode(group.attrs.get(key)), expected)
            for key, expected in expected_group.items()
            if _decode(group.attrs.get(key)) != expected
        }
        if mismatches:
            raise ValueError(f"{path} metadata mismatch: {mismatches}")
        expected_pool_attrs = {
            "pool_preregistration_id": pool["preregistration_id"],
            "pool_preregistration_sha256": pool["sha256"],
            "official_native_state_indices_json": json.dumps(
                pool["official_state_indices"]
            ),
        }
        pool_mismatches = {
            key: (_decode(group.attrs.get(key)), expected)
            for key, expected in expected_pool_attrs.items()
            if _decode(group.attrs.get(key)) != expected
        }
        if pool_mismatches:
            raise ValueError(
                f"{path} capability-conditioned pool mismatch: "
                f"{pool_mismatches}"
            )
        count = int(group.attrs.get("count", -1))
        if count != len(group) or count != pool["count"]:
            raise ValueError(f"{path} has invalid demo count")

        records = []
        for index in range(count):
            name = f"demo_{index}"
            if name not in group:
                raise ValueError(f"{path} missing contiguous {name}")
            demo = group[name]
            context = f"{path}:{name}"
            if set(demo).issuperset({"initial_state", "base_reset_state"}) is False:
                raise ValueError(f"{context} lacks serialized state datasets")
            initial = np.asarray(demo["initial_state"][:], dtype=float)
            base = np.asarray(demo["base_reset_state"][:], dtype=float)
            if initial.shape != base.shape or initial.ndim != 1:
                raise ValueError(f"{context} state shapes are invalid")
            if _decode(demo.attrs.get("condition", "")) != condition:
                raise ValueError(f"{context} condition attribute mismatch")
            if int(demo.attrs.get("episode_index", -1)) != index:
                raise ValueError(f"{context} episode index mismatch")
            if int(demo.attrs.get("native_init_state_index", -1)) != (
                pool["official_state_indices"][index]
            ):
                raise ValueError(
                    f"{context} official native state index is not locked"
                )
            if _decode(demo.attrs.get("base_state_sha256", "")) != _state_sha256(base):
                raise ValueError(f"{context} has a stale base state hash")
            if _decode(demo.attrs.get("initial_state_sha256", "")) != _state_sha256(initial):
                raise ValueError(f"{context} has a stale initial state hash")
            for flag in ("asset_inventory_changed", "prompt_changed", "bddl_changed"):
                if bool(demo.attrs.get(flag, True)):
                    raise ValueError(f"{context} prohibited flag {flag}=True")

            intervention = _json_attr(demo.attrs, "intervention_json")
            expected_body = CONDITION_INTERVENTION_BODY[condition] or ""
            expected_slot = CONDITION_SLOT[condition] or ""
            if int(demo.attrs.get("design_version", -1)) != DESIGN_VERSION:
                raise ValueError(f"{context} design version mismatch")
            if _decode(demo.attrs.get("condition_label", "")) != (
                CONDITION_LABEL[condition]
            ):
                raise ValueError(f"{context} external condition label mismatch")
            if _decode(demo.attrs.get("target_slot", "")) != expected_slot:
                raise ValueError(f"{context} target slot mismatch")
            if _decode(demo.attrs.get("intervention_body", "")) != expected_body:
                raise ValueError(f"{context} intervention body mismatch")
            if intervention.get("intervention_body", "") != expected_body:
                raise ValueError(f"{context} intervention metadata mismatch")
            if intervention.get("target_slot", "") != expected_slot:
                raise ValueError(f"{context} intervention slot mismatch")
            if condition == "native":
                if not np.array_equal(initial, base):
                    raise ValueError(f"{context} native state is not bit-exact")
                qpos_start = qvel_start = None
            else:
                qpos_start = int(intervention.get("qpos_flat_start", -1))
                qvel_start = int(intervention.get("qvel_flat_start", -1))
                if qpos_start < 0 or qvel_start < 0:
                    raise ValueError(f"{context} lacks intervention state slices")
                allowed = np.zeros(initial.shape, dtype=bool)
                allowed[qpos_start : qpos_start + 7] = True
                allowed[qvel_start : qvel_start + 6] = True
                if not np.array_equal(initial[~allowed], base[~allowed]):
                    raise ValueError(
                        f"{context} changes state outside the one moka free joint"
                    )
                if np.array_equal(initial[allowed], base[allowed]):
                    raise ValueError(f"{context} intervention changed no state")

            fixtures = _json_attr(demo.attrs, "fixture_replay_bodies_json")
            fixture_positions = np.asarray(
                demo.attrs.get("fixture_replay_positions"), dtype=float
            )
            fixture_quaternions = np.asarray(
                demo.attrs.get("fixture_replay_quaternions"), dtype=float
            )
            if set(fixtures) != EXPECTED_FIXTURE_ROOTS:
                raise ValueError(f"{context} fixture replay inventory mismatch")
            if fixture_positions.shape != (len(fixtures), 3):
                raise ValueError(f"{context} fixture position shape mismatch")
            if fixture_quaternions.shape != (len(fixtures), 4):
                raise ValueError(f"{context} fixture quaternion shape mismatch")
            physical = _validate_physical_record(demo.attrs, condition, context)
            image_paths = _json_attr(demo.attrs, "policy_images_json")
            image_hashes = _validate_policy_images(image_paths, context)
            records.append(
                {
                    "initial": initial,
                    "base": base,
                    "qpos_start": qpos_start,
                    "qvel_start": qvel_start,
                    "target_xy": np.asarray(
                        intervention.get("target_xy", []), dtype=float
                    ),
                    "slots": _json_attr(demo.attrs, "slot_targets_json"),
                    "fixture_names": fixtures,
                    "fixture_positions": fixture_positions,
                    "fixture_quaternions": fixture_quaternions,
                    "native_init_state_index": int(
                        demo.attrs.get("native_init_state_index", -1)
                    ),
                    "image_paths": image_paths,
                    "image_hashes": image_hashes,
                    "physical": physical,
                }
            )
    return records


def _validate_initial_manifest(
    path: str | Path,
    state_paths: dict[str, Path],
) -> dict:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != INITIAL_GATE_VERDICT:
        raise ValueError("L3-B initial physical gate verdict is missing")
    if (
        record.get("scenario") != SCENE_ID
        or int(record.get("design_version", -1)) != DESIGN_VERSION
        or record.get("native_suite") != SUITE
        or int(record.get("native_task_id", -1)) != TASK_ID
        or record.get("native_prompt") != TASK_PROMPT
        or record.get("custom_assets") is not False
        or record.get("custom_bddl") is not False
        or record.get("prompt_changed") is not False
        or record.get("asset_inventory_changed") is not False
    ):
        raise ValueError("L3-B initial gate native-only identity mismatch")
    pool = validate_design_preregistration(DESIGN_PREREGISTRATION)
    if record.get("official_native_state_indices") != pool[
        "official_state_indices"
    ] or record.get("pool_preregistration") != pool:
        raise ValueError(
            "L3-B initial gate capability-conditioned pool mismatch"
        )
    for condition, state_path in state_paths.items():
        binding = record.get("state_bundles", {}).get(condition, {})
        if Path(binding.get("path", "")).resolve() != state_path:
            raise ValueError(f"initial manifest {condition} state path mismatch")
        if binding.get("sha256") != sha256_path(state_path):
            raise ValueError(f"initial manifest {condition} state hash mismatch")
    return record


def validate_pairing(
    native_path: str | Path,
    near_first_path: str | Path,
    far_first_path: str | Path,
    *,
    initial_manifest: str | Path,
) -> dict:
    state_paths = {
        "native": Path(native_path).resolve(strict=True),
        "near_first": Path(near_first_path).resolve(strict=True),
        "far_first": Path(far_first_path).resolve(strict=True),
    }
    bundles = {
        condition: _validate_one(path, condition)
        for condition, path in state_paths.items()
    }
    counts = {name: len(records) for name, records in bundles.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"paired condition counts differ: {counts}")
    for index in range(counts["native"]):
        native, near, far = (
            bundles[name][index]
            for name in ("native", "near_first", "far_first")
        )
        if not np.array_equal(native["base"], near["base"]) or not np.array_equal(
            native["base"], far["base"]
        ):
            raise ValueError(f"paired base mismatch at demo_{index}")
        if not np.array_equal(native["initial"], native["base"]):
            raise ValueError(f"native/base mismatch at demo_{index}")
        if len(
            {
                native["native_init_state_index"],
                near["native_init_state_index"],
                far["native_init_state_index"],
            }
        ) != 1:
            raise ValueError(f"native state index mismatch at demo_{index}")
        if (
            native["fixture_names"] != near["fixture_names"]
            or native["fixture_names"] != far["fixture_names"]
        ):
            raise ValueError(f"fixture body mismatch at demo_{index}")
        for field in ("fixture_positions", "fixture_quaternions"):
            if not np.array_equal(native[field], near[field]) or not np.array_equal(
                native[field], far[field]
            ):
                raise ValueError(f"paired {field} mismatch at demo_{index}")
        if (
            near["qpos_start"] != far["qpos_start"]
            or near["qvel_start"] != far["qvel_start"]
        ):
            raise ValueError(
                f"near/far do not modify the same moka pot at demo_{index}"
            )
        if near["target_xy"].shape != (2,) or far["target_xy"].shape != (2,):
            raise ValueError(f"near/far target metadata invalid at demo_{index}")
        expected_near = np.asarray(near["slots"]["near_xyz"][:2], dtype=float)
        expected_far = np.asarray(far["slots"]["far_xyz"][:2], dtype=float)
        if not np.allclose(near["target_xy"], expected_near, atol=1e-12):
            raise ValueError(f"near target binding mismatch at demo_{index}")
        if not np.allclose(far["target_xy"], expected_far, atol=1e-12):
            raise ValueError(f"far target binding mismatch at demo_{index}")
        if np.allclose(near["target_xy"], far["target_xy"], atol=1e-9):
            raise ValueError(f"near/far target slots coincide at demo_{index}")

    initial = _validate_initial_manifest(initial_manifest, state_paths)
    pool = validate_design_preregistration(DESIGN_PREREGISTRATION)
    return {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene_labels": {
            "Eb": "native",
            "Er": "near_first",
            "Ec": "far_first",
        },
        "verdict": VERDICT,
        "count": counts["native"],
        "official_native_state_indices": pool["official_state_indices"],
        "pool_preregistration": pool,
        "initial_manifest": {
            "path": str(Path(initial_manifest).resolve()),
            "sha256": sha256_path(initial_manifest),
            "verdict": initial["verdict"],
        },
        "bindings": {
            condition: {
                "path": str(path),
                "sha256": sha256_path(path),
                "condition": condition,
            }
            for condition, path in state_paths.items()
        },
        "policy_images": {
            condition: [
                {
                    "paths": item["image_paths"],
                    "sha256": item["image_hashes"],
                }
                for item in bundles[condition]
            ]
            for condition in CONDITIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--near-first", required=True)
    parser.add_argument("--far-first", required=True)
    parser.add_argument("--initial-manifest", required=True)
    parser.add_argument("--out-json")
    args = parser.parse_args()
    result = validate_pairing(
        args.native,
        args.near_first,
        args.far_first,
        initial_manifest=args.initial_manifest,
    )
    if args.out_json:
        destination = Path(args.out_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(result["verdict"])


if __name__ == "__main__":
    main()
