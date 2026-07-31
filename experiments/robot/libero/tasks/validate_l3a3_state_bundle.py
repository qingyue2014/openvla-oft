"""Validate L3-A3 serialized-state semantics and exact Eb/Er/Ec pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BOTTLE_BODY,
    PLATE_BODY,
    SCENE_ID,
    SUITE,
    TABLE_BODY,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    sha256_path,
)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _json_attr(attrs, name: str):
    if name not in attrs:
        raise ValueError(f"missing required attribute {name!r}")
    value = _decode(attrs[name])
    return json.loads(value) if isinstance(value, str) else value


def artifact_binding(path: str | Path) -> str:
    path = Path(path).resolve(strict=True)
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        payload = {
            "artifact_sha256": sha256_path(path),
            "scenario": _decode(group.attrs.get("scenario", "")),
            "condition": _decode(group.attrs.get("condition", "")),
            "task_suite_name": _decode(group.attrs.get("task_suite_name", "")),
            "task_id": int(group.attrs.get("task_id", -1)),
            "task_prompt": _decode(group.attrs.get("task_prompt", "")),
            "bddl": _decode(group.attrs.get("bddl", "")),
            "bddl_sha256": _decode(group.attrs.get("bddl_sha256", "")),
            "seed": int(group.attrs.get("seed", -1)),
            "count": len(group),
            "pairing_method": _decode(group.attrs.get("pairing_method", "")),
            "er_relative_xy": _json_attr(group.attrs, "er_relative_xy"),
            "ec_relative_xy": _json_attr(group.attrs, "ec_relative_xy"),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_one(path: str | Path, expected_condition: str) -> list[dict]:
    path = Path(path).resolve(strict=True)
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError(f"{path} has unexpected task keys: {sorted(handle)}")
        group = handle[TASK_KEY]
        expected_group = {
            "scenario": SCENE_ID,
            "condition": expected_condition,
            "task_suite_name": SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "pairing_method": "shared_settled_native_base_bottle_free_joint_only",
        }
        mismatches = {
            key: (_decode(group.attrs.get(key)), expected)
            for key, expected in expected_group.items()
            if _decode(group.attrs.get(key)) != expected
        }
        if mismatches:
            raise ValueError(f"{path} metadata mismatch: {mismatches}")
        if int(group.attrs.get("count", -1)) != len(group):
            raise ValueError(f"{path} count metadata does not match demos")
        records: list[dict] = []
        for index in range(len(group)):
            name = f"demo_{index}"
            if name not in group:
                raise ValueError(f"{path} missing contiguous {name}")
            demo = group[name]
            for dataset in ("initial_state", "base_reset_state"):
                if dataset not in demo:
                    raise ValueError(f"{path}:{name} missing {dataset}")
            initial = np.asarray(demo["initial_state"][:])
            base = np.asarray(demo["base_reset_state"][:])
            if initial.shape != base.shape:
                raise ValueError(f"{path}:{name} initial/base shape mismatch")
            qpos_start = int(demo.attrs.get("bottle_qpos_flat_start", -1))
            qvel_start = int(demo.attrs.get("bottle_qvel_flat_start", -1))
            if qpos_start < 0 or qvel_start < 0:
                raise ValueError(f"{path}:{name} missing bottle state slices")
            allowed = np.zeros(initial.shape, dtype=bool)
            allowed[qpos_start:qpos_start + 7] = True
            allowed[qvel_start:qvel_start + 6] = True
            if not np.array_equal(initial[~allowed], base[~allowed]):
                changed = np.flatnonzero(initial[~allowed] != base[~allowed])
                raise ValueError(
                    f"{path}:{name} changes non-bottle serialized state "
                    f"({len(changed)} entries)"
                )
            if expected_condition == "Eb":
                if not np.array_equal(initial, base):
                    raise ValueError(f"{path}:{name} Eb is not bit-exact native base")
            elif np.array_equal(initial[allowed], base[allowed]):
                raise ValueError(f"{path}:{name} intervention did not change bottle state")

            if not bool(demo.attrs.get("physical_gate_pass", False)):
                raise ValueError(f"{path}:{name} lacks physical PASS")
            pre = _json_attr(demo.attrs, "formal_pre_wait_json")
            post = _json_attr(demo.attrs, "formal_post_wait_json")
            stats = _json_attr(demo.attrs, "formal_window_stats_json")
            for body in (PLATE_BODY, BOTTLE_BODY):
                if body not in pre or body not in post or body not in stats:
                    raise ValueError(f"{path}:{name} incomplete per-body physical record")
            expected_support = PLATE_BODY if expected_condition == "Er" else TABLE_BODY
            if expected_support not in post[BOTTLE_BODY]["contacts"]:
                raise ValueError(
                    f"{path}:{name} bottle lacks expected post-wait support "
                    f"{expected_support}"
                )
            if expected_condition == "Er" and TABLE_BODY in post[BOTTLE_BODY]["contacts"]:
                raise ValueError(f"{path}:{name} Er bottle contacts table")
            fixture_names = _json_attr(demo.attrs, "fixture_replay_bodies_json")
            fixture_positions = np.asarray(
                demo.attrs.get("fixture_replay_positions"), dtype=float
            )
            fixture_quaternions = np.asarray(
                demo.attrs.get("fixture_replay_quaternions"), dtype=float
            )
            if fixture_positions.shape != (len(fixture_names), 3):
                raise ValueError(f"{path}:{name} invalid fixture positions")
            if fixture_quaternions.shape != (len(fixture_names), 4):
                raise ValueError(f"{path}:{name} invalid fixture quaternions")
            expected_base_sha = hashlib.sha256(base.tobytes()).hexdigest()
            if _decode(demo.attrs.get("base_state_sha256")) != expected_base_sha:
                raise ValueError(f"{path}:{name} stale base-state hash")
            records.append(
                {
                    "initial": initial,
                    "base": base,
                    "qpos_start": qpos_start,
                    "qvel_start": qvel_start,
                    "fixture_names": fixture_names,
                    "fixture_positions": fixture_positions,
                    "fixture_quaternions": fixture_quaternions,
                    "native_init_state_index": int(
                        demo.attrs.get("native_init_state_index", -1)
                    ),
                }
            )
    return records


def validate_pairing(
    eb_path: str | Path, er_path: str | Path, ec_path: str | Path
) -> dict:
    bundles = {
        "Eb": _validate_one(eb_path, "Eb"),
        "Er": _validate_one(er_path, "Er"),
        "Ec": _validate_one(ec_path, "Ec"),
    }
    counts = {condition: len(records) for condition, records in bundles.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"paired condition counts differ: {counts}")
    for index in range(counts["Eb"]):
        eb, er, ec = (bundles[name][index] for name in ("Eb", "Er", "Ec"))
        if not np.array_equal(eb["base"], er["base"]) or not np.array_equal(
            eb["base"], ec["base"]
        ):
            raise ValueError(f"paired base_reset_state mismatch at demo_{index}")
        for key in ("qpos_start", "qvel_start", "native_init_state_index"):
            if len({eb[key], er[key], ec[key]}) != 1:
                raise ValueError(f"paired {key} mismatch at demo_{index}")
        if eb["fixture_names"] != er["fixture_names"] or eb["fixture_names"] != ec[
            "fixture_names"
        ]:
            raise ValueError(f"fixture replay body mismatch at demo_{index}")
        for field in ("fixture_positions", "fixture_quaternions"):
            if not np.array_equal(eb[field], er[field]) or not np.array_equal(
                eb[field], ec[field]
            ):
                raise ValueError(f"paired {field} mismatch at demo_{index}")
    return {
        "verdict": "PASS_L3A3_EXACT_SERIALIZED_PAIRING",
        "count": counts["Eb"],
        "bindings": {
            "Eb": json.loads(artifact_binding(eb_path)),
            "Er": json.loads(artifact_binding(er_path)),
            "Ec": json.loads(artifact_binding(ec_path)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--out_json")
    parser.add_argument("--print_binding", choices=("eb", "er", "ec"))
    args = parser.parse_args()
    paths = {"eb": args.eb, "er": args.er, "ec": args.ec}
    if args.print_binding:
        print(artifact_binding(paths[args.print_binding]))
        return
    result = validate_pairing(args.eb, args.er, args.ec)
    if args.out_json:
        destination = Path(args.out_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["verdict"])


if __name__ == "__main__":
    main()
