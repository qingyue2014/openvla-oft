"""Fail-closed native-only preflight for LIBERO L3-A3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BOTTLE_BODY,
    BDDL_PROMPT,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
    PLATE_BODY,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    parse_native_bddl,
    sha256_path,
)

VERDICT = "PASS_L3A3_NATIVE_ONLY_PREFLIGHT"


def validate_native_task(
    native_bddl: str | Path,
    evaluated_bddl: str | Path,
    evaluated_prompt: str,
) -> dict:
    native = Path(native_bddl).resolve()
    evaluated = Path(evaluated_bddl).resolve()
    if native != evaluated:
        raise ValueError(
            "evaluated BDDL is not the selected native LIBERO task; copied or "
            "project-local BDDL files are forbidden"
        )
    evidence = parse_native_bddl(native)
    if evidence["prompt"] != BDDL_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            f"prompt mismatch: expected BDDL={BDDL_PROMPT!r}, "
            f"benchmark={TASK_PROMPT!r}, "
            f"native={evidence['prompt']!r}, evaluated={evaluated_prompt!r}"
        )
    if evidence["fixtures"] != EXPECTED_FIXTURES:
        raise ValueError(
            f"fixture inventory mismatch: {evidence['fixtures']!r} != {EXPECTED_FIXTURES!r}"
        )
    if evidence["objects"] != EXPECTED_OBJECTS:
        raise ValueError(
            f"object inventory mismatch: {evidence['objects']!r} != {EXPECTED_OBJECTS!r}"
        )
    if native.name != TASK_FILE or native.parent.name != SUITE:
        raise ValueError(
            f"wrong native task identity: expected {SUITE}/{TASK_FILE}, got {native}"
        )
    evidence.update(
        {
            "scenario": SCENE_ID,
            "task_suite_name": SUITE,
            "task_id": TASK_ID,
            "task_file": TASK_FILE,
            "bddl_prompt": BDDL_PROMPT,
            "prompt": TASK_PROMPT,
            "native_bddl": str(native),
            "evaluated_bddl": str(evaluated),
            "asset_inventory": {
                "fixtures": evidence["fixtures"],
                "objects": evidence["objects"],
            },
            "asset_inventory_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "fixtures": evidence["fixtures"],
                        "objects": evidence["objects"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "verdict": VERDICT,
        }
    )
    return evidence


def build_manifest(evidence: dict, initial_states: str | Path, condition: str) -> dict:
    condition = condition.lower()
    if condition not in {"eb", "er", "ec"}:
        raise ValueError(f"condition must be eb/er/ec, got {condition!r}")
    states = Path(initial_states).resolve(strict=True)
    return {
        **evidence,
        "condition": condition,
        "initial_states_path": str(states),
        "initial_states_sha256": sha256_path(states),
        "allowed_intervention_object": "wine_bottle_1",
        "custom_bddl": False,
        "custom_assets": False,
        "prompt_override": False,
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
) -> dict:
    """Revalidate native identity and exact artifact bytes inside evaluation."""
    manifest = Path(manifest_path).resolve(strict=True)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("verdict") != VERDICT:
        raise ValueError(f"invalid L3-A3 native preflight verdict: {manifest}")
    actual_identity = (task_suite_name, int(task_id), task_language)
    expected_identity = (SUITE, TASK_ID, TASK_PROMPT)
    if actual_identity != expected_identity:
        raise ValueError(
            f"L3-A3 native task identity mismatch: "
            f"{actual_identity!r} != {expected_identity!r}"
        )
    if policy_prompt != TASK_PROMPT:
        raise ValueError("L3-A3 policy prompt is not the exact native benchmark prompt")
    runtime_bddl = Path(task_bddl).resolve(strict=True)
    if str(runtime_bddl) != record.get("native_bddl"):
        raise ValueError("L3-A3 runtime BDDL path does not match preflight")
    if sha256_path(runtime_bddl) != record.get("bddl_sha256"):
        raise ValueError("L3-A3 runtime BDDL bytes changed after preflight")
    runtime_states = Path(initial_states_path).resolve(strict=True)
    if str(runtime_states) != record.get("initial_states_path"):
        raise ValueError("L3-A3 runtime state artifact does not match preflight")
    if sha256_path(runtime_states) != record.get("initial_states_sha256"):
        raise ValueError("L3-A3 runtime state artifact changed after preflight")
    if (
        record.get("fixtures") != EXPECTED_FIXTURES
        or record.get("objects") != EXPECTED_OBJECTS
    ):
        raise ValueError("L3-A3 manifest native asset inventory mismatch")
    expected_inventory_sha = hashlib.sha256(
        json.dumps(
            {"fixtures": EXPECTED_FIXTURES, "objects": EXPECTED_OBJECTS},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if record.get("asset_inventory_sha256") != expected_inventory_sha:
        raise ValueError("L3-A3 manifest inventory hash mismatch")
    return record


def verify_runtime_asset_inventory(
    manifest_path: str | Path, model
) -> dict[str, str]:
    """Check required native object/fixture roots in the compiled model."""
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        record.get("fixtures") != EXPECTED_FIXTURES
        or record.get("objects") != EXPECTED_OBJECTS
    ):
        raise ValueError("L3-A3 manifest inventory is stale or modified")
    expected_object_bodies = {
        "akita_black_bowl_1_main",
        "cream_cheese_1_main",
        BOTTLE_BODY,
        PLATE_BODY,
    }
    expected_fixture_roots = {
        "table",
        "wooden_cabinet_1_main",
        "flat_stove_1_main",
        "wine_rack_1_main",
    }
    resolved = {}
    for body in sorted(expected_object_bodies | expected_fixture_roots):
        try:
            body_id = int(model.body_name2id(body))
        except Exception as exc:
            raise ValueError(
                f"native L3-A3 body missing from compiled model: {body}"
            ) from exc
        resolved[body] = str(body_id)
    compiled_free_roots = {
        model.body_id2name(int(model.jnt_bodyid[joint_id]))
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == 0
    }
    if compiled_free_roots != expected_object_bodies:
        raise ValueError(
            "L3-A3 compiled movable-object inventory mismatch: "
            f"compiled={sorted(compiled_free_roots)}, "
            f"expected={sorted(expected_object_bodies)}"
        )
    world_children = {
        model.body_id2name(body_id)
        for body_id in range(1, model.nbody)
        if int(model.body_parentid[body_id]) == 0
    }
    compiled_fixture_roots = (
        world_children - compiled_free_roots - {"robot0_base"}
    )
    if compiled_fixture_roots != expected_fixture_roots:
        raise ValueError(
            "L3-A3 compiled fixture-root inventory mismatch: "
            f"compiled={sorted(compiled_fixture_roots)}, "
            f"expected={sorted(expected_fixture_roots)}"
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_bddl", required=True)
    parser.add_argument("--evaluated_bddl", required=True)
    parser.add_argument("--evaluated_prompt", required=True)
    parser.add_argument("--initial_states", required=True)
    parser.add_argument("--condition", choices=("eb", "er", "ec"), required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    evidence = validate_native_task(
        args.native_bddl, args.evaluated_bddl, args.evaluated_prompt
    )
    evidence = build_manifest(evidence, args.initial_states, args.condition)
    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(evidence["verdict"])


if __name__ == "__main__":
    main()
