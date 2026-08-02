"""Native-only preflight and runtime identity check for L3-B bowl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    CONDITIONS,
    CONDITION_LABEL,
    DESIGN_VERSION,
    EXPECTED_FIXTURES,
    EXPECTED_FIXTURE_ROOTS,
    EXPECTED_MOVABLE_ROOTS,
    EXPECTED_OBJECTS,
    INITIAL_GATE_VERDICT,
    NATIVE_PREFLIGHT_VERDICT,
    PAIRING_VERDICT,
    RUNTIME_REPLAY_VERDICT,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    inventory_sha256,
    sha256_path,
    validate_native_bddl,
)


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _load_gate(path: str | Path, verdict: str) -> tuple[Path, dict]:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != verdict or record.get("scenario") != SCENE_ID:
        raise ValueError(f"invalid bound gate: {path}")
    if record.get("formal_authorized") is not False:
        raise ValueError(f"bound gate unexpectedly authorizes formal: {path}")
    return path, record


def _validate_state_group(path: Path, condition: str) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError("state artifact task-key mismatch")
        group = handle[TASK_KEY]
        expected = {
            "scenario": SCENE_ID,
            "design_version": DESIGN_VERSION,
            "condition": condition,
            "condition_label": CONDITION_LABEL[condition],
            "task_suite_name": SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "task_file": TASK_FILE,
            "custom_bddl": False,
            "custom_assets": False,
            "prompt_override": False,
        }
        mismatches = {
            key: (_decode(group.attrs.get(key)), value)
            for key, value in expected.items()
            if _decode(group.attrs.get(key)) != value
        }
        if mismatches:
            raise ValueError(f"state artifact metadata mismatch: {mismatches}")


def build_manifest(
    *,
    native_bddl: str | Path,
    evaluated_bddl: str | Path,
    evaluated_prompt: str,
    initial_states: str | Path,
    condition: str,
    pairing_gate: str | Path,
    initial_gate: str | Path,
    runtime_replay_gate: str | Path,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    native = Path(native_bddl).resolve(strict=True)
    evaluated = Path(evaluated_bddl).resolve(strict=True)
    if native != evaluated:
        raise ValueError("custom or replaced BDDL detected")
    bddl = validate_native_bddl(native)
    if evaluated_prompt != TASK_PROMPT:
        raise ValueError("evaluated prompt differs from native prompt")
    states = Path(initial_states).resolve(strict=True)
    _validate_state_group(states, condition)
    pairing_path, pairing = _load_gate(pairing_gate, PAIRING_VERDICT)
    initial_path, initial = _load_gate(initial_gate, INITIAL_GATE_VERDICT)
    runtime_path, runtime = _load_gate(runtime_replay_gate, RUNTIME_REPLAY_VERDICT)
    for gate_name, gate in (("pairing", pairing), ("initial", initial), ("runtime", runtime)):
        binding = gate.get("bindings", gate.get("state_bundles", {})).get(condition, {})
        if (
            Path(binding.get("path", "")).resolve() != states
            or binding.get("sha256") != sha256_path(states)
        ):
            raise ValueError(f"{gate_name} gate state binding mismatch")
    runtime_matches = [
        episode for episode in runtime.get("episodes", []) if episode.get("condition") == condition
    ]
    if not runtime_matches or any(
        item.get("runtime_gate", {}).get("physical_gate_pass") is not True
        for item in runtime_matches
    ):
        raise ValueError("runtime replay lacks per-episode physical PASS")
    return {
        "scenario": SCENE_ID,
        "scene_id": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "condition": condition,
        "condition_label": CONDITION_LABEL[condition],
        "task_suite_name": SUITE,
        "task_id": TASK_ID,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_bddl": str(native),
        "bddl_sha256": bddl["bddl_sha256"],
        "fixtures": EXPECTED_FIXTURES,
        "objects": EXPECTED_OBJECTS,
        "asset_inventory_sha256": inventory_sha256(),
        "initial_states_path": str(states),
        "initial_states_sha256": sha256_path(states),
        "pairing_gate_path": str(pairing_path),
        "pairing_gate_sha256": sha256_path(pairing_path),
        "initial_gate_path": str(initial_path),
        "initial_gate_sha256": sha256_path(initial_path),
        "runtime_replay_gate_path": str(runtime_path),
        "runtime_replay_gate_sha256": sha256_path(runtime_path),
        "custom_asset_xml_audit": "not_applicable_native_assets_only",
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "collision_oracle": "none",
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": NATIVE_PREFLIGHT_VERDICT,
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
    manifest = Path(manifest_path).resolve(strict=True)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("verdict") != NATIVE_PREFLIGHT_VERDICT:
        raise ValueError("invalid L3-B bowl preflight verdict")
    if (task_suite_name, int(task_id), task_language, policy_prompt) != (
        SUITE,
        TASK_ID,
        TASK_PROMPT,
        TASK_PROMPT,
    ):
        raise ValueError("L3-B bowl runtime task/prompt identity mismatch")
    runtime_bddl = Path(task_bddl).resolve(strict=True)
    if str(runtime_bddl) != record.get("native_bddl") or sha256_path(runtime_bddl) != record.get("bddl_sha256"):
        raise ValueError("runtime BDDL differs from preflight")
    states = Path(initial_states_path).resolve(strict=True)
    if str(states) != record.get("initial_states_path") or sha256_path(states) != record.get("initial_states_sha256"):
        raise ValueError("runtime state artifact differs from preflight")
    _validate_state_group(states, record.get("condition", ""))
    for path_key, hash_key, verdict in (
        ("pairing_gate_path", "pairing_gate_sha256", PAIRING_VERDICT),
        ("initial_gate_path", "initial_gate_sha256", INITIAL_GATE_VERDICT),
        ("runtime_replay_gate_path", "runtime_replay_gate_sha256", RUNTIME_REPLAY_VERDICT),
    ):
        path, _ = _load_gate(record[path_key], verdict)
        if sha256_path(path) != record.get(hash_key):
            raise ValueError(f"bound gate changed: {path_key}")
    if (
        record.get("fixtures") != EXPECTED_FIXTURES
        or record.get("objects") != EXPECTED_OBJECTS
        or record.get("asset_inventory_sha256") != inventory_sha256()
    ):
        raise ValueError("native asset inventory mismatch")
    return record


def verify_runtime_asset_inventory(manifest_path: str | Path, model) -> dict[str, int]:
    record = json.loads(Path(manifest_path).resolve(strict=True).read_text(encoding="utf-8"))
    if record.get("verdict") != NATIVE_PREFLIGHT_VERDICT:
        raise ValueError("runtime inventory check lacks valid preflight")
    movable = {
        str(model.body_id2name(int(model.jnt_bodyid[joint_id])))
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == 0
    }
    if movable != EXPECTED_MOVABLE_ROOTS:
        raise ValueError(f"compiled movable inventory mismatch: {sorted(movable)}")
    world_children = {
        str(model.body_id2name(body_id))
        for body_id in range(1, int(model.nbody))
        if int(model.body_parentid[body_id]) == 0
    }
    fixtures = world_children - movable - {"robot0_base"}
    if fixtures != EXPECTED_FIXTURE_ROOTS:
        raise ValueError(f"compiled fixture inventory mismatch: {sorted(fixtures)}")
    return {
        name: int(model.body_name2id(name))
        for name in sorted(EXPECTED_MOVABLE_ROOTS | EXPECTED_FIXTURE_ROOTS)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-bddl", required=True)
    parser.add_argument("--evaluated-bddl", required=True)
    parser.add_argument("--evaluated-prompt", required=True)
    parser.add_argument("--initial-states", required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--pairing-gate", required=True)
    parser.add_argument("--initial-gate", required=True)
    parser.add_argument("--runtime-replay-gate", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = build_manifest(
        native_bddl=args.native_bddl,
        evaluated_bddl=args.evaluated_bddl,
        evaluated_prompt=args.evaluated_prompt,
        initial_states=args.initial_states,
        condition=args.condition,
        pairing_gate=args.pairing_gate,
        initial_gate=args.initial_gate,
        runtime_replay_gate=args.runtime_replay_gate,
    )
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(NATIVE_PREFLIGHT_VERDICT)


if __name__ == "__main__":
    main()
