"""Native-only preflight for the provisional L3-B moka order experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITION_INTERVENTION_BODY,
    CONDITIONS,
    DESIGN_VERSION,
    EXPECTED_FIXTURE_ROOTS,
    EXPECTED_FIXTURES,
    EXPECTED_MOVABLE_ROOTS,
    EXPECTED_OBJECTS,
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
from experiments.robot.libero.tasks.validate_l3b_moka_state_bundles import (
    INITIAL_GATE_VERDICT,
    VERDICT as PAIRING_VERDICT,
)
from experiments.robot.libero.tasks.validate_l3b_moka_runtime_replay import (
    VERDICT as RUNTIME_REPLAY_VERDICT,
)


VERDICT = "PASS_L3B_MOKA_NATIVE_ONLY_PREFLIGHT"


def validate_native_task(
    native_bddl: str | Path,
    evaluated_bddl: str | Path,
    evaluated_prompt: str,
) -> dict:
    native = Path(native_bddl).resolve(strict=True)
    evaluated = Path(evaluated_bddl).resolve(strict=True)
    if native != evaluated:
        raise ValueError(
            "evaluated BDDL is not the selected native LIBERO task"
        )
    evidence = validate_native_bddl(native)
    if evaluated_prompt != TASK_PROMPT:
        raise ValueError("evaluated prompt is not the exact native prompt")
    return {
        **evidence,
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "task_suite_name": SUITE,
        "task_id": TASK_ID,
        "task_file": TASK_FILE,
        "prompt": TASK_PROMPT,
        "native_bddl": str(native),
        "evaluated_bddl": str(evaluated),
        "asset_inventory_sha256": inventory_sha256(),
        "verdict": VERDICT,
    }


def _load_gate(path: str | Path, expected_verdict: str) -> tuple[Path, dict]:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != expected_verdict:
        raise ValueError(f"{path} lacks verdict {expected_verdict}")
    if (
        record.get("scenario") != SCENE_ID
        or int(record.get("design_version", -1)) != DESIGN_VERSION
    ):
        raise ValueError(f"{path} belongs to a different scenario")
    return path, record


def _validate_state_group(path: Path, condition: str) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError("state artifact task key mismatch")
        group = handle[TASK_KEY]
        expected = {
            "scenario": SCENE_ID,
            "design_version": DESIGN_VERSION,
            "condition": condition,
            "task_suite_name": SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "task_file": TASK_FILE,
            "custom_bddl": False,
            "custom_assets": False,
            "prompt_override": False,
        }
        for key, value in expected.items():
            actual = group.attrs.get(key)
            if isinstance(actual, bytes):
                actual = actual.decode()
            if actual != value:
                raise ValueError(
                    f"state artifact {key} mismatch: {actual!r} != {value!r}"
                )


def build_manifest(
    evidence: dict,
    initial_states: str | Path,
    condition: str,
    *,
    pairing_gate: str | Path,
    initial_gate: str | Path,
    runtime_replay_gate: str | Path,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"invalid L3-B moka condition: {condition!r}")
    states = Path(initial_states).resolve(strict=True)
    _validate_state_group(states, condition)
    pairing_path, pairing = _load_gate(pairing_gate, PAIRING_VERDICT)
    initial_path, initial = _load_gate(initial_gate, INITIAL_GATE_VERDICT)
    runtime_path, runtime = _load_gate(
        runtime_replay_gate, RUNTIME_REPLAY_VERDICT
    )
    binding = pairing.get("bindings", {}).get(condition, {})
    if (
        Path(binding.get("path", "")).resolve() != states
        or binding.get("sha256") != sha256_path(states)
    ):
        raise ValueError("state artifact does not match the pairing gate")
    initial_binding = initial.get("state_bundles", {}).get(condition, {})
    if (
        Path(initial_binding.get("path", "")).resolve() != states
        or initial_binding.get("sha256") != sha256_path(states)
    ):
        raise ValueError("state artifact does not match the initial gate")
    runtime_matches = [
        episode
        for episode in runtime.get("episodes", [])
        if episode.get("condition") == condition
    ]
    if not runtime_matches:
        raise ValueError("runtime replay gate omits this condition")
    if any(
        Path(episode.get("state_artifact", "")).resolve() != states
        or episode.get("state_artifact_sha256") != sha256_path(states)
        or episode.get("runtime_gate", {}).get("physical_gate_pass") is not True
        for episode in runtime_matches
    ):
        raise ValueError("runtime replay evidence does not bind this state")
    return {
        **evidence,
        "condition": condition,
        "initial_states_path": str(states),
        "initial_states_sha256": sha256_path(states),
        "pairing_gate_path": str(pairing_path),
        "pairing_gate_sha256": sha256_path(pairing_path),
        "initial_gate_path": str(initial_path),
        "initial_gate_sha256": sha256_path(initial_path),
        "runtime_replay_gate_path": str(runtime_path),
        "runtime_replay_gate_sha256": sha256_path(runtime_path),
        "allowed_intervention_body": (
            CONDITION_INTERVENTION_BODY[condition] or ""
        ),
        "custom_bddl": False,
        "custom_assets": False,
        "prompt_override": False,
        "asset_inventory_changed": False,
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
    if record.get("verdict") != VERDICT:
        raise ValueError("invalid L3-B moka preflight verdict")
    if (
        task_suite_name,
        int(task_id),
        task_language,
        policy_prompt,
    ) != (SUITE, TASK_ID, TASK_PROMPT, TASK_PROMPT):
        raise ValueError("L3-B moka runtime task or prompt identity mismatch")
    runtime_bddl = Path(task_bddl).resolve(strict=True)
    if (
        str(runtime_bddl) != record.get("native_bddl")
        or sha256_path(runtime_bddl) != record.get("bddl_sha256")
    ):
        raise ValueError("L3-B moka runtime BDDL differs from preflight")
    states = Path(initial_states_path).resolve(strict=True)
    if (
        str(states) != record.get("initial_states_path")
        or sha256_path(states) != record.get("initial_states_sha256")
    ):
        raise ValueError("L3-B moka runtime state artifact differs from preflight")
    _validate_state_group(states, record.get("condition", ""))
    for path_key, hash_key, verdict in (
        ("pairing_gate_path", "pairing_gate_sha256", PAIRING_VERDICT),
        ("initial_gate_path", "initial_gate_sha256", INITIAL_GATE_VERDICT),
        (
            "runtime_replay_gate_path",
            "runtime_replay_gate_sha256",
            RUNTIME_REPLAY_VERDICT,
        ),
    ):
        gate_path, _ = _load_gate(record[path_key], verdict)
        if sha256_path(gate_path) != record.get(hash_key):
            raise ValueError(f"L3-B moka bound {path_key} changed")
    if (
        record.get("fixtures") != EXPECTED_FIXTURES
        or record.get("objects") != EXPECTED_OBJECTS
        or record.get("asset_inventory_sha256") != inventory_sha256()
    ):
        raise ValueError("L3-B moka native asset inventory mismatch")
    return record


def verify_runtime_asset_inventory(
    manifest_path: str | Path,
    model,
) -> dict[str, int]:
    record = json.loads(
        Path(manifest_path).resolve(strict=True).read_text(encoding="utf-8")
    )
    if record.get("verdict") != VERDICT:
        raise ValueError("L3-B moka runtime inventory check lacks preflight")
    compiled_free_roots = {
        str(model.body_id2name(int(model.jnt_bodyid[joint_id])))
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == 0
    }
    if compiled_free_roots != EXPECTED_MOVABLE_ROOTS:
        raise ValueError(
            "L3-B moka compiled movable inventory mismatch: "
            f"{sorted(compiled_free_roots)} != {sorted(EXPECTED_MOVABLE_ROOTS)}"
        )
    world_children = {
        str(model.body_id2name(body_id))
        for body_id in range(1, int(model.nbody))
        if int(model.body_parentid[body_id]) == 0
    }
    fixtures = world_children - compiled_free_roots - {"robot0_base"}
    if fixtures != EXPECTED_FIXTURE_ROOTS:
        raise ValueError(
            "L3-B moka compiled fixture inventory mismatch: "
            f"{sorted(fixtures)} != {sorted(EXPECTED_FIXTURE_ROOTS)}"
        )
    return {
        body: int(model.body_name2id(body))
        for body in sorted(EXPECTED_MOVABLE_ROOTS | EXPECTED_FIXTURE_ROOTS)
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
    evidence = validate_native_task(
        args.native_bddl,
        args.evaluated_bddl,
        args.evaluated_prompt,
    )
    result = build_manifest(
        evidence,
        args.initial_states,
        args.condition,
        pairing_gate=args.pairing_gate,
        initial_gate=args.initial_gate,
        runtime_replay_gate=args.runtime_replay_gate,
    )
    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result["verdict"])


if __name__ == "__main__":
    main()
