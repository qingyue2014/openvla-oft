"""Fail-closed native-task and evaluated-artifact gate for L3-A4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from experiments.robot.libero.tasks.l3a4_microwave_common import (
    PORCELAIN_BODY,
    SCENARIO,
    TARGET_BODY,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    resolve_microwave_names,
)


EXPECTED_FIXTURES = {
    "kitchen_table": "kitchen_table",
    "microwave_1": "microwave",
}
EXPECTED_OBJECTS = {
    "porcelain_mug_1": "porcelain_mug",
    "white_yellow_mug_1": "white_yellow_mug",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                return text[start:index + 1]
    raise ValueError(f"unterminated :{name} section")


def _typed_inventory(text: str, section_name: str) -> dict[str, str]:
    return {
        instance: asset_type
        for instance, asset_type in re.findall(
            r"(?m)^\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*$",
            _section(text, section_name),
        )
    }


def validate_native_task(
    native_bddl: Path,
    evaluated_bddl: Path,
    evaluated_prompt: str,
) -> dict[str, object]:
    native = native_bddl.resolve(strict=True)
    evaluated = evaluated_bddl.resolve(strict=True)
    if native.name != TASK_FILE or native.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native task source: {native}")
    if "bddl_files" not in native.parts:
        raise ValueError(f"native task is not under LIBERO bddl_files: {native}")
    if not native.samefile(evaluated):
        raise ValueError(
            "evaluated BDDL is not the selected native LIBERO task: "
            f"evaluated={evaluated}, native={native}"
        )
    text = native.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+([^)]+)\)", text)
    if match is None:
        raise ValueError("native BDDL has no :language prompt")
    prompt = " ".join(match.group(1).split())
    if prompt != TASK_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            f"prompt mismatch: native={prompt!r}, evaluated={evaluated_prompt!r}, "
            f"expected={TASK_PROMPT!r}"
        )
    fixtures = _typed_inventory(text, "fixtures")
    objects = _typed_inventory(text, "objects")
    if fixtures != EXPECTED_FIXTURES:
        raise ValueError(f"native fixture inventory mismatch: {fixtures!r}")
    if objects != EXPECTED_OBJECTS:
        raise ValueError(f"native object inventory mismatch: {objects!r}")
    return {
        "scenario": SCENARIO,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "native_bddl": str(native),
        "evaluated_bddl": str(evaluated),
        "bddl_sha256": sha256(native),
        "prompt": prompt,
        "fixtures": fixtures,
        "objects": objects,
        "custom_bddl": False,
        "custom_assets": False,
    }


def build_manifest(
    evidence: dict[str, object],
    initial_states: Path,
    condition: str,
) -> dict[str, object]:
    states = initial_states.resolve(strict=True)
    if condition not in {"eb", "er", "ec"}:
        raise ValueError(f"condition must be eb/er/ec, got {condition!r}")
    return {
        **evidence,
        "condition": condition,
        "initial_states_path": str(states),
        "initial_states_sha256": sha256(states),
        "allowed_intervention_object": "porcelain_mug_1",
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
) -> dict[str, object]:
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = (TASK_SUITE, TASK_ID, TASK_PROMPT)
    actual = (task_suite_name, int(task_id), task_language)
    if actual != expected:
        raise ValueError(f"L3-A4 native task identity mismatch: {actual!r} != {expected!r}")
    if policy_prompt != TASK_PROMPT:
        raise ValueError("L3-A4 policy prompt is not the exact native prompt")
    task_path = Path(task_bddl).resolve(strict=True)
    if str(task_path) != record.get("native_bddl"):
        raise ValueError("L3-A4 runtime BDDL path does not match preflight")
    if sha256(task_path) != record.get("bddl_sha256"):
        raise ValueError("L3-A4 runtime BDDL bytes changed after preflight")
    states = Path(initial_states_path).resolve(strict=True)
    if str(states) != record.get("initial_states_path"):
        raise ValueError("L3-A4 runtime initial-state artifact does not match preflight")
    if sha256(states) != record.get("initial_states_sha256"):
        raise ValueError("L3-A4 initial-state artifact changed after preflight")
    if record.get("objects") != EXPECTED_OBJECTS or record.get("fixtures") != EXPECTED_FIXTURES:
        raise ValueError("L3-A4 manifest asset inventory mismatch")
    return record


def verify_runtime_asset_inventory(manifest_path: str | Path, model) -> dict[str, str]:
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if record.get("objects") != EXPECTED_OBJECTS:
        raise ValueError("L3-A4 manifest object inventory mismatch")
    for body in (PORCELAIN_BODY, TARGET_BODY):
        try:
            model.body_name2id(body)
        except Exception as exc:
            raise ValueError(f"native object body missing from compiled model: {body}") from exc
    free_joint_bodies = {
        model.body_id2name(int(model.jnt_bodyid[joint_id]))
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id]) == 0
    }
    expected_free_joint_bodies = {PORCELAIN_BODY, TARGET_BODY}
    if free_joint_bodies != expected_free_joint_bodies:
        raise ValueError(
            "compiled movable-object inventory mismatch: "
            f"{sorted(free_joint_bodies)} != "
            f"{sorted(expected_free_joint_bodies)}"
        )
    names = resolve_microwave_names(model)
    return {
        **names,
        "compiled_free_joint_bodies": ",".join(
            sorted(free_joint_bodies)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_bddl", required=True)
    parser.add_argument("--evaluated_bddl", required=True)
    parser.add_argument("--evaluated_prompt", required=True)
    parser.add_argument("--initial_states", required=True)
    parser.add_argument("--condition", choices=("eb", "er", "ec"), required=True)
    parser.add_argument("--out_manifest", required=True)
    args = parser.parse_args()
    evidence = validate_native_task(
        Path(args.native_bddl), Path(args.evaluated_bddl), args.evaluated_prompt
    )
    manifest = build_manifest(evidence, Path(args.initial_states), args.condition)
    output = Path(args.out_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("PASS_L3A4_NATIVE_ONLY_PREFLIGHT")


if __name__ == "__main__":
    main()
