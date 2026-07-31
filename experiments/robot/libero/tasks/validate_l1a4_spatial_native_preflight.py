"""Native-only hard gate for the spatial-suite L1-A4 replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from experiments.robot.libero.tasks.validate_l1a3_native_preflight import (
    _inventory,
    _prompt,
    _read_attr,
    _resolve_libero_root,
    _sha256,
)


TASK_SUITE = "libero_spatial"
TASK_ID = 0
TASK_FILE = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate.bddl"
)
TASK_PROMPT = (
    "pick up the black bowl between the plate and the ramekin "
    "and place it on the plate"
)
BDDL_PROMPT = (
    "Pick the akita black bowl between the plate and the ramekin "
    "and place it on the plate"
)
EXPECTED_FIXTURES = {
    "main_table": "table",
    "wooden_cabinet_1": "wooden_cabinet",
    "flat_stove_1": "flat_stove",
}
EXPECTED_OBJECTS = {
    "akita_black_bowl_1": "akita_black_bowl",
    "akita_black_bowl_2": "akita_black_bowl",
    "cookies_1": "cookies",
    "glazed_rim_porcelain_ramekin_1": "glazed_rim_porcelain_ramekin",
    "plate_1": "plate",
}
VERDICT = "PASS_L1A4_SPATIAL_NATIVE_ONLY_PREFLIGHT"
INTERVENTION_ID = "l1a4_spatial_native_flat_postwait_v5"
PHYSICAL_GATE_VERDICT = "PASS_L1A4_SPATIAL_POSTWAIT_PHYSICAL_GATE"
FORMAL_WAIT_STEPS = 10
MAX_RECEPTACLE_TILT_DEG = 1.0


def resolve_native_bddl() -> Path:
    path = (
        _resolve_libero_root()
        / "libero"
        / "libero"
        / "bddl_files"
        / TASK_SUITE
        / TASK_FILE
    )
    return path.resolve(strict=True)


def validate_native_task(
    native_bddl: Path,
    evaluated_bddl: Path,
    evaluated_prompt: str,
) -> dict[str, object]:
    native = native_bddl.resolve(strict=True)
    evaluated = evaluated_bddl.resolve(strict=True)
    if native.name != TASK_FILE or native.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native task source: {native}")
    if "bddl_files" not in native.parts or not native.samefile(evaluated):
        raise ValueError(
            f"evaluated BDDL is not the selected native task: {evaluated}"
        )
    text = native.read_text(encoding="utf-8")
    bddl_prompt = _prompt(text)
    if bddl_prompt != BDDL_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            "prompt mismatch: "
            f"native_bddl={bddl_prompt!r}, native_benchmark={TASK_PROMPT!r}, "
            f"evaluated={evaluated_prompt!r}"
        )
    fixtures = _inventory(text, "fixtures")
    objects = _inventory(text, "objects")
    if fixtures != EXPECTED_FIXTURES or objects != EXPECTED_OBJECTS:
        raise ValueError(
            "native asset inventory mismatch: "
            f"fixtures={fixtures}, objects={objects}"
        )
    inventory_sha = hashlib.sha256(
        json.dumps(
            {"fixtures": fixtures, "objects": objects},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "verdict": VERDICT,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": TASK_FILE,
        "prompt": TASK_PROMPT,
        "bddl_prompt": bddl_prompt,
        "native_bddl": str(native),
        "evaluated_bddl": str(evaluated),
        "bddl_sha256": _sha256(native),
        "fixtures": fixtures,
        "objects": objects,
        "asset_inventory_sha256": inventory_sha,
    }


def write_preflight(
    manifest_path: Path, report_path: Path
) -> dict[str, object]:
    native = resolve_native_bddl()
    record = validate_native_task(native, native, TASK_PROMPT)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            (
                "# L1-A4 Spatial Native-Only Preflight",
                "",
                f"- Verdict: **{VERDICT}**",
                f"- Selected native task: `{TASK_SUITE}/{TASK_FILE}`",
                f"- Native task id: `{TASK_ID}`",
                f"- Original benchmark prompt: `{TASK_PROMPT}`",
                f"- Native BDDL `:language`: `{record['bddl_prompt']}`",
                f"- Native BDDL: `{record['native_bddl']}`",
                f"- BDDL SHA-256: `{record['bddl_sha256']}`",
                "- Native fixtures: "
                + ", ".join(
                    f"`{key}:{value}`"
                    for key, value in EXPECTED_FIXTURES.items()
                ),
                "- Native objects: "
                + ", ".join(
                    f"`{key}:{value}`"
                    for key, value in EXPECTED_OBJECTS.items()
                ),
                "- Custom assets: `none`",
                "- Custom BDDL: `none`",
                "- Prompt override: `none`",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Verdict: {VERDICT}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    return record


def verify_state_file(
    state_path: Path, record: Mapping[str, object]
) -> None:
    import h5py

    state_path = state_path.resolve(strict=True)
    with h5py.File(state_path, "r") as handle:
        required = {
            "native_only",
            "task_suite_name",
            "task_id",
            "task_file",
            "native_prompt",
            "native_bddl_sha256",
            "asset_inventory_sha256",
            "condition",
            "intervention_id",
            "physical_gate_verdict",
            "formal_wait_steps",
            "max_receptacle_tilt_deg",
        }
        missing = sorted(required - set(handle.attrs))
        if missing:
            raise ValueError(
                f"{state_path} missing native-only attrs: {missing}"
            )
        if not bool(_read_attr(handle.attrs, "native_only")):
            raise ValueError(f"{state_path} is not marked native_only")
        expected = {
            "task_suite_name": record["task_suite_name"],
            "task_id": record["task_id"],
            "task_file": record["task_file"],
            "native_prompt": record["prompt"],
            "native_bddl_sha256": record["bddl_sha256"],
            "asset_inventory_sha256": record["asset_inventory_sha256"],
            "intervention_id": INTERVENTION_ID,
            "physical_gate_verdict": PHYSICAL_GATE_VERDICT,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "max_receptacle_tilt_deg": MAX_RECEPTACLE_TILT_DEG,
        }
        mismatches = {
            name: (_read_attr(handle.attrs, name), value)
            for name, value in expected.items()
            if _read_attr(handle.attrs, name) != value
        }
        if mismatches:
            raise ValueError(
                f"{state_path} native-only metadata mismatch: {mismatches}"
            )


def verify_evaluation_request(
    manifest_path: str,
    *,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    task_bddl: str,
    policy_prompt: str,
    initial_states_path: str,
) -> None:
    """Fail closed before the evaluator creates publishable evidence."""
    path = Path(manifest_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("verdict") != VERDICT:
        raise ValueError(f"invalid L1-A4 spatial preflight verdict in {path}")
    fresh = validate_native_task(
        resolve_native_bddl(), Path(task_bddl), task_language
    )
    for key in (
        "task_suite_name",
        "task_id",
        "task_file",
        "prompt",
        "bddl_sha256",
        "asset_inventory_sha256",
    ):
        if record.get(key) != fresh.get(key):
            raise ValueError(
                f"stale or mismatched L1-A4 spatial preflight field: {key}"
            )
    if (
        task_suite_name != TASK_SUITE
        or int(task_id) != TASK_ID
        or task_language != TASK_PROMPT
        or policy_prompt != TASK_PROMPT
    ):
        raise ValueError(
            "L1-A4 spatial runtime task/prompt mismatch: "
            f"suite={task_suite_name!r}, task_id={task_id}, "
            f"task={task_language!r}, policy={policy_prompt!r}"
        )
    if initial_states_path == "DEFAULT":
        raise ValueError(
            "L1-A4 spatial requires a preflight-stamped paired HDF5 state file"
        )
    verify_state_file(Path(initial_states_path), fresh)
    print(f"Verdict: {VERDICT} (runtime recheck)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "experiments/robot/libero/tasks/"
            "l1a4_spatial_native_preflight.json"
        ),
    )
    parser.add_argument(
        "--report",
        default="experiments/logs/l1a4_spatial_native_preflight.md",
    )
    args = parser.parse_args()
    write_preflight(Path(args.manifest), Path(args.report))


if __name__ == "__main__":
    main()
