#!/usr/bin/env python3
"""Hard native-only preflight for the L1-C1 paired state bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


NATIVE_SUITE = "libero_spatial"
NATIVE_TASK_ID = 2
NATIVE_PROMPT = (
    "pick up the black bowl from table center and place it on the plate"
)


def _attribute_text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_state_file(path: Path, condition: str, context: dict) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing {condition} state file: {path}")
    key = NATIVE_PROMPT.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        if list(handle.keys()) != [key]:
            raise RuntimeError(
                f"{condition} prompt group mismatch: {list(handle.keys())}"
            )
        group = handle[key]
        required = {
            "native_suite": context["native_suite"],
            "native_task_id": str(context["native_task_id"]),
            "native_prompt": context["native_prompt"],
            "native_bddl": context["native_bddl"],
            "native_bddl_sha256": context["native_bddl_sha256"],
            "native_asset_inventory_sha256": context[
                "native_asset_inventory_sha256"
            ],
            "custom_assets": "[]",
        }
        for name, expected in required.items():
            actual = group.attrs.get(name)
            if actual is None or _attribute_text(actual) != expected:
                raise RuntimeError(
                    f"{condition} {name} mismatch: "
                    f"{_attribute_text(actual) if actual is not None else None!r} "
                    f"!= {expected!r}"
                )
        demos = sorted(group.keys())
        if not demos:
            raise RuntimeError(f"{condition} state file is empty: {path}")
        states = [group[name]["initial_state"][()] for name in demos]
        variant = _attribute_text(group.attrs["state_intervention_variant"])
    return {
        "states": states,
        "record": {
            "state_file": str(path),
            "state_sha256": _file_sha256(path),
            "num_states": len(states),
            "prompt": context["native_prompt"],
            "bddl": context["native_bddl"],
            "asset_inventory_sha256": context[
                "native_asset_inventory_sha256"
            ],
            "state_intervention_variant": variant,
        },
    }


def run(args) -> None:
    from experiments.robot.libero.tasks.generate_l1c1_initial_states import (
        OffScreenRenderEnv,
        benchmark,
        get_libero_path,
        json_sha256,
        native_task_context,
        runtime_asset_inventory,
    )

    suite = benchmark.get_benchmark_dict()[NATIVE_SUITE]()
    task = suite.get_task(NATIVE_TASK_ID)
    if task.language != NATIVE_PROMPT:
        raise RuntimeError(
            f"Native prompt mismatch: {task.language!r} != {NATIVE_PROMPT!r}"
        )
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=256, camera_widths=256
    )
    evaluated = {}
    try:
        env.reset()
        context = native_task_context(
            NATIVE_SUITE, NATIVE_TASK_ID, task, bddl, env
        )
        for condition, path_value in (
            ("Eb", args.eb_states),
            ("Er", args.er_states),
            ("Ec", args.ec_states),
        ):
            validated = validate_state_file(
                Path(path_value), condition, context
            )
            env.set_init_state(validated["states"][0])
            env.sim.forward()
            inventory_hash = json_sha256(runtime_asset_inventory(env.sim.model))
            if inventory_hash != context["native_asset_inventory_sha256"]:
                raise RuntimeError(
                    f"{condition} evaluated asset inventory mismatch"
                )
            evaluated[condition.lower()] = validated["record"]
    finally:
        env.close()

    manifest = {
        "schema_version": 1,
        "verdict": "PASS_NATIVE_ONLY_PREFLIGHT",
        **context,
        "custom_assets": [],
        "evaluated_conditions": evaluated,
        "allowed_intervention": (
            "serialized free-joint pose/state of the native second black bowl "
            "only; task assets, BDDL, prompt, goal, and semantics unchanged"
        ),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    fixtures = ", ".join(
        row["asset_class"]
        for row in context["native_declared_asset_inventory"]["fixtures"]
    )
    objects = ", ".join(
        row["asset_class"]
        for row in context["native_declared_asset_inventory"]["objects"]
    )
    report = [
        "# L1-C1 Native-Only Preflight",
        "",
        "- Verdict: **PASS_NATIVE_ONLY_PREFLIGHT**",
        f"- Native suite/task: `{NATIVE_SUITE}` / `{NATIVE_TASK_ID}`",
        f"- Native prompt: `{context['native_prompt']}`",
        "- Native BDDL-declared language: "
        f"`{context['native_bddl_declared_language']}`",
        f"- Native BDDL: `{context['native_bddl']}`",
        f"- Native BDDL SHA-256: `{context['native_bddl_sha256']}`",
        f"- Declared fixtures: {fixtures}",
        f"- Declared objects: {objects}",
        "- Runtime asset inventory SHA-256: "
        f"`{context['native_asset_inventory_sha256']}`",
        "- Custom-asset XML audit: not applicable; no custom assets are present.",
        "- Eb/Er/Ec prompt, BDDL metadata, and compiled asset inventories are identical.",
        "- Allowed intervention: serialized pose/state of the native second black bowl only.",
    ]
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report) + "\n")
    print("Verdict: PASS_NATIVE_ONLY_PREFLIGHT")
    print(f"JSON: {out_json}")
    print(f"Report: {out_report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb_states", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_report", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
