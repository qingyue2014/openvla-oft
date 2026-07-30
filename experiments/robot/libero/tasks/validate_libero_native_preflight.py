#!/usr/bin/env python3
"""Hard preflight for native-only LIBERO experiments.

The preflight resolves the selected task through LIBERO's benchmark registry,
records the exact benchmark prompt and official BDDL source, inventories every
declared fixture/object, and fingerprints the MuJoCo body/geom inventory built
from that official BDDL.  A project-local BDDL override is an unconditional
failure, even when it happens to declare the same object classes.

The JSON output is also consumed by policy evaluation so the environment that
actually receives model actions must match this exact native contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STANDARD_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def seed_native_layout(seed: int) -> None:
    """Seed layout sampling before a LIBERO environment constructs its model."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _balanced_section(text: str, section: str) -> str:
    start = text.find(f"(:{section}")
    if start < 0:
        raise ValueError(f"Native BDDL is missing (:{section} ...)")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"Unbalanced (:{section} ...) section in native BDDL")


def declared_asset_inventory(bddl_text: str) -> list[dict[str, str]]:
    """Return the exact fixture/object declarations in a native BDDL file."""
    inventory: list[dict[str, str]] = []
    for kind in ("fixtures", "objects"):
        section = _balanced_section(bddl_text, kind)
        lines = []
        for raw_line in section.splitlines()[1:]:
            line = raw_line.split(";", 1)[0].strip()
            if line and line != ")":
                lines.append(line)
        declaration_text = " ".join(lines)
        for match in re.finditer(
            r"((?:[A-Za-z0-9_]+\s+)+)-\s*([A-Za-z0-9_]+)",
            declaration_text,
        ):
            asset_type = match.group(2)
            for name in match.group(1).split():
                inventory.append(
                    {"kind": kind[:-1], "name": name, "type": asset_type}
                )
    if not inventory:
        raise ValueError("Native BDDL asset inventory is empty")
    return sorted(inventory, key=lambda row: (row["kind"], row["name"], row["type"]))


def bddl_language(bddl_text: str) -> str:
    match = re.search(r"\(:language\s+([^)]+)\)", bddl_text, flags=re.IGNORECASE)
    return " ".join(match.group(1).split()) if match else ""


def _model_names(model, kind: str) -> list[str]:
    count = int(getattr(model, f"n{kind}"))
    resolver = getattr(model, f"{kind}_id2name")
    return sorted(
        name
        for index in range(count)
        if (name := resolver(index))
    )


def resolve_native_task(task_suite_name: str, task_id: int):
    if task_suite_name not in STANDARD_SUITES:
        raise ValueError(
            f"Native-only preflight rejects suite {task_suite_name!r}; "
            f"allowed suites are {STANDARD_SUITES}"
        )
    from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
        _import_libero_modules,
    )

    benchmark, get_libero_path, _ = _import_libero_modules()
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = suite.get_task(int(task_id))
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    ).resolve()
    if not bddl_path.is_file():
        raise FileNotFoundError(f"Native BDDL does not exist: {bddl_path}")
    return suite, task, bddl_path


def build_native_contract(
    task_suite_name: str,
    task_id: int,
    seed: int,
) -> dict[str, Any]:
    """Build the auditable contract from the official native task only."""
    suite, task, bddl_path = resolve_native_task(task_suite_name, task_id)
    del suite
    bddl_payload = bddl_path.read_bytes()
    bddl_text = bddl_payload.decode("utf-8")
    assets = declared_asset_inventory(bddl_text)

    seed_native_layout(seed)
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    try:
        env.seed(0)
        env.reset()
        body_names = _model_names(env.sim.model, "body")
        geom_names = _model_names(env.sim.model, "geom")
    finally:
        env.close()

    return {
        "schema_version": 1,
        "task_suite_name": task_suite_name,
        "task_id": int(task_id),
        "task_language": task.language,
        "task_problem_folder": task.problem_folder,
        "task_bddl_file": task.bddl_file,
        "native_bddl_source": str(bddl_path),
        "native_bddl_sha256": _sha256_bytes(bddl_payload),
        # Some upstream LIBERO BDDL files contain a stale :language field.
        # Record it verbatim, but the benchmark registry's task.language is
        # the prompt passed to the policy and is the exact-prompt gate.
        "native_bddl_language_field": bddl_language(bddl_text),
        "declared_asset_inventory": assets,
        "declared_asset_inventory_sha256": _sha256_json(assets),
        "model_body_names": body_names,
        "model_body_inventory_sha256": _sha256_json(body_names),
        "model_geom_names": geom_names,
        "model_geom_inventory_sha256": _sha256_json(geom_names),
        "layout_seed": int(seed),
    }


def verify_manifest_against_native_task(
    manifest: dict[str, Any],
    task_suite_name: str,
    task_id: int,
    *,
    task=None,
    env=None,
) -> None:
    """Raise if an evaluated task/env differs from a passing native manifest."""
    if manifest.get("verdict") != "PASS":
        raise RuntimeError("Native task preflight manifest is not PASS")
    if manifest.get("task_suite_name") != task_suite_name:
        raise RuntimeError("Evaluated task suite differs from native preflight")
    if int(manifest.get("task_id", -1)) != int(task_id):
        raise RuntimeError("Evaluated task ID differs from native preflight")

    _, native_task, bddl_path = resolve_native_task(task_suite_name, task_id)
    active_task = task if task is not None else native_task
    if (
        active_task.problem_folder != manifest.get("task_problem_folder")
        or active_task.bddl_file != manifest.get("task_bddl_file")
    ):
        raise RuntimeError(
            "Evaluated task definition differs from native preflight BDDL identity"
        )
    if active_task.language != manifest.get("task_language"):
        raise RuntimeError(
            "Evaluated prompt differs from exact native prompt: "
            f"{active_task.language!r} != {manifest.get('task_language')!r}"
        )
    if Path(manifest.get("native_bddl_source", "")).resolve() != bddl_path:
        raise RuntimeError("Evaluated BDDL source differs from native preflight")
    bddl_payload = bddl_path.read_bytes()
    if _sha256_bytes(bddl_payload) != manifest.get("native_bddl_sha256"):
        raise RuntimeError("Native BDDL content changed after preflight")
    assets = declared_asset_inventory(bddl_payload.decode("utf-8"))
    if _sha256_json(assets) != manifest.get(
        "declared_asset_inventory_sha256"
    ):
        raise RuntimeError("Evaluated asset inventory differs from native task")

    if env is not None:
        body_names = _model_names(env.sim.model, "body")
        geom_names = _model_names(env.sim.model, "geom")
        if _sha256_json(body_names) != manifest.get(
            "model_body_inventory_sha256"
        ):
            raise RuntimeError(
                "Evaluated MuJoCo body inventory differs from native task"
            )
        if _sha256_json(geom_names) != manifest.get(
            "model_geom_inventory_sha256"
        ):
            raise RuntimeError(
                "Evaluated MuJoCo geom inventory differs from native task"
            )


def load_passing_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("verdict") != "PASS":
        raise RuntimeError(f"Native preflight is not PASS: {path}")
    return manifest


def run_preflight(args: argparse.Namespace) -> bool:
    from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
        FAMILIES,
    )

    spec = dict(FAMILIES[args.family])
    contract = build_native_contract(
        args.task_suite_name,
        args.task_id,
        args.seed,
    )
    declared_names = {
        row["name"] for row in contract["declared_asset_inventory"]
    }
    required_assets = {
        value.strip()
        for value in args.required_assets.split(",")
        if value.strip()
    }
    custom_bddl_absent = spec.get("bddl_file") is None
    exact_prompt = contract["task_language"] == args.expected_prompt
    required_assets_present = required_assets.issubset(declared_names)
    custom_model_names = sorted(
        name
        for name in contract["model_body_names"]
        if name.startswith(("l1_b_", "physcog_"))
    )
    native_inventory_only = not custom_model_names
    suite_allowed = args.task_suite_name in STANDARD_SUITES
    passed = bool(
        suite_allowed
        and custom_bddl_absent
        and exact_prompt
        and required_assets_present
        and native_inventory_only
    )
    manifest = {
        **contract,
        "family": args.family,
        "expected_prompt": args.expected_prompt,
        "required_native_assets": sorted(required_assets),
        "active_bddl_override": spec.get("bddl_file"),
        "custom_model_body_names": custom_model_names,
        "gates": {
            "standard_suite": suite_allowed,
            "official_bddl_only": custom_bddl_absent,
            "exact_native_prompt": exact_prompt,
            "required_assets_present": required_assets_present,
            "evaluated_inventory_matches_native": native_inventory_only,
        },
        "verdict": "PASS" if passed else "FAIL",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inventory_lines = [
        f"  - `{row['kind']} {row['name']} - {row['type']}`"
        for row in manifest["declared_asset_inventory"]
    ]
    report = [
        "# L1-B3 native LIBERO preflight",
        "",
        f"Verdict: **{manifest['verdict']}**",
        "",
        f"- Selected native task: `{args.task_suite_name}` task `{args.task_id}`",
        f"- Exact original benchmark prompt: `{manifest['task_language']}`",
        f"- Evaluated prompt exactly matches native prompt: `{exact_prompt}`",
        f"- Native BDDL source: `{manifest['native_bddl_source']}`",
        f"- Native BDDL SHA-256: `{manifest['native_bddl_sha256']}`",
        f"- Upstream BDDL `:language` field (recorded, not substituted): "
        f"`{manifest['native_bddl_language_field']}`",
        f"- Project-local/custom BDDL absent from active family: "
        f"`{custom_bddl_absent}`",
        f"- Evaluated asset inventory identical to native task: "
        f"`{native_inventory_only}`",
        f"- Declared asset inventory SHA-256: "
        f"`{manifest['declared_asset_inventory_sha256']}`",
        f"- MuJoCo body inventory SHA-256: "
        f"`{manifest['model_body_inventory_sha256']}`",
        f"- MuJoCo geom inventory SHA-256: "
        f"`{manifest['model_geom_inventory_sha256']}`",
        f"- Required native assets present: `{required_assets_present}`",
        "",
        "## Native asset inventory",
        "",
        *inventory_lines,
        "",
        "This manifest is a hard input to policy evaluation. Any prompt, BDDL,",
        "body, geom, or declared-asset mismatch stops the run before rollout.",
    ]
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--task_suite_name",
        required=True,
        choices=STANDARD_SUITES,
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_prompt", required=True)
    parser.add_argument("--required_assets", default="")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run_preflight(args) else 2)


if __name__ == "__main__":
    main()
