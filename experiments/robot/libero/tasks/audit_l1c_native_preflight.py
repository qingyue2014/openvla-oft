"""Audit native-task identity and asset inventory for L1-C paired states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_states(path: Path, prompt: str) -> list:
    key = prompt.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        if set(handle) != {key}:
            raise ValueError(
                f"{path}: expected only HDF5 key {key!r}, found {sorted(handle)}"
            )
        group = handle[key]
        names = sorted(group, key=lambda value: int(value.rsplit("_", 1)[1]))
        if names != [f"demo_{index}" for index in range(len(names))]:
            raise ValueError(f"{path}: demo keys are not contiguous from demo_0")
        return [group[name]["initial_state"][:] for name in names]


def _model_inventory(env) -> dict[str, list[str]]:
    bodies = sorted(
        name
        for index in range(env.sim.model.nbody)
        if (name := env.sim.model.body_id2name(index))
    )
    geoms = sorted(
        name
        for index in range(env.sim.model.ngeom)
        if (name := env.sim.model.geom_id2name(index))
    )
    return {"bodies": bodies, "geoms": geoms}


def _resolve_task(scenario: str):
    from libero.libero import benchmark, get_libero_path

    if scenario == "l1c1":
        suite_name = "libero_spatial"
        task_id = 2
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        relevant_bodies = (
            "akita_black_bowl_1_main",
            "akita_black_bowl_2_main",
            "plate_1_main",
        )
        return suite_name, task_id, task, bddl.resolve(), relevant_bodies

    from experiments.robot.libero.tasks.l1c_occupied_common import (
        get_spec,
        resolve_bddl,
    )

    spec = get_spec(scenario)
    suite_name = "libero_90"
    suite = benchmark.get_benchmark_dict()[suite_name]()
    matches = [
        (task_id, suite.get_task(task_id))
        for task_id in range(suite.n_tasks)
        if suite.get_task(task_id).language.strip().lower()
        == spec.prompt.strip().lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{scenario}: expected one native task for prompt {spec.prompt!r}, "
            f"found {len(matches)}"
        )
    task_id, task = matches[0]
    bddl = Path(resolve_bddl(spec)).resolve()
    official_root = Path(get_libero_path("bddl_files")).resolve()
    if official_root not in bddl.parents:
        raise ValueError(f"{scenario}: BDDL is outside native LIBERO root: {bddl}")
    relevant_bodies = (spec.target_body, spec.occupant_body, spec.anchor_body)
    return suite_name, task_id, task, bddl, relevant_bodies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("l1c1", "l1c2", "l1c3"), required=True)
    parser.add_argument("--state", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument("--expected_episodes", type=int, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    args = parser.parse_args()

    states = {}
    for assignment in args.state:
        condition, separator, raw_path = assignment.partition("=")
        if not separator or condition not in {"eb", "er", "ec"}:
            raise ValueError(f"Expected eb|er|ec=PATH, got {assignment!r}")
        states[condition] = Path(raw_path)
    if set(states) != {"eb", "er", "ec"}:
        raise ValueError("Exactly one eb, er, and ec state path is required")

    suite_name, task_id, task, bddl, relevant_bodies = _resolve_task(args.scenario)
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    try:
        env.reset()
        native_inventory = _model_inventory(env)
        missing_bodies = sorted(set(relevant_bodies) - set(native_inventory["bodies"]))
        if missing_bodies:
            raise ValueError(
                f"{args.scenario}: relevant native bodies are missing: {missing_bodies}"
            )
        condition_records = {}
        for condition, path in sorted(states.items()):
            rows = _load_states(path, task.language)
            if len(rows) != args.expected_episodes:
                raise ValueError(
                    f"{path}: expected {args.expected_episodes} episodes, "
                    f"found {len(rows)}"
                )
            for row in rows:
                env.reset()
                env.set_init_state(row)
                if _model_inventory(env) != native_inventory:
                    raise ValueError(
                        f"{condition}: serialized state changed native asset inventory"
                    )
            condition_records[condition] = {
                "path": str(path),
                "episodes": len(rows),
                "sha256": _sha256(path),
                "inventory_matches_native": True,
            }
    finally:
        env.close()

    result = {
        "verdict": "PASS_NATIVE_TASK_PREFLIGHT",
        "scenario": args.scenario,
        "suite": suite_name,
        "task_id": task_id,
        "prompt": task.language,
        "bddl": str(bddl),
        "bddl_sha256": _sha256(bddl),
        "native_assets_only": True,
        "relevant_bodies": list(relevant_bodies),
        "body_inventory": native_inventory["bodies"],
        "geom_inventory": native_inventory["geoms"],
        "conditions": condition_records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.scenario.upper()} native-task preflight",
        "",
        "- Verdict: **PASS_NATIVE_TASK_PREFLIGHT**",
        f"- Suite/task: `{suite_name}` / `{task_id}`",
        f"- Original prompt: `{task.language}`",
        f"- Native BDDL: `{bddl}`",
        f"- Body inventory: `{len(native_inventory['bodies'])}`",
        f"- Geom inventory: `{len(native_inventory['geoms'])}`",
        "- Asset inventory after every Eb/Er/Ec serialized-state restore: `identical`",
        "- Custom assets or BDDL: `none`",
    ]
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
