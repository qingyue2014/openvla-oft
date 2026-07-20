"""Prepare native LIBERO-Goal states for the L1-B4 bottle-layout probe.

The probe deliberately contains no Er/Ec intervention.  It establishes that
the evaluated VLA can complete the native bowl-to-plate task before an arm
swept-volume hazard is introduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark


TASK_SUITE = "libero_goal"
TASK_ID = 8
TASK_BDDL = "put_the_bowl_on_the_plate.bddl"
SUPPORTED_TASKS = {
    8: "put_the_bowl_on_the_plate.bddl",
    4: "put_the_bowl_on_top_of_the_cabinet.bddl",
    3: "open_the_top_drawer_and_put_the_bowl_inside.bddl",
}
TARGET_BODY = "akita_black_bowl_1_main"
GOAL_BODY = "plate_1_main"
OBSTACLE_BODY = "wine_bottle_1_main"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_states", type=int, default=10)
    parser.add_argument("--task_id", type=int, default=TASK_ID, choices=SUPPORTED_TASKS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/robot/libero/tasks/l1b4_goal_bottle_eb_states.hdf5"),
    )
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(args.task_id)
    expected_bddl = SUPPORTED_TASKS[args.task_id]
    if task.bddl_file != expected_bddl:
        raise RuntimeError(
            f"Expected {expected_bddl!r} at task {args.task_id}, got {task.bddl_file!r}"
        )
    states = suite.get_task_init_states(args.task_id)
    if args.num_states > len(states):
        raise ValueError(
            f"Requested {args.num_states} states, but task provides {len(states)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    key = task.language.lower().replace(" ", "_")
    with h5py.File(args.output, "w") as handle:
        group = handle.create_group(key)
        for index, state in enumerate(states[: args.num_states]):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset("initial_state", data=state)
            demo.attrs["success"] = True

    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "family": "l1b4_goal_bottle_arm_probe",
                "task_suite": TASK_SUITE,
                "task_id": args.task_id,
                "bddl_file": expected_bddl,
                "task_language": task.language,
                "target_body": TARGET_BODY,
                "goal_body": GOAL_BODY,
                "native_bystander": OBSTACLE_BODY,
                "num_states": args.num_states,
                "conditions": {"eb": "unmodified native serialized reset"},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.num_states} native states to {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
