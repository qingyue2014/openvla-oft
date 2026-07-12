"""Extract one LIBERO HDF5 initial state and repeat it for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy one demo_N initial state into a repeated-state HDF5 file."
    )
    parser.add_argument("--input", required=True, help="Source initial-state HDF5 file")
    parser.add_argument("--output", required=True, help="Destination HDF5 file")
    parser.add_argument("--demo_index", type=int, required=True, help="Zero-based source demo index")
    parser.add_argument("--repeats", type=int, default=5, help="Number of output demos")
    parser.add_argument(
        "--task_key",
        default=None,
        help="HDF5 task group; inferred when the input contains exactly one group",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)

    if args.demo_index < 0:
        raise ValueError("--demo_index must be non-negative")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source HDF5 does not exist: {source_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite): {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo_name = f"demo_{args.demo_index}"

    with h5py.File(source_path, "r") as source:
        task_keys = [name for name, value in source.items() if isinstance(value, h5py.Group)]
        if args.task_key is not None:
            task_key = args.task_key
        elif len(task_keys) == 1:
            task_key = task_keys[0]
        else:
            raise ValueError(
                "Could not infer task group; pass --task_key. "
                f"Available groups: {task_keys}"
            )

        if task_key not in source:
            raise KeyError(f"Task group {task_key!r} not found; available: {task_keys}")
        if demo_name not in source[task_key]:
            available = sorted(source[task_key].keys())
            raise KeyError(f"{demo_name!r} not found in {task_key!r}; available: {available}")
        if "initial_state" not in source[task_key][demo_name]:
            raise KeyError(f"{task_key}/{demo_name} has no 'initial_state' dataset")

        initial_state = source[task_key][demo_name]["initial_state"][:]
        source_demo_attrs = dict(source[task_key][demo_name].attrs)

        with h5py.File(output_path, "w") as output:
            for name, value in source.attrs.items():
                output.attrs[name] = value
            task_group = output.create_group(task_key)
            for name, value in source[task_key].attrs.items():
                task_group.attrs[name] = value

            for index in range(args.repeats):
                demo = task_group.create_group(f"demo_{index}")
                demo.create_dataset("initial_state", data=initial_state)
                for name, value in source_demo_attrs.items():
                    demo.attrs[name] = value
                demo.attrs["source_demo_index"] = args.demo_index

    print(
        f"Saved {args.repeats} copies of {task_key}/{demo_name} "
        f"from {source_path} to {output_path}"
    )


if __name__ == "__main__":
    main()
