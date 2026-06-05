"""Inspect the LIBERO-Spatial task suite loaded by the current environment.

Run this on the server inside the OpenVLA-OFT / LIBERO environment:

    python experiments/robot/libero/inspect_libero_spatial_suite.py

The script prints the installed LIBERO package path, benchmark module path,
task languages, and BDDL files. It is intended to verify whether the
black-bowl-to-plate tasks are loaded from the installed LIBERO suite.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path


def _import_libero_with_fallback():
    try:
        import libero
        from libero.libero import benchmark
        from libero.libero.utils.bddl_generation_utils import get_libero_path

        return libero, benchmark, get_libero_path
    except ModuleNotFoundError as exc:
        if exc.name != "libero":
            raise

    repo_root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        repo_root.parent / "LIBERO",
        repo_root.parent / "libero",
    ]
    for candidate in candidate_paths:
        if (candidate / "libero").is_dir():
            sys.path.insert(0, str(candidate))
            import libero
            from libero.libero import benchmark
            from libero.libero.utils.bddl_generation_utils import get_libero_path

            print(f"[info] Added LIBERO path to sys.path: {candidate}")
            return libero, benchmark, get_libero_path

    raise ModuleNotFoundError(
        "Could not import the 'libero' package. Install LIBERO in this conda "
        "environment with `pip install -e ~/04-mycode/LIBERO`, or run with "
        "`PYTHONPATH=~/04-mycode/LIBERO:$PYTHONPATH`."
    )


def _read_bddl_language(bddl_path: str) -> str | None:
    try:
        with open(bddl_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":language" in line:
                    return line.strip()
    except OSError as exc:
        return f"<could not read BDDL: {exc}>"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print tasks and BDDL sources for a LIBERO benchmark suite.",
    )
    parser.add_argument(
        "--suite",
        default="libero_spatial",
        help="LIBERO benchmark suite name to inspect (default: libero_spatial).",
    )
    parser.add_argument(
        "--show-bddl-language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the :language line read directly from each BDDL file.",
    )
    args = parser.parse_args()

    libero, benchmark, get_libero_path = _import_libero_with_fallback()

    print(f"libero package: {libero.__file__}")
    print(f"benchmark module: {inspect.getfile(benchmark)}")
    print(f"bddl root: {get_libero_path('bddl_files')}")

    benchmark_dict = benchmark.get_benchmark_dict()
    print(f"available suites: {sorted(benchmark_dict.keys())}")
    if args.suite not in benchmark_dict:
        raise SystemExit(f"Unknown suite '{args.suite}'.")

    suite = benchmark_dict[args.suite]()
    print(f"suite: {args.suite}")
    print(f"suite class: {type(suite)}")
    print(f"n_tasks: {suite.n_tasks}")

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        problem_folder = getattr(task, "problem_folder", None)
        bddl_file = getattr(task, "bddl_file", None)
        bddl_path = None
        if problem_folder and bddl_file:
            bddl_path = os.path.join(get_libero_path("bddl_files"), problem_folder, bddl_file)

        print("=" * 80)
        print(f"task_id: {task_id}")
        print(f"task.language: {getattr(task, 'language', None)}")
        print(f"problem_folder: {problem_folder}")
        print(f"bddl_file: {bddl_file}")
        print(f"bddl_path: {bddl_path}")
        if bddl_path:
            print(f"bddl_exists: {Path(bddl_path).exists()}")
            if args.show_bddl_language:
                print(f"bddl language: {_read_bddl_language(bddl_path)}")


if __name__ == "__main__":
    main()
