"""Find native LIBERO tasks by keyword.

Example:
    python experiments/robot/libero/tasks/find_libero_native_tasks.py \
        --keywords cookie,box,book,plate,stove
"""

from __future__ import annotations

import argparse

from experiments.robot.libero.inspect_libero_spatial_suite import _import_libero_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter native LIBERO tasks by language keyword.")
    parser.add_argument(
        "--keywords",
        default="cookie,box,book,plate,stove,cream cheese",
        help="Comma-separated case-insensitive keywords.",
    )
    parser.add_argument(
        "--suites",
        default="",
        help="Optional comma-separated suite names. Defaults to every available suite.",
    )
    args = parser.parse_args()

    _, benchmark, _ = _import_libero_with_fallback()
    benchmark_dict = benchmark.get_benchmark_dict()
    keywords = [keyword.strip().lower() for keyword in args.keywords.split(",") if keyword.strip()]
    suites = [suite.strip() for suite in args.suites.split(",") if suite.strip()]
    if not suites:
        suites = sorted(benchmark_dict.keys())

    for suite_name in suites:
        if suite_name not in benchmark_dict:
            print(f"[skip] unknown suite: {suite_name}")
            continue
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            language = getattr(task, "language", "")
            lower_language = language.lower()
            if any(keyword in lower_language for keyword in keywords):
                print(
                    f"{suite_name:14s} task_id={task_id:<3d} "
                    f"bddl={getattr(task, 'problem_folder', '')}/{getattr(task, 'bddl_file', '')} "
                    f"language={language}"
                )


if __name__ == "__main__":
    main()
