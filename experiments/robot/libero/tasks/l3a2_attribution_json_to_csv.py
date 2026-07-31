"""Flatten an L3-A2 attribution JSON report to a two-column CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            yield from _flatten(
                child, f"{prefix}.{key}" if prefix else str(key)
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(_flatten(data))


if __name__ == "__main__":
    main()
