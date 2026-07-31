"""Resolve the native L3-A3 parking support from generated physical records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BOTTLE_BODY,
    TASK_KEY,
)


def _supported_fixture(path: str) -> set[str]:
    with h5py.File(path, "r") as handle:
        demo = handle[TASK_KEY]["demo_0"]
        fixtures = set(json.loads(str(demo.attrs["fixture_replay_bodies_json"])))
        post = json.loads(str(demo.attrs["formal_post_wait_json"]))
        contacts = set(post[BOTTLE_BODY]["contacts"])
    return contacts & fixtures


def resolve(eb: str, ec: str) -> str:
    common = _supported_fixture(eb) & _supported_fixture(ec)
    if len(common) != 1:
        raise ValueError(
            f"expected one native bottle parking support shared by Eb/Ec, got {sorted(common)}"
        )
    return next(iter(common))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--ec", required=True)
    args = parser.parse_args()
    print(resolve(args.eb, args.ec))


if __name__ == "__main__":
    main()
