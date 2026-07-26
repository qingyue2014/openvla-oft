#!/usr/bin/env python3
"""Fail-closed serialized-state pairing validation for L3-A4 Eb/Er/Ec."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3a4_momentum import (
    A_BODY,
    B_BODY,
    C_BODY,
    CHAIN_BODIES,
    EC_SENTINEL_PARK_DXY,
    SCHEMA_VERSION,
    TASK_DESCRIPTION,
    TOPOLOGY_ID,
    contract_sha256,
)


def _load(path: str, variant: str):
    key = TASK_DESCRIPTION.replace(" ", "_")
    handle = h5py.File(path, "r")
    group = handle[key]
    checks = {
        "schema": int(group.attrs.get("l3a4_schema_version", -1)) == SCHEMA_VERSION,
        "topology": str(group.attrs.get("l3a4_topology_id", "")) == TOPOLOGY_ID,
        "variant": str(group.attrs.get("l3a4_variant", "")) == variant,
        "contract": str(group.attrs.get("contract_sha256", "")) == contract_sha256(),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        handle.close()
        raise ValueError(f"{path}: binding failed {failed}")
    layout = json.loads(group.attrs["flattened_state_layout_json"])
    return handle, group, layout


def _allowed_indices(layout: dict, bodies: tuple[str, ...]) -> set[int]:
    nq = int(layout["nq"])
    allowed = set()
    for body in bodies:
        qadr = int(layout["qpos"][body])
        vadr = int(layout["qvel"][body])
        allowed.update(range(1 + qadr, 1 + qadr + 7))
        allowed.update(range(1 + nq + vadr, 1 + nq + vadr + 6))
    return allowed


def _state(group, index: int) -> np.ndarray:
    return np.asarray(group[f"demo_{index}"]["initial_state"][:])


def _sha(state) -> str:
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--eb", required=True)
    parser.add_argument("--report", default="experiments/logs/l3a4_pairing.md")
    args = parser.parse_args()

    handles = []
    rows = []
    try:
        er_h, er, er_layout = _load(args.er, "risk")
        ec_h, ec, ec_layout = _load(args.ec, "stable")
        eb_h, eb, eb_layout = _load(args.eb, "baseline")
        handles.extend((er_h, ec_h, eb_h))
        if er_layout != ec_layout or er_layout != eb_layout:
            raise ValueError("flattened MuJoCo state layouts differ across conditions")
        if not (len(er) == len(ec) == len(eb)):
            raise ValueError("Eb/Er/Ec episode counts differ")

        ec_allowed = _allowed_indices(er_layout, (C_BODY,))
        eb_allowed = _allowed_indices(er_layout, CHAIN_BODIES)
        c_qadr = int(er_layout["qpos"][C_BODY])
        for index in range(len(er)):
            er_state = _state(er, index)
            ec_state = _state(ec, index)
            eb_state = _state(eb, index)
            if not (er_state.shape == ec_state.shape == eb_state.shape):
                raise ValueError(f"episode {index}: state shape mismatch")

            ec_changed = set(np.flatnonzero(er_state != ec_state).tolist())
            eb_changed = set(np.flatnonzero(er_state != eb_state).tolist())
            ec_outside = sorted(ec_changed - ec_allowed)
            eb_outside = sorted(eb_changed - eb_allowed)
            ec_delta = (
                ec_state[1 + c_qadr:1 + c_qadr + 2]
                - er_state[1 + c_qadr:1 + c_qadr + 2]
            )
            attempts = {
                variant: int(group[f"demo_{index}"].attrs["reset_attempt"])
                for variant, group in (("er", er), ("ec", ec), ("eb", eb))
            }
            passed = (
                not ec_outside
                and not eb_outside
                and np.allclose(
                    ec_delta, EC_SENTINEL_PARK_DXY, atol=1e-12, rtol=0.0
                )
                and len(set(attempts.values())) == 1
            )
            rows.append(
                {
                    "episode": index,
                    "passed": passed,
                    "ec_changed_count": len(ec_changed),
                    "eb_changed_count": len(eb_changed),
                    "ec_outside_allowed": ec_outside,
                    "eb_outside_allowed": eb_outside,
                    "ec_C_xy_delta": ec_delta.tolist(),
                    "reset_attempts": attempts,
                    "er_sha256": _sha(er_state),
                    "ec_sha256": _sha(ec_state),
                    "eb_sha256": _sha(eb_state),
                }
            )
        passed = all(row["passed"] for row in rows)
    finally:
        for handle in handles:
            handle.close()

    verdict = (
        "PASS_L3A4_PAIRED_SERIALIZED_STATES"
        if passed
        else "FAIL_L3A4_PAIRED_SERIALIZED_STATES"
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# L3-A4 serialized-state pairing",
                "",
                f"- Verdict: **{verdict}**",
                f"- Episodes: {len(rows)}",
                "- Ec allowed difference: C free-joint qpos/qvel only.",
                "- Eb allowed difference: A/B/C free-joint qpos/qvel only.",
                "- Prompt, fixture, robot, native objects, time, and all other "
                "serialized values must be bit-identical.",
                "",
                "```json",
                json.dumps(rows, indent=2, sort_keys=True),
                "```",
            ]
        )
        + "\n"
    )
    print(f"{verdict} count={len(rows)} report={report}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
