#!/usr/bin/env python3
"""Verify the immutable job-512800 handoff before pi0.5 execution.

This check is intentionally simulator-free.  It binds the six archived files
to the Outcome V2 preregistration and checks the task, pairing, native-asset,
and model-selection contracts encoded in their JSON records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FROZEN_DIR = Path(
    "experiments/robot/libero/tasks/frozen/"
    "l1b3_task4_outcome_v2_job512800"
)
DEFAULT_PREREGISTRATION = Path(
    "experiments/robot/libero/tasks/"
    "l1b3_task4_outcome_v2_design_prereg.json"
)
VERDICT = "PASS_L1B3_TASK4_PI05_HANDOFF"
FAMILY = "l1b3_task4_outcome_v2"
SCENE_CONTRACT = "l1b3_task4_swept_outcome_v2_matched_ec_v4"
TASK_PROMPT = "put the bowl on top of the cabinet"
GOAL_SIGNATURE = "1fb102d5ac008631703f77d891abb739004a388f14ea0abb2b25d62a32513237"
INVENTORY_SIGNATURE = "7940bfcbe1981bee868e16d31a04f4449026febff444e5d9bd97072d8162ead5"
NATIVE_BDDL_SHA256 = "2ffba859a154f50c3c99ffb3420743fa5aa65c70bf7d4cd26f5bc81d07be5713"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_handoff(
    frozen_dir: Path = DEFAULT_FROZEN_DIR,
    preregistration: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    """Return a manifest only when every frozen handoff contract matches."""
    frozen_dir = frozen_dir.resolve()
    preregistration = preregistration.resolve()
    prereg = _load_json(preregistration)
    selection = prereg.get("selection_contract", {})
    handoff = selection.get("frozen_pi0_5_scene_handoff", {})
    _require(isinstance(handoff, dict), "missing frozen pi0.5 scene handoff")
    expected_hashes = handoff.get("artifact_sha256", {})
    _require(
        isinstance(expected_hashes, dict) and len(expected_hashes) == 6,
        "handoff must bind exactly six artifacts",
    )
    _require(handoff.get("source_job_id") == "512800", "source job changed")
    _require(
        handoff.get("source_commit")
        == "9a6516b184cd6695f502a17fd7cb011eaf8a9a75",
        "source commit changed",
    )
    _require(
        handoff.get("scene_contract") == SCENE_CONTRACT,
        "frozen scene contract changed",
    )
    _require(handoff.get("pair_count") == 5, "frozen pair count changed")
    _require(
        handoff.get("reuse_without_regeneration_required") is True,
        "frozen scene is not marked reuse-only",
    )
    _require(
        handoff.get("human_approval_present") is False,
        "handoff must not claim human approval",
    )
    _require(
        selection.get("primary_evaluated_model") == "pi0.5",
        "pi0.5 is not the primary evaluated model",
    )
    _require(
        selection.get("retired_evaluated_model") == "OpenVLA-OFT",
        "OpenVLA-OFT retirement contract changed",
    )
    _require(
        selection.get("formal_model_order") == ["pi0.5", "Cosmos"],
        "formal model order must be pi0.5 then Cosmos",
    )

    observed_hashes: dict[str, str] = {}
    for name, expected in sorted(expected_hashes.items()):
        path = frozen_dir / name
        _require(path.is_file(), f"missing frozen artifact: {path}")
        observed = _sha256(path)
        _require(
            observed == expected,
            f"frozen artifact hash mismatch for {name}: "
            f"expected={expected} observed={observed}",
        )
        observed_hashes[name] = observed

    pairing = _load_json(frozen_dir / f"{FAMILY}_pairing.json")
    _require(pairing.get("family") == FAMILY, "pairing family mismatch")
    _require(pairing.get("component") == "outcome", "pairing construct mismatch")
    _require(pairing.get("task_suite") == "libero_goal", "task suite mismatch")
    _require(pairing.get("task_id") == 4, "native task id mismatch")
    _require(pairing.get("task_language") == TASK_PROMPT, "native prompt mismatch")
    _require(
        pairing.get("scene_contract") == SCENE_CONTRACT,
        "pairing scene contract mismatch",
    )
    _require(pairing.get("seed") == 42, "paired reset seed changed")
    pairs = pairing.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == 5, "pairing must contain five pairs")
    _require(pairing.get("num_states") == len(pairs), "pairing state count mismatch")
    _require(
        [pair.get("episode_idx") for pair in pairs] == list(range(5)),
        "paired episode indices must be 0..4",
    )
    _require(
        len({pair.get("source_state_index") for pair in pairs}) == 5,
        "source state indices are not unique",
    )
    _require(
        all(pair.get("only_obstacle_pose_changed") is True for pair in pairs),
        "a pair changes state outside the registered obstacle pose",
    )
    _require(
        all(
            pair.get("matched_control_mode") == "dual_radius_reflection"
            for pair in pairs
        ),
        "a pair does not use the frozen dual-radius reflection control",
    )

    archived_preflight = _load_json(
        frozen_dir / f"{FAMILY}_native_preflight.json"
    )
    _require(
        archived_preflight.get("verdict")
        == "PASS_L1B3_TASK4_OUTCOME_V2_NATIVE_PREFLIGHT",
        "archived native preflight did not pass",
    )
    _require(archived_preflight.get("family") == FAMILY, "preflight family mismatch")
    _require(
        archived_preflight.get("task_suite_name") == "libero_goal"
        and archived_preflight.get("task_id") == 4,
        "preflight native task mismatch",
    )
    _require(
        archived_preflight.get("benchmark_prompt") == TASK_PROMPT,
        "preflight benchmark prompt mismatch",
    )
    _require(
        archived_preflight.get("native_bddl_sha256") == NATIVE_BDDL_SHA256
        and archived_preflight.get("evaluated_bddl_sha256") == NATIVE_BDDL_SHA256,
        "native/evaluated BDDL hash mismatch",
    )
    _require(
        archived_preflight.get("goal_signature_sha256") == GOAL_SIGNATURE,
        "native goal signature mismatch",
    )
    _require(
        archived_preflight.get("inventory_signature") == INVENTORY_SIGNATURE,
        "native inventory signature mismatch",
    )
    condition_inventory = archived_preflight.get(
        "condition_inventory_signatures", {}
    )
    _require(
        condition_inventory
        == {condition: INVENTORY_SIGNATURE for condition in ("eb", "er", "ec")},
        "EB/ER/EC inventory signatures differ",
    )
    _require(
        archived_preflight.get("source_to_project_bddl_delta")
        == "none; evaluated BDDL is the selected native source",
        "evaluated BDDL is not the native source",
    )

    return {
        "verdict": VERDICT,
        "model_family": "pi05",
        "source_job_id": handoff["source_job_id"],
        "source_commit": handoff["source_commit"],
        "family": FAMILY,
        "scene_contract": SCENE_CONTRACT,
        "task_suite": "libero_goal",
        "task_id": 4,
        "task_prompt": TASK_PROMPT,
        "pair_count": 5,
        "seed": 42,
        "native_bddl_sha256": NATIVE_BDDL_SHA256,
        "goal_signature_sha256": GOAL_SIGNATURE,
        "inventory_signature": INVENTORY_SIGNATURE,
        "artifact_sha256": observed_hashes,
        "reuse_without_regeneration": True,
        "human_approval_present": False,
        "formal_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = verify_handoff(args.frozen_dir, args.preregistration)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"Verdict: {VERDICT}")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
