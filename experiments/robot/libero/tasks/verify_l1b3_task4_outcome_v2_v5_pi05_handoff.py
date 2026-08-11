#!/usr/bin/env python3
"""Verify the immutable job-514502 v5 handoff before pi0.5 smoke.

This verifier is simulator-free. It binds the frozen states, pairing,
model-independent selection evidence, v5 preregistration, and the explicit
scene-review approval. It authorizes smoke only; formal and Cosmos remain
blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FAMILY = "l1b3_task4_outcome_v2_v5"
SCENE_CONTRACT = "l1b3_task4_swept_outcome_v2_model_independent_v5"
TASK_PROMPT = "put the bowl on top of the cabinet"
GOAL_SIGNATURE = "1fb102d5ac008631703f77d891abb739004a388f14ea0abb2b25d62a32513237"
INVENTORY_SIGNATURE = "7940bfcbe1981bee868e16d31a04f4449026febff444e5d9bd97072d8162ead5"
NATIVE_BDDL_SHA256 = "2ffba859a154f50c3c99ffb3420743fa5aa65c70bf7d4cd26f5bc81d07be5713"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
SOURCE_INDICES = [2, 4, 5, 7, 8]
VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V5_PI05_HANDOFF"

DEFAULT_FROZEN_DIR = Path(
    "experiments/robot/libero/tasks/frozen/"
    "l1b3_task4_outcome_v2_v5_job514502"
)
DEFAULT_PREREGISTRATION = Path(
    "experiments/robot/libero/tasks/"
    "l1b3_task4_outcome_v2_v5_design_prereg.json"
)
DEFAULT_APPROVAL = DEFAULT_FROZEN_DIR / "scene_human_review_approval.json"

CORE_NAMES = (
    f"{FAMILY}_native_preflight.json",
    f"{FAMILY}_native_source_states.hdf5",
    f"{FAMILY}_eb_states.hdf5",
    f"{FAMILY}_er_states.hdf5",
    f"{FAMILY}_ec_states.hdf5",
    f"{FAMILY}_pairing.json",
)


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


def _expected_by_basename(prepare: dict[str, Any], name: str) -> str:
    matches = [
        digest
        for path, digest in prepare.get("artifact_sha256", {}).items()
        if Path(path).name == name
    ]
    _require(len(matches) == 1, f"prepare manifest does not bind exactly one {name}")
    return str(matches[0])


def verify_handoff(
    frozen_dir: Path = DEFAULT_FROZEN_DIR,
    preregistration: Path = DEFAULT_PREREGISTRATION,
    approval_path: Path = DEFAULT_APPROVAL,
) -> dict[str, Any]:
    frozen_dir = frozen_dir.resolve()
    preregistration = preregistration.resolve()
    approval_path = approval_path.resolve()

    prereg = _load_json(preregistration)
    approval = _load_json(approval_path)
    prepare_path = frozen_dir / f"{FAMILY}_prepare_manifest.json"
    selection_path = frozen_dir / f"{FAMILY}_selection_manifest.json"
    pairing_path = frozen_dir / f"{FAMILY}_pairing.json"
    preflight_path = frozen_dir / f"{FAMILY}_native_preflight.json"
    prepare = _load_json(prepare_path)
    selection = _load_json(selection_path)
    pairing = _load_json(pairing_path)
    preflight = _load_json(preflight_path)

    _require(prereg.get("family") == FAMILY, "preregistration family mismatch")
    _require(
        prereg.get("native_task", {}).get("suite") == "libero_goal"
        and prereg.get("native_task", {}).get("task_id") == 4,
        "preregistered native task mismatch",
    )
    _require(
        prereg.get("native_task", {}).get("benchmark_prompt") == TASK_PROMPT,
        "preregistered native prompt mismatch",
    )
    selection_contract = prereg.get("selection_contract", {})
    learned_contract = prereg.get("learned_policy_contract", {})
    _require(selection_contract.get("selection_model") == "none", "selection model changed")
    _require(
        selection_contract.get("trajectory_source")
        == "model_independent_scripted_osc_v1",
        "selection trajectory source changed",
    )
    _require(
        selection_contract.get("scene_selection_penetration_buffer_m") == 0.001
        and selection_contract.get("formal_rollout_physics_limit_m") == 0.002,
        "penetration contract changed",
    )
    pi05 = learned_contract.get("pi0_5", {})
    _require(
        learned_contract.get("primary_evaluated_model") == "pi0.5"
        and learned_contract.get("retired_model") == "OpenVLA-OFT",
        "model-order contract changed",
    )
    _require(pi05.get("openpi_commit") == OPENPI_COMMIT, "OpenPI commit changed")
    _require(pi05.get("replan_steps") == 1, "pi0.5 replan_steps changed")
    _require(pi05.get("may_be_changed_after_v5_outcomes") is False, "pi0.5 protocol is mutable")

    _require(approval.get("family") == FAMILY, "approval family mismatch")
    _require(approval.get("source_job_id") == "514502", "approval source job mismatch")
    _require(approval.get("approved") is True, "scene human review is not approved")
    _require(bool(approval.get("reviewer")), "scene approval reviewer missing")
    _require(bool(approval.get("reviewed_at_utc")), "scene approval time missing")
    authorizes = approval.get("authorizes", {})
    _require(authorizes.get("pi05_smoke") is True, "pi0.5 smoke not authorized")
    _require(authorizes.get("pi05_formal") is False, "approval must not authorize formal")
    _require(authorizes.get("cosmos") is False, "approval must not authorize Cosmos")
    _require(authorizes.get("openvla_oft") is False, "approval must not authorize OpenVLA-OFT")
    _require(
        _sha256(prepare_path) == approval.get("frozen_prepare_manifest_sha256"),
        "approved prepare manifest hash mismatch",
    )
    _require(
        _sha256(selection_path) == approval.get("frozen_selection_manifest_sha256"),
        "approved selection manifest hash mismatch",
    )
    _require(
        _sha256(pairing_path) == approval.get("frozen_pairing_sha256"),
        "approved pairing hash mismatch",
    )
    _require(
        approval.get("reviewed_initial_gate_manifest_sha256")
        == _expected_by_basename(
            prepare, "L1-B3-task4-outcome-v2-v5_initial_gate_manifest.json"
        ),
        "approved initial-gate hash mismatch",
    )
    _require(
        approval.get("reviewed_bundle_manifest_sha256")
        == _expected_by_basename(prepare, "REVIEW_BUNDLE_MANIFEST.json"),
        "approved review-bundle hash mismatch",
    )

    _require(prepare.get("family") == FAMILY, "prepare family mismatch")
    _require(prepare.get("selection_model") == "none", "prepare selection model changed")
    _require(prepare.get("learned_policy_executed") is False, "prepare used a learned policy")
    _require(
        prepare.get("pi0_5_replan_steps_frozen_for_future_smoke") == 1,
        "prepare did not freeze replan_steps=1",
    )
    _require(prepare.get("openvla_oft_retired") is True, "OpenVLA-OFT is not retired")
    _require(prepare.get("formal_authorized") is False, "prepare improperly authorizes formal")
    _require(prepare.get("cosmos_authorized") is False, "prepare improperly authorizes Cosmos")

    observed_hashes: dict[str, str] = {}
    for name in CORE_NAMES:
        path = frozen_dir / name
        _require(path.is_file(), f"missing frozen artifact: {path}")
        observed = _sha256(path)
        _require(observed == _expected_by_basename(prepare, name), f"hash mismatch: {name}")
        observed_hashes[name] = observed
    _require(
        _sha256(selection_path)
        == _expected_by_basename(prepare, selection_path.name),
        "selection manifest is not the prepare-bound artifact",
    )

    _require(
        selection.get("verdict")
        == "PASS_L1B3_TASK4_OUTCOME_V2_V5_MODEL_INDEPENDENT_SELECTION",
        "model-independent selection did not pass",
    )
    _require(selection.get("selection_model") == "none", "selection manifest names a model")
    _require(
        selection.get("learned_policy_trajectory_used_for_selection") is False,
        "learned trajectory used for selection",
    )
    _require(selection.get("source_state_indices") == SOURCE_INDICES, "selected source order changed")
    _require(selection.get("expected_pairs") == 5, "selected pair count changed")
    _require(selection.get("maximum_selection_penetration_m") == 0.001, "selection penetration limit changed")
    pair_records = selection.get("pair_records")
    _require(isinstance(pair_records, list) and len(pair_records) == 5, "selection pair records incomplete")
    _require(all(record.get("valid") is True for record in pair_records), "invalid selected pair")
    _require(
        max(
            record.get("selection_replay", {}).get(
                "maximum_contact_penetration_m", float("inf")
            )
            for record in pair_records
        )
        <= 0.001,
        "selected replay penetration exceeds 1 mm",
    )

    trajectory_hashes: dict[str, str] = {}
    trajectory_records = selection.get("trajectory_records")
    _require(isinstance(trajectory_records, list) and len(trajectory_records) == 5, "trajectory evidence incomplete")
    for record in trajectory_records:
        name = Path(str(record.get("path", ""))).name
        path = frozen_dir / "scripted_selection_trajectories" / name
        _require(path.is_file(), f"missing frozen selection trajectory: {name}")
        observed = _sha256(path)
        _require(observed == record.get("sha256"), f"selection trajectory hash mismatch: {name}")
        _require(record.get("model_trajectory_used") is False, "learned trajectory provenance found")
        _require(record.get("controller_obstacle_adaptive") is False, "selection controller became obstacle-adaptive")
        _require(record.get("cross_episode_grasp_cache_disabled") is True, "selection grasp cache was enabled")
        _require(record.get("valid") is True, f"invalid selection trajectory: {name}")
        trajectory_hashes[name] = observed

    _require(pairing.get("family") == FAMILY, "pairing family mismatch")
    _require(pairing.get("scene_contract") == SCENE_CONTRACT, "scene contract mismatch")
    _require(pairing.get("task_suite") == "libero_goal" and pairing.get("task_id") == 4, "pairing native task mismatch")
    _require(pairing.get("task_language") == TASK_PROMPT, "pairing native prompt mismatch")
    pairs = pairing.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == 5, "pairing must contain five pairs")
    _require([pair.get("episode_idx") for pair in pairs] == list(range(5)), "paired episode order changed")
    _require([pair.get("source_state_index") for pair in pairs] == SOURCE_INDICES, "paired source indices changed")
    _require(all(pair.get("only_obstacle_pose_changed") is True for pair in pairs), "pair changes non-obstacle state")
    _require(all(pair.get("learned_policy_trajectory_used_for_selection") is False for pair in pairs), "pair contains learned selection provenance")
    _require(all(pair.get("matched_control_mode") == "dual_radius_reflection" for pair in pairs), "matched-control mode changed")

    _require(
        preflight.get("verdict")
        == "PASS_L1B3_TASK4_OUTCOME_V2_V5_NATIVE_PREFLIGHT",
        "archived native preflight did not pass",
    )
    _require(preflight.get("family") == FAMILY, "preflight family mismatch")
    _require(preflight.get("task_suite_name") == "libero_goal" and preflight.get("task_id") == 4, "preflight native task mismatch")
    _require(preflight.get("benchmark_prompt") == TASK_PROMPT, "preflight prompt mismatch")
    _require(preflight.get("native_bddl_sha256") == NATIVE_BDDL_SHA256 and preflight.get("evaluated_bddl_sha256") == NATIVE_BDDL_SHA256, "native BDDL hash mismatch")
    _require(preflight.get("goal_signature_sha256") == GOAL_SIGNATURE, "native goal signature mismatch")
    _require(preflight.get("inventory_signature") == INVENTORY_SIGNATURE, "inventory signature mismatch")
    _require(preflight.get("custom_assets") == [] and preflight.get("custom_bddl") is False, "custom scene content found")
    _require(preflight.get("all_selected_native_files_unmodified") is True, "modified native asset found")

    return {
        "verdict": VERDICT,
        "model_family": "pi05",
        "family": FAMILY,
        "scene_contract": SCENE_CONTRACT,
        "source_job_id": "514502",
        "source_commit": approval.get("source_commit"),
        "task_suite": "libero_goal",
        "task_id": 4,
        "task_prompt": TASK_PROMPT,
        "pair_count": 5,
        "source_state_indices": SOURCE_INDICES,
        "seed": 42,
        "pi05_replan_steps": 1,
        "rollout_physics_limit_m": 0.002,
        "artifact_sha256": observed_hashes,
        "selection_trajectory_sha256": trajectory_hashes,
        "scene_human_approval_present": True,
        "scene_human_approval_sha256": _sha256(approval_path),
        "pi05_smoke_authorized": True,
        "formal_authorized": False,
        "cosmos_authorized": False,
        "openvla_oft_authorized": False,
        "reuse_without_regeneration": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = verify_handoff(args.frozen_dir, args.preregistration, args.approval)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"Verdict: {VERDICT}")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
