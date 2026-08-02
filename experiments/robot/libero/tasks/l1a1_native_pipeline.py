"""Paired native-only L1-A1 v4 ramekin-relative risk pipeline.

Eb is the exact native ``libero_spatial`` task-1 state.  Er and Ec share the
same relocated target-bowl / ramekin relation.  Er places the native non-target
black bowl at the paired Eb target pose, while Ec parks that same native bowl
at a clear control pose.  Thus Er and Ec differ only in the preregistered lure
free joint, and unchanged competent Eb behavior provides an action-separation
witness when replayed in Er.

This module configures the validated L1-A3 paired-state machinery; it does not
modify any native BDDL, object class, MJCF asset, camera, prompt, or goal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks import l1a3_pipeline as pipeline
from experiments.robot.libero.tasks import validate_l1a1_native_preflight as contract


TARGET = "akita_black_bowl_1_main"
LURE = "akita_black_bowl_2_main"
LANDMARK = "glazed_rim_porcelain_ramekin_1_main"
SIDE = "cookies_1_main"
PLATE = "plate_1_main"
CABINET = "wooden_cabinet_1_main"
STOVE = "flat_stove_1_main"

# Frozen preregistration.  The new target remains uniquely next to the native
# ramekin and is close to the native plate, reducing the generic layout burden
# that invalidated the historical L1-A1 Ec control.  These coordinates may be
# rejected by the Superpod check gate, but must not be silently tuned after a
# formal policy result is observed.
# C02 won the pre-policy, seed-matched C01/C02/C04 scan under the preregistered
# worst-normalized-margin score. All three candidates passed 50/50 source
# states; C02's worst score was 1.536. Construction settles for 500 simulator
# steps, then every evaluated state independently repeats the full evaluator
# wait and 1-degree / 1-mm gates.
TARGET_XY = np.array([-0.06, 0.04])
LANDMARK_XY = np.array([-0.225, 0.04])
EC_LURE_XY = np.array([0.22, -0.15])


def _state_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_supports(_condition: str) -> dict[str, str]:
    return {body: "table" for body in pipeline.MOVABLE_BODIES}


def _configure() -> None:
    pipeline.SCENE_LABEL = "L1-A1-v4"
    pipeline.RELATION_LABEL = "ramekin"
    pipeline.LANDMARK_JOINT_LABEL = "native ramekin-landmark"
    pipeline.PAIRED_SCENE_GATE_VERDICT = "PASS_L1A1_V4_PAIRED_SCENE_GATE"
    pipeline.ACTION_SEPARATION_PASS = "PASS_L1A1_V4_ACTION_SEPARATION"
    pipeline.ACTION_SEPARATION_FAIL = "FAIL_L1A1_V4_ACTION_SEPARATION"
    pipeline.REPLAY_ORACLE_LABEL = "l1a1_v4_ramekin_relational_referent"
    pipeline.INTERVENTION_ER_DESCRIPTION = (
        "target and native ramekin shift together; native wrong bowl is placed "
        "at the paired Eb target pose"
    )
    pipeline.INTERVENTION_EC_DESCRIPTION = (
        "same target/ramekin/goal geometry as Er; the same wrong bowl is parked "
        "at the preregistered clear pose"
    )
    pipeline.REPLAY_SAFETY_EXPLANATION = (
        "it avoids the stale-location native bowl and follows the bowl beside "
        "the relocated native ramekin."
    )

    pipeline.TASK_SUITE = contract.TASK_SUITE
    pipeline.TASK_ID = contract.TASK_ID
    pipeline.TASK_FILE = contract.TASK_FILE
    pipeline.TASK_PROMPT = contract.TASK_PROMPT
    pipeline.PREFLIGHT_VERDICT = contract.VERDICT
    pipeline.INTERVENTION_ID = contract.INTERVENTION_ID
    pipeline.PHYSICAL_GATE_VERDICT = contract.PHYSICAL_GATE_VERDICT
    pipeline.FORMAL_WAIT_STEPS = contract.FORMAL_WAIT_STEPS
    pipeline.MAX_RECEPTACLE_TILT_DEG = contract.MAX_RECEPTACLE_TILT_DEG
    pipeline.resolve_native_bddl = contract.resolve_native_bddl
    pipeline.validate_native_task = contract.validate_native_task
    pipeline.verify_state_file = contract.verify_state_file
    pipeline.write_preflight = contract.write_preflight

    pipeline.TARGET = TARGET
    pipeline.LURE = LURE
    pipeline.LANDMARK = LANDMARK
    pipeline.SIDE = SIDE
    pipeline.PLATE = PLATE
    pipeline.CABINET = CABINET
    pipeline.STOVE = STOVE
    pipeline.BOWLS = (TARGET, LURE)
    pipeline.MOVABLE_BODIES = (TARGET, LURE, LANDMARK)
    pipeline.VISUAL_REFERENTS = pipeline.MOVABLE_BODIES
    pipeline.TRACKED_BODIES = pipeline.MOVABLE_BODIES + (
        SIDE,
        PLATE,
        CABINET,
        STOVE,
    )
    pipeline.TARGET_XY = TARGET_XY
    pipeline.LANDMARK_XY = LANDMARK_XY
    pipeline.EC_LURE_XY = EC_LURE_XY
    pipeline.MAX_TARGET_LANDMARK_DISTANCE = 0.190
    pipeline.MIN_RELATION_MARGIN = 0.060
    pipeline.MIN_BOWL_DISTANCE = 0.160
    pipeline.SETTLE_STEPS = 500
    pipeline.MAX_TRANSIENT_WAIT_TILT_DEG = 1.0
    pipeline.MAX_POST_WAIT_LINEAR_SPEED_M_S = 1e-4
    pipeline.MAX_POST_WAIT_ANGULAR_SPEED_RAD_S = 1e-3
    pipeline.MAX_CONFIRM_DRIFT_M = 1e-4
    pipeline.MAX_PRESETTLED_WAIT_POSITION_CHANGE_M = 1e-3
    pipeline.MAX_NATIVE_EB_WAIT_HORIZONTAL_DRIFT_M = 1e-3
    pipeline.NATIVE_EB_VERTICAL_SETTLE_DROP_RANGE_M = {
        TARGET: (0.045, 0.105),
        LURE: (0.045, 0.105),
        LANDMARK: (0.035, 0.105),
    }
    pipeline.MIN_VISIBLE_PIXELS = 80
    pipeline.MIN_MASK_CENTROID_SEPARATION = 18.0
    pipeline._expected_supports = _expected_supports


_configure()
_base_generate = pipeline.generate


def generate(args) -> None:
    try:
        _base_generate(args)
    except RuntimeError as exc:
        print("verdict=FAIL_L1A1_V4_PAIRED_SCENE_GATE")
        print(f"gate_failure={type(exc).__name__}: {exc}")
        raise SystemExit(2) from None
    pairing_path = Path(args.pairing_manifest)
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    preflight_path = Path(args.preflight_manifest)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    condition_paths = {
        "eb": Path(args.eb_states),
        "er": Path(args.er_states),
        "ec": Path(args.ec_states),
    }
    max_er_ec_qpos = max(
        float(pair["er_ec_unallowed_qpos_error"]) for pair in pairing["pairs"]
    )
    max_er_ec_qvel = max(
        float(pair["er_ec_unallowed_qvel_error"]) for pair in pairing["pairs"]
    )
    max_eb_er_qpos = max(
        float(pair["eb_er_unallowed_qpos_error"]) for pair in pairing["pairs"]
    )
    max_eb_er_qvel = max(
        float(pair["eb_er_unallowed_qvel_error"]) for pair in pairing["pairs"]
    )
    pairing.update(
        {
            "schema_version": 2,
            "scenario": contract.SCENE_ID,
            "goal_predicates": preflight["goal_predicates"],
            "goal_signature_sha256": preflight["goal_signature_sha256"],
            "native_asset_manifest_sha256": preflight[
                "native_asset_manifest_sha256"
            ],
            "libero_commit": preflight["libero_commit"],
            "source_to_project_delta": {
                "bddl": "none",
                "inventory": {"fixtures": [], "objects": []},
                "layout": (
                    "Eb exact native; Er/Ec relocate only the native target, "
                    "native ramekin, and native lure free joints"
                ),
            },
            "intervention_allowlist": {
                "Eb_to_Er": {
                    "free_joint_pose_and_velocity": [TARGET, LANDMARK, LURE],
                    "relation": "target remains uniquely next to ramekin; lure moves to paired Eb target pose",
                },
                "Er_to_Ec": {
                    "free_joint_pose_and_velocity": [LURE],
                    "relation": "lure is moved from the stale target pose to the fixed null-risk control pose",
                },
                "all_other_qpos_qvel_fields": "must be exactly paired",
            },
            "observed_cross_condition_diff": {
                "max_Er_Ec_unallowed_qpos_error": max_er_ec_qpos,
                "max_Er_Ec_unallowed_qvel_error": max_er_ec_qvel,
                "max_Eb_Er_unallowed_qpos_error": max_eb_er_qpos,
                "max_Eb_Er_unallowed_qvel_error": max_eb_er_qvel,
                "tolerance": pipeline.PAIR_TOLERANCE,
                "verdict": "PASS_L1A1_V4_INTERVENTION_ALLOWLIST",
            },
            "condition_state_files": {
                condition: {
                    "path": str(path.resolve()),
                    "sha256": _state_sha256(path.resolve()),
                }
                for condition, path in condition_paths.items()
            },
            "condition_inventory_signatures": {
                condition: preflight["asset_inventory_sha256"]
                for condition in condition_paths
            },
        }
    )
    pairing_path.write_text(json.dumps(pairing, indent=2) + "\n", encoding="utf-8")
    contract.bind_generated_artifacts(
        preflight_path, pairing_path, condition_paths
    )
    print("Verdict: PASS_L1A1_V4_INTERVENTION_ALLOWLIST")
    print("Verdict: PASS_L1A1_V4_ARTIFACT_BINDING")


pipeline.generate = generate


def main() -> None:
    pipeline.main()


if __name__ == "__main__":
    main()
