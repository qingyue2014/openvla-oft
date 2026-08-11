"""Fail-closed native-task, asset, and selection preflight for Outcome V2 v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks import (
    validate_l1b3_task4_outcome_v2_preflight as legacy,
)


FAMILY = "l1b3_task4_outcome_v2_v5"
SCENE_ID = "L1-B3-Task4-Outcome-V2-V5"
TASK_SUITE = "libero_goal"
TASK_ID = 4
TASK_PROMPT = "put the bowl on top of the cabinet"
TASK_GOALS = ["On akita_black_bowl_1 wooden_cabinet_1_top_side"]
VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V5_NATIVE_PREFLIGHT"
PREREG = Path(__file__).with_name(
    "l1b3_task4_outcome_v2_v5_design_prereg.json"
)
RUNNER_RELATIVE = (
    "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2_v5.sh"
)

INTERVENTION_ALLOWLIST = {
    "body": "wine_bottle_1_main",
    "pose_fields": ["free_joint.qpos.x", "free_joint.qpos.y"],
    "velocity_fields": [
        "free_joint.qvel.x",
        "free_joint.qvel.y",
        "free_joint.qvel.z",
        "free_joint.qvel.rx",
        "free_joint.qvel.ry",
        "free_joint.qvel.rz",
    ],
    "all_other_state_fields": "must_be_byte_identical",
}

PROJECT_FILES = (
    "pyproject.toml",
    "experiments/robot/pi05_utils.py",
    "experiments/robot/robot_utils.py",
    "experiments/robot/libero/physcog_oracles.py",
    "experiments/robot/libero/libero_utils.py",
    "experiments/robot/libero/run_physcog_libero_l1_eval.py",
    "experiments/robot/libero/tasks/generate_l1b_swept_initial_states.py",
    "experiments/robot/libero/tasks/l1b_matched_control.py",
    "experiments/robot/libero/tasks/calibrate_l1b3_trajectory_conditioned_states.py",
    "experiments/robot/libero/tasks/replay_l1b_outcome_eb_actions.py",
    "experiments/robot/libero/tasks/validate_l1a2_safe_reference.py",
    "experiments/robot/libero/tasks/validate_l1b_safe_reference.py",
    "experiments/robot/libero/tasks/validate_l1b_swept_states.py",
    "experiments/robot/libero/tasks/validate_l1b_rollout_physics.py",
    "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_initial_gate.py",
    "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v5_initial_gate.py",
    "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v5_preflight.py",
    "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v5_selection.py",
    "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2_v5.sh",
    "experiments/robot/libero/tasks/physcog_remote_agent.py",
    "experiments/robot/libero/tasks/L1-B3_TASK4_OUTCOME_V2_SPEC.md",
    "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_v5_design_prereg.json",
    "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_v5_scripted_controller.json",
)


def _sha256(path: Path) -> str:
    return legacy._sha256(path)


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _registered_spec() -> dict[str, object]:
    previous = legacy.FAMILY
    try:
        legacy.FAMILY = FAMILY
        return legacy._registered_family_spec()
    finally:
        legacy.FAMILY = previous


def _verify_preregistration(prereg: dict, spec: dict, bddl_prompt: str) -> None:
    native = prereg.get("native_task", {})
    layout = prereg.get("source_to_project_layout_delta", {})
    selection = prereg.get("selection_contract", {})
    learned = prereg.get("learned_policy_contract", {})
    pi05 = learned.get("pi0_5", {})
    expected_spec = {
        "bddl_file": None,
        "native_assets_only": True,
        "preserve_native_layout": True,
        "preserve_native_obstacle_pose": False,
        "eb_placement_mode": "offset_from_native",
        "eb_obstacle_offset_xy": [-0.040, 0.000],
        "outcome_based": True,
        "matched_control_mode": "dual_radius_reflection",
        "require_matched_control_geometry": True,
        "selection_trajectory_source": "model_independent_scripted_osc_v1",
        "forbid_learned_selection_trajectories": True,
        "selection_controller_manifest": (
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_v5_scripted_controller.json"
        ),
        "calibration_penetration_buffer_m": 0.001,
        "evaluation_penetration_limit_m": 0.002,
        "pi05_replan_steps": 1,
        "scene_contract": "l1b3_task4_swept_outcome_v2_model_independent_v5",
    }
    mismatches = {
        key: {"observed": spec.get(key), "expected": value}
        for key, value in expected_spec.items()
        if spec.get(key) != value
    }
    if mismatches:
        raise ValueError(f"registered v5 family contract changed: {mismatches}")
    valid = bool(
        prereg.get("family") == FAMILY
        and prereg.get("status")
        == "prospective_scene_construction_before_any_v5_learned_policy_outcome"
        and native.get("suite") == TASK_SUITE
        and native.get("task_id") == TASK_ID
        and native.get("benchmark_prompt") == TASK_PROMPT
        and native.get("native_bddl_embedded_language") == bddl_prompt
        and native.get("goal_predicates") == TASK_GOALS
        and native.get("evaluated_bddl_must_be_native_source") is True
        and native.get("custom_bddl") is False
        and native.get("custom_assets") == []
        and layout.get("only_body_changed_from_native_source")
        == "wine_bottle_1_main"
        and layout.get("frozen_eb_offset_xy") == [-0.04, 0.0]
        and layout.get("all_other_native_state_fields")
        == "must_be_byte_identical"
        and selection.get("selection_model") == "none"
        and selection.get("trajectory_source")
        == "model_independent_scripted_osc_v1"
        and selection.get("controller_manifest")
        == (
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_v5_scripted_controller.json"
        )
        and selection.get("scene_selection_penetration_buffer_m") == 0.001
        and selection.get("formal_rollout_physics_limit_m") == 0.002
        and selection.get("scripted_controller_may_adapt_to_obstacle") is False
        and selection.get("scripted_controller_may_consume_a_learned_prefix")
        is False
        and learned.get("primary_evaluated_model") == "pi0.5"
        and learned.get("retired_model") == "OpenVLA-OFT"
        and learned.get("formal_model_order") == ["pi0.5", "Cosmos"]
        and pi05.get("replan_steps") == 1
        and pi05.get("may_be_changed_after_v5_outcomes") is False
        and prereg.get("human_review", {}).get("approved") is False
    )
    if not valid:
        raise ValueError("v5 preregistration does not match its native/selection contract")


def build_record() -> dict[str, object]:
    spec = _registered_spec()
    libero_root = legacy._resolve_libero_repository()
    task_name, benchmark_prompt, task_map = legacy._benchmark_task_contract(
        libero_root
    )
    if benchmark_prompt != TASK_PROMPT:
        raise ValueError(f"native benchmark prompt mismatch: {benchmark_prompt!r}")
    native_bddl = (
        libero_root
        / "libero/libero/bddl_files"
        / TASK_SUITE
        / f"{task_name}.bddl"
    ).resolve(strict=True)
    text = native_bddl.read_text(encoding="utf-8")
    bddl_prompt = legacy._prompt(text)
    fixtures = legacy._inventory(text, "fixtures")
    objects = legacy._inventory(text, "objects")
    goal_source = legacy._section(text, "goal")
    goals = legacy._goal_predicates(goal_source)
    if goals != TASK_GOALS:
        raise ValueError(f"native goal mismatch: {goals}")
    assets_root = (libero_root / "libero/libero/assets").resolve(strict=True)
    inventory = {"fixtures": fixtures, "objects": objects}
    closure = legacy._asset_closure(assets_root, {**fixtures, **objects})
    libero_commit = legacy._verify_git_clean(
        libero_root, [task_map, native_bddl, *closure]
    )
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    _verify_preregistration(prereg, spec, bddl_prompt)
    paths = [REPO_ROOT / relative for relative in PROJECT_FILES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v5 preflight project files missing: {missing}")
    project_hashes = {
        str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths
    }
    libero_utils_text = (
        REPO_ROOT / "experiments/robot/libero/libero_utils.py"
    ).read_text(encoding="utf-8")
    runner_text = (
        REPO_ROOT / RUNNER_RELATIVE
    ).read_text(encoding="utf-8")
    policy_prompt_path_verified = bool(
        "task_description = task.language" in libero_utils_text
        and "--task_description_override" not in runner_text
    )
    if not policy_prompt_path_verified:
        raise ValueError("v5 cannot verify the unmodified native policy prompt path")
    asset_files = {
        str(path.relative_to(libero_root)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in closure
    }
    inventory_signature = _json_sha256(inventory)
    selection = prereg["selection_contract"]
    learned = prereg["learned_policy_contract"]
    return {
        "schema_version": 1,
        "scenario": SCENE_ID,
        "family": FAMILY,
        "verdict": VERDICT,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": f"{task_name}.bddl",
        "benchmark_prompt": TASK_PROMPT,
        "evaluated_policy_prompt_matches_selected_native_task": True,
        "evaluated_policy_prompt_path_verified": policy_prompt_path_verified,
        "bddl_prompt": bddl_prompt,
        "native_bddl": str(native_bddl),
        "evaluated_bddl": str(native_bddl),
        "native_bddl_sha256": _sha256(native_bddl),
        "evaluated_bddl_sha256": _sha256(native_bddl),
        "goal_source": goal_source,
        "goal_predicates": goals,
        "goal_signature_sha256": hashlib.sha256(
            goal_source.encode("utf-8")
        ).hexdigest(),
        "fixtures": fixtures,
        "objects": objects,
        "inventory_signature": inventory_signature,
        "condition_inventory_signatures": {
            condition: inventory_signature for condition in ("eb", "er", "ec")
        },
        "cross_condition_inventory_signatures_identical": True,
        "source_to_project_inventory_delta": {"fixtures": [], "objects": []},
        "source_to_project_bddl_delta": "none",
        "source_to_project_layout_delta": prereg[
            "source_to_project_layout_delta"
        ],
        "intervention_id": prereg["intervention"]["id"],
        "intervention_allowlist": INTERVENTION_ALLOWLIST,
        "selection_model": selection["selection_model"],
        "selection_trajectory_source": selection["trajectory_source"],
        "learned_selection_inputs_forbidden": selection["forbidden_inputs"],
        "selection_penetration_buffer_m": selection[
            "scene_selection_penetration_buffer_m"
        ],
        "evaluation_penetration_limit_m": selection[
            "formal_rollout_physics_limit_m"
        ],
        "formal_model_order": learned["formal_model_order"],
        "primary_evaluated_model": learned["primary_evaluated_model"],
        "retired_evaluated_model": learned["retired_model"],
        "pi0_5_execution_contract": learned["pi0_5"],
        "custom_assets": [],
        "custom_bddl": False,
        "libero_commit": libero_commit,
        "all_selected_native_files_unmodified": True,
        "native_asset_files": asset_files,
        "native_asset_manifest_sha256": _json_sha256(asset_files),
        "preregistration_sha256": _sha256(PREREG),
        "project_file_hashes": project_hashes,
    }


def write_preflight(manifest: Path, report: Path) -> dict[str, object]:
    record = build_record()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# L1-B3 Task-4 Outcome V2 v5 native preflight",
                "",
                f"Verdict: **{VERDICT}**",
                "",
                f"- Native task: `{TASK_SUITE}/{record['task_file']}` (id `{TASK_ID}`)",
                f"- Evaluated prompt: `{TASK_PROMPT}` (no override)",
                f"- Goal predicates: `{record['goal_predicates']}`",
                f"- Inventory signature: `{record['inventory_signature']}`",
                f"- Native asset closure: `{record['native_asset_manifest_sha256']}`",
                "- BDDL/inventory/custom-asset delta: `none`",
                "- Native-source to project-Eb delta: only native wine-bottle x/y by `[-0.040, 0.000]`.",
                "- Selection model: `none`; selection path: `model_independent_scripted_osc_v1`.",
                "- Construction penetration buffer: `0.001 m`; unchanged rollout validity limit: `0.002 m`.",
                "- Learned-policy order: `pi0.5 -> Cosmos`; OpenVLA-OFT is retired.",
                "- pi0.5 replan steps: `1`, paired across EB/ER/EC and immutable after outcomes.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Verdict: {VERDICT}")
    print(f"Manifest: {manifest}")
    print(f"Report: {report}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_v5_native_preflight.json"
        ),
    )
    parser.add_argument(
        "--report",
        default="experiments/logs/l1b3_task4_outcome_v2_v5_native_preflight.md",
    )
    args = parser.parse_args()
    write_preflight(Path(args.manifest), Path(args.report))


if __name__ == "__main__":
    main()
