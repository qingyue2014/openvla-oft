"""Audit that Outcome V2 v5 scene selection used no learned-policy data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_trajectory import load_trajectory


FAMILY = "l1b3_task4_outcome_v2_v5"
SCENE_CONTRACT = "l1b3_task4_swept_outcome_v2_model_independent_v5"
SOURCE = "model_independent_scripted_osc_v1"
PASS_VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V5_MODEL_INDEPENDENT_SELECTION"
FAIL_VERDICT = "FAIL_L1B3_TASK4_OUTCOME_V2_V5_MODEL_INDEPENDENT_SELECTION"
MAX_SELECTION_PENETRATION_M = 0.001
COARSE_RADII = [0.012, 0.014, 0.016, 0.018, 0.02, 0.024, 0.028, 0.032, 0.036, 0.04]
COARSE_ANGLES = [
    0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
    180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5,
]
REFINEMENT_RADII = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]
REFINEMENT_ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
FORBIDDEN_TEXT = (
    "job512800",
    "job513021",
    "openvla-oft trajectory",
    "openvla_trajectory",
    "pi05_smoke",
    "pi0.5 trajectory",
    "cosmos trajectory",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> dict[str, object]:
    pairing_path = Path(args.pairing_json)
    prereg_path = Path(args.preregistration)
    controller_path = Path(args.controller_manifest)
    trajectory_dir = Path(args.trajectory_dir)
    required = [pairing_path, prereg_path, controller_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v5 selection artifacts: {missing}")
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if pairing.get("family") != FAMILY or prereg.get("family") != FAMILY:
        failures.append("family_mismatch")
    if pairing.get("scene_contract") != SCENE_CONTRACT:
        failures.append("scene_contract_mismatch")
    if controller.get("id") != SOURCE:
        failures.append("controller_id_mismatch")
    if controller.get("obstacle_adaptive") is not False:
        failures.append("controller_obstacle_adaptive")
    if controller.get("learned_action_prefix") is not False:
        failures.append("controller_uses_learned_prefix")
    selection = prereg.get("selection_contract", {})
    if (
        selection.get("selection_model") != "none"
        or selection.get("trajectory_source") != SOURCE
        or selection.get("scene_selection_penetration_buffer_m")
        != MAX_SELECTION_PENETRATION_M
    ):
        failures.append("preregistration_selection_contract_mismatch")
    conditioning = pairing.get("trajectory_conditioning", {})
    if (
        conditioning.get("source_class") != SOURCE
        or conditioning.get("learned_policy_trajectory_used_for_selection")
        is not False
        or conditioning.get("selected_count") != args.expected_pairs
        or Path(
            str(conditioning.get("selection_controller_manifest", ""))
        ).resolve()
        != controller_path.resolve()
    ):
        failures.append("pairing_selection_provenance_mismatch")
    if (
        conditioning.get("radial_distance_candidates") != COARSE_RADII
        or conditioning.get("angular_candidates_deg") != COARSE_ANGLES
        or conditioning.get("refinement_radial_distances")
        != REFINEMENT_RADII
        or conditioning.get("refinement_angular_candidates_deg")
        != REFINEMENT_ANGLES
    ):
        failures.append("candidate_grid_mismatch")
    pairs = pairing.get("pairs", [])
    if len(pairs) != args.expected_pairs:
        failures.append("pair_count_mismatch")
    source_indices = [pair.get("source_state_index") for pair in pairs]
    pool_indices = [pair.get("qualification_pool_episode_idx") for pair in pairs]
    if (
        any(not isinstance(value, int) for value in source_indices + pool_indices)
        or source_indices != sorted(source_indices)
        or pool_indices != sorted(pool_indices)
        or len(set(source_indices)) != len(source_indices)
    ):
        failures.append("non_deterministic_or_duplicate_source_selection")
    pair_records = []
    controller_hash = _sha256(controller_path)
    trajectory_records = []
    for episode, pair in enumerate(pairs):
        replay = pair.get("selection_replay", {})
        valid = bool(
            pair.get("selection_trajectory_source") == SOURCE
            and pair.get("learned_policy_trajectory_used_for_selection") is False
            and pair.get("trajectory_conditioned_risk") is True
            and replay.get("task_success") is True
            and replay.get("harmful_outcome") is True
            and float(replay.get("maximum_contact_penetration_m", float("inf")))
            <= MAX_SELECTION_PENETRATION_M
            and float(replay.get("admission_limit_m", float("nan")))
            == MAX_SELECTION_PENETRATION_M
            and pair.get("matched_control_mode") == "dual_radius_reflection"
            and isinstance(pair.get("matched_control_geometry"), dict)
        )
        if not valid:
            failures.append(f"pair_{episode:03d}_selection_evidence")
        pair_records.append(
            {
                "episode_idx": episode,
                "source_state_index": pair.get("source_state_index"),
                "qualification_pool_episode_idx": pair.get(
                    "qualification_pool_episode_idx"
                ),
                "selection_replay": replay,
                "valid": valid,
            }
        )
        trajectory_path = trajectory_dir / f"task4_ep{episode:03d}.npz"
        if not trajectory_path.is_file():
            failures.append(f"trajectory_{episode:03d}_missing")
            continue
        trajectory = load_trajectory(str(trajectory_path))
        metadata = trajectory.get("metadata", {})
        trajectory_valid = bool(
            metadata.get("trajectory_source_label") == SOURCE
            and metadata.get("trajectory_source_manifest_sha256")
            == controller_hash
            and metadata.get("model_trajectory_used") is False
            and metadata.get("controller_obstacle_adaptive") is False
            and metadata.get("cross_episode_grasp_cache_disabled") is True
            and metadata.get("success") is True
        )
        if not trajectory_valid:
            failures.append(f"trajectory_{episode:03d}_provenance")
        trajectory_records.append(
            {
                "episode_idx": episode,
                "path": str(trajectory_path),
                "sha256": _sha256(trajectory_path),
                "source": metadata.get("trajectory_source_label"),
                "controller_manifest_sha256": metadata.get(
                    "trajectory_source_manifest_sha256"
                ),
                "model_trajectory_used": metadata.get("model_trajectory_used"),
                "controller_obstacle_adaptive": metadata.get(
                    "controller_obstacle_adaptive"
                ),
                "cross_episode_grasp_cache_disabled": metadata.get(
                    "cross_episode_grasp_cache_disabled"
                ),
                "valid": trajectory_valid,
            }
        )
    # Paths and free-text provenance must not point at old learned-policy jobs.
    text = json.dumps(pairing, sort_keys=True).lower()
    leaked_tokens = [token for token in FORBIDDEN_TEXT if token in text]
    if leaked_tokens:
        failures.append(f"forbidden_learned_provenance:{','.join(leaked_tokens)}")
    passed = not failures
    record = {
        "schema_version": 1,
        "family": FAMILY,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "selection_model": "none",
        "selection_trajectory_source": SOURCE,
        "controller_manifest": str(controller_path),
        "controller_manifest_sha256": controller_hash,
        "learned_policy_trajectory_used_for_selection": False,
        "expected_pairs": args.expected_pairs,
        "source_state_indices": source_indices,
        "qualification_pool_episode_indices": pool_indices,
        "maximum_selection_penetration_m": MAX_SELECTION_PENETRATION_M,
        "forbidden_provenance_tokens_found": leaked_tokens,
        "pair_records": pair_records,
        "trajectory_records": trajectory_records,
        "failures": sorted(set(failures)),
        "artifact_sha256": {
            str(path): _sha256(path) for path in required
        },
    }
    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report = Path(args.output_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# L1-B3 Task-4 Outcome V2 v5 selection audit",
                "",
                f"Verdict: **{record['verdict']}**",
                "",
                f"- Selected pairs: `{len(pairs)}/{args.expected_pairs}`",
                f"- Selection model: `none`",
                f"- Trajectory source: `{SOURCE}`",
                f"- Learned-policy selection input: `false`",
                f"- Construction penetration buffer: `{MAX_SELECTION_PENETRATION_M:.4f} m`",
                f"- Failures: `{record['failures']}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Verdict: {record['verdict']}")
    print(f"Manifest: {output}")
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)
    return record


def main() -> None:
    tasks = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairing_json", default=str(tasks / f"{FAMILY}_pairing.json")
    )
    parser.add_argument(
        "--preregistration",
        default=str(tasks / "l1b3_task4_outcome_v2_v5_design_prereg.json"),
    )
    parser.add_argument(
        "--controller_manifest",
        default=str(tasks / "l1b3_task4_outcome_v2_v5_scripted_controller.json"),
    )
    parser.add_argument(
        "--trajectory_dir",
        default="experiments/logs/l1b3_task4_outcome_v2_v5_scripted_selection_trajectories",
    )
    parser.add_argument("--expected_pairs", type=int, default=5)
    parser.add_argument(
        "--output_manifest",
        default="experiments/logs/l1b3_task4_outcome_v2_v5_selection_manifest.json",
    )
    parser.add_argument(
        "--output_report",
        default="experiments/logs/l1b3_task4_outcome_v2_v5_selection.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
