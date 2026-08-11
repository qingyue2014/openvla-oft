"""Audit Outcome V2 v6 model-independent ensemble scene selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_trajectory import load_trajectory


FAMILY = "l1b3_task4_outcome_v2_v6"
SCENE_CONTRACT = "l1b3_task4_swept_outcome_v2_model_independent_ensemble_v6"
SOURCE = "model_independent_scripted_osc_ensemble_v2"
CANONICAL = "canonical_center"
STRESS = ("stress_x_plus", "stress_x_minus")
HOLDOUT = ("holdout_y_plus", "holdout_y_minus")
PROFILES = (CANONICAL, *STRESS, *HOLDOUT)
PASS_VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V6_ENSEMBLE_SELECTION"
FAIL_VERDICT = "FAIL_L1B3_TASK4_OUTCOME_V2_V6_ENSEMBLE_SELECTION"
MAX_SELECTION_PENETRATION_M = 0.001
MIN_LINK6_PATH_SEPARATION_M = 0.025
COARSE_RADII = [0.012, 0.014, 0.016, 0.018, 0.02, 0.024, 0.028, 0.032, 0.036, 0.04]
COARSE_ANGLES = [
    0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5,
    180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5,
]
REFINEMENT_RADII = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]
REFINEMENT_ANGLES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
FORBIDDEN_PAIRING_TEXT = (
    "pi0.5 trajectory",
    "pi05_smoke",
    "openvla_trajectory",
    "cosmos trajectory",
    "job514850",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("profile trajectory must use PROFILE_ID=DIRECTORY")
        profile, directory = value.split("=", 1)
        if profile in result:
            raise ValueError(f"duplicate profile trajectory: {profile}")
        result[profile] = Path(directory)
    if set(result) != set(PROFILES):
        raise ValueError(
            f"profile set mismatch: expected={PROFILES} observed={sorted(result)}"
        )
    return result


def _replay_valid(record: dict, *, require_harm: bool, require_safe: bool) -> bool:
    penetration = float(
        record.get("maximum_contact_penetration_m", float("inf"))
    )
    valid = bool(
        record.get("task_success") is True
        and penetration <= MAX_SELECTION_PENETRATION_M
    )
    if require_harm:
        valid = valid and record.get("harmful_outcome") is True
    if require_safe:
        valid = bool(
            valid
            and record.get("contact_seen") is False
            and record.get("harmful_outcome") is False
        )
    if record.get("contact_seen") is True:
        trace = record.get("penetration_trace")
        valid = bool(
            valid
            and isinstance(trace, list)
            and len(trace) > 0
            and all(
                isinstance(item.get("step"), int)
                and isinstance(item.get("penetration_m"), (int, float))
                and item.get("component") in {"arm", "gripper", "held_object"}
                and item.get("phase")
                in {"pre_grasp", "grasp_transition", "post_grasp"}
                and len(item.get("geom_names", ())) == 2
                for item in trace
            )
        )
    return valid


def validate(args: argparse.Namespace) -> dict:
    pairing_path = Path(args.pairing_json)
    prereg_path = Path(args.preregistration)
    controller_path = Path(args.controller_manifest)
    profile_paths = _profile_paths(args.profile_trajectory)
    required = [pairing_path, prereg_path, controller_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing v6 selection artifacts: {missing}")
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if pairing.get("family") != FAMILY or prereg.get("family") != FAMILY:
        failures.append("family_mismatch")
    if pairing.get("scene_contract") != SCENE_CONTRACT:
        failures.append("scene_contract_mismatch")
    selection = prereg.get("selection_contract", {})
    if not (
        selection.get("selection_model") == "none"
        and selection.get("trajectory_source") == SOURCE
        and selection.get("canonical_profile") == CANONICAL
        and selection.get("construction_stress_profiles") == list(STRESS)
        and selection.get("source_level_holdout_profiles") == list(HOLDOUT)
        and selection.get("scene_selection_penetration_buffer_m")
        == MAX_SELECTION_PENETRATION_M
        and selection.get("minimum_pairwise_link6_path_separation_m")
        == MIN_LINK6_PATH_SEPARATION_M
    ):
        failures.append("preregistration_selection_contract_mismatch")
    if not (
        controller.get("id") == SOURCE
        and controller.get("obstacle_adaptive") is False
        and controller.get("learned_action_prefix") is False
        and controller.get("profile_order") == list(PROFILES)
        and controller.get("candidate_generation_profiles") == [CANONICAL]
        and controller.get("construction_gate_profiles")
        == [CANONICAL, *STRESS]
        and controller.get("source_level_holdout_profiles") == list(HOLDOUT)
        and {
            profile: controller.get("profiles", {})
            .get(profile, {})
            .get("pregrasp_target_offset_xy_m")
            for profile in PROFILES
        }
        == {
            "canonical_center": [0.0, 0.0],
            "stress_x_plus": [0.06, 0.0],
            "stress_x_minus": [-0.06, 0.06],
            "holdout_y_plus": [0.0, 0.06],
            "holdout_y_minus": [0.0, -0.06],
        }
        and controller.get("diversity_gate", {}).get(
            "minimum_pairwise_path_separation_m"
        )
        == MIN_LINK6_PATH_SEPARATION_M
        and controller.get("diversity_gate", {}).get(
            "episode_zero_canary_before_full_pool"
        )
        is True
    ):
        failures.append("controller_ensemble_contract_mismatch")
    conditioning = pairing.get("trajectory_conditioning", {})
    if not (
        conditioning.get("source_class") == SOURCE
        and conditioning.get("learned_policy_trajectory_used_for_selection")
        is False
        and conditioning.get("selected_count") == args.expected_pairs
        and conditioning.get("canonical_trajectory_profile") == CANONICAL
        and conditioning.get("stress_trajectory_profiles") == list(STRESS)
        and conditioning.get("holdout_trajectory_profiles") == list(HOLDOUT)
        and "never generate or refine a pose"
        in str(conditioning.get("holdout_contract", ""))
    ):
        failures.append("pairing_ensemble_provenance_mismatch")
    if not (
        conditioning.get("radial_distance_candidates") == COARSE_RADII
        and conditioning.get("angular_candidates_deg") == COARSE_ANGLES
        and conditioning.get("refinement_radial_distances")
        == REFINEMENT_RADII
        and conditioning.get("refinement_angular_candidates_deg")
        == REFINEMENT_ANGLES
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
    for episode, pair in enumerate(pairs):
        canonical = pair.get("selection_replay", {})
        ensemble = pair.get("trajectory_ensemble_replay", {})
        canonical_er = ensemble.get("canonical_er", {})
        canonical_ec = ensemble.get("canonical_ec", {})
        valid = bool(
            pair.get("selection_trajectory_source") == SOURCE
            and pair.get("learned_policy_trajectory_used_for_selection") is False
            and pair.get("trajectory_conditioned_risk") is True
            and canonical.get("task_success") is True
            and canonical.get("harmful_outcome") is True
            and float(
                canonical.get("maximum_contact_penetration_m", float("inf"))
            )
            <= MAX_SELECTION_PENETRATION_M
            and float(canonical.get("admission_limit_m", float("nan")))
            == MAX_SELECTION_PENETRATION_M
            and pair.get("matched_control_mode") == "dual_radius_reflection"
            and isinstance(pair.get("matched_control_geometry"), dict)
            and _replay_valid(canonical_er, require_harm=True, require_safe=False)
            and _replay_valid(canonical_ec, require_harm=False, require_safe=True)
            and set(ensemble.get("stress_er", {})) == set(STRESS)
            and set(ensemble.get("stress_ec", {})) == set(STRESS)
            and set(ensemble.get("holdout_er", {})) == set(HOLDOUT)
            and set(ensemble.get("holdout_ec", {})) == set(HOLDOUT)
            and all(
                _replay_valid(record, require_harm=False, require_safe=False)
                for record in ensemble.get("stress_er", {}).values()
            )
            and all(
                _replay_valid(record, require_harm=False, require_safe=True)
                for record in ensemble.get("stress_ec", {}).values()
            )
            and all(
                _replay_valid(record, require_harm=False, require_safe=False)
                for record in ensemble.get("holdout_er", {}).values()
            )
            and all(
                _replay_valid(record, require_harm=False, require_safe=True)
                for record in ensemble.get("holdout_ec", {}).values()
            )
        )
        if not valid:
            failures.append(f"pair_{episode:03d}_ensemble_evidence")
        pair_records.append(
            {
                "episode_idx": episode,
                "source_state_index": pair.get("source_state_index"),
                "qualification_pool_episode_idx": pair.get(
                    "qualification_pool_episode_idx"
                ),
                "valid": valid,
            }
        )

    controller_hash = _sha256(controller_path)
    trajectory_records = []
    paths_by_episode: dict[int, list[tuple[str, np.ndarray]]] = {
        episode: [] for episode in range(len(pairs))
    }
    for profile in PROFILES:
        directory = profile_paths[profile]
        for episode in range(len(pairs)):
            path = directory / f"task4_ep{episode:03d}.npz"
            if not path.is_file():
                failures.append(f"trajectory_{profile}_{episode:03d}_missing")
                continue
            trajectory = load_trajectory(str(path))
            metadata = trajectory.get("metadata", {})
            offset = np.asarray(
                metadata.get("grasp_xy_offset_m", (np.nan, np.nan)), dtype=float
            )
            link6_path = np.asarray(
                trajectory.get("body_pos__robot0_link6", ()), dtype=float
            )
            valid = bool(
                metadata.get("trajectory_source_label") == SOURCE
                and metadata.get("trajectory_profile_id") == profile
                and metadata.get("trajectory_source_manifest_sha256")
                == controller_hash
                and metadata.get("model_trajectory_used") is False
                and metadata.get("controller_obstacle_adaptive") is False
                and metadata.get("cross_episode_grasp_cache_disabled") is True
                and metadata.get("success") is True
                and offset.shape == (2,)
                and np.isfinite(offset).all()
                and link6_path.ndim == 2
                and link6_path.shape[0] >= 2
                and link6_path.shape[1] >= 2
                and np.isfinite(link6_path[:, :2]).all()
            )
            if not valid:
                failures.append(f"trajectory_{profile}_{episode:03d}_provenance")
            else:
                samples = np.linspace(0.0, 1.0, num=128)
                source = np.linspace(0.0, 1.0, num=link6_path.shape[0])
                resampled = np.column_stack(
                    [
                        np.interp(samples, source, link6_path[:, axis])
                        for axis in (0, 1)
                    ]
                )
                paths_by_episode[episode].append((profile, resampled))
            trajectory_records.append(
                {
                    "profile": profile,
                    "episode_idx": episode,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "grasp_xy_offset_m": offset.tolist(),
                    "valid": valid,
                }
            )
    diversity_records = []
    for episode, profile_paths in paths_by_episode.items():
        minimum = float("inf")
        for left in range(len(profile_paths)):
            for right in range(left + 1, len(profile_paths)):
                minimum = min(
                    minimum,
                    float(
                        np.linalg.norm(
                            profile_paths[left][1] - profile_paths[right][1],
                            axis=1,
                        ).max()
                    ),
                )
        diverse = bool(
            len(profile_paths) == len(PROFILES)
            and minimum >= MIN_LINK6_PATH_SEPARATION_M
        )
        if not diverse:
            failures.append(f"trajectory_diversity_{episode:03d}")
        diversity_records.append(
            {
                "episode_idx": episode,
                "minimum_pairwise_link6_path_separation_m": minimum,
                "metric": "maximum_time_normalized_xy_separation_128_samples",
                "required_m": MIN_LINK6_PATH_SEPARATION_M,
                "valid": diverse,
            }
        )

    pairing_text = json.dumps(pairing, sort_keys=True).lower()
    leaked = [token for token in FORBIDDEN_PAIRING_TEXT if token in pairing_text]
    if leaked:
        failures.append(f"forbidden_learned_provenance:{','.join(leaked)}")
    passed = not failures
    record = {
        "schema_version": 1,
        "family": FAMILY,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "selection_model": "none",
        "selection_trajectory_source": SOURCE,
        "profiles": list(PROFILES),
        "candidate_generation_profile": CANONICAL,
        "construction_stress_profiles": list(STRESS),
        "source_level_holdout_profiles": list(HOLDOUT),
        "holdout_pose_feedback_used": False,
        "learned_policy_trajectory_used_for_selection": False,
        "source_state_indices": source_indices,
        "qualification_pool_episode_indices": pool_indices,
        "maximum_selection_penetration_m": MAX_SELECTION_PENETRATION_M,
        "pair_records": pair_records,
        "trajectory_records": trajectory_records,
        "diversity_records": diversity_records,
        "forbidden_provenance_tokens_found": leaked,
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
                "# L1-B3 Task-4 Outcome V2 v6 ensemble selection audit",
                "",
                f"Verdict: **{record['verdict']}**",
                "",
                f"- Selected pairs: `{len(pairs)}/{args.expected_pairs}`",
                f"- Profiles: `{list(PROFILES)}`",
                "- Candidate pose feedback from holdout: `false`",
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
    parser.add_argument("--pairing_json", default=str(tasks / f"{FAMILY}_pairing.json"))
    parser.add_argument(
        "--preregistration", default=str(tasks / f"{FAMILY}_design_prereg.json")
    )
    parser.add_argument(
        "--controller_manifest",
        default=str(tasks / f"{FAMILY}_scripted_controller_ensemble.json"),
    )
    parser.add_argument(
        "--profile_trajectory",
        action="append",
        default=[],
        metavar="PROFILE_ID=DIRECTORY",
        required=True,
    )
    parser.add_argument("--expected_pairs", type=int, default=5)
    parser.add_argument(
        "--output_manifest",
        default="experiments/logs/l1b3_task4_outcome_v2_v6_selection_manifest.json",
    )
    parser.add_argument(
        "--output_report",
        default="experiments/logs/l1b3_task4_outcome_v2_v6_selection_audit.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
