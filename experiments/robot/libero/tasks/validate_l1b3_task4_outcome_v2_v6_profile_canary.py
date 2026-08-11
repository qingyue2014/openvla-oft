"""Fail-fast audit of the five Outcome V2 v6 scripted approach corridors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks import (
    validate_l1b3_task4_outcome_v2_v6_selection as selection,
)


PASS_VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V6_PROFILE_CANARY"
FAIL_VERDICT = "FAIL_L1B3_TASK4_OUTCOME_V2_V6_PROFILE_CANARY"
EXPECTED_OFFSETS = {
    "canonical_center": [0.0, 0.0],
    "stress_x_plus": [0.06, 0.0],
    "stress_x_minus": [-0.06, 0.06],
    "holdout_y_plus": [0.0, 0.06],
    "holdout_y_minus": [0.0, -0.06],
}


def _resample_link6_xy(trajectory: dict) -> np.ndarray | None:
    path = np.asarray(trajectory.get("body_pos__robot0_link6", ()), dtype=float)
    if (
        path.ndim != 2
        or path.shape[0] < 2
        or path.shape[1] < 2
        or not np.isfinite(path[:, :2]).all()
    ):
        return None
    samples = np.linspace(0.0, 1.0, num=128)
    source = np.linspace(0.0, 1.0, num=path.shape[0])
    return np.column_stack(
        [np.interp(samples, source, path[:, axis]) for axis in (0, 1)]
    )


def validate(args: argparse.Namespace) -> dict:
    controller_path = Path(args.controller_manifest)
    pairing_path = Path(args.pairing_json)
    missing = [path for path in (controller_path, pairing_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    profile_dirs = selection._profile_paths(args.profile_trajectory)
    controller_hash = selection._sha256(controller_path)
    failures: list[str] = []
    canary_contract = pairing.get("canary_contract", {})
    parent_pairing = Path(str(canary_contract.get("parent_pairing", "")))
    parent_hash = (
        selection._sha256(parent_pairing) if parent_pairing.is_file() else None
    )
    if not (
        pairing.get("family") == selection.FAMILY
        and pairing.get("seed") == 42
        and pairing.get("num_states") == 1
        and len(pairing.get("pairs", ())) == 1
        and pairing["pairs"][0].get("source_state_index") == 0
        and canary_contract.get("purpose") == "profile_diversity_fail_fast_only"
        and canary_contract.get("may_generate_or_refine_scene") is False
        and canary_contract.get("source_state_indices") == [0]
        and parent_hash == canary_contract.get("parent_pairing_sha256")
    ):
        failures.append("canary_pairing_contract_mismatch")
    observed_offsets = {
        profile: controller.get("profiles", {})
        .get(profile, {})
        .get("pregrasp_target_offset_xy_m")
        for profile in selection.PROFILES
    }
    if not (
        controller.get("id") == selection.SOURCE
        and controller.get("profile_order") == list(selection.PROFILES)
        and controller.get("obstacle_adaptive") is False
        and controller.get("learned_action_prefix") is False
        and observed_offsets == EXPECTED_OFFSETS
        and controller.get("diversity_gate", {}).get(
            "episode_zero_canary_before_full_pool"
        )
        is True
        and controller.get("diversity_gate", {}).get(
            "minimum_pairwise_path_separation_m"
        )
        == selection.MIN_LINK6_PATH_SEPARATION_M
    ):
        failures.append("controller_canary_contract_mismatch")

    paths: dict[str, np.ndarray] = {}
    trajectories = []
    for profile in selection.PROFILES:
        path = profile_dirs[profile] / "task4_ep000.npz"
        if not path.is_file():
            failures.append(f"trajectory_{profile}_missing")
            continue
        trajectory = load_trajectory(str(path))
        metadata = trajectory.get("metadata", {})
        link6 = _resample_link6_xy(trajectory)
        valid = bool(
            metadata.get("trajectory_source_label") == selection.SOURCE
            and metadata.get("trajectory_profile_id") == profile
            and metadata.get("trajectory_source_manifest_sha256")
            == controller_hash
            and metadata.get("model_trajectory_used") is False
            and metadata.get("controller_obstacle_adaptive") is False
            and metadata.get("cross_episode_grasp_cache_disabled") is True
            and metadata.get("success") is True
            and metadata.get("pregrasp_target_offset_xy_m")
            == EXPECTED_OFFSETS[profile]
            and link6 is not None
        )
        if not valid:
            failures.append(f"trajectory_{profile}_provenance_or_task")
        elif link6 is not None:
            paths[profile] = link6
        trajectories.append(
            {
                "profile": profile,
                "path": str(path),
                "sha256": selection._sha256(path),
                "valid": valid,
            }
        )

    pairwise = []
    minimum: float | None = None
    for left_index, left in enumerate(selection.PROFILES):
        for right in selection.PROFILES[left_index + 1 :]:
            separation: float | None = (
                float(np.linalg.norm(paths[left] - paths[right], axis=1).max())
                if left in paths and right in paths
                else None
            )
            valid = bool(
                separation is not None
                and np.isfinite(separation)
                and separation >= selection.MIN_LINK6_PATH_SEPARATION_M
            )
            if separation is not None and np.isfinite(separation):
                minimum = separation if minimum is None else min(minimum, separation)
            if not valid:
                failures.append(f"path_diversity_{left}__{right}")
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "maximum_time_normalized_xy_separation_m": separation,
                    "required_m": selection.MIN_LINK6_PATH_SEPARATION_M,
                    "valid": valid,
                }
            )
    passed = not failures
    record = {
        "schema_version": 1,
        "family": selection.FAMILY,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "episode_idx": 0,
        "selection_model": "none",
        "profile_offsets_xy_m": EXPECTED_OFFSETS,
        "metric": "maximum_time_normalized_pairwise_robot0_link6_xy_separation_128_samples",
        "minimum_observed_pairwise_separation_m": minimum,
        "required_pairwise_separation_m": selection.MIN_LINK6_PATH_SEPARATION_M,
        "trajectory_records": trajectories,
        "pairwise_records": pairwise,
        "controller_manifest": str(controller_path),
        "controller_manifest_sha256": controller_hash,
        "canary_pairing": str(pairing_path),
        "canary_pairing_sha256": selection._sha256(pairing_path),
        "native_pool_pairing": str(parent_pairing),
        "native_pool_pairing_sha256": parent_hash,
        "failures": sorted(set(failures)),
    }
    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report = Path(args.output_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# L1-B3 Task-4 Outcome V2 v6 profile canary",
                "",
                f"Verdict: **{record['verdict']}**",
                "",
                f"- Episode: `{record['episode_idx']}`",
                "- Minimum measured pairwise separation: "
                f"`{f'{minimum:.6f} m' if minimum is not None else 'n/a'}`",
                f"- Required: `{selection.MIN_LINK6_PATH_SEPARATION_M:.6f} m`",
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
        "--controller_manifest",
        default=str(
            tasks / f"{selection.FAMILY}_scripted_controller_ensemble.json"
        ),
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/logs/l1b3_task4_outcome_v2_v6_profile_canary_pairing.json",
    )
    parser.add_argument(
        "--profile_trajectory",
        action="append",
        default=[],
        metavar="PROFILE_ID=DIRECTORY",
        required=True,
    )
    parser.add_argument(
        "--output_manifest",
        default="experiments/logs/l1b3_task4_outcome_v2_v6_profile_canary.json",
    )
    parser.add_argument(
        "--output_report",
        default="experiments/logs/l1b3_task4_outcome_v2_v6_profile_canary.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
