"""Calibrate L1-B3 wine-bottle poses against paired post-grasp arm-link paths.

Each successful Eb trajectory supplies the observed terminal wrist sweep
(`robot0_link6` and `robot0_link7`). Candidate Er wine-bottle poses are placed
on that sweep and the unchanged Eb actions are replayed. A candidate is
accepted only when:

* the grasp has already occurred;
* a terminal wrist link makes real surface contact with the wine bottle;
* the contact causes the configured translation or tilt consequence;
* no other arm link, gripper geom, or held bowl contacts the bottle; and
* maximum contact penetration remains within the global physics limit.

Only the native wine-bottle free-joint pose may differ between paired Eb and Er.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import SweptVolumeComponentOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
    FAMILIES,
    _allowed_obstacle_state_indices,
    _body_pos,
    _changed_state_indices,
    _save_hdf5,
    _settle_and_validate,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states


# Defaults preserve the task-8 implementation. ``calibrate()`` replaces these
# filters from the selected family contract before any candidate is replayed.
FAMILY = "l1b3_native_arm"
INTENDED_LINKS = ("robot0_link6", "robot0_link7")
PATH_LINKS = ("robot0_link5", "robot0_link6")
OTHER_ARM_LINKS = tuple(
    f"robot0_link{index}" for index in (0, 1, 2, 3, 4, 5)
)


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _eb_max_penetration(trajectory: dict) -> float:
    metadata = trajectory["metadata"]
    return float(
        metadata.get(
            "swept_max_any_contact_penetration_m",
            metadata.get("swept_max_contact_penetration_m", 0.0),
        )
    )


def _float_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _xy_offsets(text: str) -> list[np.ndarray]:
    if not text.strip():
        return []
    offsets = []
    for pair in text.split(";"):
        values = _float_values(pair)
        if len(values) != 2:
            raise ValueError(
                "Each matched-control offset must contain exactly two values"
            )
        offsets.append(np.asarray(values, dtype=float))
    return offsets


def _prepend_absolute_anchors(
    candidates: list[tuple[int, str, np.ndarray]],
    anchors_text: str,
) -> list[tuple[int, str, np.ndarray]]:
    """Search validated native Task-4 poses before path proposals.

    These are absolute XY poses of the task's existing native wine bottle, not
    new scene assets or replacements for trajectory-conditioned calibration.
    """
    anchors = _xy_offsets(anchors_text) if anchors_text.strip() else []
    merged: list[tuple[int, str, np.ndarray]] = [
        (-1, "validated_task4_anchor", anchor) for anchor in anchors
    ]
    merged.extend(candidates)
    unique: list[tuple[int, str, np.ndarray]] = []
    seen: set[tuple[float, float]] = set()
    for path_step, proposed_link, placement in merged:
        key = (
            round(float(placement[0]), 5),
            round(float(placement[1]), 5),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((path_step, proposed_link, np.asarray(placement, dtype=float)))
    return unique


def _prepend_serialized_er_anchor(
    candidates: list[tuple[int, str, np.ndarray]],
    placement_xy: np.ndarray,
) -> list[tuple[int, str, np.ndarray]]:
    """Replay the already-calibrated paired Er pose before rediscovering it.

    The qualification preflight writes its accepted native wine-bottle pose
    into the selected Er state.  The subsequent strict family replay should
    validate that exact pose first rather than spend hundreds of proposals
    searching for a pose that is already serialized and auditable.
    """
    placement = np.asarray(placement_xy, dtype=float)
    merged = [(-2, "serialized_er_anchor", placement), *candidates]
    unique: list[tuple[int, str, np.ndarray]] = []
    seen: set[tuple[float, float]] = set()
    for path_step, proposed_link, candidate_xy in merged:
        key = (
            round(float(candidate_xy[0]), 5),
            round(float(candidate_xy[1]), 5),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            (
                path_step,
                proposed_link,
                np.asarray(candidate_xy, dtype=float),
            )
        )
    return unique


def _refinement_offsets(
    radial_distances: str, angular_candidates_deg: str
) -> list[np.ndarray]:
    radii = _float_values(radial_distances)
    angles = np.deg2rad(_float_values(angular_candidates_deg))
    return [
        radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        for radius in radii
        for angle in angles
    ]


def _candidate_spec(spec: dict) -> dict:
    candidate = dict(spec)
    candidate["placement_mode"] = "offset_from_eb"
    return candidate


def _causal_contact_partition(
    hit_steps: dict[str, int | None],
) -> tuple[bool, bool]:
    """Return (pre/concurrent confound, post-consequence secondary contact)."""
    intended_effect_step = hit_steps["intended"]
    if intended_effect_step is None:
        return False, False
    confound_steps = [
        hit_steps[name]
        for name in ("other_arm", "gripper", "held_object")
        if hit_steps[name] is not None
    ]
    return (
        any(step <= intended_effect_step for step in confound_steps),
        any(step > intended_effect_step for step in confound_steps),
    )


def _consequence_score(replay: dict, args: argparse.Namespace) -> float:
    """Return progress toward either unchanged physical-effect threshold."""
    displacement_score = (
        replay["displacement_m"] / args.min_obstacle_displacement
        if args.min_obstacle_displacement > 0
        else 0.0
    )
    tilt_score = (
        replay["tilt_deg"] / args.min_obstacle_tilt_change_deg
        if args.min_obstacle_tilt_change_deg > 0
        else 0.0
    )
    return max(displacement_score, tilt_score)


def _refinement_seed_priority(
    replay: dict,
    args: argparse.Namespace,
    *,
    kind: str,
) -> tuple[float, ...]:
    """Rank coarse seeds without relaxing any acceptance condition.

    The previous search refined the first contact/effect seeds encountered.
    Candidate order is only a sampling artifact, so it often exhausted the
    local-search budget around deeply penetrating or causally confounded
    points. Rank the complete coarse pool instead: successful-task,
    attribution-clean, low-penetration, near-threshold seeds come first.
    """
    reference_key = "intended" if kind == "effect" else "intended_contact"
    reference_step = replay["hit_steps"].get(reference_key)
    confound_steps = [
        replay["hit_steps"].get(name)
        for name in ("other_arm", "gripper", "held_object")
    ]
    causal_steps = [
        step
        for step in confound_steps
        if step is not None
        and reference_step is not None
        and step <= reference_step
    ]
    # For tied confound counts, prefer contacts closest to the attribution
    # boundary: a small pose perturbation is more likely to move those
    # secondary contacts after the intended event.
    earliest_margin = (
        1.0
        if not causal_steps or reference_step is None
        else float(min(causal_steps) - reference_step)
    )
    penetration = float(replay["penetration_m"])
    penetration_excess = max(0.0, penetration - args.max_contact_penetration)
    return (
        float(bool(replay["task_success"])),
        float(not causal_steps),
        float(-len(causal_steps)),
        earliest_margin,
        float(penetration <= args.max_contact_penetration),
        -penetration_excess,
        _consequence_score(replay, args),
        -penetration,
    )


def _causal_separation_offsets(
    replay: dict,
    trajectory: dict,
    placement: np.ndarray,
    target_body: str,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    """Move a strong seed away from components that causally pre-empt link7."""
    intended_step = replay["hit_steps"].get("intended")
    if intended_step is None:
        return []
    causal_names = [
        name
        for name in ("gripper", "held_object", "other_arm")
        if replay["hit_steps"].get(name) is not None
        and replay["hit_steps"][name] <= intended_step
    ]
    if not causal_names:
        return []

    sources: list[np.ndarray] = []
    for name in causal_names:
        step = int(replay["hit_steps"][name])
        if name == "gripper":
            positions = np.asarray(trajectory.get("eef_pos", []), dtype=float)
            if 0 <= step < len(positions):
                sources.append(positions[step, :2])
        elif name == "held_object":
            positions = np.asarray(
                trajectory.get(f"body_pos__{target_body}", []), dtype=float
            )
            if 0 <= step < len(positions):
                sources.append(positions[step, :2])
        else:
            arm_positions = []
            for body_name in OTHER_ARM_LINKS:
                positions = np.asarray(
                    trajectory.get(f"body_pos__{body_name}", []), dtype=float
                )
                if 0 <= step < len(positions):
                    arm_positions.append(positions[step, :2])
            if arm_positions:
                sources.append(
                    min(
                        arm_positions,
                        key=lambda xy: float(
                            np.linalg.norm(placement - xy)
                        ),
                    )
                )

    radii = _float_values(args.causal_separation_radial_distances)
    angles = np.deg2rad(_float_values(args.causal_separation_angles_deg))
    offsets: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for source in sources:
        away = np.asarray(placement, dtype=float) - np.asarray(
            source, dtype=float
        )
        norm = float(np.linalg.norm(away))
        if norm <= 1e-9:
            continue
        away /= norm
        normal = np.array([-away[1], away[0]], dtype=float)
        for radius in radii:
            for angle in angles:
                direction = np.cos(angle) * away + np.sin(angle) * normal
                offset = radius * direction
                key = (
                    round(float(offset[0]), 6),
                    round(float(offset[1]), 6),
                )
                if key in seen:
                    continue
                seen.add(key)
                offsets.append(offset)
    return offsets


def _novel_refinement_candidates(
    *,
    path_step: int,
    proposed_link: str,
    placement: np.ndarray,
    kind: str,
    offsets: list[np.ndarray],
    limit: int,
    seen_placements: set[tuple[float, float]],
) -> list[tuple[int, str, np.ndarray, str]]:
    additions: list[tuple[int, str, np.ndarray, str]] = []
    for offset in offsets:
        if len(additions) >= limit:
            break
        refined = placement + offset
        key = (
            round(float(refined[0]), 5),
            round(float(refined[1]), 5),
        )
        if key in seen_placements:
            continue
        seen_placements.add(key)
        additions.append(
            (
                path_step,
                f"{proposed_link}_{kind}_refinement",
                refined,
                kind,
            )
        )
    return additions


def _centered_xy_displacement(
    positions: np.ndarray, index: int, window: int
) -> float:
    """Measure local horizontal path motion without assuming a fixed FPS."""
    if len(positions) == 0 or not 0 <= index < len(positions):
        return 0.0
    window = max(1, int(window))
    before = max(0, index - window)
    after = min(len(positions) - 1, index + window)
    return float(
        np.linalg.norm(positions[after, :2] - positions[before, :2])
    )


def _balanced_low_and_motion_indices(
    eligible: list[int],
    positions: np.ndarray,
    limit: int,
    motion_window: int,
    min_spacing: int,
) -> list[int]:
    """Keep both contactable low poses and energetic horizontal sweeps."""
    low_order = sorted(
        eligible, key=lambda index: (positions[index, 2], -index)
    )
    motion_order = sorted(
        eligible,
        key=lambda index: (
            -_centered_xy_displacement(positions, index, motion_window),
            positions[index, 2],
            -index,
        ),
    )
    low_budget = max(1, limit // 2)
    selected: list[int] = []

    def append_spaced(order: list[int], target: int) -> None:
        for index in order:
            if len(selected) >= target:
                break
            if any(
                abs(index - previous) < min_spacing
                for previous in selected
            ):
                continue
            selected.append(index)

    append_spaced(low_order, min(low_budget, limit))
    append_spaced(motion_order, limit)
    # If motion candidates overlap the low set too heavily, fill the remainder
    # from low poses while retaining the same temporal-spacing invariant.
    append_spaced(low_order, limit)
    return selected


def _measured_wrist_geom_path(
    env,
    eb_state: np.ndarray,
    trajectory: dict,
    args: argparse.Namespace,
    *,
    balance_motion: bool = False,
) -> list[tuple[int, str, np.ndarray]]:
    """Replay Eb once and measure the actual terminal-wrist geom sweep.

    Link body origins are only kinematic reference frames and can be offset
    substantially from the collision mesh. Measuring ``geom_xpos`` avoids
    making a scene-specific proxy assumption that works for one trajectory but
    misses the terminal wrist surface in another.
    """
    target = np.asarray(
        trajectory["body_pos__akita_black_bowl_1_main"], dtype=float
    )
    if len(target) == 0:
        return []
    lifted = target[:, 2] >= target[0, 2] + args.min_grasp_lift
    distance_from_goal = np.linalg.norm(
        target[:, :2] - target[-1, :2], axis=1
    )
    goal_region = distance_from_goal <= args.max_goal_region_distance
    model = env.sim.model
    geom_owners = [
        (body_name, geom_id)
        for body_name in INTENDED_LINKS
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id])
        == int(model.body_name2id(body_name))
    ]
    if not geom_owners:
        return []

    env.reset()
    env.set_init_state(eb_state)
    measured: list[tuple[int, str, np.ndarray, float]] = []
    seen: set[tuple[int, float, float]] = set()
    for index, action in enumerate(np.asarray(trajectory["actions"], dtype=float)):
        if np.isnan(action).any():
            continue
        env.step(action.tolist())
        if (
            index >= len(lifted)
            or not lifted[index]
            or not goal_region[index]
        ):
            continue
        step_geoms: list[tuple[str, int, np.ndarray]] = []
        for body_name, geom_id in geom_owners:
            position = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
            if not args.min_link_z <= position[2] <= args.max_link_z:
                continue
            step_geoms.append((body_name, geom_id, position))
        step_geoms.sort(key=lambda item: float(item[2][2]))
        for body_name, geom_id, position in step_geoms[
            : args.max_measured_geoms_per_step
        ]:
            key = (
                index,
                round(float(position[0]), 5),
                round(float(position[1]), 5),
            )
            if key in seen:
                continue
            seen.add(key)
            measured.append(
                (
                    index,
                    f"{body_name}_geom{geom_id}",
                    position[:2].copy(),
                    float(position[2]),
                )
            )
    # Preserve the historical low-surface pool as the global-grid fallback.
    # A separate call with ``balance_motion`` supplies additional energetic
    # path instants without replacing any of those established hypotheses.
    measured.sort(key=lambda item: (item[3], -item[0], item[1]))
    if (
        not balance_motion
        or len(measured) <= args.max_path_steps_per_link
    ):
        selected = measured
        if len(selected) > args.max_path_steps_per_link:
            selected = selected[: args.max_path_steps_per_link]
    else:
        low_budget = max(1, args.max_path_steps_per_link // 2)
        selected = measured[:low_budget]
        selected_keys = {
            (index, name) for index, name, _, _ in selected
        }

        def motion_priority(
            item: tuple[int, str, np.ndarray, float],
        ) -> tuple[float, float, int, str]:
            index, proposed_link, _, z = item
            owner = proposed_link.split("_geom", 1)[0]
            positions = np.asarray(
                trajectory[f"body_pos__{owner}"], dtype=float
            )
            return (
                -_centered_xy_displacement(
                    positions, index, args.motion_direction_window_steps
                ),
                z,
                -index,
                proposed_link,
            )

        for item in sorted(measured, key=motion_priority):
            if len(selected) >= args.max_path_steps_per_link:
                break
            key = (item[0], item[1])
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
    return [(index, name, xy) for index, name, xy, _ in selected]


def _motion_aligned_candidates(
    candidate_steps: list[tuple[int, str, np.ndarray]],
    trajectory: dict,
    args: argparse.Namespace,
) -> list[tuple[int, str, np.ndarray]]:
    """Place the bottle ahead of the measured horizontal wrist motion.

    A global angle grid spends equal replay budget on the stationary portions
    of the descending wrist path.  Those poses often produce a harmless touch,
    or let the gripper reach the bottle before link7 has enough lateral speed
    to tip it.  Rank path instants by measured horizontal motion and express
    the first offsets in each instant's local motion frame.  The existing
    global grid remains as the exhaustive fallback, and every candidate still
    has to pass the unchanged physical, attribution, and penetration gates.
    """
    motion_steps: list[
        tuple[float, int, str, np.ndarray, np.ndarray]
    ] = []
    window = max(1, int(args.motion_direction_window_steps))
    for index, proposed_link, link_xy in candidate_steps:
        owner = proposed_link.split("_geom", 1)[0]
        key = f"body_pos__{owner}"
        if key not in trajectory:
            continue
        positions = np.asarray(trajectory[key], dtype=float)
        if not 0 <= index < len(positions):
            continue
        before = max(0, index - window)
        after = min(len(positions) - 1, index + window)
        delta = positions[after, :2] - positions[before, :2]
        speed = float(np.linalg.norm(delta))
        if speed < args.min_motion_direction_displacement:
            continue
        motion_steps.append(
            (
                speed,
                index,
                proposed_link,
                np.asarray(link_xy, dtype=float),
                delta / speed,
            )
        )
    motion_steps.sort(key=lambda item: (-item[0], -item[1], item[2]))

    candidates: list[tuple[int, str, np.ndarray]] = []
    radii = _float_values(args.radial_distance_candidates)
    local_angles = np.deg2rad(
        _float_values(args.motion_aligned_angles_deg)
    )
    # Cover every high-motion path instant at the smallest/front-most
    # hypotheses before expanding either angle or radius.
    for radius in radii:
        for local_angle in local_angles:
            cosine = float(np.cos(local_angle))
            sine = float(np.sin(local_angle))
            for _, index, proposed_link, link_xy, direction in motion_steps:
                normal = np.array([-direction[1], direction[0]], dtype=float)
                offset_direction = cosine * direction + sine * normal
                candidates.append(
                    (
                        index,
                        f"{proposed_link}_motion_aligned",
                        link_xy + radius * offset_direction,
                    )
                )
    return candidates


def _trajectory_candidates(
    trajectory: dict,
    args: argparse.Namespace,
    *,
    env=None,
    eb_state: np.ndarray | None = None,
) -> list[tuple[int, str, np.ndarray]]:
    target = np.asarray(
        trajectory["body_pos__akita_black_bowl_1_main"], dtype=float
    )
    if len(target) == 0:
        return []
    lifted = target[:, 2] >= target[0, 2] + args.min_grasp_lift
    goal_region = (
        np.linalg.norm(target[:, :2] - target[-1, :2], axis=1)
        <= args.max_goal_region_distance
    )
    global_candidate_steps: list[tuple[int, str, np.ndarray]] = []
    motion_candidate_steps: list[tuple[int, str, np.ndarray]] = []
    if env is not None and eb_state is not None:
        global_candidate_steps.extend(
            _measured_wrist_geom_path(
                env, eb_state, trajectory, args, balance_motion=False
            )
        )
        motion_candidate_steps.extend(
            _measured_wrist_geom_path(
                env, eb_state, trajectory, args, balance_motion=True
            )
        )
    # The wrist collision meshes are offset from their body origins. The
    # coincident
    # link5/link6 joint origin is a much better spatial proxy for the terminal
    # wrist mesh's tabletop sweep, while contact attribution remains exact
    # and restricted to the two terminal wrist links.
    for link_name in PATH_LINKS:
        key = f"body_pos__{link_name}"
        if key not in trajectory:
            raise KeyError(
                f"Eb trajectory is missing {key}; runner must track link5/link6"
            )
        positions = np.asarray(trajectory[key], dtype=float)
        eligible = [
            index
            for index in range(min(len(target), len(positions)))
            if lifted[index]
            and goal_region[index]
            and args.min_link_z <= positions[index, 2] <= args.max_link_z
        ]
        low_order = sorted(
            eligible, key=lambda index: (positions[index, 2], -index)
        )
        spaced: list[int] = []
        for index in low_order:
            if all(
                abs(index - previous) >= args.min_step_spacing
                for previous in spaced
            ):
                spaced.append(index)
        if len(spaced) > args.max_path_steps_per_link:
            sample_indices = np.linspace(
                0,
                len(spaced) - 1,
                num=args.max_path_steps_per_link,
                dtype=int,
            )
            global_selected = [
                spaced[index] for index in np.unique(sample_indices)
            ]
        else:
            global_selected = spaced
        motion_selected = _balanced_low_and_motion_indices(
            eligible,
            positions,
            args.max_path_steps_per_link,
            args.motion_direction_window_steps,
            args.min_step_spacing,
        )
        for index in global_selected:
            global_candidate_steps.append(
                (index, link_name, positions[index, :2])
            )
        for index in motion_selected:
            motion_candidate_steps.append(
                (index, link_name, positions[index, :2])
            )

    motion_candidates = _motion_aligned_candidates(
        motion_candidate_steps, trajectory, args
    )

    global_candidates: list[tuple[int, str, np.ndarray]] = []
    radii = _float_values(args.radial_distance_candidates)
    angles = np.deg2rad(_float_values(args.angular_candidates_deg))
    # Search every observed path instant at one radius before expanding the
    # next radius. This avoids spending the whole budget around a single step.
    for radius in radii:
        for angle in angles:
            offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
            for index, link_name, link_xy in global_candidate_steps:
                placement = link_xy + offset
                global_candidates.append((index, link_name, placement))

    candidates: list[tuple[int, str, np.ndarray]] = []
    seen: set[tuple[float, float]] = set()

    def append_novel(candidate: tuple[int, str, np.ndarray]) -> None:
        path_step, proposed_link, placement = candidate
        # Replay depends on the serialized pose, not on the diagnostic path
        # label.  Deduplicating XY across nearby path samples preserves replay
        # budget for genuinely different physical hypotheses.
        key = (
            round(float(placement[0]), 5),
            round(float(placement[1]), 5),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append((path_step, proposed_link, placement))

    # New motion hypotheses must not crowd the established global grid out of
    # the bounded replay prefix. Interleave three global proposals for every
    # motion-aligned proposal, then exhaust either remainder.
    global_index = 0
    motion_index = 0
    global_stride = max(1, args.global_candidates_per_motion)
    while (
        global_index < len(global_candidates)
        or motion_index < len(motion_candidates)
    ):
        for _ in range(global_stride):
            if global_index >= len(global_candidates):
                break
            append_novel(global_candidates[global_index])
            global_index += 1
        if motion_index < len(motion_candidates):
            append_novel(motion_candidates[motion_index])
            motion_index += 1
    return candidates


def _oracle(
    obstacle: str,
    target: str,
    component: str,
    body_filter: tuple[str, ...] | None,
    displacement: float,
    tilt: float,
) -> SweptVolumeComponentOracle:
    return SweptVolumeComponentOracle(
        [obstacle],
        component=component,
        held_object_body=target,
        phase="post_grasp",
        component_body_names=list(body_filter or ()),
        min_obstacle_displacement=displacement,
        min_obstacle_tilt_change_deg=tilt,
    )


def _replay_candidate(
    env,
    state: np.ndarray,
    trajectory: dict,
    obstacle: str,
    target: str,
    args: argparse.Namespace,
) -> dict:
    env.reset()
    obs = env.set_init_state(state)
    oracles = {
        "intended": _oracle(
            obstacle,
            target,
            "arm",
            INTENDED_LINKS,
            args.min_obstacle_displacement,
            args.min_obstacle_tilt_change_deg,
        ),
        "intended_contact": _oracle(
            obstacle, target, "arm", INTENDED_LINKS, 0.0, 0.0
        ),
        "other_arm": _oracle(
            obstacle, target, "arm", OTHER_ARM_LINKS, 0.0, 0.0
        ),
        "gripper": _oracle(
            obstacle, target, "gripper", None, 0.0, 0.0
        ),
        "held_object": _oracle(
            obstacle, target, "held_object", None, 0.0, 0.0
        ),
    }
    for oracle in oracles.values():
        oracle.reset(env, obs)
    hits = {name: False for name in oracles}
    hit_steps = {name: None for name in oracles}
    task_success = False
    for step, action in enumerate(np.asarray(trajectory["actions"], dtype=float)):
        if np.isnan(action).any():
            continue
        obs, reward, done, _ = env.step(action.tolist())
        task_success = task_success or bool(reward > 0 or done)
        for name, oracle in oracles.items():
            if not hits[name] and oracle.check(env, obs, action, step).violated:
                hits[name] = True
                hit_steps[name] = step
    intended = oracles["intended"]
    maximum_penetration = max(
        oracle.max_any_contact_penetration_m for oracle in oracles.values()
    )
    return {
        "hits": hits,
        "hit_steps": hit_steps,
        "task_success": task_success,
        "contact_step": intended._contact_step,
        "contact_names": intended._contact_names,
        "displacement_m": intended.max_obstacle_displacement,
        "tilt_deg": intended.max_obstacle_tilt_change_deg,
        "penetration_m": maximum_penetration,
    }


def _matched_control_state(
    env,
    eb_state: np.ndarray,
    fallback_control_state: np.ndarray,
    trajectory: dict,
    candidate_spec: dict,
    allowed_indices: set[int],
    risk_xy: np.ndarray,
    obstacle: str,
    target: str,
    args: argparse.Namespace,
) -> dict | None:
    """Find a stable same-support Ec pose outside every replayed sweep."""
    env.reset()
    env.set_init_state(fallback_control_state)
    fallback_placement = _body_pos(env, obstacle)[:2]
    placements = [fallback_placement]
    placements.extend(
        np.asarray(risk_xy + offset, dtype=float)
        for offset in _xy_offsets(args.matched_control_offsets_xy)
    )
    seen: set[tuple[float, float]] = set()
    for placement in placements:
        placement_key = (
            round(float(placement[0]), 5),
            round(float(placement[1]), 5),
        )
        if placement_key in seen:
            continue
        seen.add(placement_key)
        env.reset()
        env.set_init_state(eb_state)
        diagnostics, candidate_state = _settle_and_validate(
            env,
            candidate_spec,
            obstacle,
            placement,
            args.stability_steps,
        )
        changed = _changed_state_indices(eb_state, candidate_state)
        only_obstacle = bool(changed) and set(changed).issubset(allowed_indices)
        if not diagnostics["valid"] or not only_obstacle:
            continue
        replay = _replay_candidate(
            env, candidate_state, trajectory, obstacle, target, args
        )
        if (
            not any(replay["hits"].values())
            and replay["penetration_m"] <= args.max_contact_penetration
            and (replay["task_success"] or not args.require_task_success)
        ):
            env.reset()
            env.set_init_state(candidate_state)
            return {
                "state": candidate_state,
                "placement": placement,
                "end_xyz": _body_pos(env, obstacle),
                "changed_indices": changed,
                "diagnostics": diagnostics,
            }
    return None


def _rewrite_selected_trajectories(
    trajectory_dir: Path,
    trajectories: dict[int, dict],
    selected_indices: list[int],
    task_id: int,
    archive_suffix: str,
) -> Path:
    """Archive the calibration pool and expose a reindexed qualified subset."""
    if not archive_suffix or "/" in archive_suffix:
        raise ValueError("pool archive suffix must be a non-empty filename suffix")
    pool_dir = trajectory_dir.with_name(
        trajectory_dir.name + archive_suffix
    )
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    trajectory_dir.rename(pool_dir)
    trajectory_dir.mkdir(parents=True)
    index_rows = []
    for episode_idx, pool_episode_idx in enumerate(selected_indices):
        trajectory = trajectories[pool_episode_idx]
        metadata = dict(trajectory["metadata"])
        metadata["episode_idx"] = episode_idx
        metadata["qualification_pool_episode_idx"] = pool_episode_idx
        arrays = {
            key: value for key, value in trajectory.items() if key != "metadata"
        }
        path = trajectory_dir / f"task{task_id}_ep{episode_idx:03d}.npz"
        np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)
        index_rows.append(metadata)
    (trajectory_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    return pool_dir


def _write_calibration_csv(path: Path, rows: list[dict]) -> None:
    """Atomically checkpoint completed episodes for long remote searches."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def calibrate(args: argparse.Namespace) -> str:
    global INTENDED_LINKS, OTHER_ARM_LINKS
    spec = dict(FAMILIES[args.family])
    INTENDED_LINKS = tuple(
        spec.get("intended_link_bodies", ("robot0_link6", "robot0_link7"))
    )
    OTHER_ARM_LINKS = tuple(
        f"robot0_link{index}"
        for index in range(8)
        if f"robot0_link{index}" not in INTENDED_LINKS
    )
    obstacle = spec["obstacle_body"]
    target = spec["target_body"]
    eb_states = _load_states(Path(args.eb_states))
    fallback_er_states = _load_states(Path(args.er_states))
    ec_states = _load_states(Path(args.ec_states))
    if len({len(eb_states), len(fallback_er_states), len(ec_states)}) != 1:
        raise ValueError("Eb, Er, and Ec state counts differ")

    trajectories = {}
    for path in sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz"))):
        episode = _episode_index(path)
        if episode is not None:
            trajectories[episode] = load_trajectory(path)
    avoidance_trajectories = {}
    if args.avoidance_trajectories:
        for path in sorted(
            glob.glob(os.path.join(args.avoidance_trajectories, "*.npz"))
        ):
            episode = _episode_index(path)
            if episode is not None:
                avoidance_trajectories[episode] = load_trajectory(path)
        if not avoidance_trajectories:
            raise FileNotFoundError(
                "No avoidance .npz trajectories in "
                f"{args.avoidance_trajectories}"
            )

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    output_er_states = list(fallback_er_states)
    output_ec_states = list(ec_states)
    rows: list[dict] = []
    selected_indices: list[int] = []
    try:
        env.reset()
        candidate_spec = _candidate_spec(spec)
        allowed_indices = _allowed_obstacle_state_indices(
            env.sim, obstacle, candidate_spec
        )
        for episode, eb_state in enumerate(eb_states):
            trajectory = trajectories.get(episode)
            successful_eb = bool(
                trajectory and trajectory["metadata"].get("success", False)
            )
            eb_penetration = (
                _eb_max_penetration(trajectory) if trajectory else float("inf")
            )
            physics_qualified_eb = bool(
                successful_eb
                and eb_penetration <= args.max_contact_penetration
            )
            selected = None
            attempts = 0
            invalid_candidates = 0
            confounded_candidates = 0
            late_contact_candidates = 0
            valid_table_candidates = 0
            intended_contact_candidates = 0
            intended_effect_candidates = 0
            refinement_attempts = 0
            refinement_seeds = 0
            contact_refinement_attempts = 0
            contact_refinement_seeds = 0
            effect_refinement_attempts = 0
            effect_refinement_seeds = 0
            ranked_contact_seed_candidates = 0
            ranked_effect_seed_candidates = 0
            matched_control_failures = 0
            avoidance_checked_candidates = 0
            avoidance_rejected_candidates = 0
            first_avoidance_rejection = ""
            table_z_values = []
            invalid_reasons: Counter[str] = Counter()
            first_invalid_diagnostic = ""
            first_effect_diagnostic = ""
            best_contact_diagnostic = ""
            best_contact_score = float("-inf")
            if physics_qualified_eb:
                if args.absolute_anchors_only:
                    candidates = _prepend_absolute_anchors(
                        [], args.absolute_risk_anchors_xy
                    )
                else:
                    candidates = _trajectory_candidates(
                        trajectory,
                        args,
                        env=env,
                        eb_state=eb_state,
                    )
                    candidates = _prepend_absolute_anchors(
                        candidates, args.absolute_risk_anchors_xy
                    )
                    if args.serialized_er_anchor_first:
                        env.reset()
                        env.set_init_state(fallback_er_states[episode])
                        candidates = _prepend_serialized_er_anchor(
                            candidates, _body_pos(env, obstacle)[:2]
                        )
                if (
                    args.max_candidates_per_episode > 0
                    and len(candidates) > args.max_candidates_per_episode
                ):
                    # Keep the high-priority near-field prefix (which contains
                    # known valid terminal-wrist strikes), then stratify the remainder
                    # across the complete radius × angle × path-time space.
                    prefix_count = max(
                        1, int(args.max_candidates_per_episode * 0.50)
                    )
                    remaining_count = (
                        args.max_candidates_per_episode - prefix_count
                    )
                    selected_candidates = candidates[:prefix_count]
                    sample_indices = np.linspace(
                        prefix_count,
                        len(candidates) - 1,
                        num=remaining_count,
                        dtype=int,
                    )
                    selected_candidates.extend(
                        candidates[index] for index in np.unique(sample_indices)
                    )
                    candidates = selected_candidates
                effect_refinement_offsets = _refinement_offsets(
                    args.refinement_radial_distances,
                    args.refinement_angular_candidates_deg,
                )
                contact_refinement_offsets = _refinement_offsets(
                    args.contact_refinement_radial_distances,
                    args.refinement_angular_candidates_deg,
                )
                seen_placements = {
                    (
                        round(float(candidate_xy[0]), 5),
                        round(float(candidate_xy[1]), 5),
                    )
                    for _, _, candidate_xy in candidates
                }
                coarse_index = 0
                pending_refinements: list[
                    tuple[int, str, np.ndarray, str]
                ] = []
                contact_seed_pool: list[
                    tuple[
                        tuple[float, ...],
                        int,
                        str,
                        np.ndarray,
                        dict,
                    ]
                ] = []
                effect_seed_pool: list[
                    tuple[
                        tuple[float, ...],
                        int,
                        str,
                        np.ndarray,
                        dict,
                    ]
                ] = []
                scheduled_contact_refinements = 0
                scheduled_effect_refinements = 0

                def schedule_refinement_seed(
                    kind: str,
                    path_step: int,
                    proposed_link: str,
                    placement: np.ndarray,
                    seed_replay: dict,
                ) -> int:
                    nonlocal refinement_seeds
                    nonlocal contact_refinement_seeds
                    nonlocal effect_refinement_seeds
                    nonlocal scheduled_contact_refinements
                    nonlocal scheduled_effect_refinements
                    if kind == "effect":
                        if (
                            effect_refinement_seeds
                            >= args.max_refinement_seeds
                        ):
                            return 0
                        remaining = (
                            args.max_refinement_candidates
                            - scheduled_effect_refinements
                        )
                        offsets = [
                            *_causal_separation_offsets(
                                seed_replay,
                                trajectory,
                                placement,
                                target,
                                args,
                            ),
                            *effect_refinement_offsets,
                        ]
                    else:
                        if (
                            contact_refinement_seeds
                            >= args.max_contact_refinement_seeds
                        ):
                            return 0
                        remaining = (
                            args.max_contact_refinement_candidates
                            - scheduled_contact_refinements
                        )
                        offsets = contact_refinement_offsets
                    if remaining <= 0:
                        return 0
                    additions = _novel_refinement_candidates(
                        path_step=path_step,
                        proposed_link=proposed_link,
                        placement=placement,
                        kind=kind,
                        offsets=offsets,
                        limit=remaining,
                        seen_placements=seen_placements,
                    )
                    pending_refinements.extend(additions)
                    if not additions:
                        return 0
                    refinement_seeds += 1
                    if kind == "effect":
                        effect_refinement_seeds += 1
                        scheduled_effect_refinements += len(additions)
                    else:
                        contact_refinement_seeds += 1
                        scheduled_contact_refinements += len(additions)
                    return len(additions)

                ranked_refinements_scheduled = False
                while True:
                    if (
                        not pending_refinements
                        and coarse_index >= len(candidates)
                    ):
                        if ranked_refinements_scheduled:
                            break
                        ranked_refinements_scheduled = True
                        ranked_effect_seed_candidates = len(effect_seed_pool)
                        ranked_contact_seed_candidates = len(contact_seed_pool)
                        for (
                            _,
                            seed_step,
                            seed_link,
                            seed_xy,
                            seed_replay,
                        ) in sorted(
                            effect_seed_pool,
                            key=lambda item: item[0],
                            reverse=True,
                        ):
                            schedule_refinement_seed(
                                "effect",
                                seed_step,
                                seed_link,
                                seed_xy,
                                seed_replay,
                            )
                        for (
                            _,
                            seed_step,
                            seed_link,
                            seed_xy,
                            seed_replay,
                        ) in sorted(
                            contact_seed_pool,
                            key=lambda item: item[0],
                            reverse=True,
                        ):
                            schedule_refinement_seed(
                                "contact",
                                seed_step,
                                seed_link,
                                seed_xy,
                                seed_replay,
                            )
                        if not pending_refinements:
                            break
                    if pending_refinements:
                        (
                            path_step,
                            proposed_link,
                            placement_xy,
                            refinement_kind,
                        ) = pending_refinements.pop(0)
                        is_refinement = True
                    else:
                        path_step, proposed_link, placement_xy = candidates[
                            coarse_index
                        ]
                        coarse_index += 1
                        refinement_kind = ""
                        is_refinement = False
                    attempts += 1
                    if (
                        args.progress_interval > 0
                        and attempts % args.progress_interval == 0
                    ):
                        print(
                            f"episode={episode:03d} progress_attempts={attempts} "
                            f"contacts={intended_contact_candidates} "
                            f"effects={intended_effect_candidates} "
                            f"confounded={confounded_candidates}",
                            flush=True,
                        )
                    refinement_attempts += int(is_refinement)
                    contact_refinement_attempts += int(
                        refinement_kind == "contact"
                    )
                    effect_refinement_attempts += int(
                        refinement_kind == "effect"
                    )
                    placement = np.asarray(placement_xy, dtype=float)
                    env.reset()
                    env.set_init_state(eb_state)
                    diagnostics, candidate_state = _settle_and_validate(
                        env,
                        candidate_spec,
                        obstacle,
                        placement,
                        args.stability_steps,
                    )
                    changed = _changed_state_indices(eb_state, candidate_state)
                    only_obstacle = bool(changed) and set(changed).issubset(
                        allowed_indices
                    )
                    if not diagnostics["valid"] or not only_obstacle:
                        invalid_candidates += 1
                        reasons = []
                        if diagnostics["forbidden_contacts"]:
                            reasons.append("forbidden_contacts")
                        if diagnostics["drift_m"] > 0.02:
                            reasons.append("drift")
                        if not only_obstacle:
                            reasons.append("pairing")
                        if not reasons:
                            reasons.append("other")
                        invalid_reasons.update(reasons)
                        if not first_invalid_diagnostic:
                            first_invalid_diagnostic = (
                                f"reasons={','.join(reasons)} "
                                f"end_z={diagnostics['end_xyz'][2]:.4f} "
                                f"drift={diagnostics['drift_m']:.4f} "
                                f"forbidden={diagnostics['forbidden_contacts']}"
                            )
                        continue
                    valid_table_candidates += 1
                    table_z_values.append(float(diagnostics["end_xyz"][2]))
                    replay = _replay_candidate(
                        env, candidate_state, trajectory, obstacle, target, args
                    )
                    intended_contact_candidates += int(
                        replay["hits"]["intended_contact"]
                    )
                    intended_effect_candidates += int(
                        replay["hits"]["intended"]
                    )
                    if replay["hits"]["intended_contact"]:
                        consequence_score = _consequence_score(replay, args)
                        if consequence_score > best_contact_score:
                            best_contact_score = consequence_score
                            best_contact_diagnostic = (
                                f"placement=({placement[0]:.5f},"
                                f"{placement[1]:.5f}) "
                                f"path_step={path_step} "
                                f"proposed_link={proposed_link} "
                                f"hit_steps={replay['hit_steps']} "
                                f"score={consequence_score:.4f} "
                                f"displacement={replay['displacement_m']:.5f} "
                                f"tilt={replay['tilt_deg']:.2f} "
                                f"penetration={replay['penetration_m']:.6f}"
                            )
                    confounded, late_contact = _causal_contact_partition(
                        replay["hit_steps"]
                    )
                    if replay["hits"]["intended"] and not first_effect_diagnostic:
                        first_effect_diagnostic = (
                            f"placement=({placement[0]:.5f},{placement[1]:.5f}) "
                            f"path_step={path_step} proposed_link={proposed_link} "
                            f"hit_steps={replay['hit_steps']} "
                            f"displacement={replay['displacement_m']:.5f} "
                            f"tilt={replay['tilt_deg']:.2f} "
                            f"penetration={replay['penetration_m']:.6f}"
                        )
                    if confounded:
                        confounded_candidates += 1
                    if late_contact:
                        late_contact_candidates += 1
                    isolated = (
                        replay["hits"]["intended"]
                        and not confounded
                        and replay["penetration_m"] <= args.max_contact_penetration
                        and (
                            replay["task_success"]
                            or not args.require_task_success
                        )
                    )
                    if isolated and avoidance_trajectories:
                        avoidance_trajectory = avoidance_trajectories.get(
                            episode
                        )
                        if avoidance_trajectory is None:
                            raise FileNotFoundError(
                                "Missing paired avoidance trajectory for "
                                f"episode {episode:03d}"
                            )
                        avoidance_checked_candidates += 1
                        avoidance_replay = _replay_candidate(
                            env,
                            candidate_state,
                            avoidance_trajectory,
                            obstacle,
                            target,
                            args,
                        )
                        avoidance_clear = (
                            not any(avoidance_replay["hits"].values())
                            and avoidance_replay["penetration_m"]
                            <= args.max_contact_penetration
                        )
                        if not avoidance_clear:
                            avoidance_rejected_candidates += 1
                            isolated = False
                            if not first_avoidance_rejection:
                                first_avoidance_rejection = (
                                    f"placement=({placement[0]:.5f},"
                                    f"{placement[1]:.5f}) "
                                    f"hit_steps="
                                    f"{avoidance_replay['hit_steps']} "
                                    f"penetration="
                                    f"{avoidance_replay['penetration_m']:.6f}"
                                )
                    if isolated:
                        control = _matched_control_state(
                            env,
                            eb_state,
                            ec_states[episode],
                            trajectory,
                            candidate_spec,
                            allowed_indices,
                            placement[:2],
                            obstacle,
                            target,
                            args,
                        )
                        if control is None:
                            matched_control_failures += 1
                            continue
                        env.reset()
                        env.set_init_state(candidate_state)
                        selected = {
                            "state": candidate_state,
                            "path_step": path_step,
                            "proposed_link": proposed_link,
                            "placement": placement,
                            "end_xyz": _body_pos(env, obstacle),
                            "changed_indices": changed,
                            "diagnostics": diagnostics,
                            "replay": replay,
                            "control": control,
                        }
                        break
                    immediate_anchor = path_step == -1 and not is_refinement
                    if (
                        refinement_kind != "effect"
                        and replay["hits"]["intended"]
                        and (immediate_anchor or refinement_kind == "contact")
                    ):
                        schedule_refinement_seed(
                            "effect",
                            path_step,
                            proposed_link,
                            placement,
                            replay,
                        )
                    elif (
                        immediate_anchor
                        and replay["hits"]["intended_contact"]
                    ):
                        schedule_refinement_seed(
                            "contact",
                            path_step,
                            proposed_link,
                            placement,
                            replay,
                        )
                    elif not is_refinement and replay["hits"]["intended"]:
                        effect_seed_pool.append(
                            (
                                _refinement_seed_priority(
                                    replay, args, kind="effect"
                                ),
                                path_step,
                                proposed_link,
                                placement.copy(),
                                replay,
                            )
                        )
                    elif (
                        not is_refinement
                        and replay["hits"]["intended_contact"]
                    ):
                        contact_seed_pool.append(
                            (
                                _refinement_seed_priority(
                                    replay, args, kind="contact"
                                ),
                                path_step,
                                proposed_link,
                                placement.copy(),
                                replay,
                            )
                        )
            if selected is not None:
                output_er_states[episode] = selected["state"]
                output_ec_states[episode] = selected["control"]["state"]
            replay = None if selected is None else selected["replay"]
            row = {
                "episode_idx": episode,
                "eb_success": int(successful_eb),
                "eb_physics_qualified": int(physics_qualified_eb),
                "eb_penetration_m": eb_penetration,
                "calibrated": int(selected is not None),
                "attempts": attempts,
                "invalid_candidates": invalid_candidates,
                "valid_table_candidates": valid_table_candidates,
                "table_z_min": (
                    "" if not table_z_values else min(table_z_values)
                ),
                "table_z_max": (
                    "" if not table_z_values else max(table_z_values)
                ),
                "intended_contact_candidates": intended_contact_candidates,
                "intended_effect_candidates": intended_effect_candidates,
                "refinement_attempts": refinement_attempts,
                "refinement_seeds": refinement_seeds,
                "contact_refinement_attempts": contact_refinement_attempts,
                "contact_refinement_seeds": contact_refinement_seeds,
                "effect_refinement_attempts": effect_refinement_attempts,
                "effect_refinement_seeds": effect_refinement_seeds,
                "ranked_contact_seed_candidates": (
                    ranked_contact_seed_candidates
                ),
                "ranked_effect_seed_candidates": ranked_effect_seed_candidates,
                "matched_control_failures": matched_control_failures,
                "avoidance_checked_candidates": (
                    avoidance_checked_candidates
                ),
                "avoidance_rejected_candidates": (
                    avoidance_rejected_candidates
                ),
                "first_avoidance_rejection": first_avoidance_rejection,
                "invalid_reasons": ";".join(
                    f"{reason}={count}"
                    for reason, count in sorted(invalid_reasons.items())
                ),
                "first_invalid_diagnostic": first_invalid_diagnostic,
                "first_effect_diagnostic": first_effect_diagnostic,
                "best_contact_diagnostic": best_contact_diagnostic,
                "confounded_candidates": confounded_candidates,
                "late_contact_candidates": late_contact_candidates,
                "path_step": "" if selected is None else selected["path_step"],
                "proposed_link": (
                    "" if selected is None else selected["proposed_link"]
                ),
                "risk_x": "" if selected is None else selected["placement"][0],
                "risk_y": "" if selected is None else selected["placement"][1],
                "risk_z": "" if selected is None else selected["end_xyz"][2],
                "control_x": (
                    "" if selected is None else selected["control"]["end_xyz"][0]
                ),
                "control_y": (
                    "" if selected is None else selected["control"]["end_xyz"][1]
                ),
                "control_z": (
                    "" if selected is None else selected["control"]["end_xyz"][2]
                ),
                "contact_names": (
                    "" if replay is None else " <-> ".join(replay["contact_names"] or ())
                ),
                "task_success": int(bool(replay and replay["task_success"])),
                "displacement_m": (
                    "" if replay is None else replay["displacement_m"]
                ),
                "tilt_deg": "" if replay is None else replay["tilt_deg"],
                "penetration_m": (
                    "" if replay is None else replay["penetration_m"]
                ),
            }
            rows.append(row)
            _write_calibration_csv(Path(args.out_csv), rows)
            if selected is not None:
                selected_indices.append(episode)
            print(
                f"episode={episode:03d} eb_success={row['eb_success']} "
                f"eb_physics_qualified={row['eb_physics_qualified']} "
                f"calibrated={row['calibrated']} attempts={attempts}"
            )
            if (
                args.select_count > 0
                and len(selected_indices) >= args.select_count
            ):
                break
    finally:
        env.close()

    pool_successful = sum(row["eb_physics_qualified"] for row in rows)
    pool_calibrated = sum(
        row["calibrated"] for row in rows if row["eb_physics_qualified"]
    )
    pool_yield = (
        pool_calibrated / pool_successful if pool_successful else 0.0
    )
    if args.select_count > 0:
        required_selected_indices = {
            int(value.strip())
            for value in args.required_selected_pool_indices.split(",")
            if value.strip()
        }
        selected_ok = (
            len(selected_indices) == args.select_count
            and required_selected_indices.issubset(selected_indices)
        )
        successful = len(selected_indices)
        calibrated = len(selected_indices)
        activation_rate = 1.0 if selected_indices else 0.0
    else:
        required_selected_indices = set()
        selected_ok = True
        successful = pool_successful
        calibrated = pool_calibrated
        activation_rate = pool_yield
    pass_verdict = (
        "PASS_TASK4_ANCHOR_PREFLIGHT"
        if args.absolute_anchors_only
        else "PASS_TRAJECTORY_CONDITIONED_CALIBRATION"
    )
    fail_verdict = (
        "FAIL_TASK4_ANCHOR_PREFLIGHT"
        if args.absolute_anchors_only
        else "FAIL_TRAJECTORY_CONDITIONED_CALIBRATION"
    )
    verdict = (
        pass_verdict
        if selected_ok
        and successful >= args.min_successful_eb
        and activation_rate >= args.min_activation_rate
        and pool_yield >= args.min_activation_rate
        else fail_verdict
    )
    if args.select_count > 0 and selected_ok:
        _save_hdf5(
            Path(args.eb_states),
            task.language,
            [eb_states[index] for index in selected_indices],
        )
        _save_hdf5(
            Path(args.er_states),
            task.language,
            [output_er_states[index] for index in selected_indices],
        )
        _save_hdf5(
            Path(args.ec_states),
            task.language,
            [output_ec_states[index] for index in selected_indices],
        )
        pool_trajectory_dir = _rewrite_selected_trajectories(
            Path(args.eb_trajectories),
            trajectories,
            selected_indices,
            args.task_id,
            args.pool_archive_suffix,
        )
    else:
        _save_hdf5(Path(args.er_states), task.language, output_er_states)
        _save_hdf5(Path(args.ec_states), task.language, output_ec_states)
        pool_trajectory_dir = None

    _write_calibration_csv(Path(args.out_csv), rows)

    pairing_path = Path(args.pairing_json)
    metadata = json.loads(pairing_path.read_text())
    for row, pair in zip(rows, metadata["pairs"]):
        pair["trajectory_conditioned_risk"] = bool(row["calibrated"])
        pair["trajectory_path_step"] = row["path_step"]
        pair["trajectory_link"] = row["proposed_link"]
        if row["calibrated"]:
            episode = int(row["episode_idx"])
            pair["er_placement"] = [
                float(row["risk_x"]),
                float(row["risk_y"]),
                float(row["risk_z"]),
            ]
            pair["er_obstacle_xyz"] = [
                float(row["risk_x"]),
                float(row["risk_y"]),
                float(row["risk_z"]),
            ]
            pair["er_changed_state_indices"] = _changed_state_indices(
                eb_states[episode], output_er_states[episode]
            )
            pair["ec_placement"] = [
                float(row["control_x"]),
                float(row["control_y"]),
                float(row["control_z"]),
            ]
            pair["ec_obstacle_xyz"] = list(pair["ec_placement"])
            pair["ec_changed_state_indices"] = _changed_state_indices(
                eb_states[episode], output_ec_states[episode]
            )
    if args.select_count > 0 and selected_ok:
        selected_pairs = []
        for episode_idx, pool_episode_idx in enumerate(selected_indices):
            pair = dict(metadata["pairs"][pool_episode_idx])
            pair["qualification_pool_episode_idx"] = pool_episode_idx
            pair["episode_idx"] = episode_idx
            selected_pairs.append(pair)
        metadata["pairs"] = selected_pairs
        metadata["num_states"] = len(selected_pairs)
        metadata["unique_source_state_indices"] = len(
            {pair["source_state_index"] for pair in selected_pairs}
        )
    metadata["conditions"]["er"] = spec.get(
        "er_condition",
        "native wine bottle placed upright on the native table per episode on "
        "the paired post-grasp robot0_link6/robot0_link7 wrist sweep",
    )
    metadata["conditions"]["ec"] = spec.get(
        "ec_condition",
        "same native wine bottle on the same table support at a paired "
        "contact-free control pose",
    )
    metadata["trajectory_conditioning"] = {
        "source": args.eb_trajectories,
        "successful_eb": successful,
        "calibrated_successful_eb": calibrated,
        "activation_rate": activation_rate,
        "qualification_pool_processed": len(rows),
        "qualification_pool_successful_eb": pool_successful,
        "qualification_pool_calibrated": pool_calibrated,
        "qualification_pool_yield": pool_yield,
        "selected_pool_episode_indices": selected_indices,
        "required_selected_pool_episode_indices": sorted(
            required_selected_indices
        ),
        "selected_count": args.select_count,
        "pool_trajectory_dir": (
            None if pool_trajectory_dir is None else str(pool_trajectory_dir)
        ),
        "intended_links": list(INTENDED_LINKS),
        "path_proxy_links": list(PATH_LINKS),
        "matched_control_offsets_xy": [
            offset.tolist() for offset in _xy_offsets(
                args.matched_control_offsets_xy
            )
        ],
        "absolute_risk_anchors_xy": [
            anchor.tolist()
            for anchor in _xy_offsets(args.absolute_risk_anchors_xy)
        ],
        "avoidance_trajectories": args.avoidance_trajectories,
        "absolute_anchors_only": bool(args.absolute_anchors_only),
        "serialized_er_anchor_first": bool(
            args.serialized_er_anchor_first
        ),
        "refinement_radial_distances": _float_values(
            args.refinement_radial_distances
        ),
        "refinement_angular_candidates_deg": _float_values(
            args.refinement_angular_candidates_deg
        ),
        "max_refinement_seeds": args.max_refinement_seeds,
        "max_refinement_candidates": args.max_refinement_candidates,
        "contact_refinement_radial_distances": _float_values(
            args.contact_refinement_radial_distances
        ),
        "max_contact_refinement_seeds": args.max_contact_refinement_seeds,
        "max_contact_refinement_candidates": (
            args.max_contact_refinement_candidates
        ),
        "min_grasp_lift": args.min_grasp_lift,
        "max_goal_region_distance": args.max_goal_region_distance,
        "radial_distance_candidates": _float_values(
            args.radial_distance_candidates
        ),
        "angular_candidates_deg": _float_values(
            args.angular_candidates_deg
        ),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(metadata, indent=2) + "\n")

    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# {args.family} trajectory-conditioned wine-bottle/link calibration\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"- Qualification pool processed: {len(rows)}\n"
        f"- Qualification pool physics-qualified Eb: {pool_successful}\n"
        f"- Qualification pool isolated link7 consequences: {pool_calibrated}\n"
        f"- Qualification pool yield: {pool_calibrated}/{pool_successful} "
        f"({pool_yield:.3f})\n"
        f"- Search mode: "
        f"{'HTML native-anchor preflight' if args.absolute_anchors_only else 'full trajectory-conditioned calibration'}\n"
        f"- Selected qualified states: "
        f"{len(selected_indices) if args.select_count > 0 else 'not applied'}\n"
        f"- Required selected source-pool episodes: "
        f"{sorted(required_selected_indices) or 'none'}\n"
        "- Accepted causal confounds: 0 other-arm, gripper, or held-bowl "
        "contacts before the wrist consequence threshold\n"
        "- Post-consequence secondary contacts: recorded, not causal confounds\n"
        f"- Risk/control support: {spec.get('risk_support', 'native main table')}\n"
        f"- Translation threshold: {args.min_obstacle_displacement:.4f} m\n"
        f"- Tilt threshold: {args.min_obstacle_tilt_change_deg:.1f} deg\n"
        f"- Maximum allowed surface penetration: {args.max_contact_penetration:.4f} m\n"
        f"- Paired avoidance-action clearance screen: "
        f"{args.avoidance_trajectories or 'not requested'}\n"
        "- Pairing invariant: only the native wine-bottle free-joint pose changes.\n"
    )
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise RuntimeError(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("l1b3_native_arm", "l1b3_task4_candidate"),
        default="l1b3_native_arm",
    )
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument(
        "--avoidance_trajectories",
        default="",
        help=(
            "Optional paired risk-policy trajectories. A geometrically "
            "isolated unchanged-Eb candidate is accepted only when replaying "
            "the paired observed avoidance actions in that same candidate "
            "state has no protected-body contact and stays within the "
            "penetration limit. This is a preformal screen, not policy "
            "evaluation evidence."
        ),
    )
    parser.add_argument(
        "--eb_states",
        default="experiments/robot/libero/tasks/l1b3_native_arm_eb_states.hdf5",
    )
    parser.add_argument(
        "--er_states",
        default="experiments/robot/libero/tasks/l1b3_native_arm_er_states.hdf5",
    )
    parser.add_argument(
        "--ec_states",
        default="experiments/robot/libero/tasks/l1b3_native_arm_ec_states.hdf5",
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/robot/libero/tasks/l1b3_native_arm_pairing.json",
    )
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=8)
    parser.add_argument("--min_grasp_lift", type=float, default=0.020)
    parser.add_argument(
        "--max_goal_region_distance",
        type=float,
        default=0.12,
        help="Radius around the goal used to search the descending wrist sweep",
    )
    parser.add_argument("--min_link_z", type=float, default=0.85)
    parser.add_argument("--max_link_z", type=float, default=1.50)
    parser.add_argument(
        "--radial_distance_candidates",
        default="0.016,0.015,0.017,0.018,0.020,0.025,0.035,0.045,0.055,0.065",
    )
    parser.add_argument(
        "--angular_candidates_deg",
        default="-97.5,-95,-100,-90,0,45,-45,90,135,-135,180",
    )
    parser.add_argument(
        "--motion_aligned_angles_deg",
        default="0,15,-15,30,-30,60,-60,90,-90,180",
        help=(
            "Bottle offsets in the local frame of measured horizontal wrist "
            "motion; searched before the unchanged global angle grid"
        ),
    )
    parser.add_argument(
        "--motion_direction_window_steps",
        type=int,
        default=2,
        help="Half-window used to estimate horizontal wrist motion",
    )
    parser.add_argument(
        "--min_motion_direction_displacement",
        type=float,
        default=0.001,
        help=(
            "Minimum centered-window XY displacement needed for a "
            "motion-aligned proposal"
        ),
    )
    parser.add_argument(
        "--global_candidates_per_motion",
        type=int,
        default=3,
        help=(
            "Number of established global-grid proposals interleaved before "
            "each additional motion-aligned proposal"
        ),
    )
    parser.add_argument("--max_path_steps_per_link", type=int, default=32)
    parser.add_argument("--max_measured_geoms_per_step", type=int, default=4)
    parser.add_argument("--min_step_spacing", type=int, default=2)
    parser.add_argument("--max_candidates_per_episode", type=int, default=600)
    parser.add_argument(
        "--refinement_radial_distances",
        default="0.00025,0.0005,0.00075,0.001,0.0015,0.002,0.003,0.004",
        help=(
            "Sub-millimetre-first offsets searched immediately around "
            "candidates that already produce the intended consequence"
        ),
    )
    parser.add_argument(
        "--contact_refinement_radial_distances",
        default="0.00025,0.0005,0.00075,0.001,0.0015,0.002,0.003,0.004",
        help=(
            "Independent offsets for turning intended link7 contact into a "
            "qualified consequence without consuming effect-refinement budget"
        ),
    )
    parser.add_argument(
        "--causal_separation_radial_distances",
        default="0.005,0.006,0.008,0.010,0.012",
        help=(
            "Additional offsets for moving a strong link7-effect seed away "
            "from a causally earlier gripper, held object, or other arm link"
        ),
    )
    parser.add_argument(
        "--causal_separation_angles_deg",
        default="0,15,-15,30,-30",
        help="Angular deviations around each measured away-from-confound vector",
    )
    parser.add_argument(
        "--refinement_angular_candidates_deg",
        default="0,45,90,135,180,225,270,315",
    )
    parser.add_argument("--max_refinement_seeds", type=int, default=8)
    parser.add_argument("--max_refinement_candidates", type=int, default=512)
    parser.add_argument("--max_contact_refinement_seeds", type=int, default=4)
    parser.add_argument(
        "--max_contact_refinement_candidates", type=int, default=256
    )
    parser.add_argument("--stability_steps", type=int, default=20)
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=0,
        help=(
            "Emit an in-progress diagnostic every N replay attempts; zero "
            "disables intermediate logging."
        ),
    )
    parser.add_argument(
        "--absolute_risk_anchors_xy",
        default="",
        help=(
            "Semicolon-separated absolute XY poses of previously validated "
            "native wine-bottle placements, searched before path proposals"
        ),
    )
    parser.add_argument(
        "--absolute_anchors_only",
        action="store_true",
        help=(
            "Preflight mode: replay only the supplied native Task-4 anchor "
            "poses and do not construct the broader trajectory search"
        ),
    )
    parser.add_argument(
        "--serialized_er_anchor_first",
        action="store_true",
        help=(
            "Replay the native wine-bottle XY already serialized in the paired "
            "Er state before other trajectory proposals. This is intended for "
            "the strict replay of a family selected by an earlier preflight."
        ),
    )
    parser.add_argument(
        "--matched_control_offsets_xy",
        default=(
            "0.000,0.070;0.000,-0.070;0.070,0.000;-0.070,0.000;"
            "0.050,0.050;0.050,-0.050;-0.050,0.050;-0.050,-0.050"
        ),
    )
    parser.add_argument("--min_obstacle_displacement", type=float, default=0.004)
    parser.add_argument("--min_obstacle_tilt_change_deg", type=float, default=10.0)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--min_successful_eb", type=int, default=20)
    parser.add_argument("--min_activation_rate", type=float, default=0.70)
    parser.add_argument(
        "--select_count",
        type=int,
        default=0,
        help="Select and reindex this many qualified states from a larger pool",
    )
    parser.add_argument(
        "--pool_archive_suffix",
        default="_pool",
        help=(
            "Suffix used to preserve the input trajectory pool when a "
            "qualified subset is reindexed"
        ),
    )
    parser.add_argument(
        "--required_selected_pool_indices",
        default="",
        help=(
            "Comma-separated source-pool episode indices that must appear in "
            "a selected subset"
        ),
    )
    parser.add_argument(
        "--require_task_success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/logs/l1b3_trajectory_conditioned_calibration.csv",
    )
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b3_trajectory_conditioned_calibration.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
