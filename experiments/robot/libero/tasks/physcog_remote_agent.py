#!/usr/bin/env python3
"""Auditable SSH/Slurm runner for remote PhysCog scene validation.

This module intentionally exposes a small phase registry instead of arbitrary
remote shell execution.  Scene-editing agents can therefore run expensive
MuJoCo checks on Superpod while retaining a local ledger of exactly what ran.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PhaseSpec:
    command: tuple[str, ...]
    count_env: str | None = None
    artifacts: tuple[str, ...] = ()
    # Artifacts are also fetched after a run, so a later phase must not
    # automatically delete prerequisite evidence merely because it needs to
    # download that evidence for review.  ``None`` preserves the historical
    # behavior; staged pipelines can name only their phase-owned outputs.
    cleanup_artifacts: tuple[str, ...] | None = None


PHASES: Mapping[tuple[str, str], PhaseSpec] = {
    # L1-C4 uses the unmodified libero_object task "pick up the cream cheese
    # and place it in the basket".  Keep validation and model smoke separate
    # so policy rollouts cannot start before the downloaded policy views have
    # been reviewed.
    ("l1c4", "check"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh",
            "check",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1c4_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_er_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_source_indices.json",
            "experiments/robot/libero/tasks/l1c4_state_bundle.json",
            "experiments/logs/l1c4_native_preflight.json",
            "experiments/logs/l1c4_native_preflight.md",
        ),
    ),
    ("l1c4", "preview"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh",
            "preview",
        ),
        artifacts=(
            "experiments/robot/libero/tasks/l1c4_preview",
            "experiments/logs/l1c4_native_preflight.json",
            "experiments/logs/l1c4_native_preflight.md",
            "experiments/logs/l1c4_exact_state_preview.csv",
            "experiments/logs/l1c4_exact_state_preview.md",
        ),
    ),
    ("l1c4", "calibrate"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh",
            "calibrate",
        ),
        count_env="CALIBRATION_NUM_STATES",
        artifacts=(
            "experiments/logs/l1c4_native_preflight.json",
            "experiments/logs/l1c4_native_preflight.md",
            "experiments/logs/l1c4_calibration.csv",
            "experiments/logs/l1c4_calibration.md",
        ),
    ),
    ("l1c4", "safe_reference"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh",
            "safe_reference",
        ),
        count_env="CALIBRATION_NUM_STATES",
        artifacts=(
            "experiments/logs/l1c4_native_preflight.json",
            "experiments/logs/l1c4_native_preflight.md",
            "experiments/logs/l1c4_safe_reference.csv",
            "experiments/logs/l1c4_safe_reference_attempts.csv",
            "experiments/logs/l1c4_safe_reference.md",
            "experiments/logs/l1c4_safe_reference_trajectories",
            "review/L1-C4_task/safe_reference",
        ),
    ),
    ("l1c4", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1c4_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_er_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_source_indices.json",
            "experiments/robot/libero/tasks/l1c4_state_bundle.json",
            "experiments/robot/libero/tasks/l1c4_preview",
            "experiments/logs/l1c4_native_preflight.json",
            "experiments/logs/l1c4_native_preflight.md",
            "experiments/logs/l1c4_exact_state_preview.csv",
            "experiments/logs/l1c4_exact_state_preview.md",
            "experiments/logs/l1c4_calibration.csv",
            "experiments/logs/l1c4_calibration.md",
            "experiments/logs/l1c4_safe_reference.csv",
            "experiments/logs/l1c4_safe_reference_attempts.csv",
            "experiments/logs/l1c4_safe_reference.md",
            "experiments/logs/l1c4_safe_reference_trajectories",
            "experiments/logs/l1c4_eb_to_er_replay.csv",
            "experiments/logs/l1c4_eb_to_er_replay.md",
            "experiments/logs/l1c4_eb_to_ec_replay.csv",
            "experiments/logs/l1c4_eb_to_ec_replay.md",
            "experiments/logs/l1c4_attribution.csv",
            "experiments/logs/l1c4_attribution.md",
            "review/L1-C4_task",
            "rollouts/libero_object/L1-C4-occupied-basket-eb",
            "rollouts/libero_object/L1-C4-occupied-basket-risk",
            "rollouts/libero_object/L1-C4-occupied-basket-ec",
        ),
    ),
    ("l1c4", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "REVIEW_DIR=review/L1-C4_task/formal",
            "HUMAN_VISIBILITY_REVIEW=review/L1-C4_task/visibility_review.md",
            "bash",
            "experiments/robot/libero/tasks/run_l1c4_occupied_basket.sh",
            "eval",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1c4_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_er_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1c4_source_indices.json",
            "experiments/robot/libero/tasks/l1c4_state_bundle.json",
            "experiments/robot/libero/tasks/l1c4_preview",
            "experiments/logs/l1c4_native_preflight.json",
            "experiments/logs/l1c4_native_preflight.md",
            "experiments/logs/l1c4_exact_state_preview.csv",
            "experiments/logs/l1c4_exact_state_preview.md",
            "experiments/logs/l1c4_calibration.csv",
            "experiments/logs/l1c4_calibration.md",
            "experiments/logs/l1c4_safe_reference.csv",
            "experiments/logs/l1c4_safe_reference_attempts.csv",
            "experiments/logs/l1c4_safe_reference.md",
            "experiments/logs/l1c4_safe_reference_trajectories",
            "experiments/logs/l1c4_eb_to_er_replay.csv",
            "experiments/logs/l1c4_eb_to_er_replay.md",
            "experiments/logs/l1c4_eb_to_ec_replay.csv",
            "experiments/logs/l1c4_eb_to_ec_replay.md",
            "experiments/logs/l1c4_attribution.csv",
            "experiments/logs/l1c4_attribution.md",
            "review/L1-C4_task/formal",
            "rollouts/libero_object/L1-C4-occupied-basket-eb",
            "rollouts/libero_object/L1-C4-occupied-basket-risk",
            "rollouts/libero_object/L1-C4-occupied-basket-ec",
        ),
    ),
    # Native-only RoboCasa L2-A1 bring-up. These phases stop before any policy
    # rollout or formal evaluation; the reviewed phase may run only after the
    # downloaded Eb/Er/Ec policy-view triplet has been inspected.
    ("robocasa_l2a1", "probe"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "probe",
        ),
        artifacts=("experiments/logs/robocasa_superpod/probe",),
    ),
    ("robocasa_l2a1", "static_live"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "static_live",
        ),
        artifacts=("experiments/logs/robocasa_superpod/static_live",),
    ),
    ("robocasa_l2a1", "geometry"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "geometry",
        ),
        artifacts=("experiments/logs/robocasa_superpod/geometry",),
    ),
    ("robocasa_l2a1", "layout_scan"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "layout_scan",
        ),
        artifacts=("experiments/logs/robocasa_superpod/layout_scan",),
    ),
    ("robocasa_l2a1", "initial_unreviewed"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "initial_unreviewed",
        ),
        artifacts=("experiments/logs/robocasa_superpod/initial_unreviewed",),
    ),
    ("robocasa_l2a1", "initial_reviewed"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "initial_reviewed",
        ),
        artifacts=("experiments/logs/robocasa_superpod/initial_reviewed",),
    ),
    # Native MicrowaveThawing L2-A2 bring-up. As with L2-A1, this registry
    # intentionally stops before policy rollout and formal evaluation.
    ("robocasa_l2a2", "static_live"): PhaseSpec(
        command=(
            "env",
            "ROBOCASA_SCENE=L2-A2",
            "ROBOCASA_LOG_NAMESPACE=l2a2",
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "static_live",
        ),
        artifacts=(
            "experiments/logs/robocasa_superpod/l2a2/static_live",
        ),
    ),
    ("robocasa_l2a2", "initial_unreviewed"): PhaseSpec(
        command=(
            "env",
            "ROBOCASA_SCENE=L2-A2",
            "ROBOCASA_LOG_NAMESPACE=l2a2",
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "initial_unreviewed",
        ),
        artifacts=(
            "experiments/logs/robocasa_superpod/l2a2/initial_unreviewed",
        ),
    ),
    ("robocasa_l2a2", "initial_reviewed"): PhaseSpec(
        command=(
            "env",
            "ROBOCASA_SCENE=L2-A2",
            "ROBOCASA_LOG_NAMESPACE=l2a2",
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "initial_reviewed",
        ),
        artifacts=(
            "experiments/logs/robocasa_superpod/l2a2/initial_reviewed",
        ),
    ),
    # Native PrepForSanitizing L2-A3 bring-up; no policy or formal phases.
    ("robocasa_l2a3", "static_live"): PhaseSpec(
        command=(
            "env",
            "ROBOCASA_SCENE=L2-A3",
            "ROBOCASA_LOG_NAMESPACE=l2a3",
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "static_live",
        ),
        artifacts=(
            "experiments/logs/robocasa_superpod/l2a3/static_live",
        ),
    ),
    ("robocasa_l2a3", "initial_unreviewed"): PhaseSpec(
        command=(
            "env",
            "ROBOCASA_SCENE=L2-A3",
            "ROBOCASA_LOG_NAMESPACE=l2a3",
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "initial_unreviewed",
        ),
        artifacts=(
            "experiments/logs/robocasa_superpod/l2a3/initial_unreviewed",
        ),
    ),
    ("robocasa_l2a3", "initial_reviewed"): PhaseSpec(
        command=(
            "env",
            "ROBOCASA_SCENE=L2-A3",
            "ROBOCASA_LOG_NAMESPACE=l2a3",
            "bash",
            "experiments/robot/robocasa/scripts/run_superpod.sh",
            "initial_reviewed",
        ),
        artifacts=(
            "experiments/logs/robocasa_superpod/l2a3/initial_reviewed",
        ),
    ),
    ("l1b6", "calibrate"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/calibrate_l1b6_wine_bottle.sh",
        ),
        count_env="CALIBRATION_TRIALS",
        artifacts=(
            "experiments/logs/l1b6_wine_bottle_pose_search.csv",
            "experiments/logs/l1b6_native_held_object_scene_check.md",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b6_native_held_object",
            "experiments/robot/libero/tasks/l1b6_native_held_object_pairing.json",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
        ),
    ),
    ("l1b6", "search"): PhaseSpec(
        command=(
            "python",
            "experiments/robot/libero/tasks/search_l1b_native_replay_positions.py",
            "--family", "l1b6_native_held_object",
            "--eb_trajectories",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
            "--eb_states",
            "experiments/robot/libero/tasks/l1b6_native_held_object_eb_states.hdf5",
            "--task_suite_name", "libero_goal",
            "--task_id", "6",
            "--fractions=0.25,0.30,0.325,0.35,0.375,0.40",
            "--laterals=0.055,0.060,0.065,0.070,0.075",
            "--max_episodes", "50",
            "--min_obstacle_displacement", "0.0",
            "--min_obstacle_tilt_change_deg", "45.0",
            "--out_csv", "experiments/logs/l1b6_wine_bottle_pose_search.csv",
        ),
        artifacts=("experiments/logs/l1b6_wine_bottle_pose_search.csv",),
    ),
    ("l1b6", "path_calibrate"): PhaseSpec(
        command=(
            "python",
            "experiments/robot/libero/tasks/calibrate_l1b6_trajectory_conditioned_states.py",
            "--eb_trajectories",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
            "--fail_on_invalid",
        ),
        artifacts=(
            "experiments/logs/l1b6_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b6_trajectory_conditioned_calibration.csv",
            "experiments/robot/libero/tasks/l1b6_native_held_object_pairing.json",
        ),
    ),
    ("l1b6", "prepare"): PhaseSpec(
        command=(
            "env",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b6_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b6_native_held_object",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b6_native_held_object_scene_check.md",
            "experiments/logs/l1b6_native_held_object_safe_reference.md",
            "experiments/logs/l1b6_native_held_object_safe_reference.csv",
            "experiments/logs/l1b6_safe_reference_videos",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b6_native_held_object",
            "experiments/robot/libero/tasks/l1b6_native_held_object_pairing.json",
        ),
    ),
    ("l1b6", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b6_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b6_native_held_object",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1b6_native_held_object_scene_check.md",
            "experiments/logs/l1b6_native_held_object_safe_reference.md",
            "experiments/logs/l1b6_native_held_object_safe_reference.csv",
            "experiments/logs/l1b6_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b6_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b6_native_held_object_native_replay.md",
            "experiments/logs/l1b6_native_held_object_native_replay_videos",
            "experiments/logs/l1b6_safe_reference_videos",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-eb",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-er",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-ec",
        ),
    ),
    ("l1b6", "pool_smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "L1B6_CALIBRATION_POOL_SIZE=10",
            "L1B6_ER_PHYSICS_QUALIFICATION_SIZE=5",
            "REPLAY_MIN_EPISODES=2",
            "SAVE_VIDEO_MODE=none",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b6_native_held_object",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b6_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b6_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b6_er_physics_qualification.md",
            "experiments/logs/l1b6_native_held_object_scene_check.md",
            "experiments/logs/l1b6_native_held_object_safe_reference.md",
            "experiments/logs/l1b6_native_held_object_native_replay.md",
            "experiments/robot/libero/tasks/l1b6_native_held_object_pairing.json",
        ),
    ),
    ("l1b6", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b6_native_held_object",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b6_native_held_object_scene_check.md",
            "experiments/logs/l1b6_native_held_object_safe_reference.md",
            "experiments/logs/l1b6_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b6_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b6_er_physics_qualification.md",
            "experiments/logs/l1b6_native_held_object_native_replay.md",
            "experiments/logs/l1b6_native_held_object_eb_rollout_physics.md",
            "experiments/logs/l1b6_native_held_object_er_rollout_physics.md",
            "experiments/logs/l1b6_native_held_object_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b6_native_held_object_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b6_native_held_object",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-eb",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-er",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-ec",
        ),
    ),
    ("l1b6", "ec_repair"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b6_ec_repair.sh",
        ),
        artifacts=(
            "experiments/logs/l1b6_ec_control_repair.md",
            "experiments/logs/l1b6_native_held_object_scene_check.md",
            "experiments/logs/l1b6_native_held_object_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b6_native_held_object",
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-ec/trajectories",
        ),
    ),
    ("l1b6", "ec_video"): PhaseSpec(
        command=(
            "env",
            "SAVE_VIDEO_MODE=all",
            "SAVE_TRAJECTORY=False",
            "RENDER_GPU_DEVICE_ID=1",
            "RUN_ID_SUFFIX=release-video",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b6_native_held_object",
            "ec",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "rollouts/libero_goal/L1-B6-goal-cream-cheese-native-wine-bottle-knockdown-ec-release-video",
        ),
    ),
    ("l1b7", "prepare"): PhaseSpec(
        command=(
            "env",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b7_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b7_native_arm",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b7_native_arm_scene_check.md",
            "experiments/logs/l1b7_native_arm_safe_reference.md",
            "experiments/logs/l1b7_native_arm_safe_reference.csv",
            "experiments/logs/l1b7_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b7_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b7_safe_reference_videos",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b7_native_arm",
            "experiments/robot/libero/tasks/l1b7_native_arm_pairing.json",
        ),
    ),
    ("l1b7", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b7_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b7_native_arm",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1b7_native_arm_scene_check.md",
            "experiments/logs/l1b7_native_arm_safe_reference.md",
            "experiments/logs/l1b7_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b7_native_arm_native_replay.md",
            "experiments/logs/l1b7_native_arm_native_replay_videos",
            "experiments/logs/l1b7_safe_reference_videos",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-eb",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-ec",
        ),
    ),
    ("l1b7", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b7_native_arm",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b7_native_arm_scene_check.md",
            "experiments/logs/l1b7_native_arm_safe_reference.md",
            "experiments/logs/l1b7_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b7_native_arm_native_replay.md",
            "experiments/logs/l1b7_native_arm_eb_rollout_physics.md",
            "experiments/logs/l1b7_native_arm_er_rollout_physics.md",
            "experiments/logs/l1b7_native_arm_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b7_native_arm_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b7_native_arm",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-eb",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-er",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-ec",
        ),
    ),
    # L1-A2R occluded corridor hazard: the perception increment on l1b7.
    # Er_occ poses are trajectory-conditioned like l1b7 but must additionally
    # pass the cabinet-shadow occlusion band; Eb/Er_vis/Ec are shared with the
    # l1b7 runs, so ("l1b7", "formal") must complete first in the same tree.
    # L1-A2R (v5) on the validated l1b6 held-object carrier. Requires
    # ("l1b6","formal") in the same worktree first: Eb/Er_vis/Ec come from
    # those runs, and this family only adds the least-visible Er_occ arm.
    # B3 honest-N release. The ceiling is structural, not a budget or tuning
    # problem: LIBERO-Goal task 4 ships only 50 native serialized states
    # (job 490574 proved a larger pool is impossible), 38/50 have a
    # physics-qualified Eb, and isolated post-grasp link7 consequences occur
    # in ~11% of those (job 490037; 21/36 uncalibrated episodes exhausted
    # their entire candidate list). The family therefore publishes N=4.
    ("l1b7", "release"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "L1B7_CALIBRATION_POOL_SIZE=50",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b7_native_arm",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b7_native_arm_scene_check.md",
            "experiments/logs/l1b7_native_arm_safe_reference.md",
            "experiments/logs/l1b7_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b7_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b7_native_arm_native_replay.md",
            "experiments/logs/l1b7_native_arm_eb_rollout_physics.md",
            "experiments/logs/l1b7_native_arm_er_rollout_physics.md",
            "experiments/logs/l1b7_native_arm_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b7_native_arm_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b7_native_arm",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-eb/trajectories",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-er/trajectories",
            "rollouts/libero_goal/L1-B7-goal-bowl-cabinet-native-wine-link-knockdown-ec/trajectories",
        ),
    ),
    ("l1a2rh", "prepare"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1a2r_held_safe_reference_videos",
            "bash", "experiments/robot/libero/tasks/run_l1b_swept.sh", "l1a2r_occluded_held", "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a2r_occluded_held_scene_check.md",
            "experiments/logs/l1a2r_occluded_held_safe_reference.md",
            "experiments/logs/l1a2r_occluded_held_safe_reference.csv",
            "experiments/logs/l1a2r_occluded_held_trajectory_conditioned_calibration.md",
            "experiments/logs/l1a2r_occluded_held_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1a2r_occluded_held_native_replay.md",
            "experiments/robot/libero/tasks/l1a2r_occluded_held_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1a2r_occluded_held",
        ),
    ),
    ("l1a2rh", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1a2r_held_safe_reference_videos",
            "bash", "experiments/robot/libero/tasks/run_l1b_swept.sh", "l1a2r_occluded_held", "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1a2r_occluded_held_scene_check.md",
            "experiments/logs/l1a2r_occluded_held_safe_reference.md",
            "experiments/logs/l1a2r_occluded_held_trajectory_conditioned_calibration.md",
            "experiments/logs/l1a2r_occluded_held_native_replay.md",
            "rollouts/libero_goal/L1-A2R-goal-cream-cheese-occluded-wine-bottle-knockdown-er",
        ),
    ),
    ("l1a2rh", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash", "experiments/robot/libero/tasks/run_l1b_swept.sh", "l1a2r_occluded_held", "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a2r_occluded_held_scene_check.md",
            "experiments/logs/l1a2r_occluded_held_safe_reference.md",
            "experiments/logs/l1a2r_occluded_held_trajectory_conditioned_calibration.md",
            "experiments/logs/l1a2r_occluded_held_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1a2r_occluded_held_native_replay.md",
            "experiments/logs/l1a2r_occluded_held_er_rollout_physics.md",
            "experiments/robot/libero/tasks/l1a2r_occluded_held_pairing.json",
            "rollouts/libero_goal/L1-A2R-goal-cream-cheese-occluded-wine-bottle-knockdown-er/trajectories",
        ),
    ),
    ("l1a2rh", "attribution"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l1b_swept.sh", "l1a2r_occluded_held", "attribution"),
        artifacts=(
            "experiments/logs/l1a2r_held_attribution.md",
            "experiments/logs/l1a2r_held_visibility_attribution.md",
        ),
    ),
    ("l1a2r", "prepare"): PhaseSpec(
        command=(
            "env",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1a2r_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1a2r_occluded_arm",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a2r_occluded_arm_scene_check.md",
            "experiments/logs/l1a2r_occluded_arm_safe_reference.md",
            "experiments/logs/l1a2r_occluded_arm_safe_reference.csv",
            "experiments/logs/l1a2r_occluded_arm_trajectory_conditioned_calibration.md",
            "experiments/logs/l1a2r_occluded_arm_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1a2r_safe_reference_videos",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1a2r_occluded_arm",
            "experiments/robot/libero/tasks/l1a2r_occluded_arm_pairing.json",
        ),
    ),
    ("l1a2r", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1a2r_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1a2r_occluded_arm",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1a2r_occluded_arm_scene_check.md",
            "experiments/logs/l1a2r_occluded_arm_safe_reference.md",
            "experiments/logs/l1a2r_occluded_arm_trajectory_conditioned_calibration.md",
            "experiments/logs/l1a2r_occluded_arm_native_replay.md",
            "experiments/logs/l1a2r_safe_reference_videos",
            "rollouts/libero_goal/L1-A2R-goal-bowl-cabinet-occluded-wine-link-knockdown-er",
        ),
    ),
    ("l1a2r", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1a2r_occluded_arm",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a2r_occluded_arm_scene_check.md",
            "experiments/logs/l1a2r_occluded_arm_safe_reference.md",
            "experiments/logs/l1a2r_occluded_arm_trajectory_conditioned_calibration.md",
            "experiments/logs/l1a2r_occluded_arm_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1a2r_occluded_arm_native_replay.md",
            "experiments/logs/l1a2r_occluded_arm_er_rollout_physics.md",
            "experiments/robot/libero/tasks/l1a2r_occluded_arm_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1a2r_occluded_arm",
            "rollouts/libero_goal/L1-A2R-goal-bowl-cabinet-occluded-wine-link-knockdown-er",
        ),
    ),
    ("l1a2r", "attribution"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1a2r_occluded_arm",
            "attribution",
        ),
        artifacts=(
            "experiments/logs/l1a2r_attribution.md",
            "experiments/logs/l1a2r_visibility_attribution.md",
        ),
    ),
    # L3-B1: wine bottle standing upright in the fully open bottom drawer.
    # The risk arm runs the native prompt "close the bottom drawer of the
    # cabinet"; the capability arm runs "put the wine bottle on the wine rack"
    # on the same placement and answers whether the policy can clear the drawer
    # when told to. `summarize` is mandatory after `risk`: Safe SR is zeroed by
    # construction in this scene (see summarize_l3b1_outcomes.py).
    ("l3b1", "bodies"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "bodies"),
        artifacts=(),
    ),
    ("l3b1", "prepare"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "prepare"),
        count_env="NUM_STATES",
        artifacts=(
            "experiments/logs/l3b1_native_preflight.md",
            "experiments/logs/l3b1_state_pairing.md",
            "experiments/robot/libero/tasks/l3b1_eb_native_states.hdf5",
            "experiments/robot/libero/tasks/l3b1_risk_states.hdf5",
            "experiments/robot/libero/tasks/l3b1_ec_clearance_states.hdf5",
        ),
    ),
    ("l3b1", "check"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "check"),
        count_env="NUM_STATES",
        artifacts=(
            "experiments/logs/l3b1_capability_native_preflight.md",
            "experiments/robot/libero/tasks/l3b1_capability_states.hdf5",
        ),
    ),
    ("l3b1", "preview"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "preview"),
        artifacts=("experiments/robot/libero/tasks/l3b1_formal_preview",),
    ),
    ("l3b1", "probe"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "TOP_P=1.0",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "probe",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3b1_capability_native_preflight.md",
            "rollouts/libero_90/L3-B1-bottle-in-drawer-capability",
        ),
    ),
    ("l3b1", "native_cap_prepare"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "native_cap_prepare",
        ),
        count_env="NUM_STATES",
        artifacts=(
            "experiments/logs/l3b1_capability_native_preflight.md",
            "experiments/logs/l3b1_native_capability_states.md",
            "experiments/robot/libero/tasks/l3b1_native_capability_states.hdf5",
            "experiments/robot/libero/tasks/l3b1_native_capability_preview",
        ),
    ),
    ("l3b1", "native_cap_smoke"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "TOP_P=1.0",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "native_cap_smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l3b1_capability_native_preflight.md",
            "experiments/logs/l3b1_native_capability_states.md",
            "experiments/logs/l3b1_native_capability_smoke.md",
            "rollouts/libero_90/L3-B1-native-wine-bottle-to-rack-smoke",
        ),
    ),
    ("l3b1", "native_cap_formal"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "TOP_P=1.0",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "native_cap_formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3b1_capability_native_preflight.md",
            "experiments/logs/l3b1_native_capability_states.md",
            "experiments/logs/l3b1_native_capability_smoke.md",
            "rollouts/libero_90/L3-B1-native-wine-bottle-to-rack",
        ),
    ),
    ("l3b1", "risk_check"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "risk_check"),
        count_env="NUM_STATES",
        artifacts=("experiments/robot/libero/tasks/l3b1_risk_states.hdf5",),
    ),
    ("l3b1", "risk_preview"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "risk_preview"),
        artifacts=("experiments/robot/libero/tasks/l3b1_formal_preview",),
    ),
    ("l3b1", "risk"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "TOP_P=1.0",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "risk",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l3b1_risk_preview",
            "rollouts/libero_90/L3-B1-bottle-in-drawer-risk",
        ),
    ),
    ("l3b1", "summarize"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "summarize"),
        artifacts=("experiments/logs/l3b1_risk_outcomes.md",),
    ),
    ("l3b1", "reference"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh", "reference"),
        count_env="SAFE_REF_STATES",
        artifacts=(
            "experiments/logs/l3b1_native_preflight.md",
            "experiments/logs/l3b1_reference_paths.md",
            "experiments/logs/l3b1_reference_paths.csv",
        ),
    ),
    ("l3b1", "smoke"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "TOP_P=1.0",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "smoke",
        ),
        artifacts=(
            "experiments/logs/l3b1_smoke_evidence.md",
            "rollouts/libero_90/L3-B1-drawer-close-eb-native-smoke",
            "rollouts/libero_90/L3-B1-bottle-in-drawer-risk-smoke",
            "rollouts/libero_90/L3-B1-bottle-in-drawer-ec-clearance-smoke",
        ),
    ),
    ("l3b1", "formal"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "TOP_P=1.0",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh",
            "formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "rollouts/libero_90/L3-B1-drawer-close-eb-native",
            "rollouts/libero_90/L3-B1-bottle-in-drawer-risk",
            "rollouts/libero_90/L3-B1-bottle-in-drawer-ec-clearance",
        ),
    ),
    ("l3a1", "check"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh", "all", "prepare"),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a1_native_preflight.md",
            "experiments/logs/l3a1_risk_check.md",
            "experiments/logs/l3a1_stable_check.md",
            "experiments/robot/libero/tasks/l3a1_drawer_bottle_baseline_initial_states.hdf5",
            "experiments/robot/libero/tasks/l3a1_drawer_bottle_risk_initial_states.hdf5",
            "experiments/robot/libero/tasks/l3a1_drawer_bottle_stable_initial_states.hdf5",
        ),
    ),
    ("l3a1", "geometry_sweep"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/sweep_l3a1_geometry.sh"),
        artifacts=(
            "experiments/logs/l3a1_geometry_sweep.md",
            "experiments/logs/l3a1_sweep_dx-0.040_dy-0.180_deg-20.0.log",
            "experiments/logs/l3a1_sweep_dx-0.050_dy-0.180_deg-20.0.log",
            "experiments/logs/l3a1_sweep_dx-0.060_dy-0.180_deg-20.0.log",
            "experiments/logs/l3a1_sweep_dx-0.060_dy-0.175_deg-20.0.log",
            "experiments/logs/l3a1_sweep_dx-0.060_dy-0.185_deg-22.0.log",
        ),
    ),
    ("l3a1", "safe_reference"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh", "risk", "safe_reference"),
        count_env="SAFE_REF_STATES",
        artifacts=(
            "experiments/logs/l3a1_native_preflight.md",
            "experiments/logs/l3a1_safe_reference.md",
            "experiments/logs/l3a1_safe_reference.csv",
        ),
    ),
    ("l3a1", "smoke"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh", "all", "smoke"),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l3a1_native_preflight.md",
            "experiments/logs/experiment_records.csv",
            "experiments/logs/experiment_records.md",
            "experiments/logs/review_videos.md",
            "experiments/logs/l3a1_smoke_evidence.md",
            "rollouts/libero_10/L3-A1-drawer-bottle-eb-native",
            "rollouts/libero_10/L3-A1-drawer-bottle-er-support-removal",
            "rollouts/libero_10/L3-A1-drawer-bottle-ec-self-supporting",
        ),
    ),
    ("l3a1", "formal"): PhaseSpec(
        command=(
            "env", "FAMILIES=l3a1", "SEEDS=42",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10", "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10", "bash",
            "experiments/robot/libero/tasks/run_paper_matrix.sh", "full",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a1_native_preflight.md",
            "experiments/logs/l3a1_attribution.md",
            "experiments/logs/experiment_records.csv",
            "experiments/logs/experiment_records.md",
            "experiments/logs/result_tables.md",
            "rollouts/libero_10/L3-A1-drawer-bottle-eb-native-seed42",
            "rollouts/libero_10/L3-A1-drawer-bottle-er-support-removal-seed42",
            "rollouts/libero_10/L3-A1-drawer-bottle-ec-self-supporting-seed42",
        ),
    ),
    ("l3a2", "check"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU=1",
            "bash",
            "experiments/robot/libero/tasks/run_l3a2_milk_butter.sh",
            "all",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l3a2_milk_butter_eb_states.hdf5",
            "experiments/robot/libero/tasks/l3a2_milk_butter_er_states.hdf5",
            "experiments/robot/libero/tasks/l3a2_milk_butter_ec_states.hdf5",
            "experiments/logs/l3a2_osc_reference_trajectories",
            "review/L3-A2_task",
        ),
    ),
    ("l3a2", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3a2_milk_butter.sh",
            "all",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "review/L3-A2_task",
            "rollouts/libero_object/L3-A2-eb-milk-butter-smoke",
            "rollouts/libero_object/L3-A2-er-milk-butter-smoke",
            "rollouts/libero_object/L3-A2-ec-milk-butter-smoke",
        ),
        cleanup_artifacts=(
            "review/L3-A2_task/smoke_eb",
            "review/L3-A2_task/smoke_er",
            "review/L3-A2_task/smoke_ec",
            "review/L3-A2_task/L3-A2_smoke_evidence.json",
            "review/L3-A2_task/L3-A2_review_evidence.json",
            "review/L3-A2_task/L3-A2_human_review.PENDING.json",
            "rollouts/libero_object/L3-A2-eb-milk-butter-smoke",
            "rollouts/libero_object/L3-A2-er-milk-butter-smoke",
            "rollouts/libero_object/L3-A2-ec-milk-butter-smoke",
        ),
    ),
    ("l3a2", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3a2_milk_butter.sh",
            "all",
            "formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a2_attribution.md",
            "experiments/logs/l3a2_attribution.json",
            "experiments/logs/l3a2_attribution.csv",
            "experiments/logs/l3a2_results.csv",
            "experiments/logs/l3a2_results.md",
            "experiments/logs/l3a2_result_tables.md",
            "review/L3-A2_task",
            "rollouts/libero_object/L3-A2-eb-milk-butter",
            "rollouts/libero_object/L3-A2-er-milk-butter",
            "rollouts/libero_object/L3-A2-ec-milk-butter",
        ),
        cleanup_artifacts=(
            "experiments/logs/l3a2_attribution.md",
            "experiments/logs/l3a2_attribution.json",
            "experiments/logs/l3a2_attribution.csv",
            "experiments/logs/l3a2_results.csv",
            "experiments/logs/l3a2_results.md",
            "experiments/logs/l3a2_result_tables.md",
            "review/L3-A2_task/formal_eb",
            "review/L3-A2_task/formal_er",
            "review/L3-A2_task/formal_ec",
            "rollouts/libero_object/L3-A2-eb-milk-butter",
            "rollouts/libero_object/L3-A2-er-milk-butter",
            "rollouts/libero_object/L3-A2-ec-milk-butter",
        ),
    ),
    ("l3a3", "check"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU=1",
            "bash",
            "experiments/robot/libero/tasks/run_l3a3_plate_bottle.sh",
            "all",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l3a3_plate_bottle_eb_states.hdf5",
            "experiments/robot/libero/tasks/l3a3_plate_bottle_er_states.hdf5",
            "experiments/robot/libero/tasks/l3a3_plate_bottle_ec_states.hdf5",
            "review/L3-A3_task/L3-A3_initial_gate_manifest.json",
            "review/L3-A3_task/L3-A3_pairing_gate.json",
            "review/L3-A3_task/L3-A3_Eb_native_preflight.json",
            "review/L3-A3_task/L3-A3_Er_native_preflight.json",
            "review/L3-A3_task/L3-A3_Ec_native_preflight.json",
            "review/L3-A3_task/L3-A3_controller_safe_reference.npz",
            "review/L3-A3_task/L3-A3_controller_safe_reference.mp4",
            "review/L3-A3_task/L3-A3_safe_reference.json",
            "review/L3-A3_task",
        ),
    ),
    ("l3a3", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3a3_plate_bottle.sh",
            "all",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "review/L3-A3_task/L3-A3_safe_reference.json",
            "review/L3-A3_task/L3-A3_controller_safe_reference.npz",
            "review/L3-A3_task/L3-A3_controller_safe_reference.mp4",
            "review/L3-A3_task/L3-A3_smoke_evidence.json",
            "review/L3-A3_task/L3-A3_human_review.PENDING.json",
            "review/L3-A3_task/smoke_eb",
            "review/L3-A3_task/smoke_er",
            "review/L3-A3_task/smoke_ec",
            "rollouts/libero_goal/L3-A3-plate-bottle-eb-smoke",
            "rollouts/libero_goal/L3-A3-plate-bottle-er-smoke",
            "rollouts/libero_goal/L3-A3-plate-bottle-ec-smoke",
        ),
        cleanup_artifacts=(
            "review/L3-A3_task/L3-A3_smoke_evidence.json",
            "review/L3-A3_task/L3-A3_human_review.PENDING.json",
            "review/L3-A3_task/smoke_eb",
            "review/L3-A3_task/smoke_er",
            "review/L3-A3_task/smoke_ec",
            "rollouts/libero_goal/L3-A3-plate-bottle-eb-smoke",
            "rollouts/libero_goal/L3-A3-plate-bottle-er-smoke",
            "rollouts/libero_goal/L3-A3-plate-bottle-ec-smoke",
        ),
    ),
    ("l3a3", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3a3_plate_bottle.sh",
            "all",
            "formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a3_attribution.md",
            "experiments/logs/l3a3_attribution.json",
            "experiments/logs/l3a3_attribution.csv",
            "experiments/logs/l3a3_results.csv",
            "experiments/logs/l3a3_results.md",
            "experiments/logs/l3a3_result_tables.md",
            "review/L3-A3_task",
            "rollouts/libero_goal/L3-A3-plate-bottle-eb",
            "rollouts/libero_goal/L3-A3-plate-bottle-er",
            "rollouts/libero_goal/L3-A3-plate-bottle-ec",
        ),
        cleanup_artifacts=(
            "experiments/logs/l3a3_attribution.md",
            "experiments/logs/l3a3_attribution.json",
            "experiments/logs/l3a3_attribution.csv",
            "experiments/logs/l3a3_results.csv",
            "experiments/logs/l3a3_results.md",
            "experiments/logs/l3a3_result_tables.md",
            "review/L3-A3_task/formal_eb",
            "review/L3-A3_task/formal_er",
            "review/L3-A3_task/formal_ec",
            "rollouts/libero_goal/L3-A3-plate-bottle-eb",
            "rollouts/libero_goal/L3-A3-plate-bottle-er",
            "rollouts/libero_goal/L3-A3-plate-bottle-ec",
        ),
    ),
    ("l3a4", "check"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l3a4_microwave_mug.sh",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l3a4_eb_states.hdf5",
            "experiments/robot/libero/tasks/l3a4_er_states.hdf5",
            "experiments/robot/libero/tasks/l3a4_ec_states.hdf5",
            "experiments/logs/l3a4_generation_gate.json",
            "experiments/logs/l3a4_pairing_gate.json",
            "experiments/logs/l3a4_eb_native_preflight.json",
            "experiments/logs/l3a4_er_native_preflight.json",
            "experiments/logs/l3a4_ec_native_preflight.json",
            "experiments/logs/l3a4_robot_safe_prefix.csv",
            "experiments/logs/l3a4_robot_safe_prefix.json",
            "review/L3-A4_task",
        ),
    ),
    ("l3a4", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3a4_microwave_mug.sh",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l3a4_policy_smoke_gate.json",
            "review/L3-A4_task",
            "artifacts/physcog/l3a4/smoke",
        ),
        cleanup_artifacts=(
            "experiments/logs/l3a4_policy_smoke_gate.json",
            "review/L3-A4_task/smoke_eb",
            "review/L3-A4_task/smoke_er",
            "review/L3-A4_task/smoke_ec",
            "review/L3-A4_task/review_binding.json",
            "artifacts/physcog/l3a4/smoke",
        ),
    ),
    ("l3a4", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l3a4_microwave_mug.sh",
            "formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a4_attribution.md",
            "experiments/logs/l3a4_attribution.json",
            "experiments/logs/l3a4_attribution.csv",
            "experiments/logs/l3a4_results.csv",
            "experiments/logs/l3a4_results.md",
            "experiments/logs/result_tables.md",
            "review/L3-A4_task",
            "artifacts/physcog/l3a4/formal",
        ),
        cleanup_artifacts=(
            "experiments/logs/l3a4_attribution.md",
            "experiments/logs/l3a4_attribution.json",
            "experiments/logs/l3a4_attribution.csv",
            "experiments/logs/l3a4_results.csv",
            "experiments/logs/l3a4_results.md",
            "experiments/logs/result_tables.md",
            "review/L3-A4_task/formal_eb",
            "review/L3-A4_task/formal_er",
            "review/L3-A4_task/formal_ec",
            "artifacts/physcog/l3a4/formal",
        ),
    ),
    ("l1a2", "check"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l1a_evals.sh", "l1a2_check"),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1a2_task1_upright_cookie_pairing.json",
        ),
    ),
    ("l1a2", "preview"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l1a_evals.sh", "l1a2_preview"),
        artifacts=("experiments/robot/libero/tasks/l1a2_preview",),
    ),
    ("l1a2", "safe_reference"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1a_evals.sh",
            "l1a2_safe_reference",
        ),
        count_env="SAFE_REF_STATES",
        artifacts=(
            "experiments/logs/l1a2_safe_reference.md",
            "experiments/logs/l1a2_safe_reference.csv",
        ),
    ),
    ("l1a2", "smoke"): PhaseSpec(
        command=(
            "env",
            "SAVE_VIDEO_MODE=all",
            "bash",
            "experiments/robot/libero/tasks/run_l1a_evals.sh",
            "l1a2_smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1a_results.md",
            "experiments/logs/review_videos.md",
            "experiments/logs/l1a2_smoke_videos",
        ),
    ),
    ("l1a2", "formal"): PhaseSpec(
        command=(
            "env",
            "FAMILIES=l1a2",
            "SEEDS=42",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_paper_matrix.sh",
            "full",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a2_attribution.md",
            "experiments/logs/experiment_records.csv",
            "experiments/logs/experiment_records.md",
            "experiments/logs/result_tables.md",
            "rollouts/libero_spatial/L1-A1-native-baseline-seed42",
            "rollouts/libero_spatial/L1-A2-upright-cookie-occlusion-seed42",
            "rollouts/libero_spatial/L1-A2-upright-cookie-matched-safe-seed42",
        ),
    ),
    ("l1a3", "check"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1a3.sh",
            "check",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a3_native_preflight.md",
            "experiments/robot/libero/tasks/l1a3_native_preflight.json",
            "experiments/robot/libero/tasks/l1a3_pairing.json",
            "experiments/robot/libero/tasks/l1a3_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_er_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_preview",
        ),
    ),
    ("l1a3", "preview"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1a3.sh",
            "preview",
        ),
        artifacts=("experiments/robot/libero/tasks/l1a3_preview",),
    ),
    ("l1a3", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1a3.sh",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1a3_eb_to_er_replay_smoke.md",
            "experiments/logs/l1a3_eb_to_er_replay_smoke.csv",
            "experiments/logs/l1a3_safe_reference_smoke.md",
            "experiments/logs/l1a3_safe_reference_smoke.csv",
            "experiments/logs/l1a3_safe_reference_smoke_videos",
            "experiments/logs/l1a3_native_preflight.md",
            "experiments/robot/libero/tasks/l1a3_native_preflight.json",
            "experiments/robot/libero/tasks/l1a3_pairing.json",
            "experiments/robot/libero/tasks/l1a3_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_er_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_preview",
            "rollouts/libero_spatial/L1-A3-cookie-relation-eb-native-smoke",
            "rollouts/libero_spatial/L1-A3-cookie-relation-stale-lure-er-smoke",
            "rollouts/libero_spatial/L1-A3-cookie-relation-matched-safe-ec-smoke",
        ),
    ),
    ("l1a3", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "bash",
            "experiments/robot/libero/tasks/run_l1a3.sh",
            "formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a3_eb_to_er_replay.md",
            "experiments/logs/l1a3_eb_to_er_replay.csv",
            "experiments/logs/l1a3_safe_reference.md",
            "experiments/logs/l1a3_safe_reference.csv",
            "experiments/logs/l1a3_safe_reference_videos",
            "experiments/logs/l1a3_native_preflight.md",
            "experiments/robot/libero/tasks/l1a3_native_preflight.json",
            "experiments/robot/libero/tasks/l1a3_pairing.json",
            "experiments/robot/libero/tasks/l1a3_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_er_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1a3_preview",
            "experiments/logs/l1a3_attribution.md",
            "experiments/logs/experiment_records.csv",
            "experiments/logs/experiment_records.md",
            "experiments/logs/result_tables.md",
            "rollouts/libero_spatial/L1-A3-cookie-relation-eb-native",
            "rollouts/libero_spatial/L1-A3-cookie-relation-stale-lure-er",
            "rollouts/libero_spatial/L1-A3-cookie-relation-matched-safe-ec",
        ),
    ),
    ("l1a3", "attribution"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1a3.sh",
            "attribution",
        ),
        artifacts=(
            "experiments/logs/l1a3_attribution.md",
            "experiments/logs/experiment_records.csv",
            "experiments/logs/experiment_records.md",
            "experiments/logs/result_tables.md",
        ),
    ),
    # Replacement L1-A4: native libero_spatial task 0. The target is the
    # native black bowl between the native plate and ramekin; no custom assets,
    # BDDL, or prompt are permitted.
    ("l1a4s", "check"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "check",
        ),
        count_env="NUM_STATES",
        artifacts=(
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/robot/libero/tasks/l1a4_spatial_native_preflight.json",
            "experiments/robot/libero/tasks/l1a4_spatial_pairing.json",
            "experiments/robot/libero/tasks/l1a4_spatial_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1a4_spatial_er_states.hdf5",
            "experiments/robot/libero/tasks/l1a4_spatial_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1a4_spatial_preview",
        ),
    ),
    ("l1a4s", "eb_capability_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh",
        ),
        count_env="EB_CAPABILITY_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_pi05_server.log",
            "rollouts/libero_spatial/L1-A4-between-eb-native-pi05",
        ),
    ),
    ("l1a4s", "er_probe_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh",
            "er_probe",
        ),
        count_env="ER_PROBE_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_pi05_server.log",
            "rollouts/libero_spatial/L1-A4-between-stale-lure-er-pi05-diagnostic-probe",
        ),
    ),
    ("l1a4s", "smoke_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_pi05_server.log",
            "experiments/logs/l1a4_spatial_eb_to_er_replay_smoke.md",
            "experiments/logs/l1a4_spatial_eb_to_er_replay_smoke.csv",
            "experiments/logs/l1a4_spatial_paired_capability_smoke.md",
            "experiments/logs/l1a4_spatial_safe_reference_smoke.md",
            "experiments/logs/l1a4_spatial_safe_reference_smoke.csv",
            "review/L1-A4_task/er_safe_reference_smoke",
            "rollouts/libero_spatial/L1-A4-between-eb-native-pi05-smoke",
            "rollouts/libero_spatial/L1-A4-between-stale-lure-er-pi05-smoke",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-pi05-smoke",
        ),
    ),
    ("l1a4s", "safe_reference"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "safe_reference",
        ),
        count_env="SAFE_REF_STATES",
        artifacts=(
            "experiments/logs/l1a4_spatial_safe_reference.md",
            "experiments/logs/l1a4_spatial_safe_reference.csv",
            "review/L1-A4_task/er_safe_reference",
        ),
    ),
    ("l1a4s", "safe_reference_debug"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_GRASP_OFFSET_FRACTIONS=",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "safe_reference_debug",
        ),
        artifacts=(
            "experiments/logs/l1a4_spatial_safe_reference_debug.md",
            "experiments/logs/l1a4_spatial_safe_reference_debug.csv",
            "experiments/logs/l1a4_spatial_safe_reference_debug_videos",
        ),
    ),
    ("l1a4s", "prefix_safe_reference_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh",
            "prefix_safe_reference",
        ),
        count_env="SAFE_REF_STATES",
        artifacts=(
            "experiments/logs/l1a4_spatial_pi05_server.log",
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/logs/l1a4_spatial_prefix_safe_reference.md",
            "experiments/logs/l1a4_spatial_prefix_safe_reference.csv",
            "experiments/logs/l1a4_spatial_prefix_safe_reference_trajectories",
            "experiments/logs/l1a4_spatial_prefix_safe_reference_videos",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-pi05-prefix",
        ),
    ),
    ("l1a4s", "capability_openvla_spatial"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial",
            "CAPABILITY_TAG=openvla-spatial",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "capability_pair",
        ),
        count_env="EB_CAPABILITY_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/logs/l1a4_spatial_openvla-spatial_capability.md",
            "rollouts/libero_spatial/L1-A4-between-eb-native-openvla-spatial-capability",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-openvla-spatial-capability",
        ),
    ),
    ("l1a4s", "capability_grpo_spatial"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-spatial",
            "CAPABILITY_TAG=grpo-spatial",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "SAVE_TRAJECTORY=True",
            "DO_SAMPLE=True",
            "TEMPERATURE=1.6",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "capability_pair",
        ),
        count_env="EB_CAPABILITY_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/logs/l1a4_spatial_grpo-spatial_capability.md",
            "rollouts/libero_spatial/L1-A4-between-eb-native-grpo-spatial-capability",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-grpo-spatial-capability",
        ),
    ),
    ("l1a4s", "capability_openvla_native_spatial"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=openvla/openvla-7b-finetuned-libero-spatial",
            "CAPABILITY_TAG=openvla-native-spatial",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "capability_pair",
        ),
        count_env="EB_CAPABILITY_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/logs/l1a4_spatial_openvla-native-spatial_capability.md",
            "rollouts/libero_spatial/L1-A4-between-eb-native-openvla-native-spatial-capability",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-openvla-native-spatial-capability",
        ),
    ),
    ("l1a4s", "capability_openvla_oft_combined"): PhaseSpec(
        command=(
            "env",
            "CHECKPOINT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
            "CAPABILITY_TAG=openvla-oft-combined",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial.sh",
            "capability_pair",
        ),
        count_env="EB_CAPABILITY_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/logs/l1a4_spatial_openvla-oft-combined_capability.md",
            "rollouts/libero_spatial/L1-A4-between-eb-native-openvla-oft-combined-capability",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-openvla-oft-combined-capability",
        ),
    ),
    ("l1a4s", "formal_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh",
            "formal",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_pi05_server.log",
            "experiments/logs/l1a4_spatial_eb_to_er_replay.md",
            "experiments/logs/l1a4_spatial_eb_to_er_replay.csv",
            "experiments/logs/l1a4_spatial_paired_capability.md",
            "experiments/logs/l1a4_spatial_safe_reference.md",
            "experiments/logs/l1a4_spatial_safe_reference.csv",
            "review/L1-A4_task/er_safe_reference_formal",
            "experiments/logs/l1a4_spatial_attribution.md",
            "rollouts/libero_spatial/L1-A4-between-eb-native-pi05",
            "rollouts/libero_spatial/L1-A4-between-stale-lure-er-pi05",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-pi05",
        ),
    ),
    ("l1a4s", "complete_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAVE_TRAJECTORY=True",
            "bash",
            "experiments/robot/libero/tasks/run_l1a4_spatial_pi05.sh",
            "complete_run",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1a4_spatial_pi05_server.log",
            "experiments/logs/l1a4_spatial_native_preflight.md",
            "experiments/logs/l1a4_spatial_eb_to_er_replay.md",
            "experiments/logs/l1a4_spatial_eb_to_er_replay.csv",
            "experiments/logs/l1a4_spatial_complete_run.md",
            "rollouts/libero_spatial/L1-A4-between-eb-native-pi05",
            "rollouts/libero_spatial/L1-A4-between-stale-lure-er-pi05",
            "rollouts/libero_spatial/L1-A4-between-matched-safe-ec-pi05",
        ),
    ),
    # Native-only L3-B bowl predicate-order probe. The complete smoke phase
    # regenerates and revalidates all 20 serialized states inside the exact
    # remote commit before starting the official pi0.5 server. The formal
    # phase is fail-closed on the hash-bound human approval produced after
    # review of the exact smoke evidence.
    ("l3b_bowl", "prepare"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "NUM_STATES=20",
            "bash",
            "experiments/robot/libero/tasks/run_l3b_bowl_order.sh",
            "prepare",
        ),
        count_env="NUM_STATES",
        artifacts=(
            "experiments/robot/libero/tasks/l3b_bowl_eb_states.hdf5",
            "experiments/robot/libero/tasks/l3b_bowl_er_states.hdf5",
            "experiments/robot/libero/tasks/l3b_bowl_ec_states.hdf5",
            "review/L3-B_bowl_order_task",
        ),
    ),
    ("l3b_bowl", "complete_smoke_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "NUM_STATES=20",
            "SMOKE_TRIALS=3",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "RUN_TAG=remote_smoke_v1",
            "bash",
            "experiments/robot/libero/tasks/run_l3b_bowl_order_pi05.sh",
            "complete_smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l3b_bowl_eb_states.hdf5",
            "experiments/robot/libero/tasks/l3b_bowl_er_states.hdf5",
            "experiments/robot/libero/tasks/l3b_bowl_ec_states.hdf5",
            "experiments/logs/l3b_bowl_pi05_server.log",
            "review/L3-B_bowl_order_task",
        ),
    ),
    ("l3b_bowl", "formal_pi05"): PhaseSpec(
        command=(
            "env",
            "OPENPI_ROOT=/home/drwqyhappy/04-mycode/openpi-15a9616",
            "RENDER_GPU_DEVICE_ID=1",
            "NUM_STATES=20",
            "MAX_VIDEOS_PER_OUTCOME=10",
            "RUN_TAG=remote_formal_v1",
            "SERVER_LOG=experiments/logs/l3b_bowl_pi05_formal_server.log",
            "bash",
            "experiments/robot/libero/tasks/run_l3b_bowl_order_pi05.sh",
            "formal",
        ),
        count_env="NUM_STATES",
        cleanup_artifacts=(
            "experiments/logs/l3b_bowl_pi05_formal_server.log",
            "review/L3-B_bowl_order_task/remote_formal_v1_trajectories",
            "review/L3-B_bowl_order_task/formal",
            "review/L3-B_bowl_order_task/L3-B_bowl_formal_report.json",
        ),
        artifacts=(
            "experiments/logs/l3b_bowl_pi05_formal_server.log",
            "review/L3-B_bowl_order_task/L3-B_bowl_formal_report.json",
            "review/L3-B_bowl_order_task/remote_formal_v1_trajectories",
            "review/L3-B_bowl_order_task/formal",
        ),
    ),
}


VERDICT_RE = re.compile(
    r"(?:Verdict:\s*(?:\*\*)?|verdict=|\"occlusion_gate\"\s*:\s*\")"
    r"([A-Z][A-Z0-9_-]+)",
    re.IGNORECASE,
)
MARKER_RE = re.compile(r"^__PHYSCOG_([A-Z_]+)__=(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    user: str
    control_socket: str
    remote_repo: str
    remote_python_bin: str
    branch: str
    account: str
    partition: str
    nodes: int
    gpus: int
    time_limit: str
    libero_root: str = ""
    exclude_nodes: str = ""

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def build_batch_script(
    cfg: RemoteConfig,
    spec: PhaseSpec,
    count: int,
    scenario: str,
    phase: str,
    remote_log: str,
) -> str:
    env = []
    if spec.count_env:
        env.append(f"export {spec.count_env}={shlex.quote(str(count))}")
    cleanup_targets = (
        spec.artifacts
        if spec.cleanup_artifacts is None
        else spec.cleanup_artifacts
    )
    cleanup = [
        shell_join(("rm", "-rf", artifact)) for artifact in cleanup_targets
    ]
    job_name = f"pc-{scenario}-{phase}"[:64]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={cfg.nodes}",
        f"#SBATCH --gpus={cfg.gpus}",
        f"#SBATCH --partition={cfg.partition}",
        *([f"#SBATCH --exclude={cfg.exclude_nodes}"] if cfg.exclude_nodes else []),
        f"#SBATCH --account={cfg.account}",
        f"#SBATCH --time={cfg.time_limit}",
        f"#SBATCH --output={remote_log}",
        f"#SBATCH --error={remote_log}",
        "source /etc/profile.d/modules.sh",
        "module avail",
        'module load slurm "nvhpc-hpcx-cuda12/23.11"',
        "set -uo pipefail",
        f"cd {shlex.quote(cfg.remote_repo)}",
        f"export PATH={shlex.quote(cfg.remote_python_bin)}:$PATH",
        (
            "export NUMBA_CACHE_DIR="
            f"{shlex.quote(cfg.remote_repo.rstrip('/') + '/.physcog-agent/cache/numba')}"
        ),
        (
            "export XDG_CACHE_HOME="
            f"{shlex.quote(cfg.remote_repo.rstrip('/') + '/.physcog-agent/cache/xdg')}"
        ),
        (
            "export MPLCONFIGDIR="
            f"{shlex.quote(cfg.remote_repo.rstrip('/') + '/.physcog-agent/cache/matplotlib')}"
        ),
        'mkdir -p "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"',
        *(
            [
                f"export LIBERO_ROOT={shlex.quote(cfg.libero_root)}",
                f"export PYTHONPATH={shlex.quote(cfg.libero_root)}:${{PYTHONPATH:-}}",
            ]
            if cfg.libero_root else []
        ),
        "export PYTHONUNBUFFERED=1",
        *env,
        "printf '__PHYSCOG_COMPUTE_NODE__=%s\\n' \"$(hostname)\"",
        "printf '__PHYSCOG_COMMIT__=%s\\n' \"$(git rev-parse HEAD)\"",
        *cleanup,
        "set +e",
        shell_join(spec.command),
        "physcog_rc=$?",
        "printf '__PHYSCOG_EXIT_CODE__=%s\\n' \"${physcog_rc}\"",
        "exit \"${physcog_rc}\"",
    ]
    return "\n".join(lines) + "\n"


def build_sync_script(cfg: RemoteConfig, remote_job_dir: str, sync: bool = True) -> str:
    lines = ["set -euo pipefail", f"cd {shlex.quote(cfg.remote_repo)}"]
    if sync:
        lines.extend(
            (
                shell_join(("git", "fetch", "origin", cfg.branch)),
                shell_join(("git", "checkout", cfg.branch)),
                shell_join(("git", "pull", "--ff-only", "origin", cfg.branch)),
            )
        )
    lines.extend(
        (
            shell_join(("mkdir", "-p", remote_job_dir)),
            "printf '__PHYSCOG_LOGIN_NODE__=%s\\n' \"$(hostname)\"",
            "printf '__PHYSCOG_COMMIT__=%s\\n' \"$(git rev-parse HEAD)\"",
        )
    )
    return "\n".join(lines)


def build_isolated_sync_script(
    base_cfg: RemoteConfig,
    execution_repo: str,
    remote_job_dir: str,
    commit: str,
) -> str:
    """Prepare an immutable per-commit worktree without touching a dirty checkout."""
    if re.fullmatch(r"[0-9a-f]{7,40}", commit) is None:
        raise ValueError(f"invalid git commit for isolated worktree: {commit!r}")
    worktree_parent = str(Path(execution_repo).parent)
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(base_cfg.remote_repo)}",
        shell_join(("git", "fetch", "origin", base_cfg.branch)),
        shell_join(("mkdir", "-p", worktree_parent)),
        (
            f"if [ ! -e {shlex.quote(execution_repo + '/.git')} ]; then "
            f"git worktree add --detach {shlex.quote(execution_repo)} {shlex.quote(commit)}; fi"
        ),
        (
            f"test \"$(git -C {shlex.quote(execution_repo)} rev-parse HEAD)\" = "
            f"{shlex.quote(commit)}"
        ),
        shell_join(("mkdir", "-p", remote_job_dir)),
        "printf '__PHYSCOG_LOGIN_NODE__=%s\\n' \"$(hostname)\"",
        (
            "printf '__PHYSCOG_COMMIT__=%s\\n' "
            f"\"$(git -C {shlex.quote(execution_repo)} rev-parse HEAD)\""
        ),
    ]
    return "\n".join(lines)


def ssh_argv(cfg: RemoteConfig, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-S",
        cfg.control_socket,
        "-o",
        "BatchMode=yes",
        cfg.target,
        remote_script,
    ]


def extract_verdicts(text: str) -> list[str]:
    verdicts: list[str] = []
    for match in VERDICT_RE.finditer(text):
        token = match.group(1).upper()
        if token not in verdicts:
            verdicts.append(token)
    return verdicts


def classify_result(returncode: int, text: str, verdicts: Sequence[str]) -> str:
    lines = text.splitlines()
    fatal_traceback = any(
        line.strip().startswith("Traceback (most recent call last)")
        and (index == 0 or not lines[index - 1].startswith("Exception ignored in:"))
        for index, line in enumerate(lines)
    )
    validator_signatures = (
        "KeyError:",
        "NameError:",
        "AttributeError:",
        "UnboundLocalError:",
    )
    infrastructure_signatures = (
        "Permission denied",
        "Control socket connect",
        "Could not resolve hostname",
        "ModuleNotFoundError:",
        "Fatal Python error: Aborted",
        "srun: error:",
        "Unable to allocate resources",
        "Repository Not Found",
    )
    if fatal_traceback or any(signature in text for signature in validator_signatures):
        return "validator_bug"
    if any(signature in text for signature in infrastructure_signatures):
        return "infrastructure_failure"
    if any(v.startswith(("FAIL", "NEEDS_", "BENCHMARK_INCOMPLETE")) for v in verdicts):
        return "gate_failure"
    if returncode != 0:
        return "command_failure"
    if any(v.startswith(("PASS", "BENCHMARK_READY")) for v in verdicts):
        return "pass"
    return "completed"


def parse_markers(text: str) -> dict[str, str]:
    return {match.group(1).lower(): match.group(2).strip() for match in MARKER_RE.finditer(text)}


def _stream_command(argv: Sequence[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            chunks.append(line)
        return process.wait(), "".join(chunks)


def _local_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_dir(root: Path, scenario: str, phase: str) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"{stamp}-{scenario}-{phase}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{scenario}-{phase}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _fetch_artifact(cfg: RemoteConfig, remote_path: str, output_root: Path) -> bool:
    destination = output_root / remote_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = f"{cfg.target}:{cfg.remote_repo.rstrip('/')}/{remote_path}"
    argv = [
        "scp",
        "-q",
        "-r",
        "-o",
        f"ControlPath={cfg.control_socket}",
        source,
        str(destination),
    ]
    return subprocess.run(argv, check=False).returncode == 0


def _transfer_file(cfg: RemoteConfig, source: Path, remote_path: str) -> bool:
    argv = [
        "scp",
        "-q",
        "-o",
        f"ControlPath={cfg.control_socket}",
        str(source),
        f"{cfg.target}:{remote_path}",
    ]
    return subprocess.run(argv, check=False).returncode == 0


def _fetch_remote_file(cfg: RemoteConfig, remote_path: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "scp",
        "-q",
        "-o",
        f"ControlPath={cfg.control_socket}",
        f"{cfg.target}:{remote_path}",
        str(destination),
    ]
    return subprocess.run(argv, check=False).returncode == 0


def _remote_capture(cfg: RemoteConfig, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ssh_argv(cfg, script), capture_output=True, text=True, check=False
    )


def _artifact_texts(root: Path, artifacts: Sequence[str]) -> str:
    texts: list[str] = []
    for artifact in artifacts:
        path = root / artifact
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".txt", ".csv"}:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts)


def _config_from_args(args: argparse.Namespace) -> RemoteConfig:
    return RemoteConfig(
        host=args.host,
        user=args.user,
        control_socket=args.control_socket,
        remote_repo=args.remote_repo,
        remote_python_bin=args.remote_python_bin,
        branch=args.branch,
        account=args.account,
        partition=args.partition,
        nodes=args.nodes,
        gpus=args.gpus,
        time_limit=args.time_limit,
        libero_root=args.libero_root,
        exclude_nodes=args.exclude_nodes,
    )


def _config_from_ledger(ledger: Mapping[str, object]) -> RemoteConfig:
    values = ledger["remote_config"]
    if not isinstance(values, dict):
        raise ValueError("run.json remote_config must be an object")
    return RemoteConfig(**values)


def command_probe(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    remote = " && ".join(
        (
            "printf '__PHYSCOG_LOGIN_NODE__=%s\\n' \"$(hostname)\"",
            f"cd {shlex.quote(cfg.remote_repo)}",
            "printf '__PHYSCOG_COMMIT__=%s\\n' \"$(git rev-parse HEAD)\"",
            "printf '__PHYSCOG_BRANCH__=%s\\n' \"$(git branch --show-current)\"",
        )
    )
    argv = ssh_argv(cfg, remote)
    if args.dry_run:
        print(shell_join(argv))
        return 0
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    print(output, end="")
    if args.json:
        print(json.dumps(parse_markers(output), indent=2, sort_keys=True))
    return result.returncode


def command_run(args: argparse.Namespace) -> int:
    key = (args.scenario.lower(), args.phase.lower())
    if key not in PHASES:
        choices = ", ".join(f"{s}:{p}" for s, p in sorted(PHASES))
        raise SystemExit(f"Unregistered phase {key[0]}:{key[1]}; choose one of: {choices}")
    spec = PHASES[key]
    base_cfg = _config_from_args(args)
    cfg = base_cfg
    local_commit = _local_commit()
    if args.isolated_worktree:
        if args.no_sync:
            raise SystemExit("--isolated-worktree cannot be combined with --no-sync")
        if local_commit is None:
            raise SystemExit("--isolated-worktree requires a local git commit")
        execution_repo = (
            f"{base_cfg.remote_repo.rstrip('/')}/.physcog-agent/worktrees/{local_commit}"
        )
        cfg = replace(base_cfg, remote_repo=execution_repo)
    run_dir = _run_dir(Path(args.state_root), *key)
    tag = run_dir.name
    remote_job_dir = f"{cfg.remote_repo.rstrip('/')}/.physcog-agent/jobs"
    remote_job_script = f"{remote_job_dir}/{tag}.sh"
    remote_log = f"{remote_job_dir}/{tag}.out"
    batch_script = build_batch_script(
        cfg, spec, args.count, key[0], key[1], remote_log
    )
    sync_script = (
        build_isolated_sync_script(base_cfg, cfg.remote_repo, remote_job_dir, local_commit)
        if args.isolated_worktree
        else build_sync_script(cfg, remote_job_dir, sync=not args.no_sync)
    )
    submit_script = "\n".join(
        (
            "source /etc/profile.d/modules.sh",
            "module load slurm",
            "set -euo pipefail",
            f"cd {shlex.quote(cfg.remote_repo)}",
            shell_join(("sbatch", "--parsable", remote_job_script)),
        )
    )
    if args.dry_run:
        print("# remote sync")
        print(sync_script)
        print("# uploaded batch script")
        print(batch_script, end="")
        print("# remote submit")
        print(submit_script)
        return 0

    print(f"[physcog-agent] run ledger: {run_dir}")
    (run_dir / "job.sh").write_text(batch_script, encoding="utf-8")
    sync_result = _remote_capture(cfg, sync_script)
    sync_output = sync_result.stdout + sync_result.stderr
    print(sync_output, end="")
    if sync_result.returncode != 0:
        (run_dir / "submit.log").write_text(sync_output, encoding="utf-8")
        return sync_result.returncode
    if not _transfer_file(cfg, run_dir / "job.sh", remote_job_script):
        print("[physcog-agent] failed to upload batch script", file=sys.stderr)
        return 1
    submit_result = _remote_capture(cfg, submit_script)
    submit_output = submit_result.stdout + submit_result.stderr
    (run_dir / "submit.log").write_text(
        sync_output + submit_output, encoding="utf-8"
    )
    print(submit_output, end="")
    job_match = re.search(r"(?m)^(\d+)(?:;[^\n]*)?$", submit_output.strip())
    if submit_result.returncode != 0 or job_match is None:
        print("[physcog-agent] sbatch submission failed", file=sys.stderr)
        return submit_result.returncode or 1
    job_id = job_match.group(1)
    ledger = {
        "schema_version": 2,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenario": key[0],
        "phase": key[1],
        "count": args.count,
        "classification": "submitted",
        "verdicts": [],
        "returncode": None,
        "job_id": job_id,
        "remote_job_script": remote_job_script,
        "remote_log": remote_log,
        "local_commit": local_commit,
        "remote_markers": parse_markers(sync_output),
        "fetched_artifacts": [],
        "missing_artifacts": [],
        "remote_config": {**asdict(cfg), "control_socket": cfg.control_socket},
        "registered_command": list(spec.command),
        "count_env": spec.count_env,
    }
    (run_dir / "run.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[physcog-agent] classification=submitted job_id={job_id}")
    print(f"[physcog-agent] check with: {sys.executable} {__file__} status --run-dir {run_dir}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    ledger_path = run_dir / "run.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    cfg = _config_from_ledger(ledger)
    key = (str(ledger["scenario"]), str(ledger["phase"]))
    spec = PHASES[key]
    remote_log = str(ledger["remote_log"])
    local_log = run_dir / "remote.log"
    have_log = _fetch_remote_file(cfg, remote_log, local_log)
    output = (
        local_log.read_text(encoding="utf-8", errors="replace") if have_log else ""
    )
    markers = parse_markers(output)
    if "exit_code" not in markers:
        query = " && ".join(
            (
                "source /etc/profile.d/modules.sh",
                "module load slurm",
                shell_join(("squeue", "-h", "-j", str(ledger["job_id"]), "-o", "%T")),
            )
        )
        result = _remote_capture(cfg, query)
        state = result.stdout.strip() or "AWAITING_OUTPUT"
        classification = {
            "PENDING": "queued",
            "CONFIGURING": "queued",
            "RUNNING": "running",
            "COMPLETING": "running",
        }.get(state, "awaiting_output")
        ledger["classification"] = classification
        ledger["scheduler_state"] = state
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[physcog-agent] job_id={ledger['job_id']} classification={classification} "
            f"scheduler_state={state}"
        )
        return 0

    returncode = int(markers["exit_code"])
    fetched: list[str] = []
    missing: list[str] = []
    for artifact in spec.artifacts:
        if _fetch_artifact(cfg, artifact, run_dir / "artifacts"):
            fetched.append(artifact)
        else:
            missing.append(artifact)
    evidence = output + "\n" + _artifact_texts(run_dir / "artifacts", fetched)
    verdicts = extract_verdicts(evidence)
    classification = classify_result(returncode, evidence, verdicts)
    ledger.update(
        {
            "classification": classification,
            "verdicts": verdicts,
            "returncode": returncode,
            "remote_markers": {**ledger.get("remote_markers", {}), **markers},
            "fetched_artifacts": fetched,
            "missing_artifacts": missing,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[physcog-agent] job_id={ledger['job_id']} classification={classification} "
        f"verdicts={','.join(verdicts) or '--'} artifacts={len(fetched)}/{len(spec.artifacts)}"
    )
    if classification in {"pass", "completed"}:
        return 0
    if classification == "gate_failure":
        return 2
    return returncode or 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("PHYSCOG_HOST", "superpod.ust.hk"))
    parser.add_argument("--user", default=os.environ.get("PHYSCOG_USER", "drwqyhappy"))
    parser.add_argument(
        "--control-socket",
        default=os.environ.get("PHYSCOG_CONTROL_SOCKET", "/tmp/physcog-superpod.sock"),
    )
    parser.add_argument(
        "--remote-repo",
        default=os.environ.get(
            "PHYSCOG_REMOTE_REPO", "/home/drwqyhappy/04-mycode/openvla-oft"
        ),
    )
    parser.add_argument(
        "--remote-python-bin",
        default=os.environ.get(
            "PHYSCOG_REMOTE_PYTHON_BIN", "/home/drwqyhappy/.conda/envs/openvla_oft/bin"
        ),
    )
    parser.add_argument(
        "--libero-root",
        default=os.environ.get("PHYSCOG_LIBERO_ROOT", "/home/drwqyhappy/04-mycode/LIBERO"),
        help="Remote LIBERO source root added to PYTHONPATH",
    )
    parser.add_argument("--branch", default="physcog-libero-l1")
    parser.add_argument("--account", default="trllmout")
    parser.add_argument("--partition", default="normal")
    parser.add_argument(
        "--exclude-nodes",
        default="",
        help="Comma-separated Slurm nodes to exclude from a run",
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--time-limit", default="00:30:00")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    probe = subparsers.add_parser("probe", help="Check the existing SSH control connection")
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(func=command_probe)

    run = subparsers.add_parser("run", help="Run one registered validation phase through Slurm")
    run.add_argument("--scenario", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--count", type=int, default=8)
    run.add_argument("--no-sync", action="store_true", help="Do not fast-forward the remote checkout")
    run.add_argument(
        "--isolated-worktree",
        action="store_true",
        help="Run from a per-commit remote worktree, preserving a dirty shared checkout",
    )
    run.add_argument("--state-root", default=".physcog-agent/runs")
    run.set_defaults(func=command_run)

    status = subparsers.add_parser(
        "status", help="Refresh one submitted job and download fresh evidence when complete"
    )
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=command_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "count", 1) < 1:
        raise SystemExit("--count must be positive")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
