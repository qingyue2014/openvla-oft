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


PHASES: Mapping[tuple[str, str], PhaseSpec] = {
    ("l1b1", "prepare"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b1_native_gripper_safe_reference_videos",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b1_native_gripper",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b1_native_gripper_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b1_native_gripper_er_states.hdf5",
            "experiments/robot/libero/tasks/l1b1_native_gripper_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1b1_native_gripper_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b1_native_gripper",
            "experiments/logs/l1b1_native_gripper_scene_check.md",
            "experiments/logs/l1b1_native_gripper_safe_reference.md",
            "experiments/logs/l1b1_native_gripper_safe_reference.csv",
            "experiments/logs/l1b1_native_gripper_safe_reference_videos",
        ),
    ),
    ("l1b1", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b1_native_gripper_safe_reference_videos",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b1_native_gripper",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b1_native_gripper_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b1_native_gripper",
            "experiments/logs/l1b1_native_gripper_scene_check.md",
            "experiments/logs/l1b1_native_gripper_safe_reference.md",
            "experiments/logs/l1b1_native_gripper_safe_reference.csv",
            "experiments/logs/l1b1_native_gripper_safe_reference_videos",
            "experiments/logs/l1b1_native_gripper_native_replay.md",
            "experiments/logs/l1b1_native_gripper_native_replay.csv",
            "experiments/logs/l1b1_native_gripper_native_replay_videos",
            "experiments/logs/l1b1_native_gripper_eb_rollout_physics.md",
            "experiments/logs/l1b1_native_gripper_er_rollout_physics.md",
            "experiments/logs/l1b1_native_gripper_ec_rollout_physics.md",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-eb",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-er",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-ec",
        ),
    ),
    ("l1b1", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=1",
            "MAX_SUCCESS_VIDEOS=1",
            "MAX_FAILURE_VIDEOS=1",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b1_native_gripper_safe_reference_videos",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b1_native_gripper",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b1_native_gripper_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b1_native_gripper",
            "experiments/logs/l1b1_native_gripper_scene_check.md",
            "experiments/logs/l1b1_native_gripper_safe_reference.md",
            "experiments/logs/l1b1_native_gripper_safe_reference.csv",
            "experiments/logs/l1b1_native_gripper_safe_reference_videos",
            "experiments/logs/l1b1_native_gripper_native_replay.md",
            "experiments/logs/l1b1_native_gripper_native_replay.csv",
            "experiments/logs/l1b1_action_separation_selection.md",
            "experiments/logs/l1b1_er_capture_physics_qualification.md",
            "experiments/logs/l1b1_native_gripper_eb_rollout_physics.md",
            "experiments/logs/l1b1_native_gripper_er_rollout_physics.md",
            "experiments/logs/l1b1_native_gripper_ec_rollout_physics.md",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-eb",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-er",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-ec",
        ),
    ),
    ("l1b1", "ec_calibrate"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=0",
            "SAVE_VIDEO_MODE=none",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b1_native_gripper",
            "ec_calibrate",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b1_native_gripper_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b1_native_gripper",
            "experiments/logs/l1b1_native_gripper_scene_check.md",
            "experiments/logs/l1b1_native_gripper_ec_rollout_physics.md",
            "rollouts/libero_spatial/L1-B1-task6-native-ramekin-capture-lift-v4-ec",
        ),
    ),
    ("l1b1", "ec_static_search"): PhaseSpec(
        command=(
            "python",
            "experiments/robot/libero/tasks/search_l1b1_ec_static_angles.py",
            "--states_per_angle",
            "50",
        ),
        artifacts=(
            "experiments/logs/l1b1_ec_static_angle_search.md",
            "experiments/logs/l1b1_ec_static_angle_search.csv",
        ),
    ),
    ("l1b2", "calibrate"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/calibrate_l1b2_wine_bottle.sh",
        ),
        count_env="CALIBRATION_TRIALS",
        artifacts=(
            "experiments/logs/l1b2_wine_bottle_pose_search.csv",
            "experiments/logs/l1b2_native_held_object_scene_check.md",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b2_native_held_object",
            "experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
        ),
    ),
    ("l1b2", "search"): PhaseSpec(
        command=(
            "python",
            "experiments/robot/libero/tasks/search_l1b_native_replay_positions.py",
            "--family", "l1b2_native_held_object",
            "--eb_trajectories",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
            "--eb_states",
            "experiments/robot/libero/tasks/l1b2_native_held_object_eb_states.hdf5",
            "--task_suite_name", "libero_goal",
            "--task_id", "6",
            "--fractions=0.25,0.30,0.325,0.35,0.375,0.40",
            "--laterals=0.055,0.060,0.065,0.070,0.075",
            "--max_episodes", "50",
            "--min_obstacle_displacement", "0.0",
            "--min_obstacle_tilt_change_deg", "45.0",
            "--out_csv", "experiments/logs/l1b2_wine_bottle_pose_search.csv",
        ),
        artifacts=("experiments/logs/l1b2_wine_bottle_pose_search.csv",),
    ),
    ("l1b2", "path_calibrate"): PhaseSpec(
        command=(
            "python",
            "experiments/robot/libero/tasks/calibrate_l1b2_trajectory_conditioned_states.py",
            "--eb_trajectories",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
            "--fail_on_invalid",
        ),
        artifacts=(
            "experiments/logs/l1b2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b2_trajectory_conditioned_calibration.csv",
            "experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
        ),
    ),
    ("l1b2", "prepare"): PhaseSpec(
        command=(
            "env",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b2_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b2_native_held_object",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b2_native_held_object_scene_check.md",
            "experiments/logs/l1b2_native_held_object_safe_reference.md",
            "experiments/logs/l1b2_native_held_object_safe_reference.csv",
            "experiments/logs/l1b2_safe_reference_videos",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b2_native_held_object",
            "experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
        ),
    ),
    ("l1b2", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b2_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b2_native_held_object",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1b2_native_held_object_scene_check.md",
            "experiments/logs/l1b2_native_held_object_safe_reference.md",
            "experiments/logs/l1b2_native_held_object_safe_reference.csv",
            "experiments/logs/l1b2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b2_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b2_native_held_object_native_replay.md",
            "experiments/logs/l1b2_native_held_object_native_replay_videos",
            "experiments/logs/l1b2_safe_reference_videos",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-eb",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-er",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-ec",
        ),
    ),
    ("l1b2", "pool_smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "L1B2_CALIBRATION_POOL_SIZE=10",
            "L1B2_ER_PHYSICS_QUALIFICATION_SIZE=5",
            "REPLAY_MIN_EPISODES=2",
            "SAVE_VIDEO_MODE=none",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b2_native_held_object",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b2_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b2_er_physics_qualification.md",
            "experiments/logs/l1b2_native_held_object_scene_check.md",
            "experiments/logs/l1b2_native_held_object_safe_reference.md",
            "experiments/logs/l1b2_native_held_object_native_replay.md",
            "experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
        ),
    ),
    ("l1b2", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b2_native_held_object",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b2_native_held_object_scene_check.md",
            "experiments/logs/l1b2_native_held_object_safe_reference.md",
            "experiments/logs/l1b2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b2_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b2_er_physics_qualification.md",
            "experiments/logs/l1b2_native_held_object_native_replay.md",
            "experiments/logs/l1b2_native_held_object_eb_rollout_physics.md",
            "experiments/logs/l1b2_native_held_object_er_rollout_physics.md",
            "experiments/logs/l1b2_native_held_object_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b2_native_held_object",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-eb/trajectories",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-er/trajectories",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-ec/trajectories",
        ),
    ),
    ("l1b2", "ec_repair"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b2_native_ec_repair.sh",
        ),
        artifacts=(
            "experiments/logs/l1b2_ec_control_repair.md",
            "experiments/logs/l1b2_native_held_object_scene_check.md",
            "experiments/logs/l1b2_native_held_object_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b2_native_held_object",
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-ec/trajectories",
        ),
    ),
    ("l1b2", "ec_video"): PhaseSpec(
        command=(
            "env",
            "SAVE_VIDEO_MODE=all",
            "SAVE_TRAJECTORY=False",
            "RENDER_GPU_DEVICE_ID=1",
            "RUN_ID_SUFFIX=release-video",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b2_native_held_object",
            "ec",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "rollouts/libero_goal/L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-ec-release-video",
        ),
    ),
    ("l1b3_task8", "prepare"): PhaseSpec(
        command=(
            "env",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b3_native_arm",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_native_arm_scene_check.md",
            "experiments/logs/l1b3_native_arm_safe_reference.md",
            "experiments/logs/l1b3_native_arm_safe_reference.csv",
            "experiments/logs/l1b3_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_safe_reference_videos",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_native_arm",
            "experiments/robot/libero/tasks/l1b3_native_arm_pairing.json",
        ),
    ),
    ("l1b3_task8", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_safe_reference_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b3_native_arm",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_native_arm_scene_check.md",
            "experiments/logs/l1b3_native_arm_safe_reference.md",
            "experiments/logs/l1b3_native_arm_safe_reference.csv",
            "experiments/logs/l1b3_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_native_arm_native_replay.md",
            "experiments/logs/l1b3_native_arm_native_replay.csv",
            "experiments/logs/l1b3_native_arm_native_replay_videos",
            "experiments/logs/l1b3_safe_reference_videos",
            "experiments/robot/libero/tasks/l1b3_native_arm_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_native_arm",
            "rollouts/libero_goal/L1-B3-goal-bowl-plate-native-wine-link-knockdown-eb",
            "rollouts/libero_goal/L1-B3-goal-bowl-plate-native-wine-link-knockdown-ec",
        ),
    ),
    ("l1b3_task8", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "bash",
            "experiments/robot/libero/tasks/run_l1b_swept.sh",
            "l1b3_native_arm",
            "all",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_native_arm_scene_check.md",
            "experiments/logs/l1b3_native_arm_safe_reference.md",
            "experiments/logs/l1b3_native_arm_safe_reference.csv",
            "experiments/logs/l1b3_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_native_arm_native_replay.md",
            "experiments/logs/l1b3_native_arm_native_replay.csv",
            "experiments/logs/l1b3_native_arm_eb_rollout_physics.md",
            "experiments/logs/l1b3_native_arm_er_rollout_physics.md",
            "experiments/logs/l1b3_native_arm_ec_rollout_physics.md",
            "experiments/robot/libero/tasks/l1b3_native_arm_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_native_arm",
            "rollouts/libero_goal/L1-B3-goal-bowl-plate-native-wine-link-knockdown-eb/trajectories",
            "rollouts/libero_goal/L1-B3-goal-bowl-plate-native-wine-link-knockdown-er/trajectories",
            "rollouts/libero_goal/L1-B3-goal-bowl-plate-native-wine-link-knockdown-ec/trajectories",
        ),
    ),
    ("l1b3_task4", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_candidate_safe_reference_videos",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_candidate_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_task4_candidate",
            "experiments/logs/l1b3_task4_candidate_scene_check.md",
            "experiments/logs/l1b3_task4_candidate_safe_reference.md",
            "experiments/logs/l1b3_task4_candidate_safe_reference.csv",
            "experiments/logs/l1b3_task4_candidate_safe_reference_videos",
            "experiments/logs/l1b3_task4_candidate_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_task4_candidate_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_task4_candidate_native_replay.md",
            "experiments/logs/l1b3_task4_candidate_native_replay.csv",
            "experiments/logs/l1b3_task4_candidate_native_replay_videos",
            "experiments/logs/l1b3_task4_candidate_eb_rollout_physics.md",
            "experiments/logs/l1b3_task4_candidate_er_rollout_physics.md",
            "experiments/logs/l1b3_task4_candidate_ec_rollout_physics.md",
            "rollouts/libero_goal/L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-eb",
            "rollouts/libero_goal/L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-er",
            "rollouts/libero_goal/L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-ec",
        ),
    ),
    ("l1b3_task4", "prepare"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=violation",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_candidate_safe_reference_videos",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_candidate_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_candidate_er_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_candidate_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_candidate_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_task4_candidate",
            "experiments/logs/l1b3_task4_candidate_scene_check.md",
            "experiments/logs/l1b3_task4_candidate_safe_reference.md",
            "experiments/logs/l1b3_task4_candidate_safe_reference.csv",
            "experiments/logs/l1b3_task4_candidate_safe_reference_videos",
            "experiments/logs/l1b3_task4_candidate_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_task4_candidate_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_task4_candidate_native_replay.md",
            "experiments/logs/l1b3_task4_candidate_native_replay.csv",
            "experiments/logs/l1b3_task4_candidate_eb_rollout_physics.md",
        ),
    ),
    ("l1b3_task4", "candidate_full"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=1",
            "MAX_SUCCESS_VIDEOS=1",
            "MAX_FAILURE_VIDEOS=1",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_candidate_safe_reference_videos",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "candidate_full",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_candidate_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_task4_candidate",
            "experiments/logs/l1b3_task4_candidate_scene_check.md",
            "experiments/logs/l1b3_task4_candidate_safe_reference.md",
            "experiments/logs/l1b3_task4_candidate_safe_reference.csv",
            "experiments/logs/l1b3_task4_candidate_safe_reference_videos",
            "experiments/logs/l1b3_task4_candidate_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_task4_candidate_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_task4_candidate_native_replay.md",
            "experiments/logs/l1b3_task4_candidate_native_replay.csv",
            "experiments/logs/l1b3_task4_candidate_native_replay_videos",
            "experiments/logs/l1b3_task4_candidate_eb_rollout_physics.md",
            "experiments/logs/l1b3_task4_candidate_er_rollout_physics.md",
            "experiments/logs/l1b3_task4_candidate_ec_rollout_physics.md",
            "rollouts/libero_goal/L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-eb/trajectories",
            "rollouts/libero_goal/L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-er/trajectories",
            "rollouts/libero_goal/L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-ec/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_y18"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_y18",
            "TASK4_EB_OBSTACLE_OFFSET_XY=0.000,0.180",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=y18",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-y18",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_y18_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-y18",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-y18/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_xm08y12"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_xm08y12",
            "TASK4_EB_OBSTACLE_OFFSET_XY=-0.080,0.120",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=xm08y12",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-xm08y12",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_xm08y12_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-xm08y12",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-xm08y12/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_xm12y18"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_xm12y18",
            "TASK4_EB_OBSTACLE_OFFSET_XY=-0.120,0.180",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=xm12y18",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-xm12y18",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_xm12y18_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-xm12y18",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-xm12y18/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_ym08"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_ym08",
            "TASK4_EB_OBSTACLE_OFFSET_XY=0.000,-0.080",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=ym08",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-ym08",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_ym08_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-ym08",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-ym08/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_ym12"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_ym12",
            "TASK4_EB_OBSTACLE_OFFSET_XY=0.000,-0.120",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=ym12",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-ym12",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_ym12_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-ym12",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-ym12/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_xm08"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_xm08",
            "TASK4_EB_OBSTACLE_OFFSET_XY=-0.080,0.000",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=xm08",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-xm08",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_xm08_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-xm08",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-xm08/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_xm12"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_xm12",
            "TASK4_EB_OBSTACLE_OFFSET_XY=-0.120,0.000",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=xm12",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-xm12",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_xm12_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-xm12",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-xm12/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_xm02"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_xm02",
            "TASK4_EB_OBSTACLE_OFFSET_XY=-0.020,0.000",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=xm02",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-xm02",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_xm02_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-xm02",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-xm02/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "eb_probe_xm04"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "L1B3_TUNING_ONLY=true",
            "PROBE_LABEL=eb_probe_xm04",
            "TASK4_EB_OBSTACLE_OFFSET_XY=-0.040,0.000",
            "L1B3_TASK4_FAMILY=l1b3_task4_outcome_v2",
            "L1B3_TASK4_OUTCOME_BASED=true",
            "L1B3_TASK4_RUN_NOTE_BASE=L1-B3-task4-outcome-v2-eb-anchor-probe",
            "RUN_ID_SUFFIX=xm04",
            "L1B3_TASK4_REVIEW_DIR=review/L1-B3_task/task4-outcome-v2/eb-probe-xm04",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_candidate.sh",
            "eb_probe",
        ),
        count_env="TASK4_SMOKE_POOL_SIZE",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/logs/l1b3_task4_outcome_v2_eb_probe_xm04_pairing.json",
            "review/L1-B3_task/task4-outcome-v2/eb-probe-xm04",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-eb-anchor-probe-eb-xm04/trajectories",
        ),
    ),
    ("l1b3_task4_v2", "smoke"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "smoke",
        ),
        count_env="SMOKE_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_preflight.json",
            "experiments/logs/l1b3_task4_outcome_v2_native_preflight.md",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_er_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_task4_outcome_v2",
            "experiments/logs/l1b3_task4_outcome_v2_scene_check.md",
            "experiments/logs/l1b3_task4_outcome_v2_initial_gate.md",
            "review/L1-B3_task/task4-outcome-v2/L1-B3-task4-outcome-v2_initial_gate_manifest.json",
            "review/L1-B3_task/task4-outcome-v2/HUMAN_REVIEW.json",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_task4_outcome_v2_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_task4_outcome_v2_native_replay.md",
            "experiments/logs/l1b3_task4_outcome_v2_native_replay.csv",
            "experiments/logs/l1b3_task4_outcome_v2_eb_rollout_physics.md",
            "experiments/logs/l1b3_task4_outcome_v2_er_rollout_physics.md",
            "experiments/logs/l1b3_task4_outcome_v2_ec_rollout_physics.md",
            "review/L1-B3_task/task4-outcome-v2",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-bowl-cabinet-native-wine-eb",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-bowl-cabinet-native-wine-er",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-bowl-cabinet-native-wine-ec",
        ),
    ),
    ("l1b3_task4_v2", "safe_reference"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "safe_reference",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference_trajectories",
            "review/L1-B3_task/task4-outcome-v2/safe_reference",
        ),
    ),
    ("l1b3_task4_v2", "safe_reference_high_probe"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_MAX_VIDEOS=1",
            "TASK4_SAFE_REF_REPORT_SUFFIX=high_probe",
            "TASK4_SAFE_REF_TRANSPORT_CLEARANCE=0.16",
            "TASK4_SAFE_REF_GRASP_DIAGONAL=true",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_outcome_v2_safe_reference_high_probe_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "safe_reference",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_task4_outcome_v2_high_probe_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_high_probe_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_high_probe_safe_reference_trajectories",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference_high_probe_videos",
        ),
    ),
    ("l1b3_task4_v2", "safe_reference_top_probe"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_MAX_VIDEOS=1",
            "TASK4_SAFE_REF_REPORT_SUFFIX=top_probe",
            "TASK4_SAFE_REF_TRANSPORT_CLEARANCE=0.04",
            "TASK4_SAFE_REF_GRASP_DIAGONAL=true",
            "TASK4_SAFE_REF_REQUIRE_SUPPORT_CONTACT=true",
            "TASK4_SAFE_REF_PLACE_OFFSET_Y=0.03",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_outcome_v2_safe_reference_top_probe_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "safe_reference",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_task4_outcome_v2_top_probe_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_top_probe_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_top_probe_safe_reference_trajectories",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference_top_probe_videos",
        ),
    ),
    ("l1b3_task4_v2", "safe_reference_release_probe"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_MAX_VIDEOS=1",
            "TASK4_SAFE_REF_REPORT_SUFFIX=release_probe",
            "TASK4_SAFE_REF_TRANSPORT_CLEARANCE=0.02",
            "TASK4_SAFE_REF_PREPLACE_HEIGHT=0.03",
            "TASK4_SAFE_REF_GRASP_DIAGONAL=true",
            "TASK4_SAFE_REF_CONFIRM_SUPPORT_AFTER_RELEASE=true",
            "TASK4_SAFE_REF_MAX_POST_RELEASE_DISPLACEMENT=0.05",
            "TASK4_SAFE_REF_PLACE_OFFSET_Y=0.03",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_outcome_v2_safe_reference_release_probe_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "safe_reference",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_task4_outcome_v2_release_probe_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_release_probe_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_release_probe_safe_reference_trajectories",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference_release_probe_videos",
        ),
    ),
    ("l1b3_task4_v2", "safe_reference_approach_probe"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAFE_REF_MAX_VIDEOS=1",
            "TASK4_SAFE_REF_REPORT_SUFFIX=approach_probe",
            "TASK4_SAFE_REF_APPROACH_HEIGHT=0.24",
            "TASK4_SAFE_REF_GRASP_FRACTIONS=0.80,0.90,1.00,1.10",
            "TASK4_SAFE_REF_GRASP_DIAGONAL=true",
            "TASK4_SAFE_REF_GRASP_AWAY_ORDER=true",
            "TASK4_SAFE_REF_TRANSPORT_CLEARANCE=0.02",
            "TASK4_SAFE_REF_PREPLACE_HEIGHT=0.03",
            "TASK4_SAFE_REF_CONFIRM_SUPPORT_AFTER_RELEASE=true",
            "TASK4_SAFE_REF_MAX_POST_RELEASE_DISPLACEMENT=0.05",
            "TASK4_SAFE_REF_PLACE_OFFSET_Y=0.03",
            "SAFE_REF_VIDEO_DIR=experiments/logs/l1b3_task4_outcome_v2_safe_reference_approach_probe_videos",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "safe_reference",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1b3_task4_outcome_v2_approach_probe_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_approach_probe_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_approach_probe_safe_reference_trajectories",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference_approach_probe_videos",
        ),
    ),
    ("l1b3_task4_v2", "prepare"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=violation",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "prepare",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_preflight.json",
            "experiments/logs/l1b3_task4_outcome_v2_native_preflight.md",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_er_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_task4_outcome_v2",
            "experiments/logs/l1b3_task4_outcome_v2_scene_check.md",
            "experiments/logs/l1b3_task4_outcome_v2_initial_gate.md",
            "review/L1-B3_task/task4-outcome-v2/L1-B3-task4-outcome-v2_initial_gate_manifest.json",
            "review/L1-B3_task/task4-outcome-v2/HUMAN_REVIEW.json",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_task4_outcome_v2_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_task4_outcome_v2_native_replay.md",
            "experiments/logs/l1b3_task4_outcome_v2_native_replay.csv",
            "experiments/logs/l1b3_task4_outcome_v2_eb_rollout_physics.md",
            "review/L1-B3_task/task4-outcome-v2",
        ),
    ),
    ("l1b3_task4_v2", "candidate_full"): PhaseSpec(
        command=(
            "env",
            "PHYSCG_EXECUTION_HOST=superpod",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "MAX_VIOLATION_VIDEOS=10",
            "MAX_SUCCESS_VIDEOS=10",
            "MAX_FAILURE_VIDEOS=10",
            "SAFE_REF_MAX_VIDEOS=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2.sh",
            "candidate_full",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_preflight.json",
            "experiments/logs/l1b3_task4_outcome_v2_native_preflight.md",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_native_source_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_er_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_ec_states.hdf5",
            "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_pairing.json",
            "experiments/robot/libero/tasks/l1b_swept_preview/l1b3_task4_outcome_v2",
            "experiments/logs/l1b3_task4_outcome_v2_scene_check.md",
            "experiments/logs/l1b3_task4_outcome_v2_initial_gate.md",
            "review/L1-B3_task/task4-outcome-v2/L1-B3-task4-outcome-v2_initial_gate_manifest.json",
            "review/L1-B3_task/task4-outcome-v2/HUMAN_REVIEW.json",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.md",
            "experiments/logs/l1b3_task4_outcome_v2_safe_reference.csv",
            "experiments/logs/l1b3_task4_outcome_v2_trajectory_conditioned_calibration.md",
            "experiments/logs/l1b3_task4_outcome_v2_trajectory_conditioned_calibration.csv",
            "experiments/logs/l1b3_task4_outcome_v2_native_replay.md",
            "experiments/logs/l1b3_task4_outcome_v2_native_replay.csv",
            "experiments/logs/l1b3_task4_outcome_v2_eb_rollout_physics.md",
            "experiments/logs/l1b3_task4_outcome_v2_er_rollout_physics.md",
            "experiments/logs/l1b3_task4_outcome_v2_ec_rollout_physics.md",
            "review/L1-B3_task/task4-outcome-v2",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-bowl-cabinet-native-wine-eb/trajectories",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-bowl-cabinet-native-wine-er/trajectories",
            "rollouts/libero_goal/L1-B3-task4-outcome-v2-bowl-cabinet-native-wine-ec/trajectories",
        ),
    ),
    ("l3a1", "check"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh", "all", "prepare"),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a1_risk_check.md",
            "experiments/logs/l3a1_stable_check.md",
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
            "experiments/logs/l3a1_safe_reference.md",
            "experiments/logs/l3a1_safe_reference.csv",
        ),
    ),
    ("l3a1", "smoke"): PhaseSpec(
        command=("bash", "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh", "all", "smoke"),
        count_env="SMOKE_TRIALS",
        artifacts=(
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
            "env", "FAMILIES=l3a1", "SEEDS=42", "SAVE_VIDEO_MODE=none", "bash",
            "experiments/robot/libero/tasks/run_paper_matrix.sh", "full",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l3a1_attribution.md",
            "experiments/logs/experiment_records.csv",
            "experiments/logs/experiment_records.md",
            "experiments/logs/result_tables.md",
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
            "SAVE_VIDEO_MODE=none",
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
    cleanup = [shell_join(("rm", "-rf", artifact)) for artifact in spec.artifacts]
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
        "export CC=/usr/bin/gcc",
        "export CXX=/usr/bin/g++",
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
        (
            "export TRITON_CACHE_DIR="
            f"{shlex.quote(cfg.remote_repo.rstrip('/') + '/.physcog-agent/cache/triton')}"
        ),
        (
            'mkdir -p "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" '
            '"$MPLCONFIGDIR" "$TRITON_CACHE_DIR"'
        ),
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
        "cannot cache function",
    )
    if any(signature in text for signature in validator_signatures):
        return "validator_bug"
    if any(signature in text for signature in infrastructure_signatures):
        return "infrastructure_failure"
    if any(v.startswith(("FAIL", "NEEDS_", "BENCHMARK_INCOMPLETE")) for v in verdicts):
        return "gate_failure"
    if fatal_traceback:
        return "validator_bug"
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
