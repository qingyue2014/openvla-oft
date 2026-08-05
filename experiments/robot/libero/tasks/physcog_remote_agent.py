#!/usr/bin/env python3
"""Auditable SSH/Slurm runner for remote PhysCog scene validation.

This module intentionally exposes a small phase registry instead of arbitrary
remote shell execution.  Scene-editing agents can therefore run expensive
MuJoCo checks on Superpod while retaining a local ledger of exactly what ran.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PhaseSpec:
    command: tuple[str, ...]
    count_env: str | None = None
    artifacts: tuple[str, ...] = ()
    inputs: tuple[tuple[str, str], ...] = ()


PHASES: Mapping[tuple[str, str], PhaseSpec] = {
    ("l1c1", "repair_eb"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_repair_eb",
        ),
        count_env="NUM_TRIALS",
        inputs=(
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_candidate_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_first_policy_inputs/l1c1_task2_bowl_stack_candidate_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_eb_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_first_policy_inputs/l1c1_task2_bowl_stack_eb_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_ec_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_first_policy_inputs/l1c1_task2_bowl_stack_ec_states.hdf5",
            ),
        ),
        artifacts=(
            "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "experiments/logs/l1c1_eb_repair_build.json",
            "experiments/logs/l1c1_native_preflight.json",
            "experiments/logs/l1c1_native_preflight.md",
            "experiments/logs/l1c1_repaired_eb_first_policy_gate.json",
            "experiments/logs/l1c1_repaired_eb_first_policy_gate.csv",
            "experiments/logs/l1c1_repaired_eb_first_policy_gate.md",
            "review/L1-C1_task/repaired_eb_gate",
        ),
    ),
    ("l1c1", "first_policy_gate"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_first_policy_gate",
        ),
        count_env="NUM_TRIALS",
        inputs=(
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_candidate_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_first_policy_inputs/l1c1_task2_bowl_stack_candidate_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_eb_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_first_policy_inputs/l1c1_task2_bowl_stack_eb_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_ec_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_first_policy_inputs/l1c1_task2_bowl_stack_ec_states.hdf5",
            ),
        ),
        artifacts=(
            "experiments/logs/l1c1_native_preflight.json",
            "experiments/logs/l1c1_native_preflight.md",
            "experiments/logs/l1c1_first_policy_gate.json",
            "experiments/logs/l1c1_first_policy_gate.csv",
            "experiments/logs/l1c1_first_policy_gate.md",
            "review/L1-C1_task/first_policy_gate",
        ),
    ),
    ("l1c1", "init"): PhaseSpec(
        command=(
            "env",
            "RISK_DEPENDENT_XY_OFFSET=0.0125",
            "RISK_DEPENDENT_XY_ANGLE_DEG=135",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_check",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
            "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_states.hdf5",
            "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_ec_states.hdf5",
            "experiments/logs/l1c1_native_preflight.json",
            "experiments/logs/l1c1_native_preflight.md",
        ),
    ),
    ("l1c1", "preview"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_preview",
        ),
        count_env="PREVIEW_NUM_STATES",
        artifacts=(
            "experiments/robot/libero/tasks/l1c1_implicit_stack_preview",
            "experiments/logs/l1c1_native_preflight.json",
            "experiments/logs/l1c1_native_preflight.md",
        ),
    ),
    ("l1c1", "validate_layout"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_validate",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1c1_bowl_stack_calibration.md",
            "experiments/logs/l1c1_bowl_stack_calibration.csv",
            "experiments/logs/l1c1_safe_reference.md",
            "experiments/logs/l1c1_safe_reference.csv",
            "experiments/logs/l1c1_safe_reference_videos",
            "experiments/logs/l1c1_bowl_stack_eb_replay.md",
            "experiments/logs/l1c1_bowl_stack_eb_replay.csv",
            "experiments/logs/l1c1_bowl_stack_ec_replay.md",
            "experiments/logs/l1c1_bowl_stack_ec_replay.csv",
            "experiments/logs/l1c1_native_preflight.json",
            "experiments/logs/l1c1_native_preflight.md",
        ),
    ),
    ("l1c1", "safe_reference"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_safe_reference",
        ),
        count_env="CALIBRATION_NUM_STATES",
        artifacts=(
            "experiments/logs/l1c1_safe_reference.md",
            "experiments/logs/l1c1_safe_reference.csv",
            "experiments/logs/l1c1_safe_reference_trajectories",
            "experiments/logs/l1c1_safe_reference_videos",
            "experiments/logs/l1c1_native_preflight.json",
            "experiments/logs/l1c1_native_preflight.md",
        ),
    ),
    ("l1c1", "smoke"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "BOWL_STACK_EB_STATE_PATH=experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "REPAIRED_EB_STATE_PATH=experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "REPAIRED_EB_BUILD_MANIFEST=experiments/robot/libero/tasks/l1c1_smoke_inputs/l1c1_eb_repair_build.json",
            "REPAIRED_EB_FIRST_POLICY_MANIFEST=experiments/robot/libero/tasks/l1c1_smoke_inputs/l1c1_repaired_eb_first_policy_gate.json",
            "BOWL_STACK_EB_NOTE=L1-C1-hidden-bowl-stack-eb-smoke-repaired",
            "BOWL_STACK_ER_NOTE=L1-C1-hidden-bowl-stack-risk-smoke-frozen",
            "BOWL_STACK_EC_NOTE=L1-C1-hidden-bowl-stack-ec-smoke-frozen",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_smoke",
        ),
        count_env="SMOKE_TRIALS",
        inputs=(
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/initial_layouts/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_candidate_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_ec_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_ec_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/reports/l1c1_eb_repair_build.json",
                "experiments/robot/libero/tasks/l1c1_smoke_inputs/l1c1_eb_repair_build.json",
            ),
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/reports/l1c1_repaired_eb_first_policy_gate.json",
                "experiments/robot/libero/tasks/l1c1_smoke_inputs/l1c1_repaired_eb_first_policy_gate.json",
            ),
        ),
        artifacts=(
            "experiments/logs",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb-smoke-repaired",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk-smoke-frozen",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-ec-smoke-frozen",
        ),
    ),
    ("l1c1", "eligibility"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=none",
            "BOWL_STACK_EB_STATE_PATH=experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "REPAIRED_EB_STATE_PATH=experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "REPAIRED_EB_BUILD_MANIFEST=experiments/robot/libero/tasks/l1c1_eligibility_inputs/l1c1_eb_repair_build.json",
            "REPAIRED_EB_FIRST_POLICY_MANIFEST=experiments/robot/libero/tasks/l1c1_eligibility_inputs/l1c1_repaired_eb_first_policy_gate.json",
            "ELIGIBILITY_EB_NOTE=L1-C1-hidden-bowl-stack-eb-eligibility-repaired",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_eligibility",
        ),
        count_env="NUM_TRIALS",
        inputs=(
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/initial_layouts/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_candidate_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_ec_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_ec_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/reports/l1c1_eb_repair_build.json",
                "experiments/robot/libero/tasks/l1c1_eligibility_inputs/l1c1_eb_repair_build.json",
            ),
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/reports/l1c1_repaired_eb_first_policy_gate.json",
                "experiments/robot/libero/tasks/l1c1_eligibility_inputs/l1c1_repaired_eb_first_policy_gate.json",
            ),
        ),
        artifacts=(
            "experiments/logs",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb-eligibility-repaired",
        ),
    ),
    ("l1c1", "formal"): PhaseSpec(
        command=(
            "env",
            "RENDER_GPU_DEVICE_ID=1",
            "SAVE_VIDEO_MODE=all",
            "BOWL_STACK_EB_STATE_PATH=experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "REPAIRED_EB_STATE_PATH=experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            "REPAIRED_EB_BUILD_MANIFEST=experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_eb_repair_build.json",
            "REPAIRED_EB_FIRST_POLICY_MANIFEST=experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_repaired_eb_first_policy_gate.json",
            "L1C1_SMOKE_MANIFEST=experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_repaired_bundle_smoke.json",
            "L1C1_SMOKE_PHYSICS_MANIFEST=experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_smoke_actual_first_policy_physics.json",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_eval",
        ),
        count_env="NUM_TRIALS",
        inputs=(
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/initial_layouts/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_eb_repaired_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_candidate_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/formal/20260727T091817Z-d6ec632a/initial_layouts/l1c1_task2_bowl_stack_ec_states.hdf5",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_ec_states.hdf5",
            ),
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/reports/l1c1_eb_repair_build.json",
                "experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_eb_repair_build.json",
            ),
            (
                "artifacts/physcog/l1c1/repair_eb/20260804T104908Z-5203bf7a/reports/l1c1_repaired_eb_first_policy_gate.json",
                "experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_repaired_eb_first_policy_gate.json",
            ),
            (
                "review/L1-C1_task/repaired_eb_gate/HUMAN_REVIEW.json",
                "review/L1-C1_task/repaired_eb_gate/HUMAN_REVIEW.json",
            ),
            (
                "artifacts/physcog/l1c1/smoke/20260805T074838Z-0456600f/reports/l1c1_repaired_bundle_smoke.json",
                "experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_repaired_bundle_smoke.json",
            ),
            (
                "artifacts/physcog/l1c1/smoke/20260805T074838Z-0456600f/reports/l1c1_smoke_actual_first_policy_physics.json",
                "experiments/robot/libero/tasks/l1c1_formal_inputs/l1c1_smoke_actual_first_policy_physics.json",
            ),
            (
                "review/L1-C1_task/repaired_eb_smoke/HUMAN_REVIEW.json",
                "review/L1-C1_task/repaired_eb_smoke/HUMAN_REVIEW.json",
            ),
        ),
        artifacts=(
            "experiments/logs",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk",
            "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-ec",
        ),
    ),
    ("l1c1", "recalibrate"): PhaseSpec(
        command=(
            "env",
            "RISK_DEPENDENT_XY_OFFSET=0.0125",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_recalibrate",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1c1_bowl_stack_calibration.md",
            "experiments/logs/l1c1_bowl_stack_calibration.csv",
            "experiments/logs/l1c1_safe_reference.md",
            "experiments/logs/l1c1_safe_reference.csv",
            "experiments/logs/l1c1_bowl_stack_eb_replay.md",
            "experiments/logs/l1c1_bowl_stack_eb_replay.csv",
            "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
        ),
    ),
    ("l1c1", "recalibrate15"): PhaseSpec(
        command=(
            "env",
            "RISK_DEPENDENT_XY_OFFSET=0.015",
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_recalibrate",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1c1_bowl_stack_calibration.md",
            "experiments/logs/l1c1_bowl_stack_calibration.csv",
            "experiments/logs/l1c1_safe_reference.md",
            "experiments/logs/l1c1_safe_reference.csv",
            "experiments/logs/l1c1_bowl_stack_eb_replay.md",
            "experiments/logs/l1c1_bowl_stack_eb_replay.csv",
            "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
        ),
    ),
    ("l1c1", "direction_sweep"): PhaseSpec(
        command=(
            "bash",
            "experiments/robot/libero/tasks/run_l1c1_task2.sh",
            "bowl_stack_direction_sweep",
        ),
        count_env="NUM_TRIALS",
        artifacts=(
            "experiments/logs/l1c1_direction_sweep.md",
            "experiments/logs/l1c1_direction_sweep",
        ),
    ),
    **{
        ("l1c1", f"angle{angle}"): PhaseSpec(
            command=(
                "env",
                "RISK_DEPENDENT_XY_OFFSET=0.0125",
                f"RISK_DEPENDENT_XY_ANGLE_DEG={angle}",
                "bash",
                "experiments/robot/libero/tasks/run_l1c1_task2.sh",
                "bowl_stack_recalibrate",
            ),
            count_env="NUM_TRIALS",
            artifacts=(
                "experiments/logs/l1c1_bowl_stack_calibration.md",
                "experiments/logs/l1c1_bowl_stack_calibration.csv",
                "experiments/logs/l1c1_safe_reference.md",
                "experiments/logs/l1c1_safe_reference.csv",
                "experiments/logs/l1c1_bowl_stack_eb_replay.md",
                "experiments/logs/l1c1_bowl_stack_eb_replay.csv",
                "experiments/robot/libero/tasks/l1c1_task2_bowl_stack_candidate_states.hdf5",
            ),
        )
        for angle in (0, 45, 90, 135, 225, 270, 315)
    },
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

REVIEW_VIDEO_PREFIXES: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("l1c1", "smoke"): (
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb-smoke-repaired/",
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk-smoke-frozen/",
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-ec-smoke-frozen/",
    ),
    ("l1c1", "formal"): (
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-eb/",
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-risk/",
        "rollouts/libero_spatial/L1-C1-hidden-bowl-stack-ec/",
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
    shared_repo: str | None = None,
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
        # Loading NVHPC sets CC=nvc. Triton's tiny CUDA driver extension uses
        # GCC-only flags (including -Wno-psabi), so pin its host compiler to
        # the system GCC while retaining the loaded CUDA runtime/toolchain.
        "export CC=/usr/bin/gcc",
        "export CXX=/usr/bin/g++",
        "export CUDAHOSTCXX=/usr/bin/g++",
        *(
            [f"export PHYSCOG_SHARED_REPO={shlex.quote(shared_repo)}"]
            if shared_repo else []
        ),
        *(
            [f"export PYTHONPATH={shlex.quote(cfg.libero_root)}:${{PYTHONPATH:-}}"]
            if cfg.libero_root else []
        ),
        "export PYTHONUNBUFFERED=1",
        'export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/physcog-numba-${SLURM_JOB_ID:-manual}"',
        'export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/physcog-triton-${SLURM_JOB_ID:-manual}"',
        'mkdir -p "$NUMBA_CACHE_DIR"',
        'mkdir -p "$TRITON_CACHE_DIR"',
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def extract_registered_review_videos(
    text: str, scenario: str, phase: str
) -> list[str]:
    prefixes = REVIEW_VIDEO_PREFIXES.get((scenario, phase), ())
    paths: list[str] = []
    for match in re.finditer(r"`([^`]+\.mp4)`", text):
        candidate = match.group(1)
        parsed = PurePosixPath(candidate)
        if parsed.is_absolute() or ".." in parsed.parts:
            continue
        if not any(candidate.startswith(prefix) for prefix in prefixes):
            continue
        if candidate not in paths:
            paths.append(candidate)
    return paths


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
        cfg, spec, args.count, key[0], key[1], remote_log,
        shared_repo=base_cfg.remote_repo,
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
        if spec.inputs:
            print("# uploaded immutable inputs")
            for source, destination in spec.inputs:
                print(f"{source} -> {cfg.remote_repo.rstrip('/')}/{destination}")
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
    uploaded_inputs: list[dict[str, str]] = []
    for source_value, destination_value in spec.inputs:
        source = Path(source_value)
        if not source.is_file():
            print(f"[physcog-agent] missing registered input: {source}", file=sys.stderr)
            return 1
        remote_destination = f"{cfg.remote_repo.rstrip('/')}/{destination_value}"
        mkdir_result = _remote_capture(
            cfg,
            shell_join(("mkdir", "-p", str(PurePosixPath(remote_destination).parent))),
        )
        if mkdir_result.returncode != 0 or not _transfer_file(cfg, source, remote_destination):
            print(
                f"[physcog-agent] failed to upload registered input: {source}",
                file=sys.stderr,
            )
            return 1
        uploaded_inputs.append(
            {
                "source": str(source),
                "destination": destination_value,
                "sha256": _sha256_file(source),
            }
        )
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
        "uploaded_inputs": uploaded_inputs,
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


def command_fetch_videos(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    ledger_path = run_dir / "run.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    classification = str(ledger.get("classification", ""))
    if classification in {"submitted", "queued", "running", "awaiting_output"}:
        raise SystemExit(f"Run is not terminal: classification={classification}")
    scenario = str(ledger["scenario"])
    phase = str(ledger["phase"])
    if (scenario, phase) not in REVIEW_VIDEO_PREFIXES:
        raise SystemExit(f"Review-video fetching is not registered for {scenario}:{phase}")
    review_index = (
        run_dir / "artifacts" / "experiments" / "logs" / "review_videos.md"
    )
    if not review_index.is_file():
        raise SystemExit(f"Missing downloaded review index: {review_index}")
    paths = extract_registered_review_videos(
        review_index.read_text(encoding="utf-8"), scenario, phase
    )
    if not paths:
        raise SystemExit("No registered review videos found in the review index")
    cfg = _config_from_ledger(ledger)
    fetched: list[str] = []
    missing: list[str] = []
    for path in paths:
        if _fetch_artifact(cfg, path, run_dir / "artifacts"):
            fetched.append(path)
        else:
            missing.append(path)
    ledger["fetched_review_videos"] = fetched
    ledger["missing_review_videos"] = missing
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[physcog-agent] review videos fetched={len(fetched)}/{len(paths)} "
        f"missing={len(missing)}"
    )
    return 0 if not missing else 1


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

    fetch_videos = subparsers.add_parser(
        "fetch-videos",
        help="Download allowlisted MP4s from a terminal run's review index",
    )
    fetch_videos.add_argument("--run-dir", required=True)
    fetch_videos.set_defaults(func=command_fetch_videos)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "count", 1) < 1:
        raise SystemExit("--count must be positive")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
