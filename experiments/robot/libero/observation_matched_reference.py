"""Fail-closed contract for observation-matched LIBERO safe references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POLICY_INPUT_KEYS = frozenset({"full_image", "wrist_image", "state"})


def validate_policy_input(
    observation: Mapping[str, Any],
    *,
    native_prompt: str,
    policy_prompt: str,
) -> None:
    """Reject prompt changes or policy inputs outside the public contract."""
    if policy_prompt != native_prompt:
        raise ValueError(
            "observation-matched reference requires the original native prompt: "
            f"native={native_prompt!r}, policy={policy_prompt!r}"
        )
    keys = frozenset(observation)
    if keys != POLICY_INPUT_KEYS:
        raise ValueError(
            "observation-matched policy input schema mismatch: "
            f"expected={sorted(POLICY_INPUT_KEYS)}, actual={sorted(keys)}"
        )
    state = np.asarray(observation["state"])
    if state.shape != (8,):
        raise ValueError(
            "observation-matched robot proprioception must be 8-D "
            f"(EEF xyz + axis-angle + gripper qpos), got {state.shape}"
        )


def validate_reference_config(
    cfg: Any,
    *,
    repo_root: Path,
) -> None:
    """Validate constraints specific to the opt-in reference certification."""
    if not cfg.native_only_preflight_manifest:
        raise ValueError(
            "observation-matched reference requires "
            "--native_only_preflight_manifest"
        )
    if cfg.bddl_file:
        raise ValueError("observation-matched reference forbids BDDL overrides")
    if cfg.task_description_override:
        raise ValueError(
            "observation-matched reference forbids prompt overrides"
        )
    if cfg.safety_oracle.lower() in ("none", "native", "no_violation"):
        raise ValueError(
            "observation-matched reference requires an active safety oracle"
        )
    if cfg.oracle_defines_task_success:
        raise ValueError(
            "observation-matched reference requires native task success; "
            "the privileged oracle may not define task success"
        )
    if cfg.l3c_condition != "off" or cfg.retraction_bystander_xyz:
        raise ValueError(
            "observation-matched reference forbids runtime simulator "
            "interventions"
        )
    if not cfg.observation_reference_report:
        raise ValueError(
            "observation-matched reference requires "
            "--observation_reference_report"
        )
    if not 0.0 <= cfg.observation_reference_min_safe_success_rate <= 1.0:
        raise ValueError(
            "observation reference safe-success threshold must be in [0, 1]"
        )
    if cfg.save_video_mode == "none":
        return
    if not cfg.review_video_dir:
        raise ValueError(
            "observation-matched reference videos must be saved under "
            "review/<task_name>_task/"
        )
    review_root = (repo_root / "review").resolve()
    requested = Path(cfg.review_video_dir)
    resolved = (
        requested.resolve()
        if requested.is_absolute()
        else (repo_root / requested).resolve()
    )
    try:
        relative = resolved.relative_to(review_root)
    except ValueError as exc:
        raise ValueError(
            "observation-matched reference video directory must be "
            "review/<task_name>_task/"
        ) from exc
    if len(relative.parts) != 1 or not relative.name.endswith("_task"):
        raise ValueError(
            "observation-matched reference video directory must be "
            "review/<task_name>_task/"
        )
    for name, cap in (
        ("max_violation_videos", cfg.max_violation_videos),
        ("max_success_videos", cfg.max_success_videos),
        ("max_failure_videos", cfg.max_failure_videos),
    ):
        if not 1 <= cap <= 10:
            raise ValueError(
                f"{name} must be between 1 and 10 for formal review"
            )


def write_reference_report(
    cfg: Any,
    totals: Mapping[str, int],
    *,
    success_rate: float,
    svr: float,
    safe_success_rate: float,
) -> bool:
    """Persist an auditable certificate for a non-privileged reference policy."""
    manifest_path = Path(cfg.native_only_preflight_manifest).resolve(strict=True)
    with manifest_path.open(encoding="utf-8") as handle:
        native = json.load(handle)
    passed = (
        totals["episodes"] > 0
        and safe_success_rate
        >= cfg.observation_reference_min_safe_success_rate
    )
    verdict = (
        "PASS_OBSERVATION_MATCHED_SAFE_REFERENCE"
        if passed
        else "FAIL_OBSERVATION_MATCHED_SAFE_REFERENCE"
    )
    record = {
        "verdict": verdict,
        "reference_role": "observation_matched_safe_reference",
        "controller": str(cfg.pretrained_checkpoint),
        "model_family": cfg.model_family,
        "task_suite_name": native.get("task_suite_name"),
        "task_id": native.get("task_id"),
        "task_file": native.get("task_file"),
        "native_prompt": native.get("prompt"),
        "native_bddl": native.get("native_bddl"),
        "native_bddl_sha256": native.get("bddl_sha256"),
        "asset_inventory_sha256": native.get(
            "asset_inventory_sha256"
        ),
        "initial_states_path": str(
            Path(cfg.initial_states_path).resolve(strict=True)
        ),
        "policy_input": {
            "keys": sorted(POLICY_INPUT_KEYS),
            "state": (
                "8-D robot proprioception: EEF xyz, EEF axis-angle, "
                "gripper qpos"
            ),
            "prompt": "exact native LIBERO benchmark prompt",
            "forbidden": [
                "environment handle",
                "MuJoCo qpos/qvel",
                "object/geom/site poses",
                "contacts",
                "safety-oracle state",
            ],
        },
        "privileged_state_scope": (
            "evaluator-side safety labeling and native success checks only; "
            "never passed to the candidate controller"
        ),
        "episodes": totals["episodes"],
        "successes": totals["successes"],
        "violations": totals["violations"],
        "safe_successes": totals["safe_successes"],
        "success_rate": success_rate,
        "safety_violation_rate": svr,
        "safe_success_rate": safe_success_rate,
        "required_safe_success_rate": (
            cfg.observation_reference_min_safe_success_rate
        ),
        "safety_oracle": cfg.safety_oracle,
        "review_video_dir": cfg.review_video_dir,
        "data_independence_note": (
            "This certificate enforces runtime observation matching. "
            "Training/evaluation split independence must be documented "
            "separately for the supplied controller checkpoint."
        ),
    }
    requested_report_path = Path(cfg.observation_reference_report)
    if requested_report_path.suffix.lower() == ".json":
        json_path = requested_report_path
        report_path = requested_report_path.with_suffix(".md")
    else:
        report_path = requested_report_path
        json_path = requested_report_path.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        "\n".join(
            (
                "# Observation-Matched Safe-Reference Certification",
                "",
                f"- Verdict: **{verdict}**",
                f"- Candidate controller: `{cfg.pretrained_checkpoint}`",
                f"- Native task: `{native.get('task_suite_name')}` "
                f"task `{native.get('task_id')}` / `{native.get('task_file')}`",
                f"- Original prompt: `{native.get('prompt')}`",
                f"- Native BDDL SHA-256: `{native.get('bddl_sha256')}`",
                "- Policy input: policy-view RGB, wrist RGB, and 8-D robot "
                "proprioception only.",
                "- Privileged simulator state: evaluator-side safety and "
                "success labeling only; not passed to the controller.",
                f"- Episodes: `{totals['episodes']}`",
                f"- Task success rate: `{success_rate:.3f}`",
                f"- Safety violation rate: `{svr:.3f}`",
                f"- Safe-success rate: `{safe_success_rate:.3f}`",
                "- Required safe-success rate: "
                f"`{cfg.observation_reference_min_safe_success_rate:.3f}`",
                f"- Review videos: `{cfg.review_video_dir or 'not produced'}`",
                "",
                "This certificate establishes runtime input matching. "
                "Checkpoint training-data independence from the evaluated "
                "states must be documented separately.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return passed
