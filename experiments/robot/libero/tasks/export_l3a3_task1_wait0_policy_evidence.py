"""Export two exact wait0 task1 policy views for independent visibility review.

This script does not construct or evaluate any leaning-chain candidate. It
only captures the official raw state after evaluator wait10, restores that
settled state twice with the required future evaluator num_steps_wait=0
contract, and exports policy RGB plus aligned instance-segmentation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.generate_l3a3_task1_leaning_chain_candidate import (
    A,
    B,
    BDDL_LANGUAGE,
    GOAL_SHA256,
    NATIVE_BDDL_SHA256,
    PLATE,
    POLICY_ENTRY_DUMMY_ACTION,
    POLICY_ENTRY_WAIT_STEPS,
    POLICY_PROMPT,
    POLICY_PROMPT_SHA256,
    S,
    STOVE,
    SUITE,
    TABLE,
    TASK_ID,
    capture_policy_entry_base,
    contact_rows,
    descendants_geoms,
    policy_image,
    refresh,
    sha256,
)
from experiments.robot.libero.tasks.probe_l3a3_task57_native import balanced_form


ROLES = {"S": S, "A": A, "B": B, "goal": PLATE}
PALETTE_RGB = {
    "S": [230, 57, 70],
    "A": [46, 196, 94],
    "B": [55, 125, 230],
    "goal": [245, 194, 66],
}
PENDING = "PENDING_INDEPENDENT_MANUAL_POLICY_VIEW_REVIEW"


def render_policy_segmentation_ids(env) -> np.ndarray:
    segmentation = env.sim.render(
        width=256,
        height=256,
        camera_name="agentview",
        segmentation=True,
    )
    if segmentation is None:
        raise RuntimeError("agentview segmentation render returned None")
    segmentation = np.asarray(segmentation)
    if segmentation.ndim == 3:
        segmentation = segmentation[..., -1]
    if segmentation.shape != (256, 256):
        raise RuntimeError(
            f"expected 256x256 segmentation, got {segmentation.shape}"
        )
    return np.ascontiguousarray(segmentation[::-1, ::-1])


def role_segmentation(env, segmentation: np.ndarray) -> tuple[dict, np.ndarray]:
    combined = np.zeros((256, 256, 3), dtype=np.uint8)
    rows = {}
    for role, body in ROLES.items():
        geom_ids = descendants_geoms(env.sim, body)
        mask = np.isin(segmentation, geom_ids)
        ys, xs = np.nonzero(mask)
        pixel_count = int(mask.sum())
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if pixel_count
            else None
        )
        touches_boundary = bool(
            pixel_count
            and (
                xs.min() == 0
                or ys.min() == 0
                or xs.max() == 255
                or ys.max() == 255
            )
        )
        combined[mask] = np.asarray(PALETTE_RGB[role], dtype=np.uint8)
        rows[role] = {
            "body": body,
            "compiled_geom_ids": geom_ids,
            "visible_pixels": pixel_count,
            "policy_bbox_xyxy": bbox,
            "touches_policy_image_boundary": touches_boundary,
            "zero_pixel_hard_stop": pixel_count == 0,
        }
    return rows, combined


def capture_wait0(env, base: np.ndarray, output: Path, repeat: int) -> dict:
    env.reset()
    env.set_init_state(base)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, base):
        raise RuntimeError(f"repeat {repeat}: exact wait0 state restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(
        restored, np.asarray(env.sim.get_state().flatten())
    ):
        raise RuntimeError(f"repeat {repeat}: immediate refresh changed state")
    rgb = policy_image(obs)
    segmentation = render_policy_segmentation_ids(env)
    if not np.array_equal(
        restored, np.asarray(env.sim.get_state().flatten())
    ):
        raise RuntimeError(f"repeat {repeat}: segmentation render changed state")
    roles, mask_rgb = role_segmentation(env, segmentation)

    rgb_path = output / f"wait0_repeat{repeat}_policy.png"
    segmentation_path = output / f"wait0_repeat{repeat}_segmentation_ids.npy"
    mask_path = output / f"wait0_repeat{repeat}_role_mask.png"
    imageio.imwrite(rgb_path, rgb)
    np.save(segmentation_path, segmentation, allow_pickle=False)
    imageio.imwrite(mask_path, mask_rgb)
    return {
        "repeat": repeat,
        "exact_state_restore": True,
        "immediate_refresh_state_unchanged": True,
        "segmentation_render_state_unchanged": True,
        "policy_png": rgb_path.name,
        "policy_png_sha256": sha256(rgb_path.read_bytes()),
        "segmentation_ids_npy": segmentation_path.name,
        "segmentation_ids_npy_sha256": sha256(
            segmentation_path.read_bytes()
        ),
        "segmentation_id_plane_sha256": sha256(
            segmentation.astype(np.int64, copy=False).tobytes()
        ),
        "role_mask_png": mask_path.name,
        "role_mask_png_sha256": sha256(mask_path.read_bytes()),
        "roles": roles,
        "all_required_roles_have_visible_pixels": all(
            row["visible_pixels"] > 0 for row in roles.values()
        ),
        "S_table_contact_count": len(
            contact_rows(env.sim, S, TABLE, right_exact_body=True)
        ),
        "S_stove_contact_count": len(
            contact_rows(env.sim, S, STOVE, right_exact_body=True)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a3_task1_wait0_policy_evidence",
    )
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != POLICY_PROMPT:
        raise RuntimeError(f"task1 suite prompt drift: {task.language!r}")
    if sha256(task.language.encode()) != POLICY_PROMPT_SHA256:
        raise RuntimeError("task1 suite prompt hash drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError("task1 native BDDL byte drift")
    bddl_text = bddl_bytes.decode()
    if balanced_form(bddl_text, "language")[len("(:language ") : -1] != BDDL_LANGUAGE:
        raise RuntimeError("task1 BDDL language drift")
    if sha256(balanced_form(bddl_text, "goal").encode()) != GOAL_SHA256:
        raise RuntimeError("task1 native goal drift")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    try:
        raw_source = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        base, _, entry_capture = capture_policy_entry_base(env, raw_source)
        captures = [
            capture_wait0(env, base, output, repeat)
            for repeat in (1, 2)
        ]
    finally:
        env.close()

    zero_pixel_roles = sorted(
        {
            role
            for capture in captures
            for role, row in capture["roles"].items()
            if row["zero_pixel_hard_stop"]
        }
    )
    evidence = {
        "schema": "l3a3_task1_wait0_policy_evidence_v1",
        "status": (
            "FAIL_ZERO_VISIBLE_PIXELS"
            if zero_pixel_roles
            else PENDING
        ),
        "scope": "export_only_no_candidate_construction_no_physical_grid_no_vla",
        "suite": SUITE,
        "task_id": TASK_ID,
        "policy_prompt": POLICY_PROMPT,
        "bddl_language_not_policy_prompt": BDDL_LANGUAGE,
        "prompt_override": None,
        "policy_camera": "agentview",
        "policy_transform": "agentview_image[::-1, ::-1]",
        "resolution": [256, 256],
        "base_capture": {
            "raw_source_wait_steps": POLICY_ENTRY_WAIT_STEPS,
            "dummy_action": POLICY_ENTRY_DUMMY_ACTION.tolist(),
            "policy_entry_state_sha256": entry_capture[
                "policy_entry_state_sha256"
            ],
            "required_evaluator_num_steps_wait": 0,
            "S_policy_entry_support_body": entry_capture[
                "S_policy_entry_support_body"
            ],
        },
        "required_roles": ROLES,
        "segmentation_palette_rgb": PALETTE_RGB,
        "captures": captures,
        "automated_zero_pixel_gate": {
            "passed": not zero_pixel_roles,
            "zero_pixel_roles": zero_pixel_roles,
        },
        "manual_review": {
            "status": "NOT_YET_REVIEWED",
            "reviewer_must_be_independent": True,
            "per_role_required_fields": [
                "complete",
                "recognizable",
                "unoccluded",
                "inside_frame",
                "visible_at_policy_entry",
            ],
            "review_must_cover_both_policy_pngs": True,
        },
        "repeat_render_psnr_ssim": "DIAGNOSTIC_ONLY_NOT_A_VISIBILITY_GATE",
        "physical_grid_status": "NOT_RUN",
        "vla_status": "NOT_RUN",
    }
    evidence_path = output / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    print(evidence["status"])
    if zero_pixel_roles:
        raise RuntimeError(
            f"required roles absent from policy segmentation: {zero_pixel_roles}"
        )


if __name__ == "__main__":
    main()
