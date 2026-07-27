#!/usr/bin/env python3
"""Render the official task49 policy-entry state without changing the scene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_ID = 49
TASK_STEM = (
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket"
)
TASK_PROMPT = "pick up the tomato sauce and put it in the basket"
BDDL_SHA256 = "cce015229a021baf1124562dd5efbc5bc65195926ecce34254c9da5728690816"
BODY_NAMES = (
    "tomato_sauce_1_main",
    "alphabet_soup_1_main",
    "cream_cheese_1_main",
    "ketchup_1_main",
    "basket_1_main",
)
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _body_xyz(env: OffScreenRenderEnv, name: str) -> list[float]:
    body_id = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).tolist()


def main() -> None:
    out = Path("experiments/logs/l3a2_task49_native_preview")
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError(
            f"task49 contract mismatch: name={task.name!r} language={task.language!r}"
        )
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    bddl_hash = _sha(bddl.read_bytes())
    if bddl_hash != BDDL_SHA256:
        raise RuntimeError(
            f"task49 BDDL hash mismatch: expected {BDDL_SHA256}, got {bddl_hash}"
        )
    states = suite.get_task_init_states(TASK_ID)
    if len(states) == 0:
        raise RuntimeError("task49 has no official initial states")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        image = np.ascontiguousarray(
            np.asarray(obs["agentview_image"])[::-1, ::-1]
        )
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected policy image contract: {image.shape} {image.dtype}"
            )
        state = np.asarray(env.sim.get_state().flatten()).copy()
        image_path = out / "task49_native_policy_entry_agentview.png"
        imageio.imwrite(image_path, image)
        report = {
            "verdict": "PASS_L3A2_TASK49_NATIVE_PREVIEW_EXPORTED",
            "scope": "official_native_init_state_0_read_only",
            "task_id_zero_based": TASK_ID,
            "task_stem": task.name,
            "prompt": task.language,
            "bddl_path": str(bddl),
            "bddl_sha256": bddl_hash,
            "official_init_index": 0,
            "policy_entry_dummy_steps": 10,
            "policy_image_shape": list(image.shape),
            "policy_image_orientation": "agentview rotated 180deg as get_libero_image",
            "policy_image_sha256": _sha(image.tobytes()),
            "settled_state_sha256": _sha(state.tobytes()),
            "body_xyz_m": {name: _body_xyz(env, name) for name in BODY_NAMES},
            "scene_or_asset_modified": False,
            "manual_target_visibility_review": "PENDING",
        }
        (out / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report["verdict"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
