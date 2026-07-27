"""Read-only suite-map and native-asset audit for LIBERO-90 task ID 57."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


TASK_ID = 57
EXPECTED_STEMS = (
    "cream_cheese",
    "alphabet_soup",
    "tomato_sauce",
    "ketchup",
    "butter",
    "wooden_tray",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def balanced_form(text: str, keyword: str) -> str:
    match = re.search(rf"\(\s*:{re.escape(keyword)}\b", text)
    if match is None:
        raise ValueError(f"missing (:{keyword} ...) form")
    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise ValueError(f"unterminated (:{keyword} ...) form")


def policy_image(obs) -> np.ndarray:
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 policy RGB, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def refresh(env, state: np.ndarray):
    if hasattr(env, "regenerate_obs_from_state"):
        return env.regenerate_obs_from_state(state)
    if hasattr(env, "get_observation"):
        return env.get_observation()
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    return env.set_init_state(state)


def descendants(env, body_name: str) -> set[int]:
    model = env.sim.model
    root = int(model.body_name2id(body_name))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if (
                body_id not in bodies
                and int(model.body_parentid[body_id]) in bodies
            ):
                bodies.add(body_id)
                changed = True
    return bodies


def geom_contract(env, body_name: str) -> dict:
    model = env.sim.model
    body_ids = descendants(env, body_name)
    rows = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            continue
        rows.append(
            {
                "id": geom_id,
                "name": model.geom_id2name(geom_id) or f"geom_{geom_id}",
                "group": int(model.geom_group[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "alpha": float(model.geom_rgba[geom_id][3]),
            }
        )
    collision = [
        row
        for row in rows
        if row["group"] == 0 and row["contype"] != 0 and row["conaffinity"] != 0
    ]
    visual = [
        row for row in rows if row["group"] == 1 and row["alpha"] >= 0.95
    ]
    return {
        "passed": bool(collision and visual),
        "collision_group0_count": len(collision),
        "opaque_visual_group1_count": len(visual),
        "geoms": rows,
    }


def pose(env, body_name: str) -> dict:
    body_id = int(env.sim.model.body_name2id(body_name))
    return {
        "xyz": np.asarray(env.sim.data.body_xpos[body_id], dtype=float).tolist(),
        "quat_wxyz": np.asarray(
            env.sim.data.body_xquat[body_id], dtype=float
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a3_task57_native_probe"
    )
    parser.add_argument("--passive_steps", type=int, default=60)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    bddl_bytes = bddl_path.read_bytes()
    bddl_text = bddl_bytes.decode()
    language_form = balanced_form(bddl_text, "language")
    goal_form = balanced_form(bddl_text, "goal")
    bddl_prompt = re.sub(
        r"^\(\s*:language\s+|\)$", "", language_form.strip()
    ).strip()
    if task.language != bddl_prompt:
        raise RuntimeError(
            f"suite task57 language differs from BDDL: "
            f"{task.language!r} != {bddl_prompt!r}"
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "native_task57.bddl").write_bytes(bddl_bytes)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.reset()
        state = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        env.set_init_state(state)
        env.sim.forward()
        restored = np.asarray(env.sim.get_state().flatten()).copy()
        if not np.array_equal(state, restored):
            raise RuntimeError("native task57 serialized state did not restore exactly")
        obs = refresh(env, restored)
        if not np.array_equal(restored, np.asarray(env.sim.get_state().flatten())):
            raise RuntimeError("observation refresh changed simulator state")
        png = output / "task57_ep000_policy.png"
        mp4 = output / "task57_ep000_passive.mp4"
        initial = policy_image(obs)
        imageio.imwrite(png, initial)

        all_bodies = sorted(
            env.sim.model.body_id2name(body_id)
            for body_id in range(int(env.sim.model.nbody))
            if env.sim.model.body_id2name(body_id)
        )
        resolved = {}
        for stem in EXPECTED_STEMS:
            matches = [
                name for name in all_bodies if stem in name and name.endswith("_main")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"task57 expected one native {stem} main body, got {matches}"
                )
            resolved[stem] = matches[0]
        initial_poses = {
            stem: pose(env, body_name) for stem, body_name in resolved.items()
        }
        contracts = {
            stem: geom_contract(env, body_name)
            for stem, body_name in resolved.items()
        }
        if not all(row["passed"] for row in contracts.values()):
            raise RuntimeError(f"native task57 asset contract failed: {contracts}")

        frames = [initial]
        for _ in range(args.passive_steps):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            frames.append(policy_image(obs))
        imageio.mimsave(mp4, frames, fps=20, macro_block_size=1)
        final_poses = {
            stem: pose(env, body_name) for stem, body_name in resolved.items()
        }
        passive_displacement = {
            stem: float(
                np.linalg.norm(
                    np.asarray(final_poses[stem]["xyz"])
                    - np.asarray(initial_poses[stem]["xyz"])
                )
            )
            for stem in resolved
        }
    finally:
        env.close()

    report = {
        "verdict": "PASS_L3A3_TASK57_NATIVE_READ_ONLY_AUDIT",
        "scope": "read_only_no_candidate_state_no_vla_policy",
        "task_id": TASK_ID,
        "suite_prompt": task.language,
        "prompt_sha256": sha256(task.language.encode()),
        "language_form": language_form,
        "goal_form": goal_form,
        "goal_form_sha256": sha256(goal_form.encode()),
        "native_bddl_relative": f"{task.problem_folder}/{task.bddl_file}",
        "native_bddl_sha256": sha256(bddl_bytes),
        "initial_state_sha256": sha256(state.tobytes()),
        "resolved_bodies": resolved,
        "native_asset_contracts": contracts,
        "initial_poses": initial_poses,
        "passive_final_poses": final_poses,
        "passive_displacement_m": passive_displacement,
        "policy_camera_evidence": {
            "transform": "agentview_image[::-1, ::-1]",
            "resolution": [256, 256],
            "png": png.name,
            "png_sha256": sha256(png.read_bytes()),
            "mp4": mp4.name,
            "mp4_sha256": sha256(mp4.read_bytes()),
        },
        "candidate_status": "NOT_CONSTRUCTED",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(
        "# L3-A3 task57 native-only read-only audit\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Suite prompt: `{report['suite_prompt']}`\n"
        f"- Prompt SHA-256: `{report['prompt_sha256']}`\n"
        f"- Native BDDL SHA-256: `{report['native_bddl_sha256']}`\n"
        f"- Goal SHA-256: `{report['goal_form_sha256']}`\n"
        f"- Goal: `{report['goal_form']}`\n"
        "- Candidate state: not constructed; no VLA policy executed.\n"
    )
    print(report["verdict"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
