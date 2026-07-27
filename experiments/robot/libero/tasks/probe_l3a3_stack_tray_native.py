"""Read-only native task-63 semantic, body, and policy-view audit for L3-A3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


TASK_ID = 63
EXPECTED_PROMPT = "stack the left bowl on the right bowl and place them in the tray"
RELEVANT_BODIES = (
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "wooden_tray_1_main",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def balanced_form(text: str, keyword: str) -> str:
    match = re.search(rf"\(\s*:{re.escape(keyword)}\b", text)
    if match is None:
        raise ValueError(f"BDDL has no (:{keyword} ...) form")
    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise ValueError(f"unterminated (:{keyword} ...) form")


def body_geoms(env, body_name: str) -> list[dict]:
    model = env.sim.model
    root = int(model.body_name2id(body_name))
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for candidate in range(int(model.nbody)):
            if (
                candidate not in descendants
                and int(model.body_parentid[candidate]) in descendants
            ):
                descendants.add(candidate)
                changed = True
    rows = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) not in descendants:
            continue
        rows.append(
            {
                "id": geom_id,
                "name": model.geom_id2name(geom_id) or f"geom_{geom_id}",
                "group": int(model.geom_group[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "rgba": np.asarray(model.geom_rgba[geom_id], dtype=float).tolist(),
            }
        )
    return rows


def body_contacts(env, first: str, second: str) -> bool:
    first_ids = {row["id"] for row in body_geoms(env, first)}
    second_ids = {row["id"] for row in body_geoms(env, second)}
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & first_ids and pair & second_ids:
            return True
    return False


def refreshed_observation(env, state: np.ndarray):
    if hasattr(env, "regenerate_obs_from_state"):
        return env.regenerate_obs_from_state(state)
    if hasattr(env, "get_observation"):
        return env.get_observation()
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    return env.set_init_state(state)


def policy_image(obs) -> np.ndarray:
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(f"actual policy RGB must be 256x256x3, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def pose(env, body: str) -> dict:
    body_id = int(env.sim.model.body_name2id(body))
    return {
        "xyz": np.asarray(env.sim.data.body_xpos[body_id], dtype=float).tolist(),
        "quat_wxyz": np.asarray(
            env.sim.data.body_xquat[body_id], dtype=float
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a3_stack_tray_native_probe"
    )
    parser.add_argument("--passive_steps", type=int, default=60)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.language != EXPECTED_PROMPT:
        raise RuntimeError(
            f"task63 prompt drift: expected {EXPECTED_PROMPT!r}, got {task.language!r}"
        )
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    bddl_bytes = bddl_path.read_bytes()
    bddl_text = bddl_bytes.decode()
    language_form = balanced_form(bddl_text, "language")
    goal_form = balanced_form(bddl_text, "goal")
    if EXPECTED_PROMPT not in language_form:
        raise RuntimeError("suite prompt and native BDDL (:language ...) differ")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "native_task63.bddl").write_bytes(bddl_bytes)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.reset()
        state = np.asarray(suite.get_task_init_states(TASK_ID)[0])
        env.set_init_state(state)
        env.sim.forward()
        restored = np.asarray(env.sim.get_state().flatten()).copy()
        if not np.array_equal(restored, state):
            raise RuntimeError("native serialized task63 state was not restored exactly")
        obs = refreshed_observation(env, restored)
        if not np.array_equal(np.asarray(env.sim.get_state().flatten()), restored):
            raise RuntimeError("policy observation refresh changed simulator state")
        initial = policy_image(obs)
        imageio.imwrite(output / "task63_ep000_policy.png", initial)
        start_poses = {body: pose(env, body) for body in RELEVANT_BODIES}
        starts = {
            body: np.asarray(start_poses[body]["xyz"]) for body in RELEVANT_BODIES
        }
        initial_contacts = {
            "bowl1_bowl2": body_contacts(
                env, "akita_black_bowl_1_main", "akita_black_bowl_2_main"
            ),
            "bowl1_tray": body_contacts(
                env, "akita_black_bowl_1_main", "wooden_tray_1_main"
            ),
            "bowl2_tray": body_contacts(
                env, "akita_black_bowl_2_main", "wooden_tray_1_main"
            ),
        }
        frames = [initial]
        for _ in range(args.passive_steps):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            frames.append(policy_image(obs))
        imageio.mimsave(
            output / "task63_ep000_passive.mp4",
            frames,
            fps=20,
            macro_block_size=1,
        )
        report = {
            "verdict": "PASS_L3A3_TASK63_NATIVE_READ_ONLY_PROBE",
            "task_id": TASK_ID,
            "prompt": task.language,
            "prompt_sha256": sha256_bytes(task.language.encode()),
            "native_bddl_relative": f"{task.problem_folder}/{task.bddl_file}",
            "native_bddl_sha256": sha256_bytes(bddl_bytes),
            "language_form": language_form,
            "goal_form": goal_form,
            "goal_form_sha256": sha256_bytes(goal_form.encode()),
            "initial_state_sha256": sha256_bytes(state.tobytes()),
            "policy_transform": "agentview_image[::-1, ::-1]",
            "policy_resolution": [256, 256],
            "bodies": {
                body: {
                    "initial_pose": start_poses[body],
                    "final_pose": pose(env, body),
                    "passive_displacement_m": float(
                        np.linalg.norm(np.asarray(pose(env, body)["xyz"]) - starts[body])
                    ),
                    "geoms": body_geoms(env, body),
                }
                for body in RELEVANT_BODIES
            },
            "initial_contacts": initial_contacts,
        }
    finally:
        env.close()
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(
        "# L3-A3 native task63 read-only audit\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Prompt: `{report['prompt']}`\n"
        f"- Prompt SHA-256: `{report['prompt_sha256']}`\n"
        f"- Native BDDL SHA-256: `{report['native_bddl_sha256']}`\n"
        f"- Goal form SHA-256: `{report['goal_form_sha256']}`\n"
        f"- Goal form: `{report['goal_form']}`\n"
        f"- Exact state SHA-256: `{report['initial_state_sha256']}`\n"
        "- Evidence: exact-state actual 256px policy PNG plus 60-step passive MP4.\n"
    )
    print(report["verdict"])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
