"""Read-only suite-map and native-asset audit for the task59 L3-A3 candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from probe_l3a3_task57_native import (
    balanced_form,
    geom_contract,
    policy_image,
    pose,
    refresh,
    sha256,
)


EXPECTED_TASK_ID = 59
EXPECTED_PROMPT = "pick up the tomato sauce and put it in the tray"
EXPECTED_STEMS = (
    "cream_cheese",
    "alphabet_soup",
    "tomato_sauce",
    "ketchup",
    "butter",
    "wooden_tray",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a3_task59_native_probe"
    )
    parser.add_argument("--passive_steps", type=int, default=60)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    prompt_matches = [
        task_id
        for task_id in range(suite.n_tasks)
        if suite.get_task(task_id).language == EXPECTED_PROMPT
    ]
    if prompt_matches != [EXPECTED_TASK_ID]:
        raise RuntimeError(
            f"expected exact prompt only at task {EXPECTED_TASK_ID}, "
            f"found {prompt_matches}"
        )

    task = suite.get_task(EXPECTED_TASK_ID)
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
    if task.language != bddl_prompt or task.language != EXPECTED_PROMPT:
        raise RuntimeError(
            "suite, BDDL, and expected task59 prompts are not byte-equivalent: "
            f"{task.language!r}, {bddl_prompt!r}, {EXPECTED_PROMPT!r}"
        )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "native_task59.bddl").write_bytes(bddl_bytes)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.reset()
        state = np.asarray(
            suite.get_task_init_states(EXPECTED_TASK_ID)[0]
        ).copy()
        env.set_init_state(state)
        env.sim.forward()
        restored = np.asarray(env.sim.get_state().flatten()).copy()
        if not np.array_equal(state, restored):
            raise RuntimeError("native task59 serialized state did not restore exactly")
        obs = refresh(env, restored)
        if not np.array_equal(
            restored, np.asarray(env.sim.get_state().flatten())
        ):
            raise RuntimeError("observation refresh changed simulator state")

        png = output / "task59_ep000_policy.png"
        mp4 = output / "task59_ep000_passive.mp4"
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
                name
                for name in all_bodies
                if stem in name and name.endswith("_main")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"task59 expected one native {stem} main body, got {matches}"
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
            raise RuntimeError(f"native task59 asset contract failed: {contracts}")

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
        "verdict": "PASS_L3A3_TASK59_NATIVE_READ_ONLY_AUDIT",
        "scope": "read_only_no_candidate_state_no_vla_policy",
        "task_id": EXPECTED_TASK_ID,
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
        "proposed_native_roles": {
            "S": resolved["tomato_sauce"],
            "A": resolved["alphabet_soup"],
            "B": resolved["butter"],
            "goal": resolved["wooden_tray"],
        },
        "candidate_status": "NOT_CONSTRUCTED",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(
        "# L3-A3 task59 native-only read-only audit\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Verified suite task ID: `{report['task_id']}`\n"
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
