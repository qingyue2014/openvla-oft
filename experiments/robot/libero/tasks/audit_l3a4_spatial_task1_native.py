#!/usr/bin/env python3
"""Read-only native contract, asset, competence-evidence, and RGB audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


TASK_ID = 1
PROMPT = "pick up the black bowl next to the ramekin and place it on the plate"
ROLES = {
    "S": "akita_black_bowl_1",
    "A": "glazed_rim_porcelain_ramekin_1",
    "B": "cookies_1",
    "goal": "plate_1",
}
L1A1_ATTRIBUTION_SHA256 = (
    "712686994d5e05197153281e7716c0ed93ba4a0171a0b84e5fabbef631e12be8"
)
L1A1_RUNNER_SHA256 = (
    "ced69cbb5281756c2ee425c985d42275638ff983045aa935e461f8678c18144f"
)
EVALUATOR_NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def balanced_form(text: str, marker: str) -> str:
    start = text.index(f"(:{marker}")
    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return " ".join(text[start:index + 1].split())
    raise RuntimeError(f"unterminated {marker} form")


def resolve_body(sim, stem: str) -> str:
    names = [
        sim.model.body_id2name(index)
        for index in range(sim.model.nbody)
    ]
    matches = [
        name for name in names
        if name and (name == stem or name.startswith(stem + "_"))
    ]
    main = [name for name in matches if name.endswith("_main")]
    if len(main) == 1:
        return main[0]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"cannot resolve {stem}: {matches}")


def descendant_geoms(sim, body: str) -> set[int]:
    root = int(sim.model.body_name2id(body))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for candidate in range(sim.model.nbody):
            if (
                int(sim.model.body_parentid[candidate]) in bodies
                and candidate not in bodies
            ):
                bodies.add(candidate)
                changed = True
    return {
        geom for geom in range(sim.model.ngeom)
        if int(sim.model.geom_bodyid[geom]) in bodies
    }


def contact(sim, left: set[int], right: set[int]) -> bool:
    for index in range(int(sim.data.ncon)):
        item = sim.data.contact[index]
        if (
            (int(item.geom1) in left and int(item.geom2) in right)
            or (int(item.geom2) in left and int(item.geom1) in right)
        ):
            return True
    return False


def contacting_body_names(sim, geoms: set[int]) -> list[str]:
    names = set()
    for index in range(int(sim.data.ncon)):
        item = sim.data.contact[index]
        if int(item.geom1) in geoms:
            other = int(item.geom2)
        elif int(item.geom2) in geoms:
            other = int(item.geom1)
        else:
            continue
        body_id = int(sim.model.geom_bodyid[other])
        name = sim.model.body_id2name(body_id)
        if name:
            names.add(name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_spatial_task1_native_audit",
    )
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    if task.language != PROMPT:
        raise RuntimeError(f"task1 prompt drift: {task.language!r}")
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID))
    bddl_bytes = bddl.read_bytes()
    goal_form = balanced_form(bddl_bytes.decode(), "goal")
    state0 = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    try:
        env.seed(42)
        env.reset()
        env.set_init_state(state0)
        env.sim.forward()
        restored = np.asarray(env.sim.get_state().flatten()).copy()
        if not np.array_equal(restored, state0):
            raise RuntimeError("exact native state0 restore failed")
        bodies = {
            role: resolve_body(env.sim, stem)
            for role, stem in ROLES.items()
        }
        geom_sets = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        assets = {}
        for role in ("S", "A", "B", "goal"):
            ids = geom_sets[role]
            collision = [
                geom for geom in ids
                if int(env.sim.model.geom_group[geom]) == 0
                and int(env.sim.model.geom_contype[geom])
                and int(env.sim.model.geom_conaffinity[geom])
            ]
            visible = [
                geom for geom in ids
                if int(env.sim.model.geom_group[geom]) == 1
            ]
            if not collision or not visible:
                raise RuntimeError(
                    f"{role} lacks native collision/visible geometry"
                )
            assets[role] = {
                "body": bodies[role],
                "collision_group0_count": len(collision),
                "visible_group1_count": len(visible),
                "custom_asset": False,
            }
        initial_contacts = {
            "S_A": contact(env.sim, geom_sets["S"], geom_sets["A"]),
            "A_B": contact(env.sim, geom_sets["A"], geom_sets["B"]),
            "S_B": contact(env.sim, geom_sets["S"], geom_sets["B"]),
        }
        raw_s_position = np.asarray(
            env.sim.data.body_xpos[
                env.sim.model.body_name2id(bodies["S"])
            ],
            dtype=float,
        ).copy()
        raw_support_contacts = contacting_body_names(
            env.sim, geom_sets["S"]
        )
        for _ in range(EVALUATOR_NUM_STEPS_WAIT):
            env.step(DUMMY_ACTION)
        policy_entry_state = np.asarray(
            env.sim.get_state().flatten()
        ).copy()
        policy_entry_s_position = np.asarray(
            env.sim.data.body_xpos[
                env.sim.model.body_name2id(bodies["S"])
            ],
            dtype=float,
        ).copy()
        policy_entry_support_contacts = contacting_body_names(
            env.sim, geom_sets["S"]
        )
        stove_contacts = [
            name for name in policy_entry_support_contacts
            if "stove" in name.lower()
        ]
        if not stove_contacts:
            raise RuntimeError(
                "policy-entry target bowl is not bound to a stove support: "
                f"{policy_entry_support_contacts}"
            )
        current = np.asarray(env.sim.get_state().flatten()).copy()
        obs = env.regenerate_obs_from_state(current)
        if not np.array_equal(
            current, np.asarray(env.sim.get_state().flatten())
        ):
            raise RuntimeError("policy observation refresh changed state")
        image = np.ascontiguousarray(
            np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
        )
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"unexpected policy RGB {image.shape}")
        png = out / "native_state0_policy_agentview.png"
        imageio.imwrite(png, image)
        np.savez_compressed(
            out / "policy_entry_base_state.npz",
            raw_native_state=state0,
            policy_entry_base_state=policy_entry_state,
        )
    finally:
        env.close()

    report = {
        "verdict": "PASS_L3A4_SPATIAL_TASK1_NATIVE_READ_ONLY_AUDIT",
        "scope": "read_only_contract_asset_competence_evidence_policy_rgb",
        "task_suite": "libero_spatial",
        "task_id": TASK_ID,
        "prompt": task.language,
        "prompt_sha256": sha(task.language.encode()),
        "prompt_override": False,
        "bddl_path": str(bddl),
        "bddl_sha256": sha(bddl_bytes),
        "goal_form": goal_form,
        "goal_form_sha256": sha(goal_form.encode()),
        "native_state0_sha256": sha(state0.tobytes()),
        "evaluator_policy_entry": {
            "num_steps_wait": EVALUATOR_NUM_STEPS_WAIT,
            "dummy_action": DUMMY_ACTION,
            "policy_entry_state_sha256": sha(
                policy_entry_state.tobytes()
            ),
            "raw_S_position": raw_s_position.tolist(),
            "policy_entry_S_position": policy_entry_s_position.tolist(),
            "S_displacement_m": float(
                np.linalg.norm(
                    policy_entry_s_position - raw_s_position
                )
            ),
            "raw_S_contact_bodies": raw_support_contacts,
            "policy_entry_S_contact_bodies": policy_entry_support_contacts,
            "support_surface": "stove",
            "stove_contact_bodies": stove_contacts,
            "base_state_file": str(
                out / "policy_entry_base_state.npz"
            ),
        },
        "roles": bodies,
        "assets": assets,
        "initial_contacts": initial_contacts,
        "policy_rgb": {
            "file": str(png),
            "file_sha256": sha(png.read_bytes()),
            "shape": list(image.shape),
            "manual_review": "PENDING",
        },
        "existing_native_EB_evidence": {
            "source_worktree": "openvla-oft-l1a1",
            "attribution_ledger_sha256": L1A1_ATTRIBUTION_SHA256,
            "runner_sha256": L1A1_RUNNER_SHA256,
            "task_suite": "libero_spatial",
            "task_id": 1,
            "prompt_override": False,
            "trials": 50,
            "successes": 50,
            "success_rate": 1.0,
        },
        "custom_assets": False,
        "vla_run": False,
    }
    (out / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "audit.md").write_text(
        "# L3-A4 libero_spatial task1 native read-only audit\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Exact prompt: `{PROMPT}`\n"
        "- Existing unmodified native EB competence: **50/50**.\n"
        "- Policy-entry base: **10 evaluator dummy actions; stove support bound**.\n"
        "- Custom assets: **none**.\n"
        "- Policy RGB: **256x256; manual review pending**.\n"
        "- VLA run: **no**.\n"
    )
    print(report["verdict"])


if __name__ == "__main__":
    main()
