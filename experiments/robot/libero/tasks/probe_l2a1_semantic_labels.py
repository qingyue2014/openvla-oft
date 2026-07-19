#!/usr/bin/env python3
"""Render and verify the L2-A1 geometry-matched semantic-label scenes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.run_physcog_libero_l1_eval import _ensure_libero_importable  # noqa: E402

_ensure_libero_importable()

import experiments.robot.libero.physcog_objects as physcog_objects  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

BDDLS = {
    "pair_a": "experiments/robot/libero/tasks/PHYSCOG_L2A1_pair_a_safe_goal.bddl",
    "pair_b": "experiments/robot/libero/tasks/PHYSCOG_L2A1_pair_b_safe_goal.bddl",
}
NEUTRAL_BODY = "neutral_salad_dressing_1_main"
HAZARD_BODY = "hazard_salad_dressing_1_main"
DUMMY_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def _body_tree(model, root_id: int) -> set[int]:
    ids = {root_id}
    changed = True
    while changed:
        changed = False
        for candidate in range(model.nbody):
            if int(model.body_parentid[candidate]) in ids and candidate not in ids:
                ids.add(candidate)
                changed = True
    return ids


def _collision_signature(env, body_name: str) -> list[tuple]:
    model = env.sim.model
    root_id = model.body_name2id(body_name)
    body_ids = _body_tree(model, root_id)
    signature = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in body_ids or int(model.geom_group[geom_id]) != 0:
            continue
        signature.append((
            int(model.geom_type[geom_id]),
            tuple(np.asarray(model.geom_size[geom_id], dtype=float).round(8)),
            tuple(np.asarray(model.geom_friction[geom_id], dtype=float).round(8)),
            int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id]),
        ))
    return sorted(signature, key=repr)


def _body_mass(env, body_name: str) -> float:
    model = env.sim.model
    ids = _body_tree(model, model.body_name2id(body_name))
    return float(sum(model.body_mass[body_id] for body_id in ids))


def _position(env, body_name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _write_uv_previews(out_dir: Path) -> None:
    source = physcog_objects._libero_package_root() / "assets" / "stable_hope_objects" / "salad_dressing" / "texture_map.png"
    for variant in ("neutral", "hazard"):
        destination = out_dir / f"uv_{variant}.png"
        physcog_objects.render_l2a_salad_dressing_texture(source, destination, variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="experiments/robot/libero/tasks/l2a1_preview")
    parser.add_argument("--report", default="experiments/logs/l2a1_scene_check.md")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=80,
        help="Warm-up steps before measuring stability (the reset pose starts above the floor).",
    )
    parser.add_argument("--stability-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    _write_uv_previews(out_dir)

    rows = []
    failures = []
    for label, bddl in BDDLS.items():
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            hard_reset=False,
            render_gpu_device_id=int(os.environ.get("RENDER_GPU", "-1")),
        )
        env.seed(args.seed)
        obs = env.reset()
        for _ in range(args.settle_steps):
            env.sim.step()
        initial = {body: _position(env, body) for body in (NEUTRAL_BODY, HAZARD_BODY)}
        for _ in range(args.stability_steps):
            env.sim.step()
        obs, _, _, _ = env.step(DUMMY_ACTION)
        final = {body: _position(env, body) for body in (NEUTRAL_BODY, HAZARD_BODY)}
        drift = {body: float(np.linalg.norm(final[body] - initial[body])) for body in final}

        neutral_mass = _body_mass(env, NEUTRAL_BODY)
        hazard_mass = _body_mass(env, HAZARD_BODY)
        same_mass = abs(neutral_mass - hazard_mass) <= 1e-8
        same_collision = _collision_signature(env, NEUTRAL_BODY) == _collision_signature(env, HAZARD_BODY)
        stable = all(value <= 0.02 for value in drift.values())
        if not same_mass:
            failures.append(f"{label}: body mass mismatch")
        if not same_collision:
            failures.append(f"{label}: collision signature mismatch")
        if not stable:
            failures.append(f"{label}: settle drift exceeded 0.02 m")

        image_path = out_dir / f"{label}_agentview.png"
        imageio.imwrite(image_path, obs["agentview_image"])
        rows.append({
            "layout": label,
            "neutral_xyz": final[NEUTRAL_BODY],
            "hazard_xyz": final[HAZARD_BODY],
            "neutral_drift": drift[NEUTRAL_BODY],
            "hazard_drift": drift[HAZARD_BODY],
            "neutral_mass": neutral_mass,
            "hazard_mass": hazard_mass,
            "same_collision": same_collision,
            "image": image_path,
        })
        env.close()

    verdict = "PASS_L2A_GEOMETRY_TEXTURE" if not failures else "FAIL_L2A_GEOMETRY_TEXTURE"
    lines = [
        "# L2-A1 semantic-label scene check",
        "",
        f"Verdict: **{verdict}**",
        "",
        "The check verifies identical compiled mass and collision signatures for the neutral and hazard candidates, physics settling, both counterbalanced layouts, and render generation. Semantic legibility still requires visual review of the fetched previews.",
        "",
        "| Layout | Neutral xyz | Hazard xyz | Neutral drift (m) | Hazard drift (m) | Mass pair (kg) | Collision identical | Preview |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        fmt = lambda values: "[" + ", ".join(f"{value:.4f}" for value in values) + "]"
        lines.append(
            f"| {row['layout']} | {fmt(row['neutral_xyz'])} | {fmt(row['hazard_xyz'])} | "
            f"{row['neutral_drift']:.5f} | {row['hazard_drift']:.5f} | "
            f"{row['neutral_mass']:.6f} / {row['hazard_mass']:.6f} | "
            f"{'yes' if row['same_collision'] else 'NO'} | `{row['image']}` |"
        )
    if failures:
        lines.extend(("", "## Failures", "", *[f"- {failure}" for failure in failures]))
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verdict: {verdict}")
    print(f"Report: {args.report}")
    print(f"Previews: {out_dir}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
