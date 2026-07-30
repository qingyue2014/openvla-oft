#!/usr/bin/env python3
"""Record L2-A1 native body/geom identities and burner distances.

This is a read-only calibration diagnostic. It does not change fixture state,
teleport objects, run a policy, or produce publication evidence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog import oracles as OR  # noqa: E402
from experiments.robot.robocasa.physcog.preflight import (  # noqa: E402
    initial_max_penetration,
)
from experiments.robot.robocasa.scripts.run_condition import (  # noqa: E402
    make_env,
    run_native_preflight,
)


def _name(model, kind: str, index: int) -> str:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        return str(method(index))
    return str(model.id2name(index, kind))


def _id(model, kind: str, name: str) -> int:
    method = getattr(model, f"{kind}_name2id", None)
    if callable(method):
        return int(method(name))
    return int(model.name2id(name, kind))


def condition_geometry(scene_id: str, condition: str, seed: int) -> dict:
    env = make_env(scene_id, condition, seed, render=False)
    try:
        env.reset()
        hazard = "distr_counter_0"
        hazard_model = env.objects[hazard]
        body_id = int(env.obj_body_id[hazard])
        body_pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
        hazard_geoms = sorted(OR.obj_geoms(env, hazard))
        burner_rows = []
        for location, site in sorted(env.stove.burner_sites.items()):
            if site is None:
                continue
            site_name = site.get("name")
            if not site_name:
                continue
            site_id = _id(env.sim.model, "site", site_name)
            site_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
            burner_rows.append(
                {
                    "location": str(location),
                    "site_name": str(site_name),
                    "site_id": site_id,
                    "world_xyz_m": site_pos.tolist(),
                    "hazard_xy_distance_m": float(
                        np.linalg.norm(body_pos[:2] - site_pos[:2])
                    ),
                }
            )

        contacts = []
        for index in range(int(env.sim.data.ncon)):
            contact = env.sim.data.contact[index]
            geom1 = _name(env.sim.model, "geom", int(contact.geom1))
            geom2 = _name(env.sim.model, "geom", int(contact.geom2))
            if geom1 not in hazard_geoms and geom2 not in hazard_geoms:
                continue
            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "distance_m": float(contact.dist),
                    "penetration_m": max(0.0, -float(contact.dist)),
                }
            )

        distances = [row["hazard_xy_distance_m"] for row in burner_rows]
        stove_geoms = sorted(env.stove.contact_geoms)
        geom_gap = OR.min_signed_geom_distance(
            env,
            hazard_geoms,
            stove_geoms,
            distmax_m=1.0,
        )
        return {
            "condition": condition,
            "native_prompt": env.native_lang,
            "hazard_object": hazard,
            "hazard_body_id": body_id,
            "hazard_body_name": _name(env.sim.model, "body", body_id),
            "hazard_body_world_xyz_m": body_pos.tolist(),
            "hazard_geom_names": hazard_geoms,
            "stove_geom_names": stove_geoms,
            "hazard_horizontal_radius_m": float(hazard_model.horizontal_radius),
            "hazard_bottom_offset_m": np.asarray(
                hazard_model.bottom_offset, dtype=float
            ).tolist(),
            "hazard_top_offset_m": np.asarray(
                hazard_model.top_offset, dtype=float
            ).tolist(),
            "burner_sites": burner_rows,
            "min_hazard_to_burner_xy_distance_m": min(distances, default=None),
            "hazard_to_stove_geom_min_signed_distance_m": geom_gap,
            "geom_distance_api": "mujoco.mj_geomDistance",
            "burner_center_exclusion_threshold_m": getattr(
                env, "physcog_burner_exclusion_m", None
            ),
            "stove_surface_clearance_threshold_m": getattr(
                env, "physcog_stove_surface_clearance_m", None
            ),
            "knob_states": {
                str(key): float(value)
                for key, value in env.stove.get_knobs_state(env=env).items()
            },
            "initial_hazard_contacts": contacts,
            "initial_max_penetration_m": initial_max_penetration(env),
            "safety_violated": bool(env.safety_violated),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="L2-A1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.scene != "L2-A1":
        raise SystemExit("this diagnostic is registered only for L2-A1")

    native = run_native_preflight(args.scene, args.seed)
    rows = [
        condition_geometry(args.scene, condition, args.seed)
        for condition in ("Eb", "Er", "Ec")
    ]
    payload = {
        "valid_native_preflight": True,
        "scene_id": args.scene,
        "seed": args.seed,
        "native_preflight_sha256": native["preflight_sha256"],
        "native_prompt": native["native_prompt"],
        "conditions": rows,
    }
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"geometry diagnostic -> {output}")


if __name__ == "__main__":
    main()
