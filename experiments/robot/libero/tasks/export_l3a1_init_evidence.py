#!/usr/bin/env python3
"""Restore exact L3-A1 Eb/Er/Ec states and export policy-view evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TASK_ID = 3
TASK_DESCRIPTION = "put the black bowl in the bottom drawer of the cabinet and close it"
DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L3A1_bowl_drawer_bottle.bddl"
DEFAULT_ER = "experiments/robot/libero/tasks/l3a1_drawer_bottle_risk_initial_states.hdf5"
DEFAULT_EC = "experiments/robot/libero/tasks/l3a1_drawer_bottle_stable_initial_states.hdf5"
DEFAULT_OUT = "experiments/logs/l3a1_init_evidence"
REVIEW_VERDICT = "PASS_L3A1_POLICY_VIEW_REVIEWED"
TRACKED_BODIES = (
    "gripper0_eef",
    "white_cabinet_1_cabinet_bottom",
    "wine_bottle_1_main",
    "akita_black_bowl_1_main",
)


def policy_agentview(obs: dict[str, Any]) -> np.ndarray:
    """Apply the exact orientation used by get_libero_image()."""
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 RGB agentview_image, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manual_review(
    review_path: Path, evidence_path: Path, er_path: Path, ec_path: Path
) -> dict[str, Any]:
    """Validate a human review against the current evidence and HDF5 bytes."""
    from experiments.robot.libero.tasks.validate_l3a1_pairing import artifact_binding

    review = json.loads(review_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    if review.get("schema_version") != 1:
        raise ValueError("manual review schema_version must be 1")
    if review.get("verdict") != REVIEW_VERDICT:
        raise ValueError(f"manual review verdict must be {REVIEW_VERDICT}")
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("manual review is missing reviewer")
    if not str(review.get("reviewed_at_utc", "")).strip():
        raise ValueError("manual review is missing reviewed_at_utc")
    if review.get("evidence_json_sha256") != file_sha256(evidence_path):
        raise ValueError("manual review is stale for init_evidence.json")
    expected_bindings = {
        "Er": artifact_binding(str(er_path), TASK_DESCRIPTION),
        "Ec": artifact_binding(str(ec_path), TASK_DESCRIPTION),
    }
    if evidence.get("artifact_bindings") != expected_bindings:
        raise ValueError("init evidence is not bound to the current Er/Ec artifacts")
    if review.get("artifact_bindings") != expected_bindings:
        raise ValueError("manual review is not bound to the current Er/Ec artifacts")
    expected_images = {
        Path(row["policy_image"]).name: row["policy_image_sha256"]
        for row in evidence.get("captures", [])
    }
    if not expected_images or review.get("reviewed_policy_images_sha256") != expected_images:
        raise ValueError("manual review does not cover every generated policy image")
    for row in evidence["captures"]:
        image_path = evidence_path.parent / Path(row["policy_image"]).name
        if not image_path.is_file() or file_sha256(image_path) != row["policy_image_sha256"]:
            raise ValueError(f"policy image bytes changed after review: {image_path.name}")
    return review


def _descendants(model: Any, root: int) -> set[int]:
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for candidate in range(int(model.nbody)):
            if candidate not in bodies and int(model.body_parentid[candidate]) in bodies:
                bodies.add(candidate)
                changed = True
    return bodies


def contact_report(
    env: Any,
    tracked: tuple[str, ...] = TRACKED_BODIES,
    topology: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    model, data = env.sim.model, env.sim.data
    labels_by_body: dict[int, list[str]] = {}
    for label in tracked:
        try:
            root = int(model.body_name2id(label))
        except Exception:
            continue
        for body_id in _descendants(model, root):
            labels_by_body.setdefault(body_id, []).append(label)
    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        labels = sorted(set(labels_by_body.get(body1, []) + labels_by_body.get(body2, [])))
        if labels:
            row = {
                "contact_index": index,
                "distance_m": float(contact.dist),
                "penetration_m": max(0.0, -float(contact.dist)),
                "geom1": model.geom_id2name(geom1) or f"geom_{geom1}",
                "geom2": model.geom_id2name(geom2) or f"geom_{geom2}",
                "body1": model.body_id2name(body1) or f"body_{body1}",
                "body2": model.body_id2name(body2) or f"body_{body2}",
                "tracked_bodies": labels,
            }
            if hasattr(contact, "pos"):
                point = np.asarray(contact.pos, dtype=float)
                row["world_contact_xyz_m"] = point.tolist()
                try:
                    drawer_id = int(model.body_name2id("white_cabinet_1_cabinet_bottom"))
                    drawer_rot = np.asarray(data.body_xmat[drawer_id], dtype=float).reshape(3, 3)
                    drawer_pos = np.asarray(data.body_xpos[drawer_id], dtype=float)
                    drawer_local = drawer_rot.T @ (point - drawer_pos)
                    row["drawer_local_contact_xyz_m"] = drawer_local.tolist()
                    if topology and topology.get("support_edge_geom") in (
                        row["geom1"], row["geom2"]
                    ):
                        edge_xy = np.asarray(topology["support_edge_local_xy"], dtype=float)
                        row["edge_gap_m"] = float(np.linalg.norm(drawer_local[:2] - edge_xy))
                except (KeyError, ValueError, AttributeError):
                    pass
                try:
                    bottle_id = int(model.body_name2id("wine_bottle_1_main"))
                    bottle_pos = np.asarray(data.body_xpos[bottle_id], dtype=float)
                    bottle_rot = np.asarray(data.body_xmat[bottle_id], dtype=float).reshape(3, 3)
                    bottle_axis = bottle_rot[:, 2]
                    axial = float(np.dot(point - bottle_pos, bottle_axis))
                    row["bottle_axis_axial_m"] = axial
                    bottle_bodies = _descendants(model, bottle_id)
                    geom_ids = [
                        geom_id for geom_id in range(int(model.ngeom))
                        if int(model.geom_bodyid[geom_id]) in bottle_bodies
                    ]
                    bounds = []
                    for geom_id in geom_ids:
                        geom_rot = np.asarray(
                            data.geom_xmat[geom_id], dtype=float
                        ).reshape(3, 3)
                        aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
                        center = np.asarray(data.geom_xpos[geom_id], dtype=float) + (
                            geom_rot @ aabb[:3]
                        )
                        center_axial = float(np.dot(center - bottle_pos, bottle_axis))
                        radius = float(np.sum(
                            np.abs(bottle_axis @ geom_rot) * aabb[3:]
                        ))
                        bounds.append((center_axial - radius, center_axial + radius))
                    lower = min((bound[0] for bound in bounds), default=np.nan)
                    upper = max((bound[1] for bound in bounds), default=np.nan)
                    span = upper - lower
                    row["bottle_axis_span_m"] = span
                    row["bottle_axis_fraction"] = (
                        (axial - lower) / span if span > 0 else None
                    )
                except (KeyError, ValueError, AttributeError):
                    pass
                try:
                    import mujoco
                    wrench = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(
                        model._model, data._data, index, wrench
                    )
                    force = float(wrench[0])
                    bottle_id = int(model.body_name2id("wine_bottle_1_main"))
                    weight = float(model.body_mass[bottle_id] * np.linalg.norm(model.opt.gravity))
                    row["normal_force_n"] = force
                    row["bottle_weight_n"] = weight
                    row["normal_force_weight_fraction"] = force / weight if weight > 0 else None
                except (ImportError, KeyError, ValueError, AttributeError):
                    pass
            rows.append(row)
    return rows


def pose_report(env: Any) -> dict[str, Any]:
    model, data = env.sim.model, env.sim.data
    poses = {}
    for name in TRACKED_BODIES:
        try:
            body_id = int(model.body_name2id(name))
        except Exception:
            poses[name] = {"present": False}
            continue
        poses[name] = {
            "present": True,
            "xyz_m": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
            "quat_wxyz": np.asarray(data.body_xquat[body_id], dtype=float).tolist(),
        }
    return poses


def _delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"same_shape": True, "max_abs": float(delta.max(initial=0)),
            "nonzero_scalars": int(np.count_nonzero(delta))}


def _task_group(handle: h5py.File) -> h5py.Group:
    key = TASK_DESCRIPTION.replace(" ", "_")
    if key in handle:
        return handle[key]
    groups = [value for value in handle.values() if isinstance(value, h5py.Group)]
    if len(groups) == 1:
        return groups[0]
    raise KeyError(f"task group {key!r} not found in {handle.filename}")


def load_hdf_states(path: Path, count: int, condition: str) -> list[dict[str, Any]]:
    rows = []
    with h5py.File(path, "r") as handle:
        group = _task_group(handle)
        group_evidence = {
            "artifact_sha256": file_sha256(path),
            "l3a1_topology_schema_version": _scalar(
                group.attrs.get("l3a1_topology_schema_version")
            ),
            "l3a1_topology_id": _scalar(group.attrs.get("l3a1_topology_id")),
            "support_topology_contract_sha256": _scalar(
                group.attrs.get("support_topology_contract_sha256")
            ),
            "compiled_support_component_signatures_sha256": _scalar(
                group.attrs.get("compiled_support_component_signatures_sha256")
            ),
            "support_component_role_hashes_sha256": _scalar(
                group.attrs.get("support_component_role_hashes_sha256")
            ),
            "native_cabinet_xml_sha256": _scalar(
                group.attrs.get("native_cabinet_xml_sha256")
            ),
            "support_edge_geom": _scalar(group.attrs.get("support_edge_geom")),
            "support_inner_front_geom": _scalar(group.attrs.get("support_inner_front_geom")),
            "support_side_geom": _scalar(group.attrs.get("support_side_geom")),
            "support_edge_local_xy": np.asarray(
                group.attrs.get("support_edge_local_xy", []), dtype=float
            ).tolist(),
            "min_edge_axial_m": _scalar(group.attrs.get("min_edge_axial_m")),
            "max_edge_gap_m": _scalar(group.attrs.get("max_edge_gap_m")),
            "max_support_penetration_m": _scalar(
                group.attrs.get("max_support_penetration_m")
            ),
            "min_absolute_support_force_n": _scalar(
                group.attrs.get("min_absolute_support_force_n")
            ),
            "min_edge_force_weight_fraction": _scalar(
                group.attrs.get("min_edge_force_weight_fraction")
            ),
            "min_table_force_weight_fraction": _scalar(
                group.attrs.get("min_table_force_weight_fraction")
            ),
            "support_qualification_algorithm_version": _scalar(
                group.attrs.get("support_qualification_algorithm_version")
            ),
        }
        if group_evidence["l3a1_topology_schema_version"] != 2:
            raise ValueError(f"{condition} artifact is not topology schema v2")
        if any(value is None or value == "" for value in group_evidence.values()):
            raise ValueError(f"{condition} artifact has incomplete topology binding")
        demos = sorted((key for key in group if key.startswith("demo_")),
                       key=lambda key: int(key.split("_")[-1]))
        if len(demos) < count:
            raise ValueError(f"{path} has {len(demos)} states; {count} required")
        for demo_name in demos[:count]:
            demo = group[demo_name]
            initial_roles = str(demo.attrs.get("initial_component_roles", "missing"))
            edge_table = bool(demo.attrs.get("initial_edge_table_qualified", False))
            if condition == "Er" and (
                initial_roles != "edge/front_outer" or not edge_table
            ):
                raise ValueError(f"Er {demo_name} is not edge+table qualified")
            if condition == "Ec" and (initial_roles or edge_table):
                raise ValueError(f"Ec {demo_name} contacts removal component C")
            rows.append({
                "state": np.asarray(demo["initial_state"][:]),
                "source": f"{group.name}/{demo_name}/initial_state",
                "reset_attempt": _scalar(demo.attrs.get("reset_attempt")),
                "source_demo_index": _scalar(demo.attrs.get("source_demo_index")),
                "hdf5_topology_evidence": group_evidence,
                "hdf5_initial_component_roles": initial_roles,
                "hdf5_initial_edge_table_qualified": edge_table,
            })
    return rows


def validate_capture_topology(
    row: dict[str, Any], condition: str, topology: dict[str, Any]
) -> None:
    """Gate exact restored-state contacts independently of HDF qualification flags."""
    component = {
        topology["support_edge_geom"], topology["support_inner_front_geom"],
        topology["support_side_geom"],
    }
    bottle_contacts = [
        contact for contact in row["contacts"]
        if "wine_bottle_1_main" in contact.get("tracked_bodies", [])
    ]
    touched = {
        geom for contact in bottle_contacts for geom in (contact["geom1"], contact["geom2"])
        if geom in component
    }
    if condition == "Ec":
        if touched:
            raise RuntimeError(f"Ec exact-state capture contacts component C: {sorted(touched)}")
        return
    if condition != "Er":
        return
    forbidden = touched - {topology["support_edge_geom"]}
    if forbidden:
        raise RuntimeError(f"Er exact-state capture contacts g35/g36 roles: {sorted(forbidden)}")
    edge_rows = [
        contact for contact in bottle_contacts
        if topology["support_edge_geom"] in (contact["geom1"], contact["geom2"])
    ]
    table_rows = [
        contact for contact in bottle_contacts
        if "table_collision" in (contact["geom1"], contact["geom2"])
    ]
    for label, contacts in (("edge", edge_rows), ("table", table_rows)):
        for contact in contacts:
            missing = [
                field for field in (
                    "world_contact_xyz_m", "drawer_local_contact_xyz_m",
                    "normal_force_n", "normal_force_weight_fraction",
                    "penetration_m", "bottle_axis_axial_m",
                    "bottle_axis_fraction",
                )
                if contact.get(field) is None
            ]
            if missing:
                raise RuntimeError(
                    f"Er exact-state {label} witness lacks physical audit fields: {missing}"
                )
            if not all(np.all(np.isfinite(np.asarray(contact[field], dtype=float))) for field in (
                "world_contact_xyz_m", "drawer_local_contact_xyz_m",
                "normal_force_n", "normal_force_weight_fraction",
                "penetration_m", "bottle_axis_axial_m", "bottle_axis_fraction",
            )):
                raise RuntimeError(f"Er exact-state {label} witness has non-finite physics")

    def force_ok(contact: dict[str, Any], fraction: float) -> bool:
        required = max(
            float(topology["min_absolute_support_force_n"]),
            fraction * float(contact.get("bottle_weight_n", np.nan)),
        )
        return (
            np.isfinite(float(contact.get("normal_force_n", np.nan)))
            and float(contact["normal_force_n"]) >= required
            and float(contact.get("penetration_m", np.inf))
            <= float(topology["max_support_penetration_m"])
        )

    for contact in edge_rows:
        contact["qualified_edge_witness"] = bool(
            force_ok(contact, float(topology["min_edge_force_weight_fraction"]))
            and float(contact.get("edge_gap_m", np.inf)) <= float(topology["max_edge_gap_m"])
            and float(contact.get("bottle_axis_axial_m", -np.inf))
            >= float(topology["min_edge_axial_m"])
        )
    for contact in table_rows:
        contact["qualified_table_witness"] = bool(
            force_ok(contact, float(topology["min_table_force_weight_fraction"]))
        )
    if not any(contact["qualified_edge_witness"] for contact in edge_rows):
        raise RuntimeError("Er exact-state capture has no qualified edge witness")
    if not any(contact["qualified_table_witness"] for contact in table_rows):
        raise RuntimeError("Er exact-state capture has no qualified table witness")


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def capture(
    env: Any, state: np.ndarray, condition: str, episode: int, out_dir: Path,
    topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    env.reset()
    env.set_init_state(state)
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    # Explicitly refresh the observation after the exact restore. Neither this
    # nor set_init_state advances MuJoCo with sim.step(). ControlEnv exposes
    # the refreshed observation through regenerate_obs_from_state rather than
    # forwarding robosuite's private _get_observations method.
    obs = env.regenerate_obs_from_state(restored)
    refreshed = np.asarray(env.sim.get_state().flatten()).copy()
    restore_delta = _delta(state, restored)
    refresh_delta = _delta(restored, refreshed)
    if not restore_delta.get("same_shape") or restore_delta.get("max_abs") != 0.0:
        raise RuntimeError(f"{condition}[{episode}] serialized state was not restored exactly: {restore_delta}")
    if not refresh_delta.get("same_shape") or refresh_delta.get("max_abs") != 0.0:
        raise RuntimeError(f"{condition}[{episode}] observable refresh changed simulator state: {refresh_delta}")
    image = policy_agentview(obs)
    image_path = out_dir / f"{condition.lower()}_ep{episode:03d}_policy_agentview.png"
    imageio.imwrite(image_path, image)
    return {
        "condition": condition,
        "episode": episode,
        "serialized_state_sha256": array_sha256(state),
        "restored_state_sha256": array_sha256(restored),
        "serialized_to_restored_delta": restore_delta,
        "restored_to_refreshed_delta": refresh_delta,
        "policy_image": str(image_path),
        "policy_image_sha256": file_sha256(image_path),
        "policy_image_shape": list(image.shape),
        "poses": pose_report(env),
        "contacts": contact_report(env, topology=topology),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# L3-A1 exact-state policy-view evidence", "",
        "- Verdict: **PASS_L3A1_INIT_EVIDENCE_GENERATED**",
        "- Manual visibility verdict: **PENDING_REVIEW**",
        f"- Episodes per condition: {report['episodes_per_condition']}",
        "- Policy transform: `agentview_image[::-1, ::-1]`", "- Steps after restore: **0**", "",
        "| Condition | Episode | State SHA256 | Restore max delta | Refresh max delta | Contacts | Image |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["captures"]:
        restore = row["serialized_to_restored_delta"].get("max_abs", "shape mismatch")
        refresh = row["restored_to_refreshed_delta"].get("max_abs", "shape mismatch")
        lines.append(f"| {row['condition']} | {row['episode']} | `{row['serialized_state_sha256']}` | "
                     f"{restore} | {refresh} | {len(row['contacts'])} | `{row['policy_image']}` |")
    lines += ["", "Pose/contact details and artifact hashes: `init_evidence.json`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--er", default=DEFAULT_ER)
    parser.add_argument("--ec", default=DEFAULT_EC)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--num_states", type=int, default=int(os.environ.get("PREVIEW_STATES", "5")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_gpu", type=int, default=int(os.environ.get("RENDER_GPU", "1")))
    parser.add_argument("--verify_review", type=Path)
    args = parser.parse_args()
    if args.num_states <= 0:
        parser.error("--num_states must be positive")

    er_path, ec_path, bddl_path = Path(args.er), Path(args.ec), Path(args.bddl)
    out_dir = Path(args.out_dir)
    if args.verify_review is not None:
        validate_manual_review(
            args.verify_review, out_dir / "init_evidence.json", er_path, ec_path
        )
        print(f"verdict={REVIEW_VERDICT} review={args.verify_review}")
        return

    # Defer side-effect registration so review verification and pure helpers
    # remain importable without a LIBERO runtime.
    import experiments.robot.libero.physcog_objects  # noqa: F401

    for path in (er_path, ec_path, bddl_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    from experiments.robot.libero.tasks.validate_l3a1_pairing import (
        artifact_binding,
        validate_pairing,
    )
    validate_pairing(str(er_path), str(ec_path), TASK_DESCRIPTION)
    artifact_bindings = {
        "Er": artifact_binding(str(er_path), TASK_DESCRIPTION),
        "Ec": artifact_binding(str(ec_path), TASK_DESCRIPTION),
    }

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(TASK_ID)
    native_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    native = suite.get_task_init_states(TASK_ID)
    if len(native) < args.num_states:
        raise ValueError(f"native suite has {len(native)} states; {args.num_states} required")
    state_rows = {
        "Eb": [{"state": np.asarray(state), "source": f"libero_10/task_{TASK_ID}/init_state_{i}"}
               for i, state in enumerate(native[:args.num_states])],
        "Er": load_hdf_states(er_path, args.num_states, "Er"),
        "Ec": load_hdf_states(ec_path, args.num_states, "Ec"),
    }
    native_env = OffScreenRenderEnv(bddl_file_name=str(native_bddl), camera_heights=256,
                                    camera_widths=256, render_gpu_device_id=args.render_gpu)
    custom_env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256,
                                    camera_widths=256, render_gpu_device_id=args.render_gpu)
    envs = {"Eb": native_env, "Er": custom_env, "Ec": custom_env}
    native_env.seed(args.seed)
    custom_env.seed(args.seed)
    captures = []
    try:
        for condition in ("Eb", "Er", "Ec"):
            for episode, source in enumerate(state_rows[condition]):
                topology = source.get("hdf5_topology_evidence")
                row = capture(
                    envs[condition], source["state"], condition, episode, out_dir,
                    topology=topology,
                )
                if topology is not None:
                    validate_capture_topology(row, condition, topology)
                row.update({key: value for key, value in source.items() if key != "state"})
                captures.append(row)
    finally:
        native_env.close()
        custom_env.close()

    report = {
        "verdict": "PASS_L3A1_INIT_EVIDENCE_GENERATED",
        "manual_visibility_verdict": "PENDING_REVIEW",
        "episodes_per_condition": args.num_states,
        "task_id": TASK_ID,
        "task_description": TASK_DESCRIPTION,
        "native_bddl": str(native_bddl),
        "custom_bddl": str(bddl_path),
        "seed": args.seed,
        "policy_image_transform": "agentview_image[::-1, ::-1]",
        "simulation_steps_after_restore": 0,
        "artifacts": {
            "Er": {"path": str(er_path), "sha256": file_sha256(er_path)},
            "Ec": {"path": str(ec_path), "sha256": file_sha256(ec_path)},
        },
        "artifact_bindings": artifact_bindings,
        "captures": captures,
    }
    json_path = out_dir / "init_evidence.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, out_dir / "init_evidence.md")
    print(f"verdict=PASS_L3A1_INIT_EVIDENCE_GENERATED captures={len(captures)} report={json_path}")


if __name__ == "__main__":
    main()
