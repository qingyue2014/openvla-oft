"""Native-only feasibility probes for two LIBERO L3-B order candidates.

This module does *not* define an experiment or authorize a policy rollout.  It
locks the two candidate native tasks and records the exact post-wait state seen
by the policy so that an order constraint can be accepted or rejected before
Eb/Er/Ec state generation begins.

Candidate ``mugs`` asks whether completing one mug-to-plate subgoal can create
a persistent access constraint for the other.  Candidate ``moka`` asks whether
the shared stove surface supports a meaningful far-before-near loading order.
Both candidates use unmodified ``libero_10`` BDDL files and inventories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ALLOWED_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
SUITE = "libero_10"
FORMAL_WAIT_STEPS = 10
DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


@dataclass(frozen=True)
class Candidate:
    key: str
    task_id: int
    task_file: str
    prompt: str
    fixtures: dict[str, str]
    objects: dict[str, str]
    tracked_bodies: tuple[str, ...]
    tracked_sites: tuple[str, ...]
    hypothesis: str
    rejection_rule: str


CANDIDATES = {
    "mugs": Candidate(
        key="mugs",
        task_id=4,
        task_file=(
            "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_"
            "put_the_yellow_and_white_mug_on_the_right_plate.bddl"
        ),
        prompt=(
            "put the white mug on the left plate and put the yellow and white "
            "mug on the right plate"
        ),
        fixtures={"living_room_table": "living_room_table"},
        objects={
            "porcelain_mug_1": "porcelain_mug",
            "red_coffee_mug_1": "red_coffee_mug",
            "white_yellow_mug_1": "white_yellow_mug",
            "plate_1": "plate",
            "plate_2": "plate",
        },
        tracked_bodies=(
            "porcelain_mug_1_main",
            "white_yellow_mug_1_main",
            "red_coffee_mug_1_main",
            "plate_1_main",
            "plate_2_main",
            "gripper0_eef",
        ),
        tracked_sites=(),
        hypothesis=(
            "A paired native-object pose intervention can make one completed "
            "mug-on-plate subgoal persist inside the remaining mug transport "
            "workspace, while the reverse order remains safely feasible."
        ),
        rejection_rule=(
            "Reject if both subgoal orders retain comparable collision-free "
            "clearance, if the critical mug/plate is not recognizable in the "
            "policy frame, or if the proposed constraint requires changing "
            "left/right prompt semantics."
        ),
    ),
    "moka": Candidate(
        key="moka",
        task_id=8,
        task_file="KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl",
        prompt="put both moka pots on the stove",
        fixtures={
            "kitchen_table": "kitchen_table",
            "flat_stove_1": "flat_stove",
        },
        objects={
            "moka_pot_1": "moka_pot",
            "moka_pot_2": "moka_pot",
        },
        tracked_bodies=(
            "moka_pot_1_main",
            "moka_pot_2_main",
            "flat_stove_1_main",
            "gripper0_eef",
        ),
        tracked_sites=("flat_stove_1_cook_region",),
        hypothesis=(
            "The native shared stove surface can support a meaningful "
            "far-before-near two-pot loading order without adding slots, "
            "obstacles, or prompt semantics."
        ),
        rejection_rule=(
            "Reject if the native cook region admits both pots with no "
            "order-dependent access constraint, or if declaring front/back "
            "slots would add goal semantics absent from the native BDDL."
        ),
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_bddl_path(candidate: Candidate) -> Path:
    return (
        repository_root()
        / "_deps"
        / "LIBERO"
        / "libero"
        / "libero"
        / "bddl_files"
        / SUITE
        / candidate.task_file
    )


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typed_inventory(text: str, section: str) -> dict[str, str]:
    match = re.search(
        rf"\(:{re.escape(section)}\s+(.*?)\n\s*\)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing BDDL :{section} section")
    inventory: dict[str, str] = {}
    for line in match.group(1).splitlines():
        declaration = line.strip()
        if not declaration or "-" not in declaration:
            continue
        names, kind = declaration.split("-", maxsplit=1)
        for name in names.split():
            inventory[name] = kind.strip()
    return inventory


def parse_native_bddl(candidate: Candidate) -> dict:
    path = native_bddl_path(candidate).resolve()
    text = path.read_text(encoding="utf-8")
    prompt_match = re.search(r"\(:language\s+([^)]+)\)", text)
    if not prompt_match:
        raise ValueError(f"{path} has no native :language prompt")
    record = {
        "path": str(path),
        "sha256": sha256_path(path),
        "prompt": prompt_match.group(1).strip(),
        "fixtures": _typed_inventory(text, "fixtures"),
        "objects": _typed_inventory(text, "objects"),
    }
    if SUITE not in ALLOWED_SUITES:
        raise ValueError(f"forbidden LIBERO suite: {SUITE}")
    if record["prompt"] != candidate.prompt:
        raise ValueError(
            f"{candidate.key} prompt mismatch: "
            f"{record['prompt']!r} != {candidate.prompt!r}"
        )
    if record["fixtures"] != candidate.fixtures:
        raise ValueError(
            f"{candidate.key} fixture inventory mismatch: "
            f"{record['fixtures']} != {candidate.fixtures}"
        )
    if record["objects"] != candidate.objects:
        raise ValueError(
            f"{candidate.key} object inventory mismatch: "
            f"{record['objects']} != {candidate.objects}"
        )
    return record


def static_preflight(candidate: Candidate) -> dict:
    return {
        "candidate": asdict(candidate),
        "native_suite": SUITE,
        "native_bddl": parse_native_bddl(candidate),
        "native_only": True,
        "custom_assets": [],
        "custom_bddl": False,
        "modified_prompt": False,
        "status": "PASS_STATIC_NATIVE_ONLY_PREFLIGHT",
    }


def _all_model_names(model, kind: str) -> list[str]:
    count = int(getattr(model, f"n{kind}"))
    lookup = getattr(model, f"{kind}_id2name")
    return [name for index in range(count) if (name := lookup(index))]


def _descendant_body_ids(model, root_id: int) -> set[int]:
    result = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if int(model.body_parentid[body_id]) in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def _contact_body_names(env, body_name: str) -> list[str]:
    model, data = env.sim.model, env.sim.data
    root_id = int(model.body_name2id(body_name))
    descendants = _descendant_body_ids(model, root_id)
    geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in descendants
    }
    result: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in geom_ids:
            other = geom2
        elif geom2 in geom_ids:
            other = geom1
        else:
            continue
        other_name = model.body_id2name(int(model.geom_bodyid[other]))
        if other_name:
            result.add(other_name)
    return sorted(result)


def _free_joint_velocity(env, body_name: str) -> tuple[float, float] | None:
    model = env.sim.model
    body_id = int(model.body_name2id(body_name))
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != 0:
        return None
    velocity_address = int(model.jnt_dofadr[joint_id])
    velocity = np.asarray(
        env.sim.data.qvel[velocity_address : velocity_address + 6], dtype=float
    )
    return (
        float(np.linalg.norm(velocity[:3])),
        float(np.linalg.norm(velocity[3:])),
    )


def _body_bounding_radius(env, body_name: str) -> float:
    model = env.sim.model
    body_id = int(model.body_name2id(body_name))
    descendants = _descendant_body_ids(model, body_id)
    radii = [
        float(model.geom_rbound[geom_id])
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in descendants
    ]
    return max(radii, default=0.0)


def body_measurement(env, body_name: str) -> dict:
    model, data = env.sim.model, env.sim.data
    body_id = int(model.body_name2id(body_name))
    rotation = np.asarray(data.body_xmat[body_id], dtype=float).reshape(3, 3)
    velocity = _free_joint_velocity(env, body_name)
    return {
        "position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(
            data.body_xquat[body_id], dtype=float
        ).tolist(),
        "tilt_deg": float(
            np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        ),
        "linear_speed_mps": None if velocity is None else velocity[0],
        "angular_speed_radps": None if velocity is None else velocity[1],
        "bounding_radius_m": _body_bounding_radius(env, body_name),
        "contacts": _contact_body_names(env, body_name),
    }


def _refresh_observation(env):
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    raw_env = getattr(env, "env", env)
    return raw_env._get_observations()


def _pairwise_xy(measurements: dict[str, dict]) -> dict[str, float]:
    names = list(measurements)
    result = {}
    for index, first in enumerate(names):
        first_position = np.asarray(measurements[first]["position"], dtype=float)
        for second in names[index + 1 :]:
            second_position = np.asarray(
                measurements[second]["position"], dtype=float
            )
            result[f"{first}::{second}"] = float(
                np.linalg.norm(first_position[:2] - second_position[:2])
            )
    return result


def _write_policy_image(path: Path, observation) -> None:
    import imageio.v2 as imageio

    image = np.asarray(observation["agentview_image"])
    image = image[::-1, ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def _trusted_native_states(task_suite, task) -> np.ndarray:
    """Load repository-owned LIBERO states under PyTorch >=2.6.

    PyTorch changed ``torch.load`` to ``weights_only=True`` by default.  These
    files are the task-locked official LIBERO ``.pruned_init`` artifacts, not
    user-supplied checkpoints, so the explicit legacy load is scoped here.
    """

    import torch
    from libero.libero import get_libero_path

    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    if path.name != Path(candidate_init_filename(task)).name:
        raise ValueError(f"unexpected native init-state filename: {path.name}")
    states = torch.load(path, weights_only=False)
    return np.asarray(states)


def candidate_init_filename(task) -> str:
    return f"{Path(task.bddl_file).stem}.pruned_init"


def dynamic_probe(
    candidate: Candidate,
    *,
    num_states: int,
    output_dir: Path,
    render_gpu_device_id: int,
) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[SUITE]()
    task = task_suite.get_task(candidate.task_id)
    if task.bddl_file != candidate.task_file or task.language != candidate.prompt:
        raise ValueError(
            f"{candidate.key} runtime task mismatch: "
            f"id={candidate.task_id} file={task.bddl_file!r} "
            f"prompt={task.language!r}"
        )
    states = _trusted_native_states(task_suite, task)
    if num_states < 1:
        raise ValueError("num_states must be positive")
    num_states = min(num_states, len(states))
    task_bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if task_bddl.resolve() != native_bddl_path(candidate).resolve():
        raise ValueError(
            f"{candidate.key} runtime BDDL path mismatch: "
            f"{task_bddl.resolve()} != {native_bddl_path(candidate).resolve()}"
        )

    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(0)
    records = []
    try:
        for state_index in range(num_states):
            env.reset()
            env.set_init_state(states[state_index])
            pre_observation = _refresh_observation(env)
            missing_bodies = [
                name
                for name in candidate.tracked_bodies
                if name not in _all_model_names(env.sim.model, "body")
            ]
            missing_sites = [
                name
                for name in candidate.tracked_sites
                if name not in _all_model_names(env.sim.model, "site")
            ]
            if missing_bodies or missing_sites:
                raise ValueError(
                    f"{candidate.key} compiled-name mismatch: "
                    f"bodies={missing_bodies}, sites={missing_sites}"
                )
            samples = [
                {
                    body: body_measurement(env, body)
                    for body in candidate.tracked_bodies
                }
            ]
            observation = pre_observation
            for _ in range(FORMAL_WAIT_STEPS):
                observation, _, _, _ = env.step(DUMMY_ACTION)
                samples.append(
                    {
                        body: body_measurement(env, body)
                        for body in candidate.tracked_bodies
                    }
                )
            sites = {
                site: np.asarray(
                    env.sim.data.site_xpos[env.sim.model.site_name2id(site)],
                    dtype=float,
                ).tolist()
                for site in candidate.tracked_sites
            }
            image_path = (
                output_dir
                / candidate.key
                / f"{candidate.key}_native_state{state_index:02d}_first_policy.png"
            )
            _write_policy_image(image_path, observation)
            start, finish = samples[0], samples[-1]
            stability = {}
            for body in candidate.tracked_bodies:
                start_position = np.asarray(start[body]["position"], dtype=float)
                translations = [
                    float(
                        np.linalg.norm(
                            np.asarray(sample[body]["position"], dtype=float)
                            - start_position
                        )
                    )
                    for sample in samples
                ]
                tilts = [float(sample[body]["tilt_deg"]) for sample in samples]
                linear_speeds = [
                    sample[body]["linear_speed_mps"]
                    for sample in samples
                    if sample[body]["linear_speed_mps"] is not None
                ]
                angular_speeds = [
                    sample[body]["angular_speed_radps"]
                    for sample in samples
                    if sample[body]["angular_speed_radps"] is not None
                ]
                stability[body] = {
                    "max_translation_drift_m": max(translations),
                    "max_tilt_deg": max(tilts),
                    "max_linear_speed_mps": (
                        max(linear_speeds) if linear_speeds else None
                    ),
                    "max_angular_speed_radps": (
                        max(angular_speeds) if angular_speeds else None
                    ),
                    "pre_wait": start[body],
                    "first_policy": finish[body],
                }
            records.append(
                {
                    "state_index": state_index,
                    "policy_image": str(image_path.resolve()),
                    "sites": sites,
                    "pairwise_xy_m": _pairwise_xy(finish),
                    "stability": stability,
                }
            )
    finally:
        env.close()

    return {
        **static_preflight(candidate),
        "runtime_task": {
            "task_id": candidate.task_id,
            "prompt": task.language,
            "bddl_file": task.bddl_file,
            "native_state_count": int(len(states)),
        },
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "states": records,
        "status": "PASS_NATIVE_DYNAMIC_PROBE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe native LIBERO L3-B order candidates"
    )
    parser.add_argument(
        "--candidate",
        choices=("all", *CANDIDATES),
        default="all",
    )
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="review/L3-B_order_candidates_task/native_probe",
    )
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    keys: Iterable[str] = (
        CANDIDATES if args.candidate == "all" else (args.candidate,)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        candidate = CANDIDATES[key]
        if args.static_only:
            report = static_preflight(candidate)
        else:
            report = dynamic_probe(
                candidate,
                num_states=args.num_states,
                output_dir=output_dir,
                render_gpu_device_id=args.render_gpu_device_id,
            )
        report_path = output_dir / f"{key}_native_probe.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{report['status']} candidate={key} report={report_path}")


if __name__ == "__main__":
    main()
