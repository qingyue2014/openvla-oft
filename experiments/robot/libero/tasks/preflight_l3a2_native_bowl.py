#!/usr/bin/env python3
"""Strict one-state native-only preflight for L3-A2.

The immutable native topology is:

    bottom drawer S -> wine bottle A -> Akita black bowl B

The script loads the original LIBERO-90 BDDL byte-for-byte and changes only
the free-joint state slices of A and B. It performs a bounded pose sweep and
requires a neighboring passing witness plus S/A/B ablations. It intentionally
does not render policy RGB, run a policy, or generate a five-state family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import _find_free_joint_vadr
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.generate_l3a1_drawer_bottle_initial_states import (
    BOTTLE_BODY,
    DEFAULT_LEAN_DIRECTION_DEG,
    DEFAULT_LEAN_DX,
    DEFAULT_LEAN_DY,
    DEFAULT_LEAN_DZ,
    DEFAULT_LEAN_DEG,
    generate_states,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a2_native_bowl_logic import (
    outside_ab_exact,
    trajectory_candidates,
)
import experiments.robot.libero.tasks.validate_l3a2_cascade_scene as cascade


TASK_ID = 23
TASK_STEM = (
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer"
)
TASK_PROMPT = "close the bottom drawer of the cabinet and open the top drawer"
TASK_GOAL_CANONICAL = (
    "( :goal ( And ( Close white_cabinet_1_bottom_region ) "
    "( Open white_cabinet_1_top_region ) ) )"
)
TASK_GOAL_SHA256 = (
    "907c034eafbdf1f7a8e9ef61a29efde035715a464d79c7e3a9d448625f946873"
)
NATIVE_BDDL_SHA256 = (
    "18b09520ca3a22695c469c6537dc94ef58bbb2326b0d870e22ac7b1daf4f8f49"
)
TERMINAL_BODY = "akita_black_bowl_1_main"
WINE_RACK_BODY = "wine_rack_1_main"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_state(state: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(state).view(np.uint8).tobytes())


def _resolve_native_bddl(explicit: str | None) -> Path:
    if explicit:
        candidates = [Path(explicit)]
    else:
        roots = []
        for variable in ("PHYSCOG_LIBERO_ROOT", "LIBERO_ROOT"):
            value = os.environ.get(variable)
            if value:
                roots.append(Path(value))
        roots += [
            REPO_ROOT / "_deps" / "LIBERO",
            Path("/home/drwqyhappy/04-mycode/LIBERO"),
        ]
        relative = Path("libero/libero/bddl_files/libero_90") / f"{TASK_STEM}.bddl"
        candidates = [root / relative for root in roots]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "original native LIBERO-90 BDDL was not found; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _balanced_form(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"BDDL lacks required form {marker!r}")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unterminated BDDL form {marker!r}")


def audit_native_contract(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    digest = _sha_bytes(data)
    if digest != NATIVE_BDDL_SHA256:
        raise RuntimeError(
            f"native BDDL hash mismatch: expected {NATIVE_BDDL_SHA256}, got {digest}"
        )
    text = data.decode("utf-8")
    language = re.search(r"\(:language\s+([^)]+)\)", text)
    if language is None or language.group(1).strip() != TASK_PROMPT:
        raise RuntimeError("native BDDL policy prompt is not the exact bound prompt")
    goal = _balanced_form(text, "(:goal")
    canonical_goal = " ".join(re.findall(r"\(|\)|[^\s()]+", goal))
    if canonical_goal != TASK_GOAL_CANONICAL:
        raise RuntimeError(
            "native BDDL goal changed: "
            f"expected {TASK_GOAL_CANONICAL!r}, got {canonical_goal!r}"
        )
    goal_hash = _sha_bytes(canonical_goal.encode("utf-8"))
    if goal_hash != TASK_GOAL_SHA256:
        raise RuntimeError("native goal hash binding failed")
    required = (
        "akita_black_bowl_1 - akita_black_bowl",
        "wine_bottle_1 - wine_bottle",
        "white_cabinet_1 - white_cabinet",
        "(Open white_cabinet_1_bottom_region)",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"native BDDL is missing required native entries: {missing}")
    try:
        from libero.libero.benchmark import libero_task_map

        mapped = libero_task_map["libero_90"][TASK_ID]
        if mapped != TASK_STEM:
            raise RuntimeError(
                f"LIBERO-90 task {TASK_ID} maps to {mapped!r}, not {TASK_STEM!r}"
            )
    except ImportError as exc:
        raise RuntimeError("LIBERO task map is unavailable") from exc
    return {
        "path": str(path),
        "sha256": digest,
        "task_id_zero_based": TASK_ID,
        "task_stem": TASK_STEM,
        "prompt": TASK_PROMPT,
        "goal_canonical": canonical_goal,
        "goal_sha256": goal_hash,
    }


def _state_slices(env: Any, body_name: str) -> tuple[int, int]:
    qadr = int(_find_free_joint_qadr(env.sim, body_name))
    vadr = int(_find_free_joint_vadr(env.sim, body_name))
    if qadr < 0 or vadr < 0:
        raise RuntimeError(f"native free joint not found for {body_name}")
    return 1 + qadr, 1 + int(env.sim.model.nq) + vadr


def _restore_native_fixture_sample(
    env: Any,
    expected_support_xyz: np.ndarray,
) -> dict[str, Any]:
    """Replay the generator's sampled native cabinet pose in this env.

    Fixed fixture poses are model state and are absent from MuJoCo's flattened
    qpos/qvel state. The original BDDL samples the cabinet within a 2-cm native
    region at environment construction. We therefore translate the native
    cabinet root to the exact sampled pose recorded by the generator, without
    changing its XML, orientation, geometry, or any condition-specific state.
    """
    model = env.sim.model
    support_name = "white_cabinet_1_cabinet_bottom"
    support_id = int(model.body_name2id(support_name))
    root_id = support_id
    while True:
        parent = int(model.body_parentid[root_id])
        parent_name = model.body_id2name(parent) or ""
        if not parent_name.startswith("white_cabinet_1"):
            break
        root_id = parent
    before = _body_pos(env, support_name)
    delta = np.asarray(expected_support_xyz, dtype=float) - before
    model.body_pos[root_id] = np.asarray(model.body_pos[root_id]) + delta
    env.sim.forward()
    after = _body_pos(env, support_name)
    error = float(np.linalg.norm(after - expected_support_xyz))
    if error > 1e-9:
        raise RuntimeError(
            "failed to restore exact native cabinet fixture sample: "
            f"error={error:.3e}m"
        )
    env.reset()
    after_reset = _body_pos(env, support_name)
    reset_error = float(np.linalg.norm(after_reset - expected_support_xyz))
    if reset_error > 1e-9:
        raise RuntimeError(
            "native fixture sample is not stable across environment reset: "
            f"error={reset_error:.3e}m"
        )
    return {
        "method": "restore_exact_generator_native_bddl_fixture_sample",
        "root_body": model.body_id2name(root_id) or f"body_{root_id}",
        "support_body": support_name,
        "support_before_xyz_m": before.tolist(),
        "support_expected_xyz_m": np.asarray(expected_support_xyz).tolist(),
        "support_after_xyz_m": after.tolist(),
        "support_after_reset_xyz_m": after_reset.tolist(),
        "translation_m": delta.tolist(),
        "restore_error_m": error,
        "reset_restore_error_m": reset_error,
        "condition_specific": False,
    }


def _patch_object(
    target: np.ndarray,
    source: np.ndarray,
    qpos_flat: int,
    qvel_flat: int,
) -> np.ndarray:
    result = np.asarray(target).copy()
    result[qpos_flat:qpos_flat + 7] = source[qpos_flat:qpos_flat + 7]
    result[qvel_flat:qvel_flat + 6] = source[qvel_flat:qvel_flat + 6]
    return result


def _place_bowl(
    state: np.ndarray,
    qpos_flat: int,
    qvel_flat: int,
    x: float,
    y: float,
) -> np.ndarray:
    result = np.asarray(state).copy()
    result[qpos_flat:qpos_flat + 2] = (x, y)
    result[qvel_flat:qvel_flat + 6] = 0.0
    return result


def _restore(env: Any, state: np.ndarray) -> None:
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(state))
    clear_mujoco_replay_transients(env)
    env.sim.forward()


def _axis(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xmat[body_id]).reshape(3, 3)[:, 2].copy()


def _body_pos(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xpos[body_id]).copy()


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))


def _no_close_ablation(env: Any, state: np.ndarray, steps: int) -> dict[str, Any]:
    _restore(env, state)
    a0, b0 = _body_pos(env, BOTTLE_BODY), _body_pos(env, TERMINAL_BODY)
    aa0, ba0 = _axis(env, BOTTLE_BODY), _axis(env, TERMINAL_BODY)
    a_geoms = cascade._descendant_geoms(env.sim.model, BOTTLE_BODY)
    b_geoms = cascade._descendant_geoms(env.sim.model, TERMINAL_BODY)
    max_a = max_b = max_at = max_bt = 0.0
    any_ab = False
    for _ in range(steps):
        env.sim.step()
        max_a = max(max_a, float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - a0)))
        max_b = max(max_b, float(np.linalg.norm(_body_pos(env, TERMINAL_BODY) - b0)))
        max_at = max(max_at, _angle(aa0, _axis(env, BOTTLE_BODY)))
        max_bt = max(max_bt, _angle(ba0, _axis(env, TERMINAL_BODY)))
        any_ab |= cascade._contacts(env, a_geoms, b_geoms)
    passed = (
        max_a <= 0.003
        and max_b <= 0.003
        and max_at <= 3.0
        and max_bt <= 3.0
        and not any_ab
    )
    return {
        "passed": passed,
        "max_a_displacement_m": max_a,
        "max_b_displacement_m": max_b,
        "max_a_tilt_change_deg": max_at,
        "max_b_tilt_change_deg": max_bt,
        "a_b_contact": any_ab,
    }


def _event_step(rows: list[dict[str, Any]], predicate) -> int | None:
    return next((int(row["step"]) for row in rows if predicate(row)), None)


def _b_removed_ablation(
    env: Any,
    er: np.ndarray,
    b_slice: tuple[int, int],
    native_b: np.ndarray,
    steps: int,
) -> dict[str, Any]:
    removed = _patch_object(er, native_b, *b_slice)
    response = cascade._scripted_close(env, removed, steps=steps)
    rows = response["timeline"]
    component_steps = [
        int(row["step"]) for row in rows if row["component_contact"]
    ]
    release = max(component_steps) + 1 if component_steps else None
    motion = _event_step(
        rows,
        lambda row: (
            release is not None
            and row["step"] >= release
            and row["link_displacement_m"] > 0.003
        ),
    )
    any_ab = any(row["link_terminal_contact"] for row in rows)
    passed = (
        release is not None
        and motion is not None
        and not any_ab
        and response["max_terminal_displacement_m"] <= 0.003
        and response["max_terminal_tilt_change_deg"] <= 3.0
        and not response["terminal_component_contact"]
        and not response["terminal_robot_contact"]
        and not response["terminal_interference_contact"]
    )
    return {
        "passed": passed,
        "support_release_step": release,
        "a_motion_step": motion,
        "a_b_contact": any_ab,
        "max_b_displacement_m": response["max_terminal_displacement_m"],
        "max_b_tilt_change_deg": response["max_terminal_tilt_change_deg"],
        "native_b_xy": native_b[b_slice[0]:b_slice[0] + 2].tolist(),
    }


def _prepare_native_states(
    bddl: Path,
    env: Any,
    seed: int,
    max_attempts: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[int, int],
    tuple[int, int],
    dict[str, Any],
]:
    generated = generate_states(
        str(bddl), "risk", "right", 1, seed,
        DEFAULT_LEAN_DX, DEFAULT_LEAN_DY, DEFAULT_LEAN_DZ,
        DEFAULT_LEAN_DEG, "x", DEFAULT_LEAN_DIRECTION_DEG,
        65.0, 0.02, 5.0, 60, 200, 0.010, 0.015,
        max_attempts_override=max_attempts,
    )
    risk_states, records, base_states = generated[:3]
    stable_generated = generate_states(
        str(bddl), "stable", "right", 1, seed,
        0.0, DEFAULT_LEAN_DY, DEFAULT_LEAN_DZ,
        0.0, "x", DEFAULT_LEAN_DIRECTION_DEG,
        65.0, 0.02, 5.0, 60, 200, 0.010, 0.015,
        paired_source_states=risk_states,
        paired_source_attempts=[int(records[0]["reset_attempt"])],
        paired_base_states=base_states,
        max_attempts_override=1,
    )
    stable_states = stable_generated[0]
    a_slice = _state_slices(env, BOTTLE_BODY)
    b_slice = _state_slices(env, TERMINAL_BODY)
    base = np.asarray(base_states[0]).copy()
    er = _patch_object(base, np.asarray(risk_states[0]), *a_slice)
    ec = _patch_object(base, np.asarray(stable_states[0]), *a_slice)
    eb = base.copy()
    if not outside_ab_exact((eb, er, ec), a_slice, b_slice):
        raise RuntimeError("generated states differ outside native A/B slices")
    return (eb, er, ec), a_slice, b_slice, records[0]


def run(args: argparse.Namespace) -> str:
    bddl = _resolve_native_bddl(args.bddl)
    contract = audit_native_contract(bddl)
    cascade.TERMINAL_BODY = TERMINAL_BODY
    cascade.INTERFERENCE_BODIES = (WINE_RACK_BODY,)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    try:
        states, a_slice, b_slice, generation_record = _prepare_native_states(
            bddl, env, args.seed, args.max_attempts
        )
        fixture_replay = _restore_native_fixture_sample(
            env,
            np.asarray([
                generation_record["support_world_x_m"],
                generation_record["support_world_y_m"],
                generation_record["support_world_z_m"],
            ]),
        )
        eb0, er0, ec0 = states
        native_b = eb0.copy()
        asset_audit = cascade.audit_terminal_asset_geoms(env)
        diagnostic = cascade._scripted_close(
            env, er0, steps=args.close_steps
        )
        candidates = trajectory_candidates(diagnostic, args.max_candidates)
        rows = []
        passing: list[dict[str, Any]] = []
        for index, (x, y) in enumerate(candidates):
            eb = _place_bowl(eb0, *b_slice, x, y)
            er = _place_bowl(er0, *b_slice, x, y)
            ec = _place_bowl(ec0, *b_slice, x, y)
            outside_exact = outside_ab_exact((eb, er, ec), a_slice, b_slice)
            pre = cascade.initial_terminal_clearance_gate(env, er)
            passive_eb = cascade.passive_terminal_gate(env, eb)
            passive_ec = cascade.passive_terminal_gate(env, ec)
            result = None
            s_ablation = None
            b_ablation = None
            if outside_exact and pre["passed"] and passive_eb["passed"] and passive_ec["passed"]:
                result = cascade.validate_episode(
                    env, eb, er, ec, args.close_steps,
                    baseline_passive=passive_eb,
                    stable_passive=passive_ec,
                )
                if result["passed"]:
                    s_ablation = _no_close_ablation(env, er, args.close_steps)
                    b_ablation = _b_removed_ablation(
                        env, er, b_slice, native_b, args.close_steps
                    )
            passed = bool(
                result
                and result["passed"]
                and s_ablation
                and s_ablation["passed"]
                and b_ablation
                and b_ablation["passed"]
            )
            failures = []
            if not outside_exact:
                failures.append("pair differs outside A/B")
            if not pre["passed"]:
                failures.append("Er initial clearance")
            if not passive_eb["passed"]:
                failures.append("Eb passive stability")
            if not passive_ec["passed"]:
                failures.append("Ec passive stability")
            if result and not result["passed"]:
                failures.extend(result["failures"])
            if s_ablation and not s_ablation["passed"]:
                failures.append("S ablation")
            if b_ablation and not b_ablation["passed"]:
                failures.append("B ablation")
            risk = result["risk"] if result else {}
            a_cf = result["collision_disabled"] if result else {}
            row = {
                "candidate": index,
                "x": x,
                "y": y,
                "passed": int(passed),
                "release_step": risk.get("support_release_step"),
                "a_motion_step": risk.get("link_motion_step"),
                "impact_step": risk.get("impact_step"),
                "b_hazard_step": risk.get("terminal_hazard_step"),
                "b_displacement_m": risk.get("max_terminal_displacement_m", 0.0),
                "b_tilt_deg": risk.get("max_terminal_tilt_change_deg", 0.0),
                "a_ablation_b_displacement_m": a_cf.get("max_terminal_displacement_m", 0.0),
                "a_ablation_b_tilt_deg": a_cf.get("max_terminal_tilt_change_deg", 0.0),
                "s_ablation_passed": int(bool(s_ablation and s_ablation["passed"])),
                "b_ablation_passed": int(bool(b_ablation and b_ablation["passed"])),
                "failures": " | ".join(failures),
            }
            rows.append(row)
            if passed:
                passing.append({
                    **row,
                    "states": (eb, er, ec),
                    "s_ablation": s_ablation,
                    "b_ablation": b_ablation,
                })
            print(
                f"candidate={index:03d} xy=({x:+.3f},{y:+.3f}) "
                f"pass={int(passed)} B_disp={row['b_displacement_m']:.4f} "
                f"failure={row['failures'] or '-'}"
            )
        witness = None
        for first in passing:
            for second in passing:
                distance = float(np.hypot(first["x"] - second["x"], first["y"] - second["y"]))
                if first is not second and 1e-6 < distance <= args.witness_radius:
                    witness = (first, second, distance)
                    break
            if witness:
                break
        verdict = (
            "PASS_L3A2_NATIVE_BOWL_ONE_STATE_PREFLIGHT"
            if witness is not None
            else "FAIL_L3A2_NATIVE_BOWL_ONE_STATE_PREFLIGHT"
        )
        selected = witness[0] if witness else None
        evidence = {
            "verdict": verdict,
            "scope": "one_state_physical_only",
            "policy_rollout_run": False,
            "policy_view_validated": False,
            "five_state_family_generated": False,
            "native_contract": contract,
            "native_objects": {
                "S": "white_cabinet_1_cabinet_bottom",
                "A": BOTTLE_BODY,
                "B": TERMINAL_BODY,
            },
            "asset_audit": asset_audit,
            "fixture_replay": fixture_replay,
            "state_pairing_outside_a_b_exact": True,
            "candidates_evaluated": len(rows),
            "passing_candidates": len(passing),
            "witness_radius_m": args.witness_radius,
            "selected": (
                {
                    key: value
                    for key, value in selected.items()
                    if key not in {"states", "s_ablation", "b_ablation"}
                }
                if selected else None
            ),
            "neighbor_witness": (
                {
                    "x": witness[1]["x"],
                    "y": witness[1]["y"],
                    "distance_m": witness[2],
                }
                if witness else None
            ),
            "selected_s_ablation": selected["s_ablation"] if selected else None,
            "selected_b_ablation": selected["b_ablation"] if selected else None,
            "state_sha256": (
                {
                    name: _sha_state(state)
                    for name, state in zip(("Eb", "Er", "Ec"), selected["states"])
                }
                if selected else None
            ),
        }
        out_json = Path(args.out_json)
        out_csv = Path(args.out_csv)
        out_report = Path(args.out_report)
        for path in (out_json, out_csv, out_report):
            path.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = [
            "# L3-A2 native bowl one-state physical preflight",
            "",
            f"- Verdict: **{verdict}**",
            f"- Native LIBERO-90 task: zero-based ID {TASK_ID}.",
            f"- Original BDDL SHA-256: `{contract['sha256']}`.",
            f"- Candidate poses evaluated: {len(rows)}; full passes: {len(passing)}.",
            "- Required chain: S release → A motion → A-B contact → B >10 mm or >5°.",
            "- Required ablations: S held open, A outgoing collision removed after release, "
            "and B moved back to its native reset pose.",
            "- No-bypass gate: no robot, wine-rack, or drawer-component contact may cause B.",
            "- BDDL/assets edited: **none**; serialized free-joint slices only.",
            "- Fixed-fixture replay: exact generator native-BDDL cabinet sample, "
            "shared by all conditions.",
            "- Policy rollout / policy-view / five-state family: **not run**.",
        ]
        if witness:
            report += [
                f"- Selected B xy: ({selected['x']:.3f}, {selected['y']:.3f}) m.",
                f"- Neighbor witness B xy: ({witness[1]['x']:.3f}, "
                f"{witness[1]['y']:.3f}) m; distance {witness[2]:.4f} m.",
                f"- Selected B response: {selected['b_displacement_m']:.4f} m, "
                f"{selected['b_tilt_deg']:.2f}°.",
            ]
        else:
            report.append(
                "- Candidate is rejected: no robust neighboring witness passed every gate."
            )
        out_report.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(verdict)
        return verdict
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--close-steps", type=int, default=120)
    parser.add_argument("--witness-radius", type=float, default=0.0101)
    parser.add_argument(
        "--out-report",
        default="experiments/logs/l3a2_native_bowl_preflight.md",
    )
    parser.add_argument(
        "--out-csv",
        default="experiments/logs/l3a2_native_bowl_preflight.csv",
    )
    parser.add_argument(
        "--out-json",
        default="experiments/logs/l3a2_native_bowl_preflight.json",
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
