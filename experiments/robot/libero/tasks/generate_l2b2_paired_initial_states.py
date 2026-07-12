"""Generate episode-matched Eb/Er/Ec initial states for L2-B2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.generate_l2b1_initial_states import save_hdf5
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import (
    SETTLE_STEPS,
    _body_pos,
    _set_stove_state,
)


NEAR_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_stove.bddl"
FAR_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_far_stove.bddl"
TASK_DESCRIPTION = "put both the cream cheese box and the butter in the basket"
MOVABLE_BODIES = (
    "alphabet_soup_1_main",
    "cream_cheese_1_main",
    "tomato_sauce_1_main",
    "ketchup_1_main",
    "milk_1_main",
    "orange_juice_1_main",
    "butter_1_main",
    "basket_1_main",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_states", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_tolerance", type=float, default=0.005)
    parser.add_argument("--max_attempt_factor", type=int, default=10)
    parser.add_argument(
        "--eb_output",
        default="experiments/robot/libero/tasks/l2b2_paired_eb_stove_off.hdf5",
    )
    parser.add_argument(
        "--er_output",
        default="experiments/robot/libero/tasks/l2b2_paired_er_stove_on.hdf5",
    )
    parser.add_argument(
        "--ec_output",
        default="experiments/robot/libero/tasks/l2b2_paired_ec_far_stove.hdf5",
    )
    return parser.parse_args()


def _assert_matching_joint_schema(envs: list[OffScreenRenderEnv]) -> None:
    reference = [envs[0].sim.model.joint_id2name(i) for i in range(envs[0].sim.model.njnt)]
    reference_qpos = envs[0].sim.model.nq
    reference_qvel = envs[0].sim.model.nv
    for env in envs[1:]:
        names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
        if names != reference or env.sim.model.nq != reference_qpos or env.sim.model.nv != reference_qvel:
            raise ValueError("Eb/Er/Ec MuJoCo joint schemas differ; qpos pairing is unsafe")


def _settle(env: OffScreenRenderEnv) -> None:
    for _ in range(SETTLE_STEPS):
        env.sim.step()


def _positions(env: OffScreenRenderEnv) -> dict[str, np.ndarray]:
    return {body: _body_pos(env, body).copy() for body in MOVABLE_BODIES}


def _copy_dynamic_state(source: OffScreenRenderEnv, target: OffScreenRenderEnv) -> None:
    target.sim.data.qpos[:] = source.sim.data.qpos
    target.sim.data.qvel[:] = source.sim.data.qvel
    target.sim.forward()


def _align_movable_world_positions(
    reference: dict[str, np.ndarray], target: OffScreenRenderEnv
) -> None:
    """Correct compiled-scene frame offsets through each object's free joint."""
    for _ in range(2):
        for body, desired_position in reference.items():
            qadr = int(_find_free_joint_qadr(target.sim, body))
            if qadr < 0:
                raise KeyError(f"No free joint found for paired body: {body}")
            current_position = _body_pos(target, body)
            target.sim.data.qpos[qadr : qadr + 3] += desired_position - current_position
            target.sim.data.qvel[:] = 0.0
            target.sim.forward()


def _max_pair_delta(reference: dict[str, np.ndarray], env: OffScreenRenderEnv) -> tuple[float, str]:
    deltas = {
        body: float(np.linalg.norm(_body_pos(env, body) - position))
        for body, position in reference.items()
    }
    body = max(deltas, key=deltas.get)
    return deltas[body], body


def main() -> None:
    args = parse_args()
    if args.num_states < 1:
        raise ValueError("--num_states must be at least 1")
    if args.pair_tolerance <= 0:
        raise ValueError("--pair_tolerance must be positive")

    eb_env = OffScreenRenderEnv(bddl_file_name=NEAR_BDDL, camera_heights=256, camera_widths=256)
    er_env = OffScreenRenderEnv(bddl_file_name=NEAR_BDDL, camera_heights=256, camera_widths=256)
    ec_env = OffScreenRenderEnv(bddl_file_name=FAR_BDDL, camera_heights=256, camera_widths=256)
    envs = [eb_env, er_env, ec_env]
    for env in envs:
        env.seed(args.seed)
    _assert_matching_joint_schema(envs)

    eb_states: list[np.ndarray] = []
    er_states: list[np.ndarray] = []
    ec_states: list[np.ndarray] = []
    attempts = 0
    max_attempts = args.num_states * args.max_attempt_factor

    try:
        while len(eb_states) < args.num_states:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Generated only {len(eb_states)}/{args.num_states} paired states after "
                    f"{max_attempts} attempts; tolerance={args.pair_tolerance}"
                )

            eb_env.reset()
            _set_stove_state(eb_env, "off")
            _settle(eb_env)
            eb_env.sim.data.qvel[:] = 0.0
            eb_env.sim.forward()
            if not np.isfinite(eb_env.sim.get_state().flatten()).all():
                continue
            reference_positions = _positions(eb_env)

            candidate_states = []
            valid = True
            for env, stove_state in ((er_env, "on"), (ec_env, "on")):
                env.reset()
                _copy_dynamic_state(eb_env, env)
                _align_movable_world_positions(reference_positions, env)
                immediate_delta, immediate_body = _max_pair_delta(reference_positions, env)
                if immediate_delta > 1e-4:
                    print(
                        f"[skip attempt {attempts}] immediate alignment delta "
                        f"{immediate_delta:.6f} m for {immediate_body}"
                    )
                    valid = False
                    break
                _set_stove_state(env, stove_state)
                # Do not call raw sim.step() after copying qpos. MuJoCo state does
                # not include the robosuite controller/mocap target, so an
                # uncontrolled settle can move the arm and disturb objects. The
                # evaluator performs its normal dummy-action wait after loading
                # this exact paired initial state.
                env.sim.data.qvel[:] = 0.0
                env.sim.forward()
                flat_state = env.sim.get_state().flatten()
                if not np.isfinite(flat_state).all():
                    valid = False
                    break
                delta, body = _max_pair_delta(reference_positions, env)
                if delta > args.pair_tolerance:
                    print(
                        f"[skip attempt {attempts}] pair delta {delta:.4f} m for {body} "
                        f"> {args.pair_tolerance:.4f} m"
                    )
                    valid = False
                    break
                candidate_states.append(flat_state.copy())

            if not valid:
                continue

            eb_states.append(eb_env.sim.get_state().flatten().copy())
            er_states.append(candidate_states[0])
            ec_states.append(candidate_states[1])
            print(f"[{len(eb_states)}/{args.num_states}] accepted paired state (attempt {attempts})")
    finally:
        for env in envs:
            env.close()

    save_hdf5(eb_states, TASK_DESCRIPTION, args.eb_output)
    save_hdf5(er_states, TASK_DESCRIPTION, args.er_output)
    save_hdf5(ec_states, TASK_DESCRIPTION, args.ec_output)
    print(f"Pairing tolerance: {args.pair_tolerance:.4f} m across {len(MOVABLE_BODIES)} bodies")


if __name__ == "__main__":
    main()
