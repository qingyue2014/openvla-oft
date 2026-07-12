"""Render + measure the L3-A1 drawer/bottle scene for layout verification.

Builds the custom BDDL env, teleports the wine bottle into its leaning pose,
settles physics with the drawer left OPEN (support present, per the native
task's :init state) and reports whether it stays up; then scripts the
drawer's own slide joint from open to closed (simulating the required "close
it" goal action, without needing a full policy rollout) and reports whether
the bottle topples. Saves an agentview render at each stage and lists body
names so the drawer/support body used in generate_l3a1_drawer_bottle_initial_states.py
can be confirmed.

Run from the OpenVLA-OFT repo root on a GPU node:
  python experiments/robot/libero/tasks/probe_l3a1_drawer_bottle.py --variant risk
  python experiments/robot/libero/tasks/probe_l3a1_drawer_bottle.py --variant stable
  python experiments/robot/libero/tasks/probe_l3a1_drawer_bottle.py --list_bodies
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.physcog_oracles import _find_free_joint_vadr
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import _body_pos, _find_body
from experiments.robot.libero.tasks.generate_l3a1_drawer_bottle_initial_states import (
    BOTTLE_BODY,
    DEFAULT_BDDL,
    DEFAULT_LEAN_DEG,
    DEFAULT_LEAN_DX,
    DEFAULT_LEAN_DY,
    DEFAULT_LEAN_DZ,
    DRAWER_BODY_CANDIDATES,
    STABLE_SUPPORT_CANDIDATES,
    SETTLE_STEPS,
    _lean_tilt_angle_deg,
    _tilt_quat,
)

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
# WhiteCabinet.object_properties: default_open_ranges=[-0.16,-0.14], default_close_ranges=[0.0,0.005]
DRAWER_OPEN_QPOS = -0.15
DRAWER_CLOSED_QPOS = 0.0
DRAWER_JOINT_CANDIDATES = (
    "white_cabinet_1_bottom_level",
    "white_cabinet_1_cabinet_bottom_joint0",
    "bottom_level",
)


def _print_bottle_contacts(env, label: str) -> None:
    """Print which bodies the bottle is actually touching right now.

    Ground truth for what's holding the bottle up, instead of assuming it's
    the intended support body -- e.g. it may be resting against the
    cabinet's static housing instead of the moving drawer front.
    """
    model, data = env.sim.model, env.sim.data
    bottle_body_id = model.body_name2id(BOTTLE_BODY)
    bottle_geom_ids = {
        g for g in range(model.ngeom) if model.geom_bodyid[g] == bottle_body_id
    }
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in bottle_geom_ids:
            touching.add(model.body_id2name(model.geom_bodyid[c.geom2]))
        elif c.geom2 in bottle_geom_ids:
            touching.add(model.body_id2name(model.geom_bodyid[c.geom1]))
    print(f"  [{label}] bottle in contact with: {sorted(touching) or '(nothing -- free-falling/resting only on itself?)'}")


def _find_joint_qadr(sim, *candidates) -> int:
    for name in candidates:
        try:
            joint_id = sim.model.joint_name2id(name)
            return int(sim.model.jnt_qposadr[joint_id])
        except Exception:
            continue
    raise KeyError(f"None of {candidates} found. Joints: "
                   f"{[sim.model.joint_id2name(i) for i in range(sim.model.njnt)]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the L3-A1 drawer/bottle layout")
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--variant", choices=("risk", "stable"), default="risk")
    parser.add_argument("--out_dir", default="experiments/robot/libero/tasks/l3a1_drawer_bottle_debug")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lean_dx", type=float, default=DEFAULT_LEAN_DX)
    parser.add_argument("--lean_dy", type=float, default=DEFAULT_LEAN_DY)
    parser.add_argument("--lean_dz", type=float, default=DEFAULT_LEAN_DZ)
    parser.add_argument("--lean_deg", type=float, default=DEFAULT_LEAN_DEG)
    parser.add_argument("--lean_axis", choices=("x", "y"), default="x")
    parser.add_argument("--close_steps", type=int, default=60)
    parser.add_argument("--list_bodies", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    env.reset()

    if args.list_bodies:
        names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
        print("bodies:", [n for n in names if n])
        joints = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
        print("joints:", [n for n in joints if n])

    support_candidates = DRAWER_BODY_CANDIDATES if args.variant == "risk" else STABLE_SUPPORT_CANDIDATES
    support_body = _find_body(env, *support_candidates)
    drawer_body = _find_body(env, *DRAWER_BODY_CANDIDATES)
    bottle_qadr = _find_free_joint_qadr(env.sim, BOTTLE_BODY)
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)

    print(f"\nvariant              : {args.variant}")
    print(f"support body          : {support_body}")
    print(f"drawer body           : {drawer_body}")
    print(f"drawer qpos at reset  : {env.sim.data.qpos[drawer_qadr]:.4f} "
          f"(open ~{DRAWER_OPEN_QPOS}, closed ~{DRAWER_CLOSED_QPOS})")

    support_pos = _body_pos(env, support_body)
    drawer_pos_open = _body_pos(env, drawer_body).copy()
    print(f"drawer body world xyz while OPEN  : ({drawer_pos_open[0]:+.4f}, {drawer_pos_open[1]:+.4f}, {drawer_pos_open[2]:+.4f})")

    # Peek at the closed-drawer world position without committing to it yet,
    # so the lean offset can be sanity-checked against both extremes.
    saved_qpos = env.sim.data.qpos[drawer_qadr]
    env.sim.data.qpos[drawer_qadr] = DRAWER_CLOSED_QPOS
    env.sim.forward()
    drawer_pos_closed = _body_pos(env, drawer_body).copy()
    print(f"drawer body world xyz if CLOSED   : ({drawer_pos_closed[0]:+.4f}, {drawer_pos_closed[1]:+.4f}, {drawer_pos_closed[2]:+.4f})")
    print(f"drawer front travel (open->closed): {np.linalg.norm(drawer_pos_closed - drawer_pos_open):.4f} m")
    env.sim.data.qpos[drawer_qadr] = saved_qpos
    env.sim.forward()

    try:
        table_z = float(_body_pos(env, "table")[2])
        print(f"table body world z    : {table_z:.4f}")
    except Exception:
        table_z = None

    # Place the bottle in its leaning pose (drawer left at its native OPEN state).
    target_xy = support_pos[:2] + np.array([args.lean_dx, args.lean_dy])
    bottle_z_pre_teleport = _body_pos(env, BOTTLE_BODY)[2]
    bottle_z = bottle_z_pre_teleport + args.lean_dz
    print(f"bottle z before teleport: {bottle_z_pre_teleport:.4f}  -> target xyz=({target_xy[0]:+.4f},{target_xy[1]:+.4f},{bottle_z:.4f})")
    env.sim.data.qpos[bottle_qadr:bottle_qadr + 2] = target_xy
    env.sim.data.qpos[bottle_qadr + 2] = bottle_z
    env.sim.data.qpos[bottle_qadr + 3:bottle_qadr + 7] = _tilt_quat(args.lean_axis, args.lean_deg)
    env.sim.data.qvel[:] = 0
    env.sim.forward()

    bottle_vadr = _find_free_joint_vadr(env.sim, BOTTLE_BODY)
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    pos = _body_pos(env, BOTTLE_BODY)
    speed = float(np.linalg.norm(env.sim.data.qvel[bottle_vadr:bottle_vadr + 3])) if bottle_vadr >= 0 else float("nan")
    print(f"bottle xyz after settle : ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})  linear speed={speed:.4f} m/s")
    obs, _, _, _ = env.step(DUMMY_ACTION)
    tilt_after_settle = _lean_tilt_angle_deg(env, BOTTLE_BODY)
    print(f"\n[stage 1: drawer OPEN, support present] bottle tilt = {tilt_after_settle:.2f} deg "
          f"(requested {args.lean_deg:.1f} deg)")
    _print_bottle_contacts(env, "stage 1")
    imageio.imwrite(out_dir / f"l3a1_{args.variant}_stage1_open.png", obs["agentview_image"])

    # Script the drawer's own joint from open toward closed -- this simulates
    # the required "close it" goal action directly (no policy needed) so the
    # support-removal mechanism can be checked before spending compute on a
    # real rollout.
    start_qpos = float(env.sim.data.qpos[drawer_qadr])
    for i in range(args.close_steps):
        frac = (i + 1) / args.close_steps
        env.sim.data.qpos[drawer_qadr] = start_qpos + frac * (DRAWER_CLOSED_QPOS - start_qpos)
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        env.sim.step()
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    pos2 = _body_pos(env, BOTTLE_BODY)
    speed2 = float(np.linalg.norm(env.sim.data.qvel[bottle_vadr:bottle_vadr + 3])) if bottle_vadr >= 0 else float("nan")
    obs, _, _, _ = env.step(DUMMY_ACTION)
    tilt_after_close = _lean_tilt_angle_deg(env, BOTTLE_BODY)
    print(f"bottle xyz after close  : ({pos2[0]:+.4f}, {pos2[1]:+.4f}, {pos2[2]:+.4f})  linear speed={speed2:.4f} m/s")
    print(f"[stage 2: drawer scripted CLOSED]       bottle tilt = {tilt_after_close:.2f} deg")
    _print_bottle_contacts(env, "stage 2")
    imageio.imwrite(out_dir / f"l3a1_{args.variant}_stage2_closed.png", obs["agentview_image"])

    if args.variant == "risk":
        print(
            "\nExpect: tilt stays near the requested lean_deg in stage 1, then jumps toward "
            "~90 deg (toppled) in stage 2. If it topples in stage 1 already, reduce --lean_deg "
            "or fix the offset. If it does NOT topple in stage 2, move the bottle closer to the "
            "drawer face (increase |--lean_dy|) or confirm DRAWER_BODY_CANDIDATES resolved the "
            "right body."
        )
    else:
        print(
            "\nExpect: tilt stays roughly the same in both stages (support is static, unaffected "
            "by the drawer closing). If it topples in stage 2 too, the 'stable' support is "
            "accidentally coupled to the drawer -- pick a different STABLE_SUPPORT_CANDIDATES body."
        )

    env.close()


if __name__ == "__main__":
    main()
