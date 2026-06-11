"""Isolate the episode-2 SIGABRT without loading the VLA model.

Cycles a BDDL env through reset + dummy-action episodes, optionally writing a
rollout video between episodes exactly like the eval script does. Run the
matrix:

  # (a) env-only: reset/step/render cycling
  python experiments/robot/libero/tasks/debug_bddl_episode_cycle.py --save_video 0

  # (b) env + video writer between episodes
  python experiments/robot/libero/tasks/debug_bddl_episode_cycle.py --save_video 1

If (a) crashes at episode 2 the env reset itself is broken; if only (b)
crashes, the in-process video encoder corrupts the renderer.
"""

import argparse
import faulthandler
import sys
from pathlib import Path

faulthandler.enable()

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.run_physcog_libero_l1_eval import _ensure_libero_importable  # noqa: E402

_ensure_libero_importable()

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_image,
    save_rollout_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl",
        default="experiments/robot/libero/tasks/PHYSCOG_L1B2_corridor_carry.bddl",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--save_video", type=int, default=1)
    args = parser.parse_args()

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        hard_reset=False,
    )
    env.seed(7)

    for episode in range(args.episodes):
        print(f"[episode {episode}] reset", flush=True)
        obs = env.reset()
        frames = []
        for t in range(args.steps):
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            frames.append(get_libero_image(obs))
            if t % 20 == 0:
                print(f"[episode {episode}] step {t} ok", flush=True)
        if args.save_video:
            path = save_rollout_video(
                frames,
                episode,
                success=False,
                task_description=f"debug episode cycle {episode}",
                rollout_dir="./rollouts/debug_episode_cycle",
            )
            print(f"[episode {episode}] video written: {path}", flush=True)
    env.close()
    print("ALL EPISODES COMPLETED WITHOUT CRASH", flush=True)


if __name__ == "__main__":
    main()
