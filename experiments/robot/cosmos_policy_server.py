"""Run official Cosmos Policy inference outside the LIBERO simulator process."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from experiments.robot.cosmos_policy_utils import (
    COSMOS_DEFAULT_CHECKPOINT,
    CosmosPolicy,
    serve_cosmos_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", default=str(COSMOS_DEFAULT_CHECKPOINT))
    parser.add_argument("--num-open-loop-steps", type=int, default=16)
    parser.add_argument("--num-denoising-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-suite-name", default="libero_object")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Cosmos policy server must bind to loopback only")
    if args.port <= 0:
        raise SystemExit("--port must be positive")
    policy = CosmosPolicy(
        SimpleNamespace(
            pretrained_checkpoint=args.checkpoint,
            num_open_loop_steps=args.num_open_loop_steps,
            cosmos_num_denoising_steps=args.num_denoising_steps,
            seed=args.seed,
            task_suite_name=args.task_suite_name,
        )
    )
    serve_cosmos_policy(policy, args.host, args.port)


if __name__ == "__main__":
    main()
