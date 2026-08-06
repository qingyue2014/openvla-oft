#!/usr/bin/env python3
"""Local inference server for the pinned NVIDIA Cosmos Policy checkpoint."""

from __future__ import annotations

import argparse
import traceback
from multiprocessing.connection import Listener
from types import SimpleNamespace
from typing import Any, Mapping

from experiments.robot.cosmos_policy_utils import (
    COSMOS_SERVER_PROTOCOL,
    CosmosPolicy,
)


def _response(*, result: Mapping[str, Any] | None = None, error: str = ""):
    return {
        "ok": not error,
        "result": {} if result is None else dict(result),
        "error": error,
    }


def serve(args: argparse.Namespace) -> None:
    cfg = SimpleNamespace(
        pretrained_checkpoint=args.pretrained_checkpoint,
        cosmos_tokenizer_path=args.cosmos_tokenizer_path,
        cosmos_num_denoising_steps=args.cosmos_num_denoising_steps,
        num_open_loop_steps=16,
        seed=args.seed,
        task_suite_name=args.task_suite_name,
    )
    policy = CosmosPolicy(cfg)
    address = (args.host, args.port)
    listener = Listener(address, authkey=args.authkey.encode())
    print(
        "PASS_L1C1_COSMOS_SERVER_READY "
        f"host={args.host} port={args.port} protocol={COSMOS_SERVER_PROTOCOL}",
        flush=True,
    )
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    if not isinstance(request, Mapping):
                        connection.send(_response(error="request must be a mapping"))
                        continue
                    operation = request.get("op")
                    try:
                        if operation == "metadata":
                            result = {
                                "protocol": COSMOS_SERVER_PROTOCOL,
                                "model": "Cosmos-Policy-LIBERO-Predict2-2B",
                                "action_chunk": 16,
                            }
                        elif operation == "infer":
                            actions = policy.infer(
                                request["observation"], request["task_label"]
                            )
                            result = {"actions": actions}
                        elif operation == "reset":
                            policy.reset()
                            result = {"reset": True}
                        else:
                            raise ValueError(f"unsupported operation: {operation!r}")
                    except Exception as exc:
                        traceback.print_exc()
                        connection.send(_response(error=f"{type(exc).__name__}: {exc}"))
                    else:
                        connection.send(_response(result=result))
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--authkey", default="l1c1-cosmos-local")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--cosmos_tokenizer_path", required=True)
    parser.add_argument("--cosmos_num_denoising_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
