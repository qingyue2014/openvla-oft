"""Post-success settling shared by LIBERO evaluation runners.

The environment can report success while an object is still held inside the
goal region.  A settling window therefore has two responsibilities: record the
states that a reviewer needs to see, and replace the transient success value
with the task value observed after the final settling step.
"""

from collections.abc import Callable
from typing import Any


def settle_after_success(
    env,
    *,
    initial_obs: Any,
    dummy_action,
    num_steps: int,
    start_step: int,
    initial_success: bool = True,
    recorder=None,
    capture_observation: Callable[[Any], None] | None = None,
    check_safety: Callable[[Any, Any, int], bool] | None = None,
    success_after_step: Callable[[bool], bool] | None = None,
) -> tuple[Any, bool]:
    """Settle a successful episode and return its final task-success value.

    ``success_after_step`` receives the latest environment ``done`` value.  A
    native LIBERO task normally uses that value directly; an oracle-defined
    task can instead re-query its own success predicate after each step.
    The initial observation is the state immediately after the success-causing
    action.  It and every post-``env.step`` settling observation are captured,
    so videos do not end on a pre-action frame.
    """

    obs = initial_obs
    success = bool(initial_success)
    if capture_observation is not None:
        capture_observation(obs)
    for offset in range(num_steps):
        step = start_step + offset
        obs, _reward, done, _info = env.step(dummy_action)
        if recorder is not None:
            recorder.record(obs, dummy_action, step, phase="settle")
        if capture_observation is not None:
            capture_observation(obs)
        success = bool(
            success_after_step(bool(done))
            if success_after_step is not None
            else done
        )
        if check_safety is not None and check_safety(obs, dummy_action, step):
            break
    return obs, success
