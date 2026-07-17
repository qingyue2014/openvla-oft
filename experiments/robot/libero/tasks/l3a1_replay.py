"""Deterministic MuJoCo state replay helpers for L3-A1.

MuJoCo's flattened ``MjSimState`` does not contain solver warm-start or
externally applied-force buffers.  A near-critical support contact can
therefore evolve differently depending on the reset that happened immediately
before ``set_init_state``.  Clear only those non-serialized transient buffers
after loading an L3-A1 state so generation, reference validation, and policy
evaluation start from the same physical state.
"""


def clear_mujoco_replay_transients(env) -> None:
    data = env.sim.data
    for name in ("qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        buffer = getattr(data, name, None)
        if buffer is not None:
            buffer[...] = 0
