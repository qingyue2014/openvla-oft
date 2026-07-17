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


L3A1_BOTTLE_FRICTION = (2.0, 0.10, 0.01)


def configure_l3a1_contact_model(env, bottle_body: str = "wine_bottle_1_main") -> None:
    """Give the support-dependent bottle a reproducible no-slip contact.

    The native wine-bottle asset is tuned as a free tabletop distractor.  At
    the near-critical L3-A1 lean its low contact friction lets solver noise
    slide it off the drawer before the policy acts.  Increasing only the
    bottle geoms' friction stabilizes the intended static support contact; it
    does not constrain the bottle or create a new support.
    """
    model = env.sim.model
    body_id = model.body_name2id(bottle_body)
    geom_ids = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == int(body_id)
    ]
    if not geom_ids:
        raise ValueError(f"no MuJoCo geoms found for L3-A1 bottle {bottle_body!r}")
    model.geom_friction[geom_ids, :] = L3A1_BOTTLE_FRICTION
