"""Shared reset/restore sequence used by formal LIBERO evaluation gates."""

from __future__ import annotations


def restore_formal_observation(env, initial_state=None):
    """Restore a state and force the observation cache to match MuJoCo state.

    The returned observation is the input to the evaluator's controller no-op
    window.  The observation returned by the final no-op step is therefore the
    exact first policy observation.
    """

    observation = env.reset()
    if initial_state is not None:
        observation = env.set_init_state(initial_state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    raw_env = getattr(env, "env", env)
    observation = raw_env._get_observations()
    return observation
