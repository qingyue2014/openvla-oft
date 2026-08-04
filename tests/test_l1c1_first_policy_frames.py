import numpy as np

from experiments.robot.libero.formal_evaluator_state import restore_formal_observation
from experiments.robot.libero.tasks.validate_l1c1_first_policy_frames import (
    CONFIRM_STEPS,
    FORMAL_WAIT_STEPS,
    TRACKED_BODIES,
    _center_crop,
    _evaluate_trace,
)


def _trace():
    sample = {
        body: {
            "position": [0.0, 0.0, 0.9],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "tilt_deg": 0.2,
            "linear_speed_mps": 0.0,
            "angular_speed_radps": 0.0,
            "contacts": ["table"],
            "expected_support": "table",
            "support_present": True,
            "forbidden_contacts": [],
        }
        for body in TRACKED_BODIES
    }
    return [
        {body: dict(measurement) for body, measurement in sample.items()}
        for _ in range(FORMAL_WAIT_STEPS + CONFIRM_STEPS + 1)
    ]


def test_trace_passes_only_when_full_wait_and_confirmation_are_stable():
    valid, failures, summary = _evaluate_trace(_trace())
    assert valid
    assert failures == []
    assert summary[TRACKED_BODIES[0]]["first_policy"]["tilt_deg"] == 0.2


def test_in_place_tipping_after_first_policy_frame_fails_receptacle_gate():
    trace = _trace()
    trace[FORMAL_WAIT_STEPS + 1]["plate_1_main"]["tilt_deg"] = 1.01
    valid, failures, _ = _evaluate_trace(trace)
    assert not valid
    assert "plate_1_main:tilt" in failures


def test_first_policy_velocity_and_forbidden_contact_fail_closed():
    trace = _trace()
    first = trace[FORMAL_WAIT_STEPS]["akita_black_bowl_1_main"]
    first["linear_speed_mps"] = 0.011
    first["forbidden_contacts"] = ["robot"]
    valid, failures, _ = _evaluate_trace(trace)
    assert not valid
    assert "akita_black_bowl_1_main:first_policy_linear_speed" in failures
    assert "akita_black_bowl_1_main:forbidden_contact" in failures


def test_center_crop_uses_openvla_point_nine_area_geometry():
    image = np.zeros((224, 224), dtype=np.uint8)
    cropped = _center_crop(image)
    assert cropped.shape == (213, 213)


def test_shared_restore_forces_forward_and_fresh_observation():
    calls = []

    class Sim:
        def forward(self):
            calls.append("forward")

    class Env:
        sim = Sim()
        env = None

        def __init__(self):
            self.env = self

        def reset(self):
            calls.append("reset")
            return {"stale": "reset"}

        def set_init_state(self, state):
            calls.append(("set", state))
            return {"stale": "set"}

        def _post_process(self):
            calls.append("post")

        def _update_observables(self, force=False):
            calls.append(("update", force))

        def _get_observations(self):
            calls.append("observe")
            return {"fresh": True}

    observation = restore_formal_observation(Env(), initial_state="frozen")
    assert observation == {"fresh": True}
    assert calls == [
        "reset",
        ("set", "frozen"),
        "forward",
        "post",
        ("update", True),
        "observe",
    ]
