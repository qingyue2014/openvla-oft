"""Exact first-policy-state runtime gate for L3-B3.

This gate runs inside the formal evaluator.  It observes the state restored
from the bound HDF5 episode before the controller wait and after every one of
the evaluator's ten no-op steps.  A failure raises and invalidates the job; it
is never converted into an ordinary policy failure.
"""

from __future__ import annotations

import json

from experiments.robot.libero.tasks.generate_l3b3_microwave_precondition_states import (
    _gate_failures,
)
from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    CONDITIONS,
    DESIGN_VERSION,
    EXPECTED_INITIAL_PREDICATES,
    FORMAL_WAIT_STEPS,
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_PROMPT,
    scene_measurement,
    validate_runtime_inventory,
)


class MicrowavePreconditionRuntimeGateError(RuntimeError):
    """Raised when the exact evaluator state violates the L3-B3 contract."""


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


class MicrowavePreconditionRuntimeGate:
    def __init__(self, env, state_record: dict[str, object]):
        self.env = env
        self.state_record = dict(state_record)
        self.condition = str(_decode(self.state_record.get("condition", "")))
        if self.condition not in CONDITIONS:
            raise MicrowavePreconditionRuntimeGateError(
                f"invalid L3-B3 HDF condition: {self.condition!r}"
            )
        exact = {
            "design_version": DESIGN_VERSION,
            "task_id": TASK_ID,
        }
        for name, expected in exact.items():
            if int(self.state_record.get(name, -1)) != int(expected):
                raise MicrowavePreconditionRuntimeGateError(
                    f"L3-B3 episode {name} mismatch"
                )
        for name, expected in {
            "scenario": SCENE_ID,
            "task_suite_name": SUITE,
            "task_prompt": TASK_PROMPT,
        }.items():
            if str(_decode(self.state_record.get(name, ""))) != expected:
                raise MicrowavePreconditionRuntimeGateError(
                    f"L3-B3 episode {name} mismatch"
                )
        if bool(self.state_record.get("physical_gate_pass", True)) is not True:
            raise MicrowavePreconditionRuntimeGateError(
                "L3-B3 serialized episode was not construction-gate qualified"
            )
        for flag in ("asset_inventory_changed", "prompt_changed", "bddl_changed"):
            if bool(self.state_record.get(flag, True)):
                raise MicrowavePreconditionRuntimeGateError(
                    f"L3-B3 episode has prohibited {flag}=True"
                )
        try:
            self.runtime_inventory = validate_runtime_inventory(env.sim.model)
            self.samples = [scene_measurement(env)]
        except Exception as exc:
            raise MicrowavePreconditionRuntimeGateError(str(exc)) from exc
        self.finalized = False

    def observe(self) -> None:
        if self.finalized:
            raise MicrowavePreconditionRuntimeGateError(
                "L3-B3 runtime gate observed after finalization"
            )
        self.samples.append(scene_measurement(self.env))
        if len(self.samples) > FORMAL_WAIT_STEPS + 1:
            raise MicrowavePreconditionRuntimeGateError(
                "L3-B3 evaluator exceeded the preregistered wait length"
            )

    def finalize(self) -> dict[str, object]:
        if len(self.samples) != FORMAL_WAIT_STEPS + 1:
            raise MicrowavePreconditionRuntimeGateError(
                f"L3-B3 expected {FORMAL_WAIT_STEPS} wait steps, observed "
                f"{len(self.samples) - 1}"
            )
        failures, stats = _gate_failures(self.samples, self.condition)
        first_policy = self.samples[-1]
        expected = EXPECTED_INITIAL_PREDICATES[self.condition]
        if first_policy["predicates"] != expected:
            failures.append("first_policy_predicate_mismatch")
        if failures:
            raise MicrowavePreconditionRuntimeGateError(
                "L3-B3 exact first-policy gate failed: "
                + json.dumps(sorted(set(failures)))
            )
        self.finalized = True
        return {
            "scenario": SCENE_ID,
            "condition": self.condition,
            "condition_label": _decode(
                self.state_record.get("condition_label", "")
            ),
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "native_init_state_index": int(
                self.state_record.get("native_init_state_index", -1)
            ),
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "pre_wait": self.samples[0],
            "first_policy": first_policy,
            "formal_window_stats": stats,
            "physical_gate_pass": True,
            "failures": [],
        }
