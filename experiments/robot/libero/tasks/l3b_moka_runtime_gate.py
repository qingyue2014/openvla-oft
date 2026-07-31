"""Exact pre-policy physical gate for L3-B moka evaluator episodes."""

from __future__ import annotations

import json

import numpy as np

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITIONS,
    CONDITION_LABEL,
    CONDITION_INTERVENTION_BODY,
    DESIGN_VERSION,
    FORMAL_WAIT_STEPS,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_PLACED_WINDOW_TRANSLATION_M,
    MAX_RECEPTACLE_TILT_DEG,
    POT_1,
    POT_2,
    POT_BODIES,
    STOVE_BODY,
    TABLE_BODY,
    measurement,
    window_stats,
)


TRACKED_BODIES = (*POT_BODIES, STOVE_BODY)


class MokaOrderRuntimeGateError(RuntimeError):
    """Raised when the exact evaluator first-policy state is invalid."""


def _json_value(record: dict, name: str):
    value = record.get(name)
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise MokaOrderRuntimeGateError(f"{name} is missing or not JSON")
    return json.loads(value)


def _support_ok(measure: dict, expected: str) -> bool:
    contacts = [str(value) for value in measure["contacts"]]
    if expected == "stove":
        return any(value.startswith("flat_stove_1_") for value in contacts)
    return TABLE_BODY in contacts


class MokaOrderRuntimeGate:
    """Collect and validate the actual evaluator stabilization window."""

    def __init__(self, env, state_record: dict):
        condition = state_record.get("condition")
        if isinstance(condition, bytes):
            condition = condition.decode()
        if condition not in CONDITIONS:
            raise MokaOrderRuntimeGateError(
                f"invalid serialized L3-B condition: {condition!r}"
            )
        if int(state_record.get("design_version", -1)) != DESIGN_VERSION:
            raise MokaOrderRuntimeGateError(
                "serialized L3-B state does not use design version 2"
            )
        condition_label = state_record.get("condition_label")
        if isinstance(condition_label, bytes):
            condition_label = condition_label.decode()
        if condition_label != CONDITION_LABEL[condition]:
            raise MokaOrderRuntimeGateError(
                "serialized L3-B external condition label mismatch"
            )
        self.env = env
        self.condition = condition
        self.finalized = False
        self.metrics: dict = {}
        names = _json_value(state_record, "fixture_replay_bodies_json")
        positions = np.asarray(
            state_record.get("fixture_replay_positions"), dtype=float
        )
        quaternions = np.asarray(
            state_record.get("fixture_replay_quaternions"), dtype=float
        )
        if positions.shape != (len(names), 3) or quaternions.shape != (
            len(names),
            4,
        ):
            raise MokaOrderRuntimeGateError("fixture replay pose shape mismatch")
        for index, body_name in enumerate(names):
            body_id = int(env.sim.model.body_name2id(body_name))
            if not np.allclose(
                env.sim.model.body_pos[body_id],
                positions[index],
                atol=1e-12,
                rtol=0.0,
            ) or not np.allclose(
                env.sim.model.body_quat[body_id],
                quaternions[index],
                atol=1e-12,
                rtol=0.0,
            ):
                raise MokaOrderRuntimeGateError(
                    f"fixture replay mismatch for {body_name}"
                )
        self.samples = [
            {body: measurement(env, body) for body in TRACKED_BODIES}
        ]

    def observe(self) -> None:
        if self.finalized:
            raise MokaOrderRuntimeGateError("runtime gate observed after finalize")
        self.samples.append(
            {
                body: measurement(self.env, body)
                for body in TRACKED_BODIES
            }
        )

    def finalize(self) -> dict:
        if self.finalized:
            return self.metrics
        if len(self.samples) != FORMAL_WAIT_STEPS + 1:
            raise MokaOrderRuntimeGateError(
                "formal evaluator wait mismatch: "
                f"observed {len(self.samples) - 1}, expected {FORMAL_WAIT_STEPS}"
            )
        stats = {
            body: window_stats(self.samples, body)
            for body in TRACKED_BODIES
        }
        first = self.samples[-1]
        failures = []
        placed_body = CONDITION_INTERVENTION_BODY[self.condition]
        for body in POT_BODIES:
            placed = body == placed_body
            body_stats = stats[body]
            translation_limit = (
                MAX_PLACED_WINDOW_TRANSLATION_M
                if placed
                else MAX_NATIVE_WINDOW_TRANSLATION_M
            )
            linear_limit = (
                MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS
                if placed
                else MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
            )
            angular_limit = (
                MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS
                if placed
                else MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
            )
            if body_stats["max_translation_drift_m"] > translation_limit:
                failures.append(f"{body}:translation")
            if body_stats["max_tilt_deg"] > MAX_RECEPTACLE_TILT_DEG:
                failures.append(f"{body}:tilt")
            if body_stats["max_linear_speed_mps"] > linear_limit:
                failures.append(f"{body}:linear_speed")
            if body_stats["max_angular_speed_radps"] > angular_limit:
                failures.append(f"{body}:angular_speed")
            if first[body]["linear_speed_mps"] > MAX_FINAL_LINEAR_SPEED_MPS:
                failures.append(f"{body}:first_policy_linear_speed")
            if first[body]["angular_speed_radps"] > MAX_FINAL_ANGULAR_SPEED_RADPS:
                failures.append(f"{body}:first_policy_angular_speed")
            expected_support = "stove" if placed else "table"
            if not _support_ok(first[body], expected_support):
                failures.append(f"{body}:missing_{expected_support}_support")
            other = POT_2 if body == POT_1 else POT_1
            other_prefix = other.removesuffix("_main")
            if any(
                str(contact).startswith(other_prefix)
                for contact in first[body]["contacts"]
            ):
                failures.append(f"{body}:initial_pot_contact")
        if stats[STOVE_BODY]["max_translation_drift_m"] > 1e-6:
            failures.append(f"{STOVE_BODY}:translation")
        if stats[STOVE_BODY]["max_tilt_deg"] > MAX_RECEPTACLE_TILT_DEG:
            failures.append(f"{STOVE_BODY}:tilt")
        if bool(self.env.check_success()):
            failures.append("partial_or_native_state_already_satisfies_goal")

        self.metrics = {
            "condition": self.condition,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "pre_wait": self.samples[0],
            "first_policy": first,
            "formal_window_stats": stats,
            "physical_gate_pass": not failures,
            "failures": sorted(set(failures)),
        }
        if failures:
            raise MokaOrderRuntimeGateError(
                "L3-B exact first-policy physical gate failed: "
                + ", ".join(sorted(set(failures)))
            )
        self.finalized = True
        return self.metrics
