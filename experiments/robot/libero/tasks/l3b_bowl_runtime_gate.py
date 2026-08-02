"""Exact evaluator-wait gate and predicate-order tracker for L3-B bowl."""

from __future__ import annotations

import json

import numpy as np

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    BOTTLE_BODY,
    BOWL_BODY,
    CONDITIONS,
    CONDITION_LABEL,
    DESIGN_VERSION,
    DRAWER_BODY,
    EXPECTED_INITIAL_PREDICATES,
    FORMAL_WAIT_STEPS,
    MAX_BOTTLE_TILT_DEG,
    MAX_BOWL_TILT_DEG,
    MAX_DRAWER_FINAL_SPEED,
    MAX_DRAWER_CABINET_PENETRATION_M,
    MAX_DRAWER_WINDOW_QPOS_DRIFT,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_PLACED_WINDOW_TRANSLATION_M,
    TABLE_BODY,
    body_window_stats,
    drawer_window_stats,
    predicate_state,
    scene_measurement,
)


class BowlOrderRuntimeGateError(RuntimeError):
    """An invalid exact evaluator state must invalidate the whole job."""


def _json_value(record: dict, name: str):
    value = record.get(name)
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise BowlOrderRuntimeGateError(f"{name} is missing or not JSON")
    return json.loads(value)


def _support_ok(measure: dict, expected: str) -> bool:
    contacts = [str(value) for value in measure.get("contacts", [])]
    if expected == "table":
        return TABLE_BODY in contacts
    return any(value.startswith("white_cabinet_1_cabinet_bottom") for value in contacts)


def _window_failures(samples: list[dict], condition: str) -> tuple[list[str], dict]:
    stats = {
        BOWL_BODY: body_window_stats(samples, BOWL_BODY),
        BOTTLE_BODY: body_window_stats(samples, BOTTLE_BODY),
        DRAWER_BODY: body_window_stats(samples, DRAWER_BODY),
        "drawer_joint": drawer_window_stats(samples),
    }
    first = samples[-1]
    failures = []
    bowl_placed = condition == "prerequisite_done"
    bowl_stats = stats[BOWL_BODY]
    bottle_stats = stats[BOTTLE_BODY]
    if bowl_stats["max_tilt_deg"] > MAX_BOWL_TILT_DEG:
        failures.append("bowl:tilt")
    if bottle_stats["max_tilt_deg"] > MAX_BOTTLE_TILT_DEG:
        failures.append("bottle:tilt")
    if bowl_stats["max_translation_drift_m"] > (
        MAX_PLACED_WINDOW_TRANSLATION_M if bowl_placed else MAX_NATIVE_WINDOW_TRANSLATION_M
    ):
        failures.append("bowl:translation")
    if bowl_stats["max_linear_speed_mps"] > (
        MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS
        if bowl_placed
        else MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
    ):
        failures.append("bowl:linear_speed")
    if bowl_stats["max_angular_speed_radps"] > (
        MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS
        if bowl_placed
        else MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
    ):
        failures.append("bowl:angular_speed")
    if bottle_stats["max_translation_drift_m"] > MAX_NATIVE_WINDOW_TRANSLATION_M:
        failures.append("bottle:translation")
    if bottle_stats["max_linear_speed_mps"] > MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS:
        failures.append("bottle:linear_speed")
    if bottle_stats["max_angular_speed_radps"] > MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS:
        failures.append("bottle:angular_speed")
    for body in (BOWL_BODY, BOTTLE_BODY):
        if first[body]["linear_speed_mps"] > MAX_FINAL_LINEAR_SPEED_MPS:
            failures.append(f"{body}:first_policy_linear_speed")
        if first[body]["angular_speed_radps"] > MAX_FINAL_ANGULAR_SPEED_RADPS:
            failures.append(f"{body}:first_policy_angular_speed")
    if not _support_ok(first[BOWL_BODY], "drawer" if bowl_placed else "table"):
        failures.append("bowl:missing_expected_support")
    if not _support_ok(first[BOTTLE_BODY], "table"):
        failures.append("bottle:missing_table_support")
    if any(
        str(name).startswith(("robot0_", "wine_bottle_1"))
        for name in first[BOWL_BODY].get("contacts", [])
    ):
        failures.append("bowl:forbidden_contact")
    if any(
        str(name).startswith(("robot0_", "akita_black_bowl_1"))
        for name in first[BOTTLE_BODY].get("contacts", [])
    ):
        failures.append("bottle:forbidden_contact")
    if stats["drawer_joint"]["max_qpos_drift"] > MAX_DRAWER_WINDOW_QPOS_DRIFT:
        failures.append("drawer:qpos_drift")
    if first["drawer_joint"]["speed"] > MAX_DRAWER_FINAL_SPEED:
        failures.append("drawer:first_policy_speed")
    if any(
        float(contact["distance_m"]) < -MAX_DRAWER_CABINET_PENETRATION_M
        for sample in samples
        for contact in sample["drawer_cabinet_self_contacts"]
    ):
        failures.append("drawer:cabinet_self_contact")
    expected = EXPECTED_INITIAL_PREDICATES[condition]
    if any(sample["predicates"] != expected for sample in samples):
        failures.append("predicate:formal_window_mismatch")
    return sorted(set(failures)), stats


class BowlOrderRuntimeGate:
    """Collect and validate the actual evaluator stabilization samples."""

    def __init__(self, env, state_record: dict):
        condition = state_record.get("condition")
        if isinstance(condition, bytes):
            condition = condition.decode()
        if condition not in CONDITIONS:
            raise BowlOrderRuntimeGateError(f"invalid L3-B bowl condition: {condition!r}")
        if int(state_record.get("design_version", -1)) != DESIGN_VERSION:
            raise BowlOrderRuntimeGateError("serialized state design version mismatch")
        label = state_record.get("condition_label")
        if isinstance(label, bytes):
            label = label.decode()
        if label != CONDITION_LABEL[condition]:
            raise BowlOrderRuntimeGateError("serialized external condition label mismatch")
        self.env = env
        self.condition = condition
        self.native_init_state_index = int(state_record.get("native_init_state_index", -1))
        if self.native_init_state_index < 0:
            raise BowlOrderRuntimeGateError("official native state index is missing")
        names = _json_value(state_record, "fixture_replay_bodies_json")
        positions = np.asarray(state_record.get("fixture_replay_positions"), dtype=float)
        quaternions = np.asarray(state_record.get("fixture_replay_quaternions"), dtype=float)
        if positions.shape != (len(names), 3) or quaternions.shape != (len(names), 4):
            raise BowlOrderRuntimeGateError("fixture replay pose shape mismatch")
        for index, body_name in enumerate(names):
            body_id = int(env.sim.model.body_name2id(body_name))
            if not np.allclose(env.sim.model.body_pos[body_id], positions[index], atol=1e-12, rtol=0.0):
                raise BowlOrderRuntimeGateError(f"fixture position mismatch: {body_name}")
            if not np.allclose(env.sim.model.body_quat[body_id], quaternions[index], atol=1e-12, rtol=0.0):
                raise BowlOrderRuntimeGateError(f"fixture quaternion mismatch: {body_name}")
        self.samples = [scene_measurement(env)]
        self.finalized = False
        self.metrics: dict = {}

    def observe(self) -> None:
        if self.finalized:
            raise BowlOrderRuntimeGateError("runtime gate observed after finalize")
        self.samples.append(scene_measurement(self.env))

    def finalize(self) -> dict:
        if self.finalized:
            return self.metrics
        if len(self.samples) != FORMAL_WAIT_STEPS + 1:
            raise BowlOrderRuntimeGateError(
                f"formal wait mismatch: {len(self.samples)-1} != {FORMAL_WAIT_STEPS}"
            )
        failures, stats = _window_failures(self.samples, self.condition)
        if bool(self.env.check_success()):
            failures.append("partial_state_already_satisfies_native_goal")
        self.metrics = {
            "condition": self.condition,
            "condition_label": CONDITION_LABEL[self.condition],
            "native_init_state_index": self.native_init_state_index,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "pre_wait": self.samples[0],
            "first_policy": self.samples[-1],
            "formal_window_stats": stats,
            "physical_gate_pass": not failures,
            "failures": sorted(set(failures)),
        }
        if failures:
            raise BowlOrderRuntimeGateError(
                "L3-B bowl first-policy gate failed: " + ", ".join(sorted(set(failures)))
            )
        self.finalized = True
        return self.metrics


class BowlOrderSequenceTracker:
    """Event-based diagnostic; it never defines native task success."""

    def __init__(self, env, condition: str, *, policy_start_step: int):
        if condition not in CONDITIONS:
            raise BowlOrderRuntimeGateError(f"invalid tracker condition: {condition}")
        self.env = env
        self.condition = condition
        self.policy_start_step = int(policy_start_step)
        self.initial = predicate_state(env)
        expected = EXPECTED_INITIAL_PREDICATES[condition]
        if self.initial != expected:
            raise BowlOrderRuntimeGateError(
                f"tracker initial predicates {self.initial} != {expected}"
            )
        self.previous = dict(self.initial)
        self.transitions: list[dict] = []
        self.rollback_step: int | None = None
        self.insertion_step: int | None = None
        self.reclose_step: int | None = None

    def observe(self, step: int) -> None:
        current = predicate_state(self.env)
        for name in ("close", "in"):
            if current[name] != self.previous[name]:
                self.transitions.append(
                    {
                        "step": int(step),
                        "predicate": name,
                        "from": bool(self.previous[name]),
                        "to": bool(current[name]),
                    }
                )
        if (
            self.condition == "premature_close"
            and self.rollback_step is None
            and self.previous["close"]
            and not current["close"]
        ):
            self.rollback_step = int(step)
        if (
            self.insertion_step is None
            and not self.previous["in"]
            and current["in"]
        ):
            self.insertion_step = int(step)
        if (
            self.condition == "premature_close"
            and self.rollback_step is not None
            and self.insertion_step is not None
            and self.reclose_step is None
            and not self.previous["close"]
            and current["close"]
            and int(step) >= self.insertion_step
        ):
            self.reclose_step = int(step)
        self.previous = current

    def finalize(self, *, task_success: bool, final_step: int) -> dict:
        final = predicate_state(self.env)
        if final != self.previous:
            self.observe(final_step)
            final = dict(self.previous)
        rollback = self.rollback_step is not None
        insertion_after_rollback = bool(
            rollback
            and self.insertion_step is not None
            and self.insertion_step >= self.rollback_step
        )
        reclose_after_insertion = bool(
            insertion_after_rollback
            and self.reclose_step is not None
            and self.reclose_step >= self.insertion_step
        )
        full_repair = bool(task_success and reclose_after_insertion)
        if self.condition != "premature_close":
            failure_stage = "not_applicable"
        elif full_repair:
            failure_stage = "full_ordered_repair"
        elif not rollback:
            failure_stage = "no_rollback"
        elif not insertion_after_rollback:
            failure_stage = "rollback_without_insertion"
        elif not reclose_after_insertion:
            failure_stage = "insertion_without_reclose"
        else:
            failure_stage = "ordered_trace_without_native_success"
        eb_in_before_close = bool(
            self.condition == "native"
            and self.insertion_step is not None
            and any(
                item["predicate"] == "close"
                and item["to"] is True
                and item["step"] >= self.insertion_step
                for item in self.transitions
            )
        )
        return {
            "condition": self.condition,
            "condition_label": CONDITION_LABEL[self.condition],
            "policy_start_step": self.policy_start_step,
            "initial_predicates": self.initial,
            "final_predicates": final,
            "transitions": self.transitions,
            "rollback_recognized": rollback,
            "rollback_step": self.rollback_step,
            "insertion_step": self.insertion_step,
            "insertion_after_rollback": insertion_after_rollback,
            "reclose_step": self.reclose_step,
            "reclose_after_insertion": reclose_after_insertion,
            "full_ordered_repair": full_repair,
            "failure_stage": failure_stage,
            "eb_in_before_close": eb_in_before_close,
            "native_task_success": bool(task_success),
            "defines_task_success": False,
            "collision_oracle_used": False,
        }
