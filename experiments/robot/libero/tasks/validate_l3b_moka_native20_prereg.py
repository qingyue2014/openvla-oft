"""Validate and bind the locked L3-B moka native-20 capability screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITIONS,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    repository_root,
    sha256_path,
)
from experiments.robot.libero.tasks.summarize_l3b_moka_order_smoke import (
    TERMINAL_MAX_DRIFT_M,
    TERMINAL_WINDOW_STEPS,
)
from experiments.robot.libero.tasks.validate_l3b_moka_native_preflight import (
    VERDICT as PREFLIGHT_VERDICT,
)
from experiments.robot.libero.tasks.validate_l3b_moka_runtime_replay import (
    VERDICT as RUNTIME_REPLAY_VERDICT,
)
from experiments.robot.libero.tasks.validate_l3b_moka_state_bundles import (
    INITIAL_GATE_VERDICT,
    VERDICT as PAIRING_VERDICT,
)


PREREGISTRATION_ID = "l3b_moka_native20_v1"
POOL_INDICES = list(range(20))
POOL_COUNT = len(POOL_INDICES)
MINIMUM_STABLE_SUCCESSES = 12
MAX_TILT_DEG = 1.0
NATIVE_INIT_FILENAME = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.pruned_init"
ARTIFACTS = {
    "native_states": ("experiments/robot/libero/tasks/" "l3b_moka_native20_native_states.hdf5"),
    "near_states": ("experiments/robot/libero/tasks/" "l3b_moka_native20_near_first_states.hdf5"),
    "far_states": ("experiments/robot/libero/tasks/" "l3b_moka_native20_far_first_states.hdf5"),
    "review_root": "review/L3-B_moka_order_task/native20_pi05_v1",
}


def _load_json(path: str | Path) -> tuple[Path, dict]:
    resolved = Path(path).resolve(strict=True)
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def validate_spec(path: str | Path) -> dict:
    prereg_path, record = _load_json(path)
    expected_task = {
        "suite": SUITE,
        "task_id": TASK_ID,
        "bddl_file": TASK_FILE,
        "prompt": TASK_PROMPT,
        "fixtures": EXPECTED_FIXTURES,
        "objects": EXPECTED_OBJECTS,
    }
    expected_native_contract = {
        "asset_inventory_changed": False,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "safety_oracle": "none",
    }
    pool = record.get("pool", {})
    acceptance = record.get("acceptance", {})
    stable = acceptance.get("stable_success_requires", {})
    protocol = record.get("protocol", {})
    checkpoint = record.get("checkpoint", {})
    disclosure = record.get("prior_observation_disclosure", {})
    failures = []
    checks = {
        "id": record.get("preregistration_id") == PREREGISTRATION_ID,
        "status": record.get("status") == "LOCKED_BEFORE_NATIVE20_RUN",
        "scope": record.get("scope") == "native_capability_screen_only",
        "task": record.get("task") == expected_task,
        "native_contract": (record.get("native_only_contract") == expected_native_contract),
        "artifacts": record.get("artifacts") == ARTIFACTS,
        "pool_count": pool.get("count") == POOL_COUNT,
        "pool_indices": pool.get("official_state_indices") == POOL_INDICES,
        "pool_order": (pool.get("selection_rule") == "first_20_official_states_in_native_file_order"),
        "pool_file": (pool.get("official_init_state_file") == NATIVE_INIT_FILENAME),
        "no_substitution": pool.get("substitution_allowed") is False,
        "prior_indices": (disclosure.get("previously_observed_state_indices") == POOL_INDICES[:5]),
        "new_indices": (disclosure.get("newly_unobserved_state_indices") == POOL_INDICES[5:]),
        "not_blinded": (disclosure.get("independent_blinded_replication") is False),
        "threshold_count": (acceptance.get("minimum_stable_successes") == MINIMUM_STABLE_SUCCESSES),
        "threshold_rate": (acceptance.get("minimum_stable_success_rate") == 0.6),
        "native_goal": stable.get("native_libero_goal_success") is True,
        "settle_window": (stable.get("minimum_settle_samples_per_pot") == TERMINAL_WINDOW_STEPS),
        "terminal_tilt": (stable.get("both_pots_terminal_max_tilt_deg_lte") == MAX_TILT_DEG),
        "terminal_drift": (stable.get("both_pots_terminal_max_drift_m_lte") == TERMINAL_MAX_DRIFT_M),
        "near_far_gate": (acceptance.get("near_far_authorized_only_if_pass") is True),
        "checkpoint": checkpoint
        == {
            "model_family": "pi05",
            "openpi_checkpoint": ("gs://openpi-assets/checkpoints/pi05_libero"),
            "replan_steps": 5,
        },
        "protocol": protocol
        == {
            "evaluation_seed": 42,
            "formal_noop_wait_steps": 10,
            "max_saved_episode_pairs_per_outcome": 5,
            "post_success_settle_steps": 100,
            "scene_seed": 42,
        },
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError("native20 preregistration mismatch: " + ", ".join(failures))
    return {
        "scenario": SCENE_ID,
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": sha256_path(prereg_path),
        "pool_indices": POOL_INDICES,
        "minimum_stable_successes": MINIMUM_STABLE_SUCCESSES,
        "verdict": "PASS_L3B_MOKA_NATIVE20_PREREG_SPEC",
    }


def _artifact_path(spec: dict, key: str) -> Path:
    path = repository_root() / spec["artifacts"][key]
    return path.resolve(strict=True)


def _validate_state_indices(path: Path, condition: str) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError(f"{path} task key mismatch")
        group = handle[TASK_KEY]
        if group.attrs.get("condition") != condition:
            raise ValueError(f"{path} condition mismatch")
        if int(group.attrs.get("count", -1)) != POOL_COUNT:
            raise ValueError(f"{path} is not the locked 20-state pool")
        indices = [int(group[f"demo_{index}"].attrs["native_init_state_index"]) for index in range(POOL_COUNT)]
        if indices != POOL_INDICES:
            raise ValueError(f"{path} native indices are not exactly 0..19")


def validate_prepared(path: str | Path) -> dict:
    binding = validate_spec(path)
    _, spec = _load_json(path)
    review_root = _artifact_path(spec, "review_root")
    state_paths = {
        "native": _artifact_path(spec, "native_states"),
        "near_first": _artifact_path(spec, "near_states"),
        "far_first": _artifact_path(spec, "far_states"),
    }
    for condition, state_path in state_paths.items():
        _validate_state_indices(state_path, condition)

    initial_path, initial = _load_json(review_root / "L3-B_moka_initial_gate_manifest.json")
    if (
        initial.get("verdict") != INITIAL_GATE_VERDICT
        or initial.get("count") != POOL_COUNT
        or initial.get("native_suite") != SUITE
        or initial.get("native_task_id") != TASK_ID
        or initial.get("native_prompt") != TASK_PROMPT
        or Path(initial.get("native_init_states_path", "")).name != NATIVE_INIT_FILENAME
        or [episode.get("state_index") for episode in initial.get("episodes", [])] != POOL_INDICES
    ):
        raise ValueError("initial gate does not bind the locked native20 pool")

    pairing_path, pairing = _load_json(review_root / "L3-B_moka_pairing_gate.json")
    if pairing.get("verdict") != PAIRING_VERDICT or pairing.get("count") != POOL_COUNT:
        raise ValueError("pairing gate does not bind 20 states")

    runtime_path, runtime = _load_json(review_root / "L3-B_moka_runtime_replay_gate.json")
    runtime_episodes = runtime.get("episodes", [])
    if (
        runtime.get("verdict") != RUNTIME_REPLAY_VERDICT
        or runtime.get("count") != POOL_COUNT * len(CONDITIONS)
        or len(runtime_episodes) != POOL_COUNT * len(CONDITIONS)
        or any(episode.get("runtime_gate", {}).get("physical_gate_pass") is not True for episode in runtime_episodes)
    ):
        raise ValueError("runtime replay does not PASS all 60 paired states")

    preflight_paths = {}
    for condition in CONDITIONS:
        preflight_path, preflight = _load_json(review_root / f"L3-B_moka_{condition}_native_preflight.json")
        if (
            preflight.get("verdict") != PREFLIGHT_VERDICT
            or preflight.get("condition") != condition
            or preflight.get("prompt") != TASK_PROMPT
            or preflight.get("task_suite_name") != SUITE
            or preflight.get("task_id") != TASK_ID
            or Path(preflight.get("initial_states_path", "")) != state_paths[condition]
            or preflight.get("initial_states_sha256") != sha256_path(state_paths[condition])
        ):
            raise ValueError(f"{condition} native-only preflight mismatch")
        preflight_paths[condition] = str(preflight_path)

    return {
        **binding,
        "native_init_states_path": initial["native_init_states_path"],
        "native_init_states_sha256": initial["native_init_states_sha256"],
        "state_artifacts": {
            condition: {
                "path": str(state_path),
                "sha256": sha256_path(state_path),
            }
            for condition, state_path in state_paths.items()
        },
        "initial_gate_path": str(initial_path),
        "initial_gate_sha256": sha256_path(initial_path),
        "pairing_gate_path": str(pairing_path),
        "pairing_gate_sha256": sha256_path(pairing_path),
        "runtime_replay_gate_path": str(runtime_path),
        "runtime_replay_gate_sha256": sha256_path(runtime_path),
        "preflight_paths": preflight_paths,
        "verdict": "PASS_L3B_MOKA_NATIVE20_PREPARED",
    }


def validate_result(
    path: str | Path,
    report_path: str | Path,
    *,
    out_binding: str | Path | None = None,
) -> dict:
    prepared = validate_prepared(path)
    report_path, report = _load_json(report_path)
    prereg = report.get("preregistration", {})
    native = report.get("native", {})
    episodes = native.get("episodes", [])
    if report.get("verdict") not in {
        "PASS_L3B_MOKA_NATIVE_CAPABILITY_SMOKE",
        "FAIL_L3B_MOKA_NATIVE_CAPABILITY",
    }:
        raise ValueError("native20 report has no terminal capability verdict")
    if (
        report.get("scenario") != SCENE_ID
        or report.get("minimum_successes") != MINIMUM_STABLE_SUCCESSES
        or native.get("count") != POOL_COUNT
        or sorted(episode.get("episode_idx") for episode in episodes) != POOL_INDICES
        or prereg.get("preregistration_id") != PREREGISTRATION_ID
        or prereg.get("sha256") != prepared["preregistration_sha256"]
    ):
        raise ValueError("native20 result does not bind the locked screen")
    stable_successes = sum(bool(episode.get("success")) and bool(episode.get("terminal_stable")) for episode in episodes)
    if native.get("stable_successes") != stable_successes or (stable_successes >= MINIMUM_STABLE_SUCCESSES) != report[
        "verdict"
    ].startswith("PASS_"):
        raise ValueError("native20 stable-success verdict is inconsistent")

    result = {
        **prepared,
        "capability_report_path": str(report_path),
        "capability_report_sha256": sha256_path(report_path),
        "stable_successes": stable_successes,
        "stable_success_rate": stable_successes / POOL_COUNT,
        "capability_verdict": report["verdict"],
        "near_far_authorized": (stable_successes >= MINIMUM_STABLE_SUCCESSES),
        "verdict": "PASS_L3B_MOKA_NATIVE20_RESULT_BINDING",
    }
    if out_binding:
        destination = Path(out_binding)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("spec", "prepared", "result"))
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--report")
    parser.add_argument("--out-binding")
    args = parser.parse_args()
    if args.mode == "spec":
        result = validate_spec(args.preregistration)
    elif args.mode == "prepared":
        result = validate_prepared(args.preregistration)
    else:
        if not args.report:
            raise ValueError("--report is required in result mode")
        result = validate_result(
            args.preregistration,
            args.report,
            out_binding=args.out_binding,
        )
    print(result["verdict"])


if __name__ == "__main__":
    main()
