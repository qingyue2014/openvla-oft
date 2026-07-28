from pathlib import Path

import h5py
import numpy as np

from experiments.robot.libero.physcog_oracles import (
    ResidualRiskClosureOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.validate_l3b1_native_preflight import (
    CAPABILITY_TASK_PROMPT,
    TASK_PROMPT,
    validate_native_task,
)
from experiments.robot.libero.tasks.validate_l3b1_states import validate_pairing
from experiments.robot.libero.tasks.validate_l3b1_native_capability_states import (
    validate as validate_native_capability_states,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh"


def _state_file(path, variant, bottle_delta):
    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        group = handle.create_group(key)
        group.attrs["l3b1_variant"] = variant
        group.attrs["task_suite_name"] = "libero_90"
        group.attrs["task_description"] = TASK_PROMPT
        group.attrs["bddl_basename"] = (
            "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl"
        )
        group.attrs["seed"] = 42
        group.attrs["intervention_body"] = "wine_bottle_1_main"
        group.attrs["custom_assets"] = False
        group.attrs["custom_bddl"] = False
        demo = group.create_group("demo_0")
        base = np.zeros(30)
        state = base.copy()
        state[2] = bottle_delta
        demo.create_dataset("initial_state", data=state)
        demo.create_dataset("base_reset_state", data=base)
        demo.attrs["source_state_index"] = 0
        demo.attrs["bottle_qpos_flat_start"] = 0
        demo.attrs["bottle_qvel_flat_start"] = 20
        demo.attrs["non_bottle_error"] = 0.0
        demo.attrs["runtime_wait_displacement_m"] = 0.0
        demo.attrs["runtime_wait_tilt_change_deg"] = 0.0


def test_native_preflight_accepts_only_exact_native_task():
    bddl = (
        ROOT
        / "_deps/LIBERO/libero/libero/bddl_files/libero_90"
        / "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl"
    )
    if not bddl.exists():
        return
    evidence = validate_native_task(bddl, bddl, TASK_PROMPT)
    assert evidence["prompt"] == TASK_PROMPT
    assert evidence["objects"]["wine_bottle_1"] == "wine_bottle"


def test_capability_preflight_accepts_exact_second_native_task():
    bddl = (
        ROOT
        / "_deps/LIBERO/libero/libero/bddl_files/libero_90"
        / "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl"
    )
    if not bddl.exists():
        return
    evidence = validate_native_task(
        bddl,
        bddl,
        CAPABILITY_TASK_PROMPT,
        task_role="capability",
    )
    assert evidence["prompt"] == CAPABILITY_TASK_PROMPT
    assert evidence["objects"]["wine_bottle_1"] == "wine_bottle"


def test_paired_states_allow_only_bottle_change(tmp_path):
    eb, er, ec = (tmp_path / name for name in ("eb.hdf5", "er.hdf5", "ec.hdf5"))
    _state_file(eb, "baseline", 0.0)
    _state_file(er, "risk", 0.2)
    _state_file(ec, "clearance", 0.3)
    assert validate_pairing(str(eb), str(er), str(ec), 1) == 1


def test_native_capability_states_are_exact_official_baselines(tmp_path):
    path = tmp_path / "native_capability.hdf5"
    key = CAPABILITY_TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        group = handle.create_group(key)
        group.attrs["l3b1_variant"] = "capability_native"
        group.attrs["task_suite_name"] = "libero_90"
        group.attrs["task_description"] = CAPABILITY_TASK_PROMPT
        group.attrs["bddl_basename"] = (
            "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl"
        )
        group.attrs["custom_assets"] = False
        group.attrs["custom_bddl"] = False
        group.attrs["intervention_body"] = ""
        demo = group.create_group("demo_0")
        state = np.arange(30, dtype=float)
        demo.create_dataset("initial_state", data=state)
        demo.create_dataset("base_reset_state", data=state)
        demo.attrs["source_state_index"] = 0
        demo.attrs["non_bottle_error"] = 0.0
    assert validate_native_capability_states(str(path), 1) == 1


def test_residual_oracle_factory_accepts_safe_handling_and_clearance_modes():
    risk = make_safety_oracle(
        "residual_risk_closure",
        held_object_body="drawer",
        distractor_body="bottle",
        l3b1_condition="risk",
    )
    clearance = make_safety_oracle(
        "residual_risk_closure",
        held_object_body="drawer",
        distractor_body="bottle",
        l3b1_condition="clearance",
    )
    assert isinstance(risk, ResidualRiskClosureOracle)
    assert risk.condition == "risk"
    assert clearance.condition == "clearance"


def test_formal_runner_enforces_video_cap_and_allows_lay_down_solution():
    text = RUNNER.read_text()
    assert (
        'CHECKPOINT="${CHECKPOINT:-RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90}"'
        in text
    )
    assert 'DO_SAMPLE="${DO_SAMPLE:-True}"' in text
    assert 'TEMPERATURE="${TEMPERATURE:-1.6}"' in text
    assert 'TOP_P="${TOP_P:-1.0}"' in text
    assert '"${POLICY_DECODE_ARGS[@]}"' in text
    assert "run_capability_native_preflight" in text
    assert "run_native_cap_prepare" in text
    assert "run_native_cap_smoke" in text
    assert "run_native_cap_formal" in text
    gate = text.split("require_native_capability_gates() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert gate.index("validate_l3b1_native_capability_states.py") < gate.index(
        "grep -q PASS_L3B1_NATIVE_CAPABILITY_STATES"
    )
    formal = text.split("run_native_cap_formal() {", 1)[1].split("\n}", 1)[0]
    assert formal.index("validate_l3b1_native_capability_smoke.py") < formal.index(
        "grep -q PASS_L3B1_NATIVE_CAPABILITY_SMOKE"
    )
    assert 'MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-10}"' in text
    assert "--max_violation_videos" in text
    assert "--max_success_videos" in text
    assert "--max_failure_videos" in text
    oracle = (ROOT / "experiments/robot/libero/physcog_oracles.py").read_text()
    assert '"laid_bottle_then_closed"' in oracle
    assert "min_clearance_tilt_deg" in oracle
