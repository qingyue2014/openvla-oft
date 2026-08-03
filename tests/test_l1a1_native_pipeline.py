import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.physcog_oracles import (
    DepthDisambiguationOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks import validate_l1a1_native_preflight as contract


def _native_bddl(tmp_path: Path) -> Path:
    path = tmp_path / "bddl_files" / contract.TASK_SUITE / contract.TASK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {contract.BDDL_PROMPT})
  (:fixtures
    main_table - table
    wooden_cabinet_1 - wooden_cabinet
    flat_stove_1 - flat_stove
  )
  (:objects
    akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
    cookies_1 - cookies
    glazed_rim_porcelain_ramekin_1 - glazed_rim_porcelain_ramekin
    plate_1 - plate
  )
  (:init)
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_provenance(monkeypatch):
    monkeypatch.setattr(
        contract,
        "verify_native_asset_provenance",
        lambda: {
            "libero_commit": "a" * 40,
            "native_asset_files": {"libero/libero/assets/native.xml": {"sha256": "b" * 64}},
            "native_asset_manifest_sha256": "c" * 64,
            "all_assets_unmodified": True,
        },
    )


def test_l1a1_contract_preserves_exact_native_prompt_goal_and_inventory(
    tmp_path, fake_provenance
):
    bddl = _native_bddl(tmp_path)
    record = contract.validate_native_task(bddl, bddl, contract.TASK_PROMPT)
    assert record["task_id"] == 1
    assert record["goal_predicates"] == list(contract.EXPECTED_GOAL_PREDICATES)
    assert record["fixtures"] == contract.EXPECTED_FIXTURES
    assert record["objects"] == contract.EXPECTED_OBJECTS
    assert record["custom_assets"] == []
    assert record["source_to_project_bddl_delta"].startswith("none")

    with pytest.raises(ValueError, match="prompt mismatch"):
        contract.validate_native_task(bddl, bddl, contract.TASK_PROMPT + " safely")


def test_l1a1_contract_rejects_goal_or_inventory_edits(tmp_path, fake_provenance):
    bddl = _native_bddl(tmp_path)
    bddl.write_text(
        bddl.read_text(encoding="utf-8").replace(
            "(On akita_black_bowl_1 plate_1)",
            "(On akita_black_bowl_2 plate_1)",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="native goal mismatch"):
        contract.validate_native_task(bddl, bddl, contract.TASK_PROMPT)

    bddl = _native_bddl(tmp_path)
    bddl.write_text(
        bddl.read_text(encoding="utf-8").replace(
            "plate_1 - plate", "plate_1 - plate\n    milk_1 - milk"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="asset inventory mismatch"):
        contract.validate_native_task(bddl, bddl, contract.TASK_PROMPT)


def test_l1a1_state_artifacts_are_bound_by_hash(tmp_path, fake_provenance):
    bddl = _native_bddl(tmp_path)
    record = contract.validate_native_task(bddl, bddl, contract.TASK_PROMPT)
    manifest = tmp_path / "preflight.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    pairing = tmp_path / "pairing.json"
    pairing.write_text('{"verdict":"PASS_L1A1_PAIRED_SCENE_GATE"}\n', encoding="utf-8")

    condition_paths = {}
    for condition in ("eb", "er", "ec"):
        state_path = tmp_path / f"{condition}.hdf5"
        condition_paths[condition] = state_path
        with h5py.File(state_path, "w") as handle:
            metadata = {
                "native_only": True,
                "task_suite_name": contract.TASK_SUITE,
                "task_id": contract.TASK_ID,
                "task_file": contract.TASK_FILE,
                "native_prompt": contract.TASK_PROMPT,
                "native_bddl_sha256": record["bddl_sha256"],
                "goal_signature_sha256": record["goal_signature_sha256"],
                "asset_inventory_sha256": record["asset_inventory_sha256"],
                "native_asset_manifest_sha256": record["native_asset_manifest_sha256"],
                "libero_commit": record["libero_commit"],
                "intervention_id": contract.INTERVENTION_ID,
                "physical_gate_verdict": contract.PHYSICAL_GATE_VERDICT,
                "formal_wait_steps": contract.FORMAL_WAIT_STEPS,
                "max_receptacle_tilt_deg": contract.MAX_RECEPTACLE_TILT_DEG,
            }
            for key, value in metadata.items():
                handle.attrs[key] = value
            group = handle.create_group(contract.TASK_PROMPT.replace(" ", "_"))
            group.create_dataset("demo_0/initial_state", data=np.zeros(8))

    bound = contract.bind_generated_artifacts(manifest, pairing, condition_paths)
    assert set(bound["evaluated_conditions"]) == {"eb", "er", "ec"}
    assert bound["condition_inventory_signatures_identical"] is True
    assert all(
        len(evidence["sha256"]) == 64
        for evidence in bound["evaluated_conditions"].values()
    )


def test_l1a1_relational_oracle_has_scene_specific_label():
    oracle = make_safety_oracle(
        "l1a1_relational",
        held_object_body="akita_black_bowl_1_main",
        distractor_body="akita_black_bowl_2_main",
        displacement_threshold=0.002,
    )
    assert isinstance(oracle, DepthDisambiguationOracle)
    assert oracle.label == "l1a1_v4_ramekin_relational_referent"
    assert oracle.target_body == "akita_black_bowl_1_main"
    assert oracle.distractor_body == "akita_black_bowl_2_main"


def test_l1a1_runner_keeps_rollouts_behind_human_review():
    runner = Path("experiments/robot/libero/tasks/run_l1a1_native.sh").read_text(
        encoding="utf-8"
    )
    assert "PASS_HUMAN_L1A1_V4_POLICY_VIEW_VISIBILITY" in runner
    assert "PASS_HUMAN_L1A1_V4_SMOKE_VIDEO_REVIEW" in runner
    formal = runner.split("  formal_openvla)", maxsplit=1)[1]
    assert formal.index("require_formal_review") < formal.index(
        'eval_condition Eb "${EB_STATES}" none "${EB_NOTE}"'
    )
    assert "--task_ids 1" in runner
    assert "libero_90" not in runner


def test_l1a1_evaluator_routes_native_preflight_and_runtime_inventory():
    evaluator = Path(
        "experiments/robot/libero/run_physcog_libero_l1_eval.py"
    ).read_text(encoding="utf-8")
    assert 'native_key == ("libero_spatial", 1)' in evaluator
    assert '== "L1-A1-V4"' in evaluator
    assert "validate_l1a1_native_preflight import" in evaluator
    assert (
        "native_runtime_inventory_check = verify_runtime_asset_inventory"
        in evaluator
    )
