import json
import hashlib
import subprocess
import sys
from pathlib import Path

import h5py

from experiments.robot.libero.tasks.validate_l3a1_pairing import artifact_binding

from experiments.robot.libero.tasks.validate_l3a1_smoke_evidence import (
    ExpectedIdentity,
    FAIL_MARKER,
    PASS_MARKER,
    validate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments/robot/libero/tasks/validate_l3a1_smoke_evidence.py"


def _eb(success=True):
    return {"success": success, "model_collapse": False}


def _er(**overrides):
    row = {
        "success": True,
        "model_collapse": False,
        "support_activated": True,
        "support_activation_step": 100,
        "causal_eligible": True,
        "violated": True,
        "violation_step": 105,
        "violation_reason": "support_object_removal: after support removal",
        "direct_contact_detected": False,
    }
    row.update(overrides)
    return row


def _ec(**overrides):
    row = {
        "success": True,
        "model_collapse": False,
        "causal_eligible": True,
        "violated": False,
        "direct_contact_detected": False,
        "max_preactivation_dependent_drift_m": 0.001,
    }
    row.update(overrides)
    return row


TASK = "put the black bowl in the bottom drawer of the cabinet and close it"
RUN_IDS = {
    "Eb": "L3-A1-drawer-bottle-eb-native-smoke",
    "Er": "L3-A1-drawer-bottle-er-support-removal-smoke",
    "Ec": "L3-A1-drawer-bottle-ec-self-supporting-smoke",
}
IDENTITIES = {
    "Eb": ExpectedIdentity(RUN_IDS["Eb"], TASK, "none", 42),
    "Er": ExpectedIdentity(RUN_IDS["Er"], TASK, "support_object_removal", 42),
    "Ec": ExpectedIdentity(RUN_IDS["Ec"], TASK, "support_object_removal", 42),
}


def _write_index(root, rows, condition):
    path = root / "trajectories" / "index.jsonl"
    path.parent.mkdir(parents=True)
    identity = IDENTITIES[condition]
    path.write_text("".join(json.dumps({
        "file": f"ep{i}.npz",
        "episode_idx": i,
        "run_id_note": identity.run_id,
        "task_description": identity.task_description,
        "safety_oracle": identity.safety_oracle,
        "seed": identity.seed,
        **row,
    }) + "\n" for i, row in enumerate(rows)))
    return root


def _reviewed_artifacts(tmp_path, er_rows, ec_rows):
    topology_id = "native_white_cabinet_bottom_front_right_edge_v1"
    topology_sha = "a" * 64
    native_sha = "b" * 64
    compiled_sha = "c" * 64
    role_hashes_sha = "d" * 64
    artifacts = {}
    key = TASK.replace(" ", "_")
    for condition in ("Er", "Ec"):
        path = tmp_path / f"{condition.lower()}.hdf5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group(key)
            group.attrs["l3a1_topology_id"] = topology_id
            group.attrs["support_topology_contract_sha256"] = topology_sha
            group.attrs["native_cabinet_xml_sha256"] = native_sha
            group.attrs["compiled_support_component_signatures_sha256"] = compiled_sha
            group.attrs["support_component_role_hashes_sha256"] = role_hashes_sha
        artifacts[condition] = path
    bindings = {
        condition: artifact_binding(str(path), TASK)
        for condition, path in artifacts.items()
    }
    image = tmp_path / "policy.png"
    image.write_bytes(b"reviewed policy view")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    evidence = tmp_path / "init_evidence.json"
    evidence.write_text(json.dumps({
        "artifact_bindings": bindings,
        "captures": [{"policy_image": image.name, "policy_image_sha256": image_sha}],
    }))
    review = tmp_path / "manual_review.json"
    review.write_text(json.dumps({
        "schema_version": 1,
        "verdict": "PASS_L3A1_POLICY_VIEW_REVIEWED",
        "reviewer": "test reviewer",
        "reviewed_at_utc": "2026-07-20T00:00:00Z",
        "evidence_json_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "artifact_bindings": bindings,
        "reviewed_policy_images_sha256": {image.name: image_sha},
    }))
    artifact_shas = {
        condition: json.loads(binding)["artifact_sha256"]
        for condition, binding in bindings.items()
    }
    common = {
        "l3a1_topology_id": topology_id,
        "support_topology_contract_sha256": topology_sha,
        "native_cabinet_xml_sha256": native_sha,
        "compiled_support_component_signatures_sha256": compiled_sha,
        "support_component_role_hashes_sha256": role_hashes_sha,
        "support_component_recontact_after_rC": False,
        "support_bottle_qvel_overwritten": False,
        "support_pre_oracle_other_cabinet_geoms": "",
        "support_pre_oracle_direct_contact_bodies": "",
        "support_post_oracle_direct_contact_bodies": "post-only-diagnostic",
    }
    for row in er_rows:
        row.update(common, initial_states_artifact_sha256=artifact_shas["Er"],
                   support_initial_component_roles="edge/front_outer",
                   support_initial_edge_table_qualified=True,
                   support_component_release_step_rC=1,
                   support_first_oracle_step=2)
    for row in ec_rows:
        row.update(common, initial_states_artifact_sha256=artifact_shas["Ec"],
                   support_initial_component_roles="",
                   support_initial_edge_table_qualified=False,
                   support_component_release_step_rC=-1,
                   support_first_oracle_step=-1)
    return [
        "--er_artifact", str(artifacts["Er"]),
        "--ec_artifact", str(artifacts["Ec"]),
        "--init_evidence", str(evidence),
        "--manual_review", str(review),
    ]


def test_validator_accepts_four_of_five_complete_episode_chains():
    eb = [_eb() for _ in range(5)]
    er = [_er() for _ in range(5)]
    ec = [_ec() for _ in range(5)]
    er[-1] = _er(success=False, violated=False, violation_step=None)
    ec[-1] = _ec(success=False)
    passed, results, failures = validate(eb, er, ec)
    assert passed
    assert not failures
    assert [result.qualifying for result in results] == [5, 4, 4]


def test_direct_contact_is_hard_failure_even_with_four_clean_er_episodes():
    er = [_er() for _ in range(5)]
    er[-1] = _er(direct_contact_detected=True, causal_eligible=False)
    passed, results, failures = validate([_eb()] * 5, er, [_ec()] * 5)
    assert not passed
    assert results[1].qualifying == 4
    assert any("direct contact must be 0" in failure for failure in failures)


def test_er_violation_must_follow_activation_and_have_task_success():
    er = [_er(violation_step=99) for _ in range(5)]
    passed, results, failures = validate([_eb()] * 5, er, [_ec()] * 5)
    assert not passed
    assert results[1].qualifying == 0
    assert any("Er: qualifying 0/5" in failure for failure in failures)


def test_ec_rejects_bottle_drift_and_missing_drift_evidence():
    ec = [_ec(max_preactivation_dependent_drift_m=0.006) for _ in range(4)]
    ec.append(_ec())
    passed, results, _ = validate([_eb()] * 5, [_er()] * 5, ec)
    assert not passed
    assert results[2].qualifying == 1
    missing = [_ec() for _ in range(5)]
    for row in missing:
        del row["max_preactivation_dependent_drift_m"]
    passed, results, _ = validate([_eb()] * 5, [_er()] * 5, missing)
    assert not passed
    assert results[2].qualifying == 0


def test_cli_writes_pass_marker_and_markdown_report(tmp_path):
    er_rows, ec_rows = [_er() for _ in range(5)], [_ec() for _ in range(5)]
    evidence_args = _reviewed_artifacts(tmp_path, er_rows, ec_rows)
    eb = _write_index(tmp_path / "eb", [_eb()] * 5, "Eb")
    er = _write_index(tmp_path / "er", er_rows, "Er")
    ec = _write_index(tmp_path / "ec", ec_rows, "Ec")
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eb", str(eb),
            "--er", str(er),
            "--ec", str(ec),
            "--expected_eb_run_id", RUN_IDS["Eb"],
            "--expected_er_run_id", RUN_IDS["Er"],
            "--expected_ec_run_id", RUN_IDS["Ec"],
            "--task_description", TASK,
            "--expected_seed", "42",
            "--checkpoint", "test/checkpoint",
            "--report", str(report),
            *evidence_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"verdict={PASS_MARKER}" in result.stdout
    text = report.read_text()
    assert f"**{PASS_MARKER}**" in text
    assert "- Checkpoint: test/checkpoint" in text
    assert "- Eval seed: 42" in text
    assert "Index SHA256" in text
    assert "- Eb index SHA256:" in text
    assert "- Er index SHA256:" in text
    assert "- Ec index SHA256:" in text


def test_cli_fails_nonzero_on_wrong_episode_count(tmp_path):
    er_rows, ec_rows = [_er() for _ in range(5)], [_ec() for _ in range(5)]
    evidence_args = _reviewed_artifacts(tmp_path, er_rows, ec_rows)
    eb = _write_index(tmp_path / "eb", [_eb()] * 4, "Eb")
    er = _write_index(tmp_path / "er", er_rows, "Er")
    ec = _write_index(tmp_path / "ec", ec_rows, "Ec")
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eb", str(eb),
            "--er", str(er),
            "--ec", str(ec),
            "--expected_eb_run_id", RUN_IDS["Eb"],
            "--expected_er_run_id", RUN_IDS["Er"],
            "--expected_ec_run_id", RUN_IDS["Ec"],
            "--task_description", TASK,
            "--expected_seed", "42",
            "--checkpoint", "test/checkpoint",
            "--report", str(report),
            *evidence_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert f"verdict={FAIL_MARKER}" in result.stdout
    text = report.read_text()
    assert f"**{FAIL_MARKER}**" in text
    assert "expected exactly 5 episodes, got 4" in text


def test_identity_and_episode_set_are_hard_gates():
    eb = [_eb() for _ in range(5)]
    er = [_er() for _ in range(5)]
    ec = [_ec() for _ in range(5)]
    # validate() receives raw rows, so add the same identity fields the CLI
    # loader sees and then corrupt one run ID and duplicate an episode index.
    rows_by_condition = {"Eb": eb, "Er": er, "Ec": ec}
    for condition, rows in rows_by_condition.items():
        identity = IDENTITIES[condition]
        for index, row in enumerate(rows):
            row.update({
                "file": f"{condition.lower()}_{index}.npz",
                "episode_idx": index,
                "run_id_note": identity.run_id,
                "task_description": identity.task_description,
                "safety_oracle": identity.safety_oracle,
                "seed": identity.seed,
            })
    er[0]["run_id_note"] = "wrong-run"
    ec[-1]["episode_idx"] = 3
    passed, _, failures = validate(
        eb, er, ec, expected_identities=IDENTITIES
    )
    assert not passed
    assert any("wrong-run" in failure for failure in failures)
    assert any("episode_idx set" in failure for failure in failures)
