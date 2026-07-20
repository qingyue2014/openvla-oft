import hashlib

import h5py

from experiments.robot.libero.physcog_trajectory import load_l3a1_episode_evidence


def test_l3a1_episode_evidence_maps_exact_hdf_demo_and_artifact(tmp_path):
    artifact = tmp_path / "states.hdf5"
    with h5py.File(artifact, "w") as handle:
        group = handle.create_group("task")
        group.attrs["l3a1_topology_schema_version"] = 2
        group.attrs["l3a1_topology_id"] = "native-edge-v1"
        group.attrs["support_topology_contract_sha256"] = "a" * 64
        group.attrs["native_cabinet_xml_sha256"] = "b" * 64
        group.attrs["compiled_support_component_signatures_sha256"] = "c" * 64
        group.attrs["support_component_role_hashes_sha256"] = "d" * 64
        group.attrs["l3a1_variant"] = "risk"
        demo = group.create_group("demo_0")
        demo.create_dataset("initial_state", data=[1.0, 2.0, 3.0])
        demo.attrs["initial_component_roles"] = "edge/front_outer"
        demo.attrs["initial_edge_table_qualified"] = True
        demo.attrs["factual_close_component_release_step_rC"] = 6
        demo.attrs["factual_close_first_oracle_step"] = 52
        demo.attrs["factual_close_component_recontact_after_rC"] = False
        demo.attrs["factual_close_pre_oracle_other_cabinet_geoms"] = ""
        demo.attrs["factual_close_pre_oracle_direct_contact_bodies"] = ""
        demo.attrs["factual_close_post_oracle_direct_contact_bodies"] = "post-contact"
        demo.attrs["factual_close_bottle_qvel_overwritten"] = False

    evidence = load_l3a1_episode_evidence(str(artifact), "task", 0)
    assert evidence["initial_states_artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert evidence["initial_states_demo_index"] == 0
    assert evidence["support_initial_component_roles"] == "edge/front_outer"
    assert evidence["support_component_release_step_rC"] == 6
    assert evidence["support_first_oracle_step"] == 52
    assert evidence["support_bottle_qvel_overwritten"] is False
    assert evidence["support_pre_oracle_direct_contact_bodies"] == ""
    assert evidence["support_post_oracle_direct_contact_bodies"] == "post-contact"
    assert evidence["support_component_role_hashes_sha256"] == "d" * 64
