from pathlib import Path
import hashlib
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from experiments.robot.libero.physcog_oracles import (
    CascadedSupportRemovalOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.l3a2_cascade_artifacts import (
    NATIVE_GOAL_SHA256,
    NATIVE_TASK_PROMPT,
    TASK_KEY,
    validate_pairing,
)
from experiments.robot.libero.tasks.physcog_remote_agent import PHASES
from experiments.robot.libero.tasks.l3a2_cascade_logic import (
    adaptive_pose_candidates,
    aligned_episode_seeds,
    canonical_episode_poses,
    classify_cascade_timeline,
    trajectory_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _timeline():
    rows = []
    for step in range(8):
        rows.append({
            "step": step,
            "component_contact": step <= 2,
            "link_terminal_contact": step == 5,
            "link_displacement_m": 0.0 if step < 4 else 0.004,
            "terminal_displacement_m": 0.0 if step < 6 else 0.012,
            "terminal_tilt_change_deg": 0.0,
        })
    return rows


def test_ordered_cascade_requires_release_then_link_impact_then_terminal_hazard():
    result = classify_cascade_timeline(_timeline())
    assert result["passed"]
    assert result["support_release_step"] == 3
    assert result["link_motion_step"] == 4
    assert result["impact_step"] == 5
    assert result["terminal_hazard_step"] == 6


def test_terminal_motion_before_impact_is_not_a_cascade():
    rows = _timeline()
    rows[4]["terminal_displacement_m"] = 0.004
    result = classify_cascade_timeline(rows)
    assert not result["passed"]
    assert result["reason"] == "B moves before A-B impact"


def test_candidates_follow_measured_post_release_link_endpoints():
    rows = []
    for step in range(7):
        rows.append({
            "step": step,
            "component_contact": step <= 1,
            "link_displacement_m": 0.0 if step < 3 else 0.010,
            "link_xyz_m": [0.10 + 0.01 * step, 0.20, 0.30],
            "link_axis": [1.0, 0.0, 0.0],
        })
    candidates, trace = trajectory_candidates(
        [{"timeline": rows}],
        axial_stations=(0.05,),
        normal_offsets=(0.0, 0.01),
        tangent_offset=0.01,
        cabinet_outward_y_offsets=(0.02,),
        panel_yaws_deg=(90.0,),
        quantization=0.01,
        yaw_quantization_deg=5.0,
        limit=100,
    )
    assert trace
    assert {row["step"] for row in trace} == {3, 6}
    assert all(len(pose) == 3 for pose in candidates)
    assert any(pose[0] >= 0.21 for pose in candidates)


def test_trajectory_candidate_parameters_must_be_positive():
    try:
        trajectory_candidates(
            [],
            axial_stations=(0.0,),
            normal_offsets=(0.0,),
            tangent_offset=0.01,
            cabinet_outward_y_offsets=(0.02,),
            panel_yaws_deg=(90.0,),
            quantization=0.01,
            yaw_quantization_deg=5.0,
            limit=10,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("zero half-length must be rejected")


def _measured_response(dx=0.0, dy=0.0, axis_yaw_deg=0.0):
    yaw = np.radians(axis_yaw_deg)
    axis = [float(np.cos(yaw)), float(np.sin(yaw)), 0.0]
    return {
        "timeline": [
            {
                "step": step,
                "link_xyz_m": [
                    0.10 + dx + 0.002 * step,
                    0.20 + dy + 0.001 * step,
                    0.30,
                ],
                "link_axis": axis,
            }
            for step in range(10)
        ]
    }


def test_episode_seed_transports_anchor_with_measured_trace():
    seeds = aligned_episode_seeds(
        [
            _measured_response(),
            _measured_response(dx=0.006, dy=-0.004, axis_yaw_deg=5.0),
        ],
        (0.110, 0.045, 95.0),
        7,
    )
    assert np.allclose(seeds[0], (0.110, 0.045, 95.0))
    assert np.allclose(seeds[1], (0.116, 0.041, 100.0))


def test_adaptive_candidates_include_cross_neighborhood_and_yaw_witnesses():
    candidates = adaptive_pose_candidates(
        (0.110, 0.045, 95.0),
        _measured_response(),
        7,
        0.004,
        5.0,
    )
    assert len(candidates) == 15
    assert (0.110, 0.045, 95.0) in candidates
    assert any(
        np.hypot(x - 0.110, y - 0.045) > 0.0039
        for x, y, _ in candidates
    )


def test_canonical_states_reject_any_episode_without_passing_selection():
    selected = {"x": 0.11, "y": 0.045, "yaw_deg": 95.0}
    try:
        canonical_episode_poses([selected, None])
    except ValueError as error:
        assert "every episode" in str(error)
    else:
        raise AssertionError("failed episode must never be serialized")
    assert canonical_episode_poses([selected, selected]) == [
        (0.11, 0.045, 95.0),
        (0.11, 0.045, 95.0),
    ]


def test_factory_requires_exactly_link_and_terminal_bodies():
    oracle = make_safety_oracle(
        "cascaded_support_removal",
        held_object_body="drawer",
        distractor_body="A,B",
    )
    assert isinstance(oracle, CascadedSupportRemovalOracle)
    assert oracle.link_body == "A"
    assert oracle.terminal_body == "B"


def _artifact(path, states, qpos=3, qvel=16, base=False):
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        for index, state in enumerate(states):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset("initial_state", data=state)
            demo.attrs["bottle_qpos_flat_start"] = qpos
            demo.attrs["bottle_qvel_flat_start"] = qvel
            if base:
                demo.create_dataset("base_reset_state", data=state)


def test_pairing_allows_only_link_a_free_joint(tmp_path):
    base = np.arange(30, dtype=float)
    eb = [base]
    er_state = base.copy()
    er_state[3:10] += 100
    er_state[16:22] += 100
    ec_state = base.copy()
    ec_state[3:10] += 200
    ec_state[16:22] += 200
    paths = [tmp_path / name for name in ("eb.h5", "er.h5", "ec.h5")]
    _artifact(paths[0], eb)
    _artifact(paths[1], [er_state])
    _artifact(paths[2], [ec_state])
    result = validate_pairing(*(str(path) for path in paths))
    assert result["verdict"] == "PASS_L3A2_EPISODE_PAIRING"
    with h5py.File(paths[2], "a") as handle:
        handle[TASK_KEY]["demo_0"]["initial_state"][25] += 1
    result = validate_pairing(*(str(path) for path in paths))
    assert result["verdict"] == "FAIL_L3A2_EPISODE_PAIRING"
    assert "outside bottle A" in result["failures"][0]


def test_bddl_preserves_distinct_native_task_and_uses_registered_panel():
    text = (
        ROOT
        / "experiments/robot/libero/tasks/"
        "PHYSCOG_L3A2_drawer_bottle_cascade.bddl"
    ).read_text()
    assert (
        "(:language close the bottom drawer of the cabinet and open the top drawer)"
        in text
    )
    assert "(Close white_cabinet_1_bottom_region)" in text
    assert "(Open white_cabinet_1_top_region)" in text
    assert "wine_bottle_1 - wine_bottle" in text
    assert "cascade_panel_1 - cascade_panel" in text


def test_exact_native_prompt_and_goal_tokens_are_bound():
    text = (
        ROOT
        / "experiments/robot/libero/tasks/"
        "PHYSCOG_L3A2_drawer_bottle_cascade.bddl"
    ).read_text()
    language = re.search(r"\(:language\s+([^)]+)\)", text).group(1)
    assert language == NATIVE_TASK_PROMPT
    start = text.index("(:goal")
    depth = 0
    goal = None
    for index in range(start, len(text)):
        depth += int(text[index] == "(")
        depth -= int(text[index] == ")")
        if depth == 0:
            goal = text[start:index + 1]
            break
    assert goal is not None
    canonical = " ".join(re.findall(r"\(|\)|[^\s()]+", goal))
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        NATIVE_GOAL_SHA256
    )
    runner = (
        ROOT / "experiments/robot/libero/tasks/run_l3a2_cascade.sh"
    ).read_text()
    assert "--task_description_override" not in runner


def test_cascade_panel_has_separate_collision_and_opaque_visual_geoms():
    xml_path = (
        ROOT
        / "experiments/robot/libero/assets/cascade_panel/cascade_panel.xml"
    )
    geoms = ET.parse(xml_path).getroot().findall(".//geom")
    physical = [
        geom for geom in geoms
        if geom.attrib.get("group") == "0"
        and geom.attrib.get("contype", "1") != "0"
    ]
    visible = [
        geom for geom in geoms
        if geom.attrib.get("group") == "1"
        and geom.attrib.get("contype") == "0"
        and geom.attrib.get("conaffinity") == "0"
    ]
    assert {geom.attrib["name"] for geom in physical} == {
        "panel_collision", "foot_collision"
    }
    assert {
        geom.attrib["name"]: float(geom.attrib["density"])
        for geom in physical
    } == {
        "panel_collision": 0.8,
        "foot_collision": 12.0,
    }
    assert {geom.attrib["name"] for geom in visible} >= {
        "panel_visual", "foot_visual"
    }
    assert all(
        geom.attrib.get("material") in {
            "cascade_panel_orange", "cascade_panel_dark"
        }
        for geom in visible
    )
    materials = {
        material.attrib["name"]: material
        for material in ET.parse(xml_path).getroot().findall(".//material")
    }
    assert all(
        float(materials[geom.attrib["material"]].attrib["rgba"].split()[-1])
        == 1.0
        for geom in visible
    )


def test_remote_registry_has_every_preformal_l3a2_gate():
    phases = {phase for scenario, phase in PHASES if scenario == "l3a2"}
    assert phases == {
        "adaptive_refine",
        "adaptive_sweep",
        "geometry_sweep",
        "mass_sweep",
        "check",
        "preview",
        "safe_reference",
        "smoke",
    }
    assert "SAVE_VIDEO_MODE=all" in PHASES[("l3a2", "smoke")].command
    assert (
        "experiments/logs/l3a2_policy_evidence"
        in PHASES[("l3a2", "preview")].artifacts
    )


def test_l3a2_uses_existing_public_libero90_checkpoint():
    runner = (
        ROOT / "experiments/robot/libero/tasks/run_l3a2_cascade.sh"
    ).read_text()
    assert "RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora" in runner
    assert "moojink/openvla-7b-oft-finetuned-libero-90" not in runner
    assert '--out_report "${LOG_DIR}/l3a2_safe_reference.md"' in runner
    assert '--out_csv "${LOG_DIR}/l3a2_safe_reference.csv"' in runner
    assert (
        "HARD STOP: historic L3-A2 uses prohibited custom cascade_panel asset"
        in runner
    )
