from collections import OrderedDict

import numpy as np
import pytest

from experiments.robot.robocasa.pi05_policy import (
    LIBERO_INITIAL_EEF_POS,
    Pi05RoboCasaPolicy,
    build_request,
    canonicalize_robocasa_state,
    map_libero_action_to_pandaomron,
    pi05_preprocessing_label,
    preprocess_camera_image,
    preprocess_camera_image_for_mode,
    resize_with_pad,
)
from experiments.robot.robocasa.scripts.run_condition import (
    load_smoke_gate_manifest,
)


class _ArmController:
    origin_pos = np.zeros(3)
    origin_ori = np.eye(3)


class _Controller:
    _action_split_indexes = OrderedDict(
        (
            ("right", (0, 6)),
            ("torso", (6, 7)),
            ("base", (7, 10)),
            ("right_gripper", (10, 11)),
        )
    )
    part_controllers = {"right": _ArmController()}


class _Robot:
    composite_controller = _Controller()


class _Env:
    action_spec = (-np.ones(12), np.ones(12))
    robots = [_Robot()]


def _obs():
    return {
        "robot0_agentview_center_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.01, -0.01]),
    }


def test_resize_with_pad_has_official_shape():
    output = resize_with_pad(np.zeros((120, 240, 3), dtype=np.uint8))
    assert output.shape == (224, 224, 3)
    assert output.dtype == np.uint8


def test_preprocess_camera_image_matches_official_180_degree_rotation():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = np.arange(6).reshape(2, 3)
    output = preprocess_camera_image(image, size=3)
    np.testing.assert_array_equal(output[0, :, 0], [5, 4, 3])
    np.testing.assert_array_equal(output[1, :, 0], [2, 1, 0])


def test_vertical_camera_mode_does_not_mirror_robocasa_image():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = np.arange(6).reshape(2, 3)
    output = preprocess_camera_image_for_mode(
        image,
        mode="vertical",
        size=3,
    )
    np.testing.assert_array_equal(output[0, :, 0], [3, 4, 5])
    np.testing.assert_array_equal(output[1, :, 0], [0, 1, 2])
    assert pi05_preprocessing_label("vertical") == (
        "pi05_robocasa_vertical_resize_with_pad_224"
    )


def test_build_request_uses_native_prompt_and_eight_dimensional_state():
    request = build_request(_obs(), "pick up the mug")
    assert request["prompt"] == "pick up the mug"
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/wrist_image"].shape == (224, 224, 3)
    assert request["observation/state"].shape == (8,)


def test_build_request_accepts_existing_native_side_camera():
    obs = _obs()
    obs["robot0_agentview_left_image"] = np.full(
        (256, 256, 3),
        17,
        dtype=np.uint8,
    )
    request = build_request(
        obs,
        "pick up the mug",
        agent_camera="robot0_agentview_left",
    )
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/image"].mean() == 17


def test_canonical_state_anchors_initial_position_to_libero_mean():
    state, anchor = canonicalize_robocasa_state(
        _obs(), _Env(), state_anchor=None
    )
    np.testing.assert_allclose(state[:3], LIBERO_INITIAL_EEF_POS)
    np.testing.assert_allclose(
        state[3:6],
        [3.1404691, -0.0022365, -0.08691545],
        atol=1e-6,
    )
    np.testing.assert_allclose(anchor.world_position, [0.1, 0.2, 0.3])

    moved = _obs()
    moved["robot0_eef_pos"] = np.array([0.11, 0.18, 0.33])
    moved_state, reused_anchor = canonicalize_robocasa_state(
        moved, _Env(), state_anchor=anchor
    )
    np.testing.assert_allclose(
        moved_state[:3],
        LIBERO_INITIAL_EEF_POS + np.array([0.01, -0.02, 0.03]),
    )
    assert reused_anchor is anchor


def test_canonical_state_preserves_world_delta_with_rotated_arm_base():
    env = _Env()
    arm = env.robots[0].composite_controller.part_controllers["right"]
    original = arm.origin_ori
    arm.origin_ori = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    try:
        _, anchor = canonicalize_robocasa_state(
            _obs(), env, state_anchor=None
        )
        moved = _obs()
        moved["robot0_eef_pos"] = np.array([0.11, 0.18, 0.33])
        moved_state, _ = canonicalize_robocasa_state(
            moved, env, state_anchor=anchor
        )
    finally:
        arm.origin_ori = original

    np.testing.assert_allclose(
        moved_state[:3],
        LIBERO_INITIAL_EEF_POS + np.array([0.01, -0.02, 0.03]),
    )


def test_map_pi05_action_freezes_mobile_base_and_torso():
    mapped = map_libero_action_to_pandaomron(np.arange(7) / 10, _Env())
    np.testing.assert_allclose(mapped[:6], np.arange(6) / 10)
    assert mapped[6] == 0.0
    np.testing.assert_allclose(mapped[7:10], 0.0)
    assert mapped[10] == pytest.approx(0.6)
    assert mapped[11] == -1.0


def test_map_rotates_libero_world_delta_into_pandaomron_base_frame():
    env = _Env()
    arm = env.robots[0].composite_controller.part_controllers["right"]
    original = arm.origin_ori
    arm.origin_ori = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    try:
        mapped = map_libero_action_to_pandaomron(
            np.array([0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -1.0]),
            env,
        )
    finally:
        arm.origin_ori = original

    np.testing.assert_allclose(mapped[:3], [0.3, -0.2, 0.4])
    np.testing.assert_allclose(mapped[3:6], [0.6, 0.5, 0.7])
    assert mapped[10] == -1.0
    assert mapped[11] == -1.0


def test_pi05_policy_matches_official_ten_step_settling():
    assert Pi05RoboCasaPolicy.settle_steps == 10
    mapped = Pi05RoboCasaPolicy.settle_action(_Env())
    np.testing.assert_allclose(mapped[:10], 0.0)
    assert mapped[10] == -1.0
    assert mapped[11] == -1.0


def test_map_rejects_unexpected_robot_interface():
    env = _Env()
    env.action_spec = (-np.ones(7), np.ones(7))
    with pytest.raises(ValueError, match="12-D PandaOmron"):
        map_libero_action_to_pandaomron(np.zeros(7), env)


def test_smoke_gate_manifest_requires_reviewed_initial_gates(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(
        """{
          "scene_id": "L1-A1",
          "native_preflight_sha256": "abc",
          "gates": {
            "G0": {"passed": true},
            "physics": {"passed": true},
            "visibility": {
              "passed": true,
              "policy_preprocessing": "pi05_libero_rotate180_resize_with_pad_224",
              "policy_cameras": [
                "robot0_agentview_center",
                "robot0_eye_in_hand"
              ],
              "wrist_initial_frame": "wrist.png",
              "paired_wrist_initial_frames": {
                "Eb": "eb-wrist.png",
                "Er": "er-wrist.png",
                "Ec": "ec-wrist.png"
              }
            }
          }
        }"""
    )
    payload = load_smoke_gate_manifest(
        str(path), scene_id="L1-A1", preflight_sha256="abc"
    )
    assert payload["gates"]["visibility"]["passed"] is True


def test_smoke_gate_manifest_rejects_unreviewed_visibility(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(
        """{
          "scene_id": "L1-A1",
          "native_preflight_sha256": "abc",
          "gates": {
            "G0": {"passed": true},
            "physics": {"passed": true},
            "visibility": {"passed": false}
          }
        }"""
    )
    with pytest.raises(Exception, match="visibility"):
        load_smoke_gate_manifest(
            str(path), scene_id="L1-A1", preflight_sha256="abc"
        )


def test_smoke_gate_manifest_rejects_missing_wrist_camera_evidence(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(
        """{
          "scene_id": "L1-A1",
          "native_preflight_sha256": "abc",
          "gates": {
            "G0": {"passed": true},
            "physics": {"passed": true},
            "visibility": {
              "passed": true,
              "policy_preprocessing": "pi05_libero_rotate180_resize_with_pad_224",
              "policy_cameras": ["robot0_agentview_center"]
            }
          }
        }"""
    )
    with pytest.raises(Exception, match="both pi0.5 policy cameras"):
        load_smoke_gate_manifest(
            str(path), scene_id="L1-A1", preflight_sha256="abc"
        )


def test_smoke_gate_manifest_matches_explicit_vertical_policy_view(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(
        """{
          "scene_id": "L1-A2",
          "native_preflight_sha256": "abc",
          "gates": {
            "G0": {"passed": true},
            "physics": {"passed": true},
            "visibility": {
              "passed": true,
              "policy_preprocessing": "pi05_robocasa_vertical_resize_with_pad_224",
              "policy_cameras": [
                "robot0_agentview_center",
                "robot0_eye_in_hand"
              ],
              "wrist_initial_frame": "wrist.png",
              "paired_wrist_initial_frames": {
                "Eb": "eb-wrist.png",
                "Er": "er-wrist.png",
                "Ec": "ec-wrist.png"
              }
            }
          }
        }"""
    )
    payload = load_smoke_gate_manifest(
        str(path),
        scene_id="L1-A2",
        preflight_sha256="abc",
        expected_policy_preprocessing=(
            "pi05_robocasa_vertical_resize_with_pad_224"
        ),
    )
    assert payload["gates"]["visibility"]["passed"] is True


def test_smoke_gate_manifest_matches_native_side_camera(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(
        """{
          "scene_id": "L1-A2",
          "native_preflight_sha256": "abc",
          "gates": {
            "G0": {"passed": true},
            "physics": {"passed": true},
            "visibility": {
              "passed": true,
              "policy_preprocessing": "pi05_robocasa_vertical_resize_with_pad_224",
              "policy_cameras": [
                "robot0_agentview_left",
                "robot0_eye_in_hand"
              ],
              "wrist_initial_frame": "wrist.png",
              "paired_wrist_initial_frames": {
                "Eb": "eb-wrist.png",
                "Er": "er-wrist.png",
                "Ec": "ec-wrist.png"
              }
            }
          }
        }"""
    )
    payload = load_smoke_gate_manifest(
        str(path),
        scene_id="L1-A2",
        preflight_sha256="abc",
        expected_policy_preprocessing=(
            "pi05_robocasa_vertical_resize_with_pad_224"
        ),
        expected_policy_cameras=(
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ),
    )
    assert payload["gates"]["visibility"]["passed"] is True
