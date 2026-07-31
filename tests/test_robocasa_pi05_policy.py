from collections import OrderedDict

import numpy as np
import pytest

from experiments.robot.robocasa.pi05_policy import (
    LIBERO_STATE_MEAN_POS,
    build_request,
    canonicalize_robocasa_state,
    map_libero_action_to_pandaomron,
    preprocess_camera_image,
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


def test_build_request_uses_native_prompt_and_eight_dimensional_state():
    request = build_request(_obs(), "pick up the mug")
    assert request["prompt"] == "pick up the mug"
    assert request["observation/image"].shape == (224, 224, 3)
    assert request["observation/wrist_image"].shape == (224, 224, 3)
    assert request["observation/state"].shape == (8,)


def test_canonical_state_anchors_initial_position_to_libero_mean():
    state, anchor = canonicalize_robocasa_state(
        _obs(), _Env(), position_anchor=None
    )
    np.testing.assert_allclose(state[:3], LIBERO_STATE_MEAN_POS)
    np.testing.assert_allclose(anchor, [0.1, 0.2, 0.3])

    moved = _obs()
    moved["robot0_eef_pos"] = np.array([0.11, 0.18, 0.33])
    moved_state, reused_anchor = canonicalize_robocasa_state(
        moved, _Env(), position_anchor=anchor
    )
    np.testing.assert_allclose(
        moved_state[:3],
        LIBERO_STATE_MEAN_POS + np.array([0.01, -0.02, 0.03]),
    )
    np.testing.assert_allclose(reused_anchor, anchor)


def test_map_pi05_action_freezes_mobile_base_and_torso():
    mapped = map_libero_action_to_pandaomron(np.arange(7) / 10, _Env())
    np.testing.assert_allclose(mapped[:6], np.arange(6) / 10)
    assert mapped[6] == 0.0
    np.testing.assert_allclose(mapped[7:10], 0.0)
    assert mapped[10] == pytest.approx(0.6)
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
            "visibility": {"passed": true}
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
