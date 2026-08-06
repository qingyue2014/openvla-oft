import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.robot.cosmos_policy_utils import (
    prepare_cosmos_libero_observation,
    validate_cosmos_actions,
)
from experiments.robot.libero.tasks.physcog_remote_agent import PHASES


def test_cosmos_action_and_observation_contract():
    actions = validate_cosmos_actions(np.zeros((16, 7), dtype=np.float64))
    assert actions.shape == (16, 7)
    assert actions.dtype == np.float32
    with pytest.raises(ValueError):
        validate_cosmos_actions(np.zeros((16, 8)))

    agent = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    wrist = agent + 20
    prepared = prepare_cosmos_libero_observation(
        {
            "agentview_image": agent,
            "robot0_eye_in_hand_image": wrist,
            "robot0_gripper_qpos": np.array([0.1, 0.2]),
            "robot0_eef_pos": np.array([0.3, 0.4, 0.5]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    )
    np.testing.assert_array_equal(prepared["primary_image"], np.flipud(agent))
    np.testing.assert_array_equal(prepared["wrist_image"], np.flipud(wrist))
    assert prepared["proprio"].shape == (9,)


def test_l1c1_cosmos_wrapper_pins_runtime_and_frozen_scene_gate():
    wrapper = Path("experiments/robot/libero/tasks/run_l1c1_cosmos.sh").read_text(
        encoding="utf-8"
    )
    assert "18a2accadf4e7a3531e56754102af5a24d2316da" in wrapper
    assert "Cosmos-Policy-LIBERO-Predict2-2B" in wrapper
    assert "MODEL_OPEN_LOOP_STEPS=16" in wrapper
    assert "MODEL_FAMILY=cosmos" in wrapper
    assert "SLURM_JOB_ID" in wrapper
    assert 'LIBERO_NATIVE_SOURCE_ROOT="${LIBERO_ROOT}"' in wrapper
    assert 'PYTHONPATH="${NATIVE_LIBERO_SITE_DIR}:${COSMOS_SOURCE_ROOT}' in wrapper
    assert 'test -f "${LIBERO_ROOT}/libero/libero/__init__.py"' in wrapper
    assert "PASS_L1C1_COSMOS_NATIVE_LIBERO_SOURCE" in wrapper

    smoke = PHASES[("l1c1", "cosmos_smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "SAVE_VIDEO_MODE=capped" in smoke.command
    assert "experiments/robot/libero/tasks/run_l1c1_cosmos.sh" in smoke.command
    assert any(item.endswith("check_l1c1_cascade_gate.py") for item in smoke.local_gate)
    assert len(smoke.inputs) == 5


def test_cosmos_evaluator_does_not_eagerly_import_openvla_stack():
    evaluator = Path("experiments/robot/libero/run_libero_eval.py").read_text(
        encoding="utf-8"
    )
    physcog = Path(
        "experiments/robot/libero/run_physcog_libero_l1_eval.py"
    ).read_text(encoding="utf-8")
    assert "from prismatic" not in evaluator
    assert "from prismatic" not in physcog
    assert "OPENVLA_LIBERO_NUM_ACTIONS_CHUNK = 8" in evaluator


def test_native_libero_site_guard_overrides_regular_venv_package(tmp_path):
    source_root = tmp_path / "approved"
    approved_package = source_root / "libero" / "libero"
    approved_package.mkdir(parents=True)
    (approved_package / "__init__.py").write_text(
        "ORIGIN = 'approved'\n", encoding="utf-8"
    )

    venv_root = tmp_path / "venv"
    regular_package = venv_root / "libero"
    regular_package.mkdir(parents=True)
    (regular_package / "__init__.py").write_text(
        "ORIGIN = 'wrong-venv-copy'\n", encoding="utf-8"
    )

    guard_dir = Path("experiments/robot/libero/native_libero_site").resolve()
    env = os.environ.copy()
    env["LIBERO_NATIVE_SOURCE_ROOT"] = str(source_root)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(guard_dir), str(venv_root), str(source_root))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import libero, libero.libero; "
                "print(libero.__file__); "
                "print(libero.libero.ORIGIN); "
                "print(libero.libero.__file__)"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "None"
    assert lines[1] == "approved"
    assert Path(lines[2]).resolve() == approved_package / "__init__.py"
