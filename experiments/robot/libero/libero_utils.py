"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time

import imageio
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

# TensorFlow is used here only for lightweight image preprocessing.  Keep it
# off the accelerator so it neither reserves OpenVLA's GPU memory nor triggers
# a long PTX JIT on newer (for example sm_90) evaluation nodes.
if tf is not None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        # A caller may already have initialized TensorFlow before importing
        # this module; its device policy can no longer be changed.
        pass

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, model_family, resolution=256, render_gpu_device_id=-1):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": False,
        "render_gpu_device_id": render_gpu_device_id,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    rollout_dir=None,
    model_family="openvla",
):
    """Saves an MP4 replay of an episode."""
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    model_label = (
        "openvla_oft"
        if model_family == "openvla"
        else str(model_family).lower().replace(".", "").replace("_", "-")
    )
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--{model_label}--episode={idx}"
        f"--success={success}--task={processed_task_description}.mp4"
    )
    # Force the imageio-ffmpeg backend, which encodes in an ffmpeg
    # subprocess. The in-process pyav backend shares the heap with MuJoCo's
    # EGL renderer and has been observed to corrupt it (SIGABRT in
    # read_pixels on the episode after the first video write).
    try:
        video_writer = imageio.get_writer(mp4_path, fps=30, format="FFMPEG")
    except Exception:
        video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
