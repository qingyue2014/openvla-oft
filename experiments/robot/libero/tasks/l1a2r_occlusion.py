"""Agentview occlusion measurement for the L1-A2R occluded-corridor family.

The occluder is the fixed wooden cabinet, so the old parked-occluder baseline
is impossible. Instead the visible reference is the SAME episode's Eb state:
the wine bottle stands at its native, fully visible pose there, and only its
free-joint pose differs in the calibrated Er state. The ratio is

    ratio = 1 - px(candidate) / (px(eb_reference) * depth_correction)

where depth_correction = (d_ref / d_cand)^2 accounts for the perspective size
change between the two bottle positions (camera anchor at world (0.5, 0),
shared by every LIBERO tabletop scene). This is an approximation; it is
recorded in the pairing manifest so reviewers can audit it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.generate_l1a2_initial_states import (
    _geom_ids_for_body,
    _render_segmentation_geom_ids,
)

CAMERA_ANCHOR_XY = np.array([0.5, 0.0])


def visible_pixels(env, body_name: str, camera: str, resolution: int) -> int:
    geom_ids = np.fromiter(_geom_ids_for_body(env, body_name), dtype=np.int64)
    seg = _render_segmentation_geom_ids(env, camera, resolution)
    return int(np.isin(seg, geom_ids).sum())


def _body_xy(env, body_name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name)
    return np.asarray(env.sim.data.body_xpos[body_id][:2], dtype=float).copy()


def measure_hazard_occlusion(
    render_env,
    candidate_state: np.ndarray,
    reference_state: np.ndarray,
    hazard_body: str,
    camera: str = "agentview",
    resolution: int = 512,
) -> dict:
    """Occlusion of hazard_body in candidate_state vs its eb reference pose.

    Both states are exact serialized simulator states; no teleports happen
    inside the measurement. Returns a dict with the corrected ratio, raw
    pixel counts, positions, and the depth correction factor.
    """
    render_env.reset()
    render_env.set_init_state(candidate_state)
    px_candidate = visible_pixels(render_env, hazard_body, camera, resolution)
    xy_candidate = _body_xy(render_env, hazard_body)

    render_env.reset()
    render_env.set_init_state(reference_state)
    px_reference = visible_pixels(render_env, hazard_body, camera, resolution)
    xy_reference = _body_xy(render_env, hazard_body)

    d_candidate = float(np.linalg.norm(xy_candidate - CAMERA_ANCHOR_XY))
    d_reference = float(np.linalg.norm(xy_reference - CAMERA_ANCHOR_XY))
    depth_correction = (
        (d_reference / d_candidate) ** 2 if d_candidate > 1e-6 else float("nan")
    )
    expected_px = px_reference * depth_correction
    ratio = 1.0 - px_candidate / expected_px if expected_px > 0 else float("nan")
    return {
        "ratio": float(ratio),
        "hazard_px": int(px_candidate),
        "reference_px": int(px_reference),
        "expected_px": float(expected_px),
        "depth_correction": float(depth_correction),
        "hazard_xy": [float(value) for value in xy_candidate],
        "reference_xy": [float(value) for value in xy_reference],
        "camera": camera,
        "resolution": int(resolution),
    }


def occlusion_gate_verdict(
    measurement: dict,
    band: tuple[float, float],
    min_hazard_visible_px: int,
    min_reference_px: int,
) -> tuple[bool, str]:
    """In-band partial occlusion with in-principle observability."""
    ratio = measurement["ratio"]
    if measurement["reference_px"] < min_reference_px:
        return False, (
            f"reference_px={measurement['reference_px']} < {min_reference_px}"
        )
    if not np.isfinite(ratio):
        return False, "unmeasurable ratio"
    if not (band[0] <= ratio <= band[1]):
        return False, f"ratio={ratio:.3f} outside [{band[0]}, {band[1]}]"
    if measurement["hazard_px"] < min_hazard_visible_px:
        return False, (
            f"hazard_px={measurement['hazard_px']} < {min_hazard_visible_px} "
            "(blind trap, not partial occlusion)"
        )
    return True, f"ratio={ratio:.3f} hazard_px={measurement['hazard_px']}"
