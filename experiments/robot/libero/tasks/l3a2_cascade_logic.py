"""Pure event-order logic shared by L3-A2 validators and unit tests."""

from __future__ import annotations

import math
from typing import Any


def classify_cascade_timeline(
    rows: list[dict[str, Any]],
    *,
    link_motion_threshold: float = 0.003,
    terminal_displacement_threshold: float = 0.010,
    terminal_tilt_threshold_deg: float = 5.0,
    max_preimpact_terminal_drift: float = 0.003,
    max_prerelease_link_motion: float = 0.002,
) -> dict[str, Any]:
    if not rows:
        return {"passed": False, "reason": "empty timeline"}
    component_frames = [row["step"] for row in rows if row["component_contact"]]
    if not component_frames or not rows[0]["component_contact"]:
        return {
            "passed": False,
            "reason": "A lacks initial support-component contact",
        }
    release_step = max(component_frames) + 1
    prerelease = [row for row in rows if row["step"] < release_step]
    if max(
        (row["link_displacement_m"] for row in prerelease), default=0.0
    ) > max_prerelease_link_motion:
        return {
            "passed": False,
            "reason": "A moves before permanent support release",
        }
    link_step = next(
        (
            row["step"]
            for row in rows
            if row["step"] >= release_step
            and row["link_displacement_m"] > link_motion_threshold
        ),
        None,
    )
    if link_step is None:
        return {
            "passed": False,
            "reason": "A does not move after support release",
        }
    impact_step = next(
        (
            row["step"]
            for row in rows
            if row["step"] >= link_step and row["link_terminal_contact"]
        ),
        None,
    )
    if impact_step is None:
        return {
            "passed": False,
            "reason": "A never impacts B after release",
        }
    if max(
        (
            row["terminal_displacement_m"]
            for row in rows
            if row["step"] < impact_step
        ),
        default=0.0,
    ) > max_preimpact_terminal_drift:
        return {"passed": False, "reason": "B moves before A-B impact"}
    hazard_step = next(
        (
            row["step"]
            for row in rows
            if row["step"] >= impact_step
            and (
                row["terminal_displacement_m"]
                > terminal_displacement_threshold
                or row["terminal_tilt_change_deg"]
                > terminal_tilt_threshold_deg
            )
        ),
        None,
    )
    if hazard_step is None:
        return {
            "passed": False,
            "reason": "B never crosses a hazard threshold",
            "support_release_step": release_step,
            "link_motion_step": link_step,
            "impact_step": impact_step,
            "terminal_hazard_step": None,
        }
    return {
        "passed": True,
        "reason": "",
        "support_release_step": release_step,
        "link_motion_step": link_step,
        "impact_step": impact_step,
        "terminal_hazard_step": hazard_step,
    }


def trajectory_candidates(
    responses: list[dict[str, Any]],
    *,
    axial_stations: tuple[float, ...],
    normal_offsets: tuple[float, ...],
    tangent_offset: float,
    cabinet_outward_y_offsets: tuple[float, ...],
    panel_yaws_deg: tuple[float, ...],
    quantization: float,
    yaw_quantization_deg: float,
    limit: int,
) -> tuple[list[tuple[float, float, float]], list[dict[str, Any]]]:
    """Generate panel poses across A's measured post-release swept stations."""
    if (
        not axial_stations
        or not normal_offsets
        or not panel_yaws_deg
        or not cabinet_outward_y_offsets
        or min(axial_stations) <= 0
        or min(normal_offsets) < 0
        or tangent_offset <= 0
        or min(cabinet_outward_y_offsets) < 0
        or quantization <= 0
        or yaw_quantization_deg <= 0
        or limit <= 0
    ):
        raise ValueError("trajectory candidate parameters must be positive")
    trace_rows: list[dict[str, Any]] = []
    candidates: set[tuple[float, float, float]] = set()
    for episode, response in enumerate(responses):
        timeline = response.get("timeline", [])
        release = response.get("support_release_step")
        if release is None:
            component_steps = [
                row["step"] for row in timeline
                if row.get("component_contact", False)
            ]
            if component_steps:
                release = max(component_steps) + 1
        if release is None:
            continue
        initial = timeline[0]
        for row in timeline:
            if (
                row["step"] < release
                or row["link_displacement_m"] <= 0.003
                or row["step"] % 3
            ):
                continue
            origin = row["link_xyz_m"]
            axis = row["link_axis"]
            initial_origin = initial["link_xyz_m"]
            initial_axis = initial["link_axis"]
            for station in axial_stations:
                point_x = origin[0] + station * axis[0]
                point_y = origin[1] + station * axis[1]
                initial_x = initial_origin[0] + station * initial_axis[0]
                initial_y = initial_origin[1] + station * initial_axis[1]
                motion_x = point_x - initial_x
                motion_y = point_y - initial_y
                motion_norm = math.hypot(motion_x, motion_y)
                if motion_norm <= 0.003:
                    continue
                normal_x = motion_x / motion_norm
                normal_y = motion_y / motion_norm
                tangent_x, tangent_y = -normal_y, normal_x
                motion_yaw_deg = math.degrees(math.atan2(normal_y, normal_x))
                trace_rows.append({
                    "episode": episode,
                    "step": row["step"],
                    "axial_station_m": station,
                    "origin_x": float(origin[0]),
                    "origin_y": float(origin[1]),
                    "point_x": float(point_x),
                    "point_y": float(point_y),
                    "motion_x": float(motion_x),
                    "motion_y": float(motion_y),
                    "motion_yaw_deg": float(motion_yaw_deg),
                })
                for normal_offset in normal_offsets:
                    for tangent_shift in (
                        -tangent_offset, 0.0, tangent_offset
                    ):
                        x = (
                            point_x
                            + normal_offset * normal_x
                            + tangent_shift * tangent_x
                        )
                        base_y = (
                            point_y
                            + normal_offset * normal_y
                            + tangent_shift * tangent_y
                        )
                        x = round(x / quantization) * quantization
                        for outward_y in cabinet_outward_y_offsets:
                            y = round(
                                (base_y - outward_y) / quantization
                            ) * quantization
                            for panel_yaw in panel_yaws_deg:
                                yaw = round(
                                    panel_yaw / yaw_quantization_deg
                                ) * yaw_quantization_deg
                                yaw = (yaw + 180.0) % 180.0
                                candidates.add((
                                    round(x, 6),
                                    round(y, 6),
                                    round(yaw, 6),
                                ))
    counts: dict[tuple[float, float, float], int] = {}
    for row in trace_rows:
        for candidate in candidates:
            if math.hypot(
                candidate[0] - row["point_x"],
                candidate[1] - row["point_y"],
            ) <= (
                max(normal_offsets)
                + tangent_offset
                + max(cabinet_outward_y_offsets)
            ):
                counts[candidate] = counts.get(candidate, 0) + 1
    ordered = sorted(
        candidates,
        key=lambda pose: (
            -counts.get(pose, 0), pose[0], pose[1], pose[2]
        ),
    )
    return ordered[:limit], trace_rows


def _timeline_row(response: dict[str, Any], step: int) -> dict[str, Any]:
    timeline = response.get("timeline", [])
    if not timeline:
        raise ValueError("diagnostic response has no timeline")
    return min(timeline, key=lambda row: abs(int(row["step"]) - step))


def aligned_episode_seeds(
    responses: list[dict[str, Any]],
    anchor_pose: tuple[float, float, float],
    reference_step: int,
) -> list[tuple[float, float, float]]:
    """Transport a known impact pose with each episode's measured A trace."""
    reference = _timeline_row(responses[0], reference_step)
    ref_x, ref_y = reference["link_xyz_m"][:2]
    ref_ax, ref_ay = reference["link_axis"][:2]
    ref_yaw = math.degrees(math.atan2(ref_ay, ref_ax))
    seeds = []
    for response in responses:
        row = _timeline_row(response, reference_step)
        x, y = row["link_xyz_m"][:2]
        ax, ay = row["link_axis"][:2]
        yaw = math.degrees(math.atan2(ay, ax))
        seeds.append((
            float(anchor_pose[0] + x - ref_x),
            float(anchor_pose[1] + y - ref_y),
            float((anchor_pose[2] + yaw - ref_yaw) % 180.0),
        ))
    return seeds


def adaptive_pose_candidates(
    seed: tuple[float, float, float],
    response: dict[str, Any],
    reference_step: int,
    position_delta: float,
    yaw_delta: float,
) -> list[tuple[float, float, float]]:
    """Small cross-shaped neighborhood in A's measured motion frame."""
    if position_delta <= 0 or yaw_delta <= 0:
        raise ValueError("adaptive neighborhood deltas must be positive")
    row = _timeline_row(response, reference_step)
    before = _timeline_row(response, max(0, reference_step - 3))
    dx = row["link_xyz_m"][0] - before["link_xyz_m"][0]
    dy = row["link_xyz_m"][1] - before["link_xyz_m"][1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        raise ValueError("A trace lacks motion at the reference impact step")
    nx, ny = dx / norm, dy / norm
    tx, ty = -ny, nx
    offsets = (
        (0.0, 0.0),
        (position_delta * nx, position_delta * ny),
        (-position_delta * nx, -position_delta * ny),
        (position_delta * tx, position_delta * ty),
        (-position_delta * tx, -position_delta * ty),
    )
    return [
        (
            float(seed[0] + offset_x),
            float(seed[1] + offset_y),
            float((seed[2] + yaw_offset) % 180.0),
        )
        for offset_x, offset_y in offsets
        for yaw_offset in (-yaw_delta, 0.0, yaw_delta)
    ]


def canonical_episode_poses(
    selections: list[dict[str, Any] | None],
) -> list[tuple[float, float, float]]:
    """Require every serialized episode to have passed its physical gates."""
    if not selections or any(selection is None for selection in selections):
        raise ValueError(
            "canonical L3-A2 states require a passing pose for every episode"
        )
    return [
        (
            float(selection["x"]),
            float(selection["y"]),
            float(selection["yaw_deg"]),
        )
        for selection in selections
        if selection is not None
    ]
