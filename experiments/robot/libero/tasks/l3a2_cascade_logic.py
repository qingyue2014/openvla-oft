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
    yaw_offsets_deg: tuple[float, ...],
    quantization: float,
    yaw_quantization_deg: float,
    limit: int,
) -> tuple[list[tuple[float, float, float]], list[dict[str, Any]]]:
    """Generate panel poses across A's measured post-release swept stations."""
    if (
        not axial_stations
        or not normal_offsets
        or not yaw_offsets_deg
        or min(axial_stations) <= 0
        or min(normal_offsets) < 0
        or tangent_offset <= 0
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
                yaw_deg = math.degrees(math.atan2(normal_y, normal_x))
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
                    "motion_yaw_deg": float(yaw_deg),
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
                        y = (
                            point_y
                            + normal_offset * normal_y
                            + tangent_shift * tangent_y
                        )
                        x = round(x / quantization) * quantization
                        y = round(y / quantization) * quantization
                        for yaw_offset in yaw_offsets_deg:
                            yaw = round(
                                (yaw_deg + yaw_offset)
                                / yaw_quantization_deg
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
            ) <= max(normal_offsets) + tangent_offset:
                counts[candidate] = counts.get(candidate, 0) + 1
    ordered = sorted(
        candidates,
        key=lambda pose: (
            -counts.get(pose, 0), pose[0], pose[1], pose[2]
        ),
    )
    return ordered[:limit], trace_rows
