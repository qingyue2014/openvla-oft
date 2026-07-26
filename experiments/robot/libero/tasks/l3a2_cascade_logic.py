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
    half_length: float,
    offset: float,
    quantization: float,
    limit: int,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    """Generate B centers around A's measured post-release swept endpoints."""
    if half_length <= 0 or offset <= 0 or quantization <= 0 or limit <= 0:
        raise ValueError("trajectory candidate parameters must be positive")
    trace_rows: list[dict[str, Any]] = []
    candidates: set[tuple[float, float]] = set()
    offsets = (
        (-offset, -offset), (-offset, 0.0), (-offset, offset),
        (0.0, -offset), (0.0, 0.0), (0.0, offset),
        (offset, -offset), (offset, 0.0), (offset, offset),
    )
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
        for row in timeline:
            if (
                row["step"] < release
                or row["link_displacement_m"] <= 0.003
                or row["step"] % 3
            ):
                continue
            center = row["link_xyz_m"]
            axis = row["link_axis"]
            for sign in (-1.0, 1.0):
                endpoint_x = center[0] + sign * half_length * axis[0]
                endpoint_y = center[1] + sign * half_length * axis[1]
                trace_rows.append({
                    "episode": episode,
                    "step": row["step"],
                    "endpoint_sign": int(sign),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "endpoint_x": float(endpoint_x),
                    "endpoint_y": float(endpoint_y),
                })
                for dx, dy in offsets:
                    x = round(
                        float(endpoint_x + dx) / quantization
                    ) * quantization
                    y = round(
                        float(endpoint_y + dy) / quantization
                    ) * quantization
                    candidates.add((round(x, 6), round(y, 6)))
    counts: dict[tuple[float, float], int] = {}
    for row in trace_rows:
        for candidate in candidates:
            if math.hypot(
                candidate[0] - row["endpoint_x"],
                candidate[1] - row["endpoint_y"],
            ) <= offset * 1.5:
                counts[candidate] = counts.get(candidate, 0) + 1
    ordered = sorted(
        candidates,
        key=lambda xy: (-counts.get(xy, 0), xy[0], xy[1]),
    )
    return ordered[:limit], trace_rows
