"""Pure event-order logic shared by L3-A2 validators and unit tests."""

from __future__ import annotations

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
