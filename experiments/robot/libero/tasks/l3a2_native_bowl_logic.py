"""Pure, dependency-light logic for the native-only L3-A2 preflight."""

from __future__ import annotations

from typing import Any

import numpy as np


def outside_ab_exact(
    states: tuple[np.ndarray, np.ndarray, np.ndarray],
    a_slice: tuple[int, int],
    b_slice: tuple[int, int],
) -> bool:
    """Return true only when paired states match outside A/B free joints."""
    mask = np.ones(len(states[0]), dtype=bool)
    for qpos, qvel in (a_slice, b_slice):
        mask[qpos:qpos + 7] = False
        mask[qvel:qvel + 6] = False
    return all(np.array_equal(states[0][mask], state[mask]) for state in states[1:])


def trajectory_candidates(
    response: dict[str, Any],
    limit: int,
) -> list[tuple[float, float]]:
    """Return a bounded 5-mm grid around the measured native-A fall trace."""
    if limit <= 0:
        raise ValueError("candidate limit must be positive")
    rows = response["timeline"]
    release = response.get("support_release_step")
    if release is None:
        component_steps = [
            int(row["step"]) for row in rows if row["component_contact"]
        ]
        release = max(component_steps) + 1 if component_steps else None
    candidates: set[tuple[float, float]] = set()
    if release is not None:
        moving = [
            row for row in rows
            if row["step"] >= release and row["link_displacement_m"] > 0.003
        ]
        for before, row in zip(moving, moving[2:]):
            dx = row["link_xyz_m"][0] - before["link_xyz_m"][0]
            dy = row["link_xyz_m"][1] - before["link_xyz_m"][1]
            norm = float(np.hypot(dx, dy))
            if norm < 1e-5:
                continue
            nx, ny = dx / norm, dy / norm
            tx, ty = -ny, nx
            for advance in (0.050, 0.060, 0.070, 0.080):
                for tangent in (
                    -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015
                ):
                    x = row["link_xyz_m"][0] + advance * nx + tangent * tx
                    y = row["link_xyz_m"][1] + advance * ny + tangent * ty
                    candidates.add((
                        round(x / 0.005) * 0.005,
                        round(y / 0.005) * 0.005,
                    ))
    # Frozen fallback window around the previously measured native bottle fall
    # trace. It changes no asset parameter and stays bounded.
    for x in np.arange(0.075, 0.141, 0.005):
        for y in np.arange(0.015, 0.081, 0.005):
            candidates.add((round(float(x), 3), round(float(y), 3)))
    ordered = sorted(
        candidates,
        key=lambda xy: (
            abs(xy[0] - 0.108) + abs(xy[1] - 0.048),
            xy,
        ),
    )
    return ordered[:limit]
