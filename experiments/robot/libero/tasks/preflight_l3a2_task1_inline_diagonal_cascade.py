#!/usr/bin/env python3
"""One bounded no-VLA scan for a near-inline native task1 diagonal cascade."""

from experiments.robot.libero.tasks.preflight_l3a2_task1_diagonal_cascade import (
    main,
)


FROZEN_A_SEEDS = (
    (16.0, 0.0),
    (16.0, -0.0015),
    (16.0, 0.0015),
)
B_TURN_DEG = (0.0, 10.0, 20.0)
B_CLEARANCE_M = (0.0, 0.002, 0.004)


if __name__ == "__main__":
    main(
        out_name="l3a2_task1_inline_diagonal_cascade",
        a_scan_points=FROZEN_A_SEEDS,
        fixed_a_seed_points=FROZEN_A_SEEDS,
        b_turn_deg=B_TURN_DEG,
        b_clearance_m=B_CLEARANCE_M,
        selection_turn_deg=10.0,
        selection_clearance_m=0.002,
        candidate_identity="task1_near_inline_diagonal_native_S_A_B",
        mechanism=(
            "remove native target S; frozen robust cookies A falls on the "
            "world -45 degree diagonal, then impacts a near-inline native "
            "bowl B at 0/10/20 degrees from the measured fall ray"
        ),
        verdict_tag="L3A2_TASK1_INLINE_DIAGONAL_ONE_STATE_NO_VLA_GATE",
        artifact_prefix="task1_inline_selected",
    )
