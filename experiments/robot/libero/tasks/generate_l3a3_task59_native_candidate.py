"""One-state static/visual precheck for the native-only task59 L3-A3 candidate."""

from __future__ import annotations

from experiments.robot.libero.tasks import (
    l3a3_native_tower_candidate_common as candidate,
)
from experiments.robot.libero.tasks.diagnose_l3a3_task59_can_alignment import (
    place_with_exact_aabb,
)


def place_on_top_exact(
    sim,
    body: str,
    support: str,
    xy,
    quat,
    clearance: float = 0.0005,
) -> None:
    place_with_exact_aabb(
        sim,
        body,
        support,
        xy,
        quat,
        clearance_m=clearance,
    )


candidate.TASK_ID = 59
candidate.TASK_LABEL = "task59"
candidate.PROMPT = "pick up the tomato sauce and put it in the tray"
candidate.PROMPT_SHA256 = (
    "289571a0f835287ad32a27e72b5f17c98ac1bc9ec772665c64884c80db4ab2c0"
)
candidate.NATIVE_BDDL_SHA256 = (
    "7580a3282b33142c441a3a4f906e7f88415a9a734b3ef14e59e22f3a8d7d3315"
)
candidate.GOAL_SHA256 = (
    "a3cb4109ca75f8e64024e9cf63066478505f44b9b95946a4d98fb95c59bb00b9"
)
candidate.SCHEMA = "physcog_l3a3_task59_native_tower_v1_one_state"
candidate.KEY = candidate.PROMPT.replace(" ", "_")
candidate.ROLE_SUMMARY = "tomato sauce / alphabet soup / butter"
candidate.PLACEMENT_AUDIT = {
    "method": "exact_compiled_collision_geom_world_aabb",
    "diagnostic_job_id": 490080,
    "diagnostic_commit": "5a771c4b44aaa6a85d65318d34d31691a05674c5",
    "grid_stable_count": 25,
    "grid_total_count": 25,
    "robust_adjacent_witness_count": 25,
    "selected_S_A_offset_m": [0.0, 0.0],
    "clearance_m": 0.0005,
}
candidate.place_on_top = place_on_top_exact

candidate.SUPPORT = "tomato_sauce_1_main"
candidate.MIDDLE = "alphabet_soup_1_main"
candidate.TOP = "butter_1_main"
candidate.TRAY = "wooden_tray_1_main"
candidate.RELEVANT = (
    candidate.SUPPORT,
    candidate.MIDDLE,
    candidate.TOP,
    candidate.TRAY,
)
candidate.OTHER_NATIVE = ("cream_cheese_1_main", "ketchup_1_main")


if __name__ == "__main__":
    candidate.main()
