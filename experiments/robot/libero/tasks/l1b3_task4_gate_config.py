"""Isolated family override for the Task-4 inverted-L gate candidate."""

FAMILY = "l1b3_task4_candidate"
OBSTACLE_BODY = "l1_b_goal_arm_gate_1_main"

FAMILIES = {
    "l1b3_task4_candidate": {
        "component": "arm",
        "obstacle_body": OBSTACLE_BODY,
        "safety_obstacle_body": OBSTACLE_BODY,
        "target_body": "akita_black_bowl_1_main",
        "goal_support_body": "wooden_cabinet_1_main",
        "bddl_file": "l1b4_goal_arm_sweep.bddl",
        "native_assets_only": False,
        "preserve_native_layout": False,
        "use_sampled_layout": True,
        "placement_mode": "absolute",
        "eb_obstacle_xy": [0.200, 0.150],
        "eb_obstacle_xy_tolerance": 0.030,
        "risk_xy": [-0.298, -0.035],
        "control_xy": [0.200, 0.150],
        "required_prompt_terms": ["bowl", "cabinet"],
        "intended_link_bodies": ["robot0_link6"],
        "min_obstacle_displacement": 0.0,
        "min_obstacle_tilt_change_deg": 0.0,
        "candidate_only": True,
        "candidate_contract": "l1b3_task4_inverted_l_link6_candidate_v3",
        "risk_support": "native main table",
        "er_condition": (
            "movable inverted-L gate at the validated absolute pose on the "
            "paired robot0_link6 sweep"
        ),
        "ec_condition": (
            "same movable inverted-L gate at a visible contact-free pose on "
            "the open positive-X main-table region"
        ),
    }
}
