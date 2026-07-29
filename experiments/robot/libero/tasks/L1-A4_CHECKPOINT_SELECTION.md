# L1-A4 checkpoint selection

Date: 2026-07-29

## Fixed native experiment

- Suite/task: `libero_90`, task 14
- Native BDDL: `KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl`
- Exact prompt: `put the middle black bowl on the plate`
- EB states: the first 50 exact native serialized initial states
- Asset, BDDL, prompt, and task inventory interventions: none

Every reported run passed the native-only runtime preflight. A run-process
`PASS_L1A4_EB_CAPABILITY_RUN` verdict means that evaluation completed; it does
not by itself mean that the policy completed the task.

## Empirical screen

| Policy checkpoint | EB trials | Task successes | Valid executions | Model collapses | Decision |
|---|---:|---:|---:|---:|---|
| `RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora` | 50 | 0 | 50 | 0 | Reject |
| `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-90` | 10 | 0 | 10 | 0 | Reject |
| `RLinf/RLinf-OpenVLAOFT-LIBERO-130` | 10 | 0 | 10 | 0 | Reject |
| `VQ-VLA/openvla-7b-finetuned-libero-90` | 10 | 0 | 10 | 0 | Reject |
| `gs://openpi-assets/checkpoints/pi05_libero` | 10 | 5 | 10 | 0 | Expand |
| `gs://openpi-assets/checkpoints/pi05_libero` | 50 | 21 | 50 | 0 | Select with qualification |

The 50-trial pi0.5 EB success estimate is 42.0%, with a 95% Wilson interval of
29.4% to 55.8%.

## Selection

Select the official OpenPI `pi05_libero` checkpoint as the current L1-A4
checkpoint candidate because it is the only screened policy with observed EB
task successes. This is a **partial-capability selection**, not evidence of a
high-reliability EB pass. If the certification rule requires an EB success rate
of 80% or higher, no screened checkpoint qualifies and L1-A4 attribution must
remain gated.

Reproducible local ledgers:

- Base-Lora formal: `.physcog-agent/runs/20260729T035603Z-l1a4-formal`
- GRPO smoke: `.physcog-agent/runs/20260729T063005Z-l1a4-eb_capability`
- LIBERO-130 smoke: `.physcog-agent/runs/20260729T064050Z-l1a4-eb_capability_130`
- VQ-VLA smoke: `.physcog-agent/runs/20260729T065257Z-l1a4-eb_capability_vqvla`
- pi0.5 smoke: `.physcog-agent/runs/20260729T070307Z-l1a4-eb_capability_pi05`
- pi0.5 formal EB: `.physcog-agent/runs/20260729T070848Z-l1a4-eb_capability_pi05`

The pi0.5 probe runs the official policy server on GPU 0 and the unchanged
native L1-A4 evaluator on GPU 1 through
`run_l1a4_pi05_capability.sh`.
