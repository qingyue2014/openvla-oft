# L1-C5 Exact Policy-View and Physics Gate

- Verdict: **PASS_EXACT_STATE_PREVIEW**
- Native suite: `libero_object`
- Native prompt: `pick up the orange juice and place it in the basket`
- States per condition: 5
- Recognizable threshold: 100 policy-crop pixels
- Policy start: t=10
- Policy view: agentview/cosmos-vertical-flip/native-256.
- Forbidden initial contacts: occupant-target and occupant-robot.
- Translation, rotation, speeds, support, region membership, and forbidden contacts must pass at every stabilization step.
- Full-window timeline: `experiments/logs/l1c5_cosmos-preview_gate/l1c5_exact_state_preview_timeline.csv`.
- Human visibility verdict: **PENDING_REVIEW**.

| Cond | Ep | Occ t0/start px | Target t0/start px | Anchor t0/start px | In goal | Drift m | Rot deg | Contact | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eb | 0 | 755/755 | 525/685 | 3099/2977 | 0 | 0.0000 | 0.00 | 0 | 1 |
| eb | 1 | 755/755 | 525/685 | 3063/2928 | 0 | 0.0000 | 0.00 | 0 | 1 |
| eb | 2 | 755/755 | 525/685 | 3142/3017 | 0 | 0.0000 | 0.00 | 0 | 1 |
| eb | 3 | 755/755 | 525/686 | 3134/3022 | 0 | 0.0000 | 0.00 | 0 | 1 |
| eb | 4 | 755/755 | 525/685 | 3170/3046 | 0 | 0.0000 | 0.00 | 0 | 1 |
| er | 0 | 232/233 | 525/685 | 2867/2743 | 1 | 0.0001 | 0.04 | 0 | 1 |
| er | 1 | 233/234 | 525/685 | 2828/2693 | 1 | 0.0001 | 0.04 | 0 | 1 |
| er | 2 | 235/236 | 525/685 | 2907/2781 | 1 | 0.0001 | 0.04 | 0 | 1 |
| er | 3 | 244/243 | 525/686 | 2890/2778 | 1 | 0.0001 | 0.04 | 0 | 1 |
| er | 4 | 248/250 | 525/685 | 2922/2796 | 1 | 0.0001 | 0.04 | 0 | 1 |
| ec | 0 | 780/780 | 525/685 | 3099/2977 | 0 | 0.0000 | 0.00 | 0 | 1 |
| ec | 1 | 780/780 | 525/685 | 3063/2928 | 0 | 0.0000 | 0.00 | 0 | 1 |
| ec | 2 | 780/780 | 525/685 | 3142/3017 | 0 | 0.0000 | 0.00 | 0 | 1 |
| ec | 3 | 780/780 | 525/686 | 3134/3022 | 0 | 0.0000 | 0.00 | 0 | 1 |
| ec | 4 | 780/780 | 525/685 | 3170/3046 | 0 | 0.0000 | 0.00 | 0 | 1 |
