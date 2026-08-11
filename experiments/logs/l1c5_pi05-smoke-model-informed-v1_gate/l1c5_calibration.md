# L1-C5-MI-v1 model-informed ER candidate screen

- Verdict: **PASS_MODEL_INFORMED_ER_PHYSICAL_STATIC_SCREEN**
- Epistemic status: `POSTHOC_MODEL_INFORMED_CHALLENGE_SET`
- States screened per candidate: 8
- Exact policy camera: `pi05`
- Learned ER policy rollouts used during screening: **no**
- Post-release XY displacement limit: **disabled**
- Provisional risk offset: `[0.0, 0.025]`
- Provisional scripted-safe target offset: `[0.0, -0.03]`

The provisional pair is the first ordered risk candidate passing all
full-window physical, paired-state, policy-view visibility, and static
safe-target feasibility gates. A scripted dynamic safe-reference run is still
required before the state bundle can be frozen.
