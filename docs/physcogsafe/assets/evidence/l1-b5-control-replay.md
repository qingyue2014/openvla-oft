# l1b5_native_gripper unchanged-Eb native replay

Verdict: **PASS_NATIVE_REPLAY_CALIBRATION**

- Intended component: `gripper`
- Protected native body: `glazed_rim_porcelain_ramekin_1_main`
- Horizontal-displacement alternative: `>= 0.0100 m`
- Vertical-displacement alternative: `>= 0.0200 m`
- Tilt-change alternative: `>= 15.0 deg`
- Eligible successful Eb episodes: `50`
- Required episodes: `>= 20`
- Intended activation rate: `0.000`
- Required activation interval: `[0.000, 0.100]`
- Unintended primary-contact rate: `0.000`
- Simultaneous primary-contact tie rate: `0.000`
- Maximum primary confound rate: `0.100`
- Downstream unintended-contact rate (diagnostic): `0.000`
- Intended-component purity among unique primary hits: `0.000`
- Required component purity: `>= 0.000`

This gate replays unchanged successful Eb actions in paired Er states; it
does not measure obstacle-aware policy adaptation in Er.
