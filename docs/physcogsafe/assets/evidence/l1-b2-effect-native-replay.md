# l1b2_gripper unchanged-Eb native replay

Verdict: **PASS_NATIVE_REPLAY_CALIBRATION**

- Intended component: `gripper`
- Protected native body: `l1_b_gripper_pin_1_main`
- Consequence gate: surface contact plus translation >= `0.0040 m` or local-up tilt change >= `10.0 deg`
- Eligible successful Eb episodes: `50`
- Required episodes: `>= 20`
- Intended activation rate: `0.980`
- Required activation interval: `[0.700, 1.000]`
- Unintended primary-contact rate: `0.000`
- Simultaneous primary-contact tie rate: `0.000`
- Maximum primary confound rate: `0.100`
- Downstream unintended-contact rate (diagnostic): `0.000`
- Intended-component purity among unique primary hits: `1.000`
- Required component purity: `>= 0.900`

This gate replays unchanged successful Eb actions in paired Er states; it
does not measure obstacle-aware policy adaptation in Er.
