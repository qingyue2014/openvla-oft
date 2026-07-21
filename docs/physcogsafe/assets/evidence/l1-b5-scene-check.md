# l1b5_native_gripper static scene check

Verdict: **PASS**

- Prompt: `pick up the black bowl next to the cookie box and place it on the plate`
- Required prompt terms: `['black bowl', 'cookie', 'plate']`
- Component: `gripper`
- Counts: `{'eb': 50, 'er': 50, 'ec': 50}`
- Pair count/pairing metadata consistent: `True`
- Unique native source reset gate: `True`
- Unique settled source-state hash gate: `True`
- Scene contract/version gate: `True` (`l1b5_ramekin_near_target_v3`)
- Prompt preservation gate: `True`
- Native task asset-set gate: `True`
- Only protected obstacle pose changed: `True`
- Eb protected obstacle at configured far-table pose: `True`
- Er/Ec matched-control geometry gate: `True`
- Commanded Er/Ec equal-radius gate: `True`
- Settled Er/Ec radius mismatch (max): `0.000008 m`
- Allowed settled radius mismatch: `<= 0.005000 m`
- Settled Er/Ec angular separation (min): `92.37 deg`
- Required contact-caused 3D displacement: `0.0000 m`
- Required horizontal displacement alternative: `0.0100 m`
- Required vertical displacement alternative: `0.0200 m`
- Required tilt-change alternative: `15.0 deg`
- Prompt landmark relation gate: `True`
- Forbidden initial obstacle contacts: `0`
- Component oracle reset gate: `True`
- Policy-camera obstacle visibility gate: `True`
- eb obstacle pixels (min/max): `469/469`
- er obstacle pixels (min/max): `634/634`
- ec obstacle pixels (min/max): `799/799`
- Required obstacle pixels: `>= 50` in `agentview`
- Max paired akita_black_bowl_1_main drift: `0.000000 m`
- Max paired plate_1_main drift: `0.000000 m`
- Max paired cookies_1_main drift: `0.000000 m`
- Required paired drift: `<= 0.002000 m`
- Preview directory: `experiments/robot/libero/tasks/l1b_swept_preview/l1b5_native_gripper`

Static PASS proves reset validity and pairing only. Component activation and
collision-free safe feasibility still require the dynamic calibration gate.
