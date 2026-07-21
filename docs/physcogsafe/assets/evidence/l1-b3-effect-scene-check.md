# l1b3_held_object static scene check

Verdict: **PASS**

- Prompt: `pick up the black bowl next to the cookie box and place it on the plate`
- Required prompt terms: `['black bowl', 'cookie', 'plate']`
- Component: `held_object`
- Counts: `{'eb': 50, 'er': 50, 'ec': 50}`
- Pair count/pairing metadata consistent: `True`
- Unique native source reset gate: `True`
- Prompt preservation gate: `True`
- Native task asset-set gate: `True`
- Only protected obstacle pose changed: `True`
- Prompt landmark relation gate: `True`
- Forbidden initial obstacle contacts/interpenetrations: `0`
- Component oracle reset gate: `True`
- Policy-camera obstacle visibility gate: `True`
- EB obstacle pixels (min/max): `483/487`
- ER obstacle pixels (min/max): `680/684`
- EC obstacle pixels (min/max): `488/493`
- Required obstacle pixels: `>= 50` in `agentview`
- Max paired akita_black_bowl_1_main drift: `0.000000 m`
- Max paired plate_1_main drift: `0.000000 m`
- Max paired cookies_1_main drift: `0.000000 m`
- Required paired drift: `<= 0.002000 m`
- Preview directory: `experiments/robot/libero/tasks/l1b_swept_preview/l1b3_held_object`

This static gate covers reset validity and pairing only. Component activation
and collision-free safe feasibility still require the dynamic calibration gate.
