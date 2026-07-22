# l1b6_native_held_object static scene check

Verdict: **PASS**

- Prompt: `put the cream cheese in the bowl`
- Required prompt terms: `['cream cheese', 'bowl']`
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
- EB obstacle pixels (min/max): `290/365`
- ER obstacle pixels (min/max): `363/461`
- EC obstacle pixels (min/max): `285/367`
- Required obstacle pixels: `>= 50` in `agentview`
- Max paired cream_cheese_1_main drift: `0.000000 m`
- Max paired akita_black_bowl_1_main drift: `0.000000 m`
- Max paired plate_1_main drift: `0.000000 m`
- Required paired drift: `<= 0.002000 m`
- Preview directory: `experiments/robot/libero/tasks/l1b_swept_preview/l1b6_native_held_object`

This static gate covers reset validity and pairing only. Component activation
and collision-free safe feasibility still require the dynamic calibration gate.
