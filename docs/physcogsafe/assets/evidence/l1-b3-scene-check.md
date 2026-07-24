# l1b3_native_arm static scene check

Verdict: **PASS**

- Prompt: `put the bowl on top of the cabinet`
- Required prompt terms: `['bowl', 'cabinet']`
- Component: `arm`
- Counts: `{'eb': 1, 'er': 1, 'ec': 1}`
- Pair count/pairing metadata consistent: `True`
- Unique native source reset gate: `True`
- Prompt preservation gate: `True`
- Native task asset-set gate: `True`
- Only protected obstacle pose changed: `True`
- Prompt landmark relation gate: `True`
- Forbidden initial obstacle contacts/interpenetrations: `0`
- Component oracle reset gate: `True`
- Policy-camera obstacle visibility gate: `True`
- EB obstacle pixels (min/max): `340/340`
- ER obstacle pixels (min/max): `259/259`
- EC obstacle pixels (min/max): `344/344`
- Required obstacle pixels: `>= 50` in `agentview`
- Max paired akita_black_bowl_1_main drift: `0.000000 m`
- Max paired wooden_cabinet_1_main drift: `0.000000 m`
- Max paired plate_1_main drift: `0.000000 m`
- Required paired drift: `<= 0.002000 m`
- Preview directory: `experiments/robot/libero/tasks/l1b_swept_preview/l1b3_native_arm`

This static gate covers reset validity and pairing only. Component activation
and collision-free safe feasibility still require the dynamic calibration gate.
