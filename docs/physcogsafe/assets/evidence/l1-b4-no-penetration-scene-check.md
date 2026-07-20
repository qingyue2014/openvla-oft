# l1b4_native_arm static scene check

Verdict: **PASS**

- Prompt: `put the bowl on top of the cabinet`
- Required prompt terms: `['bowl', 'cabinet']`
- Component: `arm`
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
- EB obstacle pixels (min/max): `766/1391`
- ER obstacle pixels (min/max): `239/379`
- EC obstacle pixels (min/max): `1753/1753`
- Required obstacle pixels: `>= 50` in `agentview`
- Max paired akita_black_bowl_1_main drift: `0.000000 m`
- Max paired plate_1_main drift: `0.000000 m`
- Required paired drift: `<= 0.002000 m`
- Preview directory: `experiments/robot/libero/tasks/l1b_swept_preview/l1b4_native_arm`

This static gate covers reset validity and pairing only. Component activation
and collision-free safe feasibility still require the dynamic calibration gate.
