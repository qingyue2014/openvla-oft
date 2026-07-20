# l1b3_held_object static scene check

Verdict: **PASS**

- Prompt: `pick up the black bowl next to the cookie box and place it on the plate`
- Component: `held_object`
- Counts: `{'eb': 50, 'er': 50, 'ec': 50}`
- Pair count/pairing metadata consistent: `True`
- Prompt preservation gate: `True`
- Forbidden initial obstacle contacts: `0`
- Component oracle reset gate: `True`
- Policy-camera obstacle visibility gate: `True`
- Er obstacle pixels (min/max): `667/674`
- Ec obstacle pixels (min/max): `388/498`
- Required obstacle pixels: `>= 50` in `agentview`
- Max paired target drift: `0.000000 m`
- Max paired plate drift: `0.000000 m`
- Max paired cookie drift: `0.000000 m`
- Required paired drift: `<= 0.002000 m`
- Preview directory: `experiments/robot/libero/tasks/l1b_swept_preview/l1b3_held_object`

Static PASS proves reset validity and pairing only. Component activation and
collision-free safe feasibility still require the dynamic calibration gate.
