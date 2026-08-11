# L1-C5 Native-Only Preflight

- Verdict: **PASS_NATIVE_ONLY_PREFLIGHT**
- Native suite/task: `libero_object` / `9`
- Native prompt: `pick up the orange juice and place it in the basket`
- Native BDDL :language: `Pick the orange juice and place it in the basket`
- Evaluated policy prompt source: official benchmark `task.language`; the unmodified native BDDL field is recorded separately.
- Project BDDL: none.
- Native BDDL: `libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl`
- Native BDDL SHA-256: `6298533e7bcfb83e40779a77fda39216e8cd53f22bf1af6388585dad0abb0b50`
- Native parsed goal: `(:goal (And (In orange_juice_1 basket_1_contain_region)) )`
- Native goal SHA-256: `81ad03892da97d439d2d9f7bd6cdf321f7a713357f141a90fbebf4d02979638e`
- Declared fixtures: floor
- Declared objects: orange_juice, basket, butter, chocolate_pudding, bbq_sauce, ketchup, salad_dressing
- Runtime asset inventory SHA-256: `adaa4ca08be6c0b51e4e87a85910b4597c077e4f2e84d9c09e5e8aaceb1d0140`
- Native referenced asset-file closure SHA-256: `1ee9ca991e700ced5cec37a6c4d4cde1baa441355b15a80c86916dc1d33a3cd6`
- Custom-asset XML audit: not applicable; no custom assets are present.
- EB/ER/EC prompts, BDDL metadata, and compiled asset inventories are identical.
- Allowed intervention: serialized pose/state of native `ketchup_1_main` only.
- Every EB→ER and EB→EC non-occupant qpos/qvel diff is exactly zero within 1e-10.
