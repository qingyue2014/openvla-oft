# L1-C4 — native occupied basket

## Native task identity

- Suite: `libero_object` (standard four-suite task; `libero_90` is forbidden)
- Native task id: `1`
- Prompt: `pick up the cream cheese and place it in the basket`
- BDDL:
  `libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket.bddl`
- BDDL SHA-256:
  `7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2`
- Native fixture inventory: `floor`
- Native object inventory: `cream_cheese`, `basket`, `alphabet_soup`,
  `milk`, `tomato_sauce`, `butter`, `orange_juice`
- Compiled inventory SHA-256:
  `c6a4b5a0f17206b06defbea762dc61acb2f1e9191c209dd965e075eb16ef907c`
- Custom assets/BDDL/prompt changes: none.

## Paired intervention

EB, ER, and EC use the same native task, prompt, BDDL, compiled model, source
state, robot state, target state, and unrelated-object states. The only
condition-varying state is the free-joint pose/velocity of native
`milk_1_main`.

- EB: milk remains at its native floor location.
- ER: milk is stably seated in native `basket_1_main`.
- EC: milk is moved locally on the native floor, outside the basket.

The generated HDF5 files, source indices, BDDL hash, prompt, native task id,
and compiled inventory hash are bound by
`l1c4_state_bundle.json` and the native-only preflight manifest.

## Gates

The implementation fails closed in this order:

1. Native-only identity, BDDL, prompt, state metadata, and compiled-inventory
   verification.
2. Exact frozen-state verification by SHA-256.
3. OpenVLA policy-view visibility at t0 and policy-start, including occupant
   and basket segmentation pixel thresholds.
4. Physical reset validity: finite state, collision geometry, correct ER
   occupancy, EB/EC non-occupancy, no occupant-target/robot initial contact,
   stable world/support-relative motion, and paired basket motion.
5. Static action-separation calibration.
6. Executable OSC safe-reference validation in ER.
7. Human visibility review before formal evaluation.
8. Short model smoke, unchanged-action replay, and only then formal
   attribution.

Any native-only mismatch writes an invalidation marker covering the scene,
jobs, metrics, videos, tables, and HTML and stops evaluation.

## Current three-state implementation evidence

- `PASS_NATIVE_ONLY_PREFLIGHT`
- `PASS_EXACT_STATE_PREVIEW`
  - milk: 273–516 policy-crop pixels at policy-start
  - basket: 2704–3175 policy-crop pixels
  - forbidden initial contacts: 0/9 state-condition pairs
- `PASS_STATIC_OCCUPANCY_LAYOUT`
  - center safe rate: 0/3
  - +Y 50 mm safe rate: 3/3
  - -Y 45 mm safe rate: 3/3
- `PASS_DYNAMIC_SAFE_REFERENCE`
  - safe success: 3/3
  - target lift: 105.1–105.2 mm
  - post-release XY drift: 3.6–11.6 mm

Formal model evaluation is intentionally not certified yet:
`review/L1-C4_task/visibility_review.md` still requires a human
`PASS_HUMAN_VISIBILITY` verdict, and EB/ER/EC model smoke videos have not been
collected.
