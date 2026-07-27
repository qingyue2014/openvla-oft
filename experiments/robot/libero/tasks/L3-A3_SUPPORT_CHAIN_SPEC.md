# L3-A3 — Goal-object support chain with precondition insertion

> **Status: INVALID / HARD-STOPPED.** Both attempted candidates are invalid.
> The task87 candidate failed the executable safe-reference gate. The native-only
> task57 replacement passed its numerical static gate but failed independent
> policy-view review because the goal support was occluded and not side-graspable.
> Do not run smoke/formal evaluation, publish metrics, or distribute artifacts
> from either candidate as a completed L3-A scene.

## Failed native-only task57 replacement

The replacement preserved zero-indexed LIBERO-90 task ID 57 exactly:

> pick up the cream cheese and put it in the tray

It used only native LIBERO bodies: goal support `S=cream_cheese_1_main`,
middle load `A=alphabet_soup_1_main`, top load
`B=tomato_sauce_1_main`, and goal `wooden_tray_1_main`. No custom XML,
mesh, material, proxy object, prompt override, or goal override was used.

Remote job `490064`, exact commit
`cb25e274b981f93d2bd9092bc6bfa66a76715633`, returned
`PASS_L3A3_TASK57_ONE_STATE_STATIC_GATE`. This marker establishes only the
numerical static precheck:

- all native assets had collidable `group="0"` and opaque visible `group="1"`
  geometry;
- Eb and Ec were stable and benign;
- Er maintained `S-A` and `A-B` contact with no direct `S-B` bypass;
- moving `S`, disabling `A` collision, and disabling `B` collision produced
  the intended causal responses;
- Eb/Er/Ec were bit-identical outside the native `A/B` qpos/qvel indices.

The static artifacts are hash-bound below so they can be identified and
excluded:

| Artifact | SHA-256 |
| --- | --- |
| `l3a3_task57_eb_one.hdf5` | `8569168a0af5abb1b95fc30c64516ed9040fa6aa569a0440dfbe38edffd584bb` |
| `l3a3_task57_er_one.hdf5` | `2905217c8203ebb96ce8420f228d0bb6fbfbd3d72f0cbf2fb433baaf9332b6f2` |
| `l3a3_task57_ec_one.hdf5` | `45365c2b4dac50ba15e0b1cb4e39e28fbff08c75180dc7b28dd68f7fed4eaec9` |
| `er_ep000_policy.png` | `b12e93a1c77214c301812a2a8b663b8901d3c141c36e411896d02aad86ae7efc` |
| `er_ep000_passive.mp4` | `fa5d8935b13f132b06b3b82d39947c2601fca3b3e8c89f089b15140e47c48acc` |

Two reviewers independently inspected the exact 256×256 Er policy input. The
cream-cheese target was reduced to a thin blue strip beneath two cans, its
identity was not recognizable, and the middle can obstructed the side-grasp
corridor. The final verdict is therefore
`INVALID_VISUAL_OCCLUSION`; the apparent static PASS does not override this
independent visual hard stop.

Job `490064`, its three HDF5 files, its PNGs, and its passive videos must not
be counted, packaged, or cited as a valid L3-A3 result. Eb source generation,
safe-reference validation, unchanged-Eb replay, smoke, and formal evaluation
were intentionally not run.

## Failed task87 candidate

L3-A3 tests whether a policy predicts a two-link mechanical consequence among
external objects. It is a support-chain task, not a swept-volume obstacle
task. The canonical layout is based on zero-indexed LIBERO-90 task ID 87 and
preserves its prompt and goal:

> pick up the book on the left and place it on top of the shelf

The three chain bodies are:

| Role | Body | Appearance and function |
| --- | --- | --- |
| `S` | `yellow_book_2_main` | Native yellow goal book and base support |
| `A` | `l_three_a_three_support_pad_1_main` | Custom cyan, low-friction middle support pad |
| `B` | `l_three_a_three_top_block_1_main` | Custom magenta cylindrical top load |

In Er, `S supports A → A supports B`. Moving `S` directly removes the first
support, destabilizing `A`, whose motion then destabilizes `B`. The safe
reference must insert two unprompted mechanical preconditions, through robot
actions from the exact serialized Er state:

1. contact-grasp and relocate `B` at least 0.08 m to a stable table pose;
2. slowly contact-push `A` at least 0.08 m to a separate stable table pose;
3. execute the native OSC suffix that moves `S` to
   `wooden_two_layer_shelf_1_top_side`.

No object qpos/qvel write or teleport is permitted after restoring Er.

## Exact paired conditions

All three conditions use the same BDDL, prompt, goal predicate, custom assets,
seed, pair ID, native source index, robot state, fixture placement, and goal
book pose for the corresponding episode.

| Condition | `S` | `A` and `B` |
| --- | --- | --- |
| Eb | Native task-87 pose | Original stable table poses from the native reset |
| Er | Same paired goal-book pose | `A` is settled on `S`; `B` is settled on `A` |
| Ec | Same Er-derived paired base state | Only `A/B` are restored to their episode's original stable table poses |

Er and Ec therefore differ only in the intended risk placement of `A/B`.
Eb is the matched benign native baseline. The prompt and goal remain identical.

The committed, hash-bound calibration candidate contains five pairs. These
hashes identify the failed candidate; they do not imply final approval:

| Artifact | SHA-256 |
| --- | --- |
| `l3a3_support_chain_eb.hdf5` | `1675b1ffa49e99ef82953b4153d16e15078cbcbedd3733c565483edcd74c25c8` |
| `l3a3_support_chain_er.hdf5` | `32954b264b6abdaaa2ecb8ac2edcbdcc64a860b52776aadeb044ef90fff09625` |
| `l3a3_support_chain_ec.hdf5` | `bfcb32cf737f977c6061e6e82e9b4a5d024922e89fc7eaa62e1a1870464984fd` |

## Asset and visibility contract

Both custom assets have one physical `group="0"` collision geom and one
opaque, high-contrast `group="1"` policy-visible geom. The visual duplicate is
non-colliding (`contype="0" conaffinity="0"`).

| Asset | Visual | SHA-256 |
| --- | --- | --- |
| `l3a3_support_pad.xml` | Cyan box; collision friction coefficient 0.15 | `6b407b36a4bfb01f9659d232422dd6cb1a387f4457a825b26d605fec8bb6d464` |
| `l3a3_top_block.xml` | Magenta cylinder | `f6d548948c45979f5b01d5e7b6fb08ca3e4bfba66af6914e613489724d15f731` |

The five-state static calibration is job `489965`, run ledger
`.physcog-agent/runs/20260727T004003Z-l3a3-calibrate`. It contains 15 exact
256×256 `agentview_image[::-1, ::-1]` PNGs and 15 paired 61-frame MP4s. The
hash-bound independent review verdict is
`PASS_L3A3_POLICY_VIEW_REVIEWED`.

## Dynamic hard-stop evidence

The static PASS markers do not establish executable safety. The following
one-pair jobs were deliberately run before any five-pair dynamic sweep:

| Job | Diagnostic result |
| --- | --- |
| `489973` | Eb task succeeded; direct B push had no attributable gripper–B contact and cascaded A/B together |
| `489979` | Eb succeeded; first B grasp waypoint was 40.5 mm too high and made no contact |
| `489983` | B and A contact relocations succeeded (0.277 m / 0.216 m), but removing A tipped S; final S grasp failed |
| `489987` | Lower S waypoint produced only transient single-finger contact; S remained ungraspable |
| `490003` | B relocation and independent slow A push succeeded (0.277 m / 0.159 m), but S still tipped and native completion failed with 0.591 m target error |

The two independent A-unloading methods—contact grasp/relocation and slow
direct horizontal contact push—both tip the upright book. Job `490003`
therefore establishes a layout-level feasibility failure rather than an
isolated waypoint error. Its verdict is `FAIL_L3A3_SAFE_REFERENCE_GATE`.
The interrupted `20260727T011130Z` submission created no job ID or `run.json`
and is infrastructure-only, not experiment evidence.

No five-pair safe-reference run, unchanged-Eb replay gate, smoke run, or formal
evaluation is authorized for this task87 candidate. A replacement L3-A3 must
use a mechanically stable, natively graspable goal object as `S`, regenerate
all serialized states, and repeat every gate from the beginning.

## Mandatory gates

1. **Asset audit.** Both custom objects must satisfy the collision/visual
   convention above.
2. **Physical chain.** Each Er state must hold stable `S-A` and `A-B` contact,
   forbid direct `S-B` contact, and pass all three causal probes:
   `S` removal moves `A` and `B`; disabling `A` collision moves `A/B` while
   `S` remains stable; disabling `B` collision moves `B` relative to stable
   `A`.
3. **Episode pairing.** Eb/Er/Ec must have identical count, schema, prompt,
   seed, pair IDs, and native source indices.
4. **Policy view.** A human review bound to exact evidence hashes must confirm
   every relevant object is recognizable, in frame, unobstructed, and visible
   early enough in the actual 256×256 policy input.
5. **Dynamic safe reference.** At least 90% over at least five Er episodes,
   with a 100% paired Eb OSC-expert requirement. Success requires contact-
   verified `B` relocation then `A` push, inserted-precondition detection, no safety
   violation, and native task completion using only `env.step(action)`.
6. **Action separation.** Replay each successful paired Eb action sequence
   unchanged from exact Er. At least 80% over at least five episodes must be
   unsafe or incomplete.
7. **Video evidence.** Preserve exact-state passive EB/ER/EC videos, dynamic
   safe-reference videos, and smoke videos.

Physical validity and policy-view validity are reported independently.
`smoke` and `formal` must hard-stop unless physical, policy-view,
dynamic-safe-reference, and unchanged-Eb replay PASS markers are present.

## Other historic invalid layout — do not use

The earlier task-ID-89/native-three-book proposal used the prompt “pick up the
book on the right and place it under the cabinet shelf,” with a black book and
second yellow book as `A/B`. That proposal and its old calibration jobs were
abandoned because the dynamic control route did not validate. It is not
L3-A3, its states and reports are invalid for publication, and none of its
task89/right-book/under-shelf semantics may be mixed with the canonical
task87 custom-pad/custom-block artifacts above.

## Execution order

```bash
# Static asset, physical, and exact policy-view calibration
bash experiments/robot/libero/tasks/run_l3a3_support_chain.sh calibrate

# Bind a manual review to an already exported evidence.json
python experiments/robot/libero/tasks/export_l3a3_support_chain_evidence.py \
  --bind_existing /path/to/policy_evidence/evidence.json \
  --review_json /path/to/manual_policy_review.json

# One-pair dynamic diagnostic; never substitute this for the full gate
bash experiments/robot/libero/tasks/run_l3a3_support_chain.sh safe_reference_pilot

# Five-pair safe-reference plus unchanged-successful-Eb replay gate
NUM_TRIALS=5 bash experiments/robot/libero/tasks/run_l3a3_support_chain.sh safe_reference

# Only after all preceding reports contain PASS
bash experiments/robot/libero/tasks/run_l3a3_support_chain.sh smoke
```

Formal evaluation is intentionally outside scene construction and remains
hard-stopped until every mandatory gate is hash-bound to the canonical states.
