# Cosmos Policy and DreamZero evaluation

This integration distinguishes a downloadable checkpoint from a checkpoint
whose embodiment/action space is valid for LIBERO.

## Model selection

### Cosmos

`cosmos` means NVIDIA's **Cosmos Policy**, not a generic Cosmos video/world
model. The selected public checkpoint is:

- model: `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`
- Superpod path:
  `/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B`
- pinned Hugging Face revision:
  `cb689ec0e3347c13667d70a78a3447388f5c3bb8`
- official source: <https://github.com/nvlabs/cosmos-policy>
- pinned source revision:
  `18a2accadf4e7a3531e56754102af5a24d2316da`

This is the correct candidate because the official repository provides a
LIBERO evaluator, a 7-D LIBERO action head, matching dataset statistics and
T5 embeddings. The adapter follows the official observation contract:

- vertically flipped agent and wrist RGB (not OpenVLA's 180-degree rotation);
- 9-D proprio in `[gripper qpos, EEF xyz, EEF quaternion]` order;
- 16-action chunks;
- already-unnormalized actions passed directly to LIBERO.

The NVIDIA model card currently identifies the model license as the NVIDIA
One-Way Noncommercial License. Confirm that the intended use complies with
that license before evaluation/publication.

### DreamZero

The public inference checkpoint is:

- model: `GEAR-Dreams/DreamZero-DROID`
- Superpod path: `/project/trllmout/models/DreamZero-DROID`
- pinned Hugging Face revision:
  `96ad344138c66e82536422432ad742f015784942`
- official source: <https://github.com/dreamzero0/dreamzero>
- pinned source revision:
  `ab790c198fbce33503358efbbd4187ce9a89adf3`

It is downloadable, but it is **not a valid LIBERO policy checkpoint**. The
official server is configured for DROID and returns an 8-D joint-position
action (joint action plus gripper). PhysCog LIBERO consumes a 7-D Franka
end-effector relative action. The official project publishes DROID simulation
evaluation and AgiBot/YAM post-training instructions, but no DreamZero-LIBERO
checkpoint/action head. Therefore `--model_family dreamzero` fails before
creating an environment or rollout. Do not report a DreamZero PhysCog result
until a LIBERO embodiment is trained and its action/statistics contract is
validated.

## Registered Superpod setup phases

The setup phases download revision-pinned checkpoint snapshots and pin the
corresponding official source checkout. They do not submit themselves, install
CUDA environments, or start evaluation.

```bash
python experiments/robot/libero/tasks/physcog_remote_agent.py run \
  --scenario models --phase setup_cosmos --count 1 --gpus 1

python experiments/robot/libero/tasks/physcog_remote_agent.py run \
  --scenario models --phase setup_dreamzero --count 1 --gpus 1
```

The underlying idempotent commands are:

```bash
bash experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh cosmos
bash experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh dreamzero
```

`HF_TOKEN` may be supplied through the job environment if Hugging Face
requires authentication. Re-running the phase resumes an interrupted
snapshot download.

## Cosmos PhysCog evaluation

Install the official Cosmos runtime using its documented Linux x86-64 CUDA
environment (the upstream project recommends its Docker image). Ensure its
`cosmos_policy` package is importable by the Python interpreter that launches
PhysCog. Then use only already-validated native frozen states:

```bash
MODEL_FAMILY=cosmos \
CHECKPOINT=/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B \
GOAL_CHECKPOINT=/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B \
RUN_ID_SUFFIX=cosmos-policy \
bash experiments/robot/libero/tasks/run_l1b_swept.sh \
  l1b1_native_gripper eb
```

Repeat with `er` and `ec`, and use `l1b2_native_held_object` for L1-B2. Before
any model rollout, retain the native-only preflight:

- native task and original prompt recorded verbatim;
- native BDDL source unchanged;
- evaluated asset inventory identical to the selected native task;
- frozen EB/ER/EC state and policy-RGB validation reports passing.

Any custom asset, custom BDDL, prompt change, or asset-inventory mismatch
invalidates the run.

## Official evidence

- Cosmos Policy source and LIBERO quick start:
  <https://github.com/nvlabs/cosmos-policy>
- Cosmos LIBERO checkpoint:
  <https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B>
- DreamZero source, action server and released embodiments:
  <https://github.com/dreamzero0/dreamzero>
- DreamZero-DROID checkpoint:
  <https://huggingface.co/GEAR-Dreams/DreamZero-DROID>

