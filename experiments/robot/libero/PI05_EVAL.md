# pi0.5 evaluation

PhysCogSafe can evaluate the official OpenPI `pi05_libero` policy through
OpenPI's websocket policy server. The simulator, PhysCog safety oracles,
trajectory recorder, initial-state pairing, and video logging remain in this
repository; only policy inference runs in the separate OpenPI environment.

This separation is intentional. OpenPI's JAX/PyTorch dependencies and this
repository's OpenVLA-specific Transformers fork should not be installed in the
same runtime.

## 1. Start the official OpenPI server

In an OpenPI checkout, follow the official installation instructions and run:

```bash
uv run scripts/serve_policy.py --env LIBERO
```

This serves the official `pi05_libero` checkpoint on port 8000. OpenPI
downloads `gs://openpi-assets/checkpoints/pi05_libero` automatically when the
checkpoint is not cached.

For a local or separately converted checkpoint:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/path/to/pi05_libero/checkpoint
```

## 2. Install the lightweight client

In the OpenVLA-OFT / PhysCogSafe environment:

```bash
pip install -e ".[pi05]"
```

The optional dependency is pinned to the OpenPI revision used when this
adapter was added. It installs only `openpi-client`, not the OpenPI model
runtime.

## 3. Smoke-test native LIBERO

With the server still running:

```bash
python -m experiments.robot.libero.run_libero_eval \
  --model_family pi05 \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --pi05_host 127.0.0.1 \
  --pi05_port 8000
```

The evaluator uses OpenPI's official LIBERO conventions:

- 224 x 224 aspect-preserving resize with zero padding
- third-person and wrist RGB observations
- 8-D end-effector/gripper state
- direct 7-D LIBERO environment actions
- replanning after the first 5 actions of every predicted chunk

Change `--pi05_replan_steps` only for a deliberate ablation.
The client fails after 900 seconds by default if the server never becomes
reachable; change `--pi05_connect_timeout_s` for unusually slow checkpoint
startup.

## 4. Run a PhysCogSafe condition

All existing PhysCogSafe evaluator arguments remain available. For example:

```bash
python -m experiments.robot.libero.run_physcog_libero_l1_eval \
  --model_family pi05 \
  --task_suite_name libero_spatial \
  --task_ids 1 \
  --num_trials_per_task 5 \
  --safety_oracle none \
  --run_id_note pi05-L1-A1-Eb-smoke \
  --save_video_mode all
```

For formal evaluation, run the same frozen `Eb`, `Er`, and `Ec` initial-state
files and the same oracle configuration used for OpenVLA-OFT. Do not
recalibrate scene geometry on pi0.5 rollouts: keeping the test family frozen is
necessary for a fair cross-model comparison.

The registered Superpod formal phases enforce 50 paired episodes per
condition, re-run the static/policy-camera gate on the exact restored state
bytes, verify the archived dynamic safe-reference gate, save one formal video
per condition, and emit machine-readable aggregate results:

```bash
python experiments/robot/libero/tasks/physcog_remote_agent.py run \
  --scenario l1b1 --phase pi05_formal --count 50
python experiments/robot/libero/tasks/physcog_remote_agent.py run \
  --scenario l1b2 --phase pi05_formal --count 50
```

The OpenPI server owns the pi0.5 checkpoint, so
`--pretrained_checkpoint` is intentionally ignored for `model_family=pi05`.
Rollout filenames and log run IDs contain `pi05` to prevent mixing results
with OpenVLA-OFT.

The L1-B runner forwards these settings through environment variables:

```bash
MODEL_FAMILY=pi05 PI05_HOST=127.0.0.1 PI05_PORT=8000 \
  bash experiments/robot/libero/tasks/run_l1b_swept.sh \
  l1b5_native_gripper eb
```

Use the canonical family name on branches where native B5/B6/B7 have already
been promoted to B1/B2/B3.
