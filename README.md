# Chinatown

Focused Genesis robot-arm pouring simulation for three same-action liquids:
water, a log-midpoint viscosity, and a honey-like liquid.

Generated particle caches, videos, and glass meshes are rebuilt under `outputs/`
and are intentionally ignored by git.

## Install

Use a Genesis-capable environment. Install PyTorch for your GPU first, then:

```bash
python -m pip install -e ".[genesis]"
```

For local test tooling:

```bash
python -m pip install -e ".[genesis,dev]"
```

For the learning scripts, use an environment with PyTorch/TorchVision and
FFmpeg available, then install:

```bash
python -m pip install -e ".[learning]"
```

## Render

Render all three liquids with the same robot trajectory and same frame count:

```bash
python scripts/render_same_action_viscosity_triplet.py
```

The scripts use GPU 1 by default. Pass another device when needed:

```bash
python scripts/render_same_action_viscosity_triplet.py --cuda-device 0
```

The default variants are:

- water: `mu=0.001`
- middle log point: `mu=sqrt(0.001 * 0.03)`
- honey-like: `mu=0.03`

Honey-like uses a smaller SPH timestep for numerical stability while keeping
the same visible robot action and video duration.

All variants use the same liquid color by default, so viscosity is not encoded
as an appearance shortcut.

## Dataset Generation

Generate a 128-video viscosity dataset with one shared liquid color and
dataset-level metadata:

```bash
python scripts/generate_viscosity_dataset.py --num-videos 128
```

The dataset is written under `outputs/viscosity_dataset_128/` with one
subdirectory per run, plus `manifest.csv` and `dataset_metadata.json`.

After a sharded generation run, merge manifests, validate videos, and create
train/validation/test split CSVs:

```bash
python scripts/prepare_viscosity_dataset.py
```

Train the first compact contrastive-regression sanity model:

```bash
python scripts/train_viscosity_contrastive.py --epochs 20 --batch-size 8 --amp
```

For the current 128-video single-viscosity dataset, the strongest simple
baseline is regression-only with longer training:

```bash
python scripts/train_viscosity_contrastive.py \
  --epochs 40 \
  --batch-size 8 \
  --amp \
  --contrastive-weight 0 \
  --regression-weight 1 \
  --output-dir outputs/viscosity_training/regression_only
```

The learning script reads `outputs/viscosity_dataset_128/manifest.csv` and
writes checkpoints, metrics, and predictions under
`outputs/viscosity_training/sanity_cnn/`.

## Direct Run

Render one custom viscosity with the reusable runner:

```bash
python scripts/run_robotic_arm_pour_viscosity_genesis.py \
  --viscosity 0.03 \
  --microsteps-per-frame 360 \
  --source-boundary-correction-interval 4 \
  --output-path outputs/robotic_arm_pour_mu_0p03.mp4
```

## Tests

```bash
python tests/test_robotic_arm_pour_genesis_smoke.py
```
