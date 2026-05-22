# Chinatown

Focused Genesis robot-arm pouring simulation for three same-action liquids:
water, a log-midpoint viscosity, and a honey-like liquid.

Generated particle caches, videos, manifests, and glass meshes are rebuilt under
`outputs/` and are intentionally ignored by git.

## Install

Use a Genesis-capable environment. Install PyTorch for your GPU first, then:

```bash
python -m pip install -e ".[genesis]"
```

For local test tooling:

```bash
python -m pip install -e ".[genesis,dev]"
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
