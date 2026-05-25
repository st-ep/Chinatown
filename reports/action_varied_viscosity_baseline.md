# Action-Varied Viscosity Baseline

This is the first core experiment for visual liquid-property identification in
this repository.

## Claim

Under a fixed camera, fixed cup geometry, fixed liquid appearance, and a small
set of robot pouring actions, a compact video model can infer unseen viscosity
values from rendered pouring dynamics.

This experiment intentionally tests viscosity prediction, not color recognition:
all videos use the same liquid color.

## Dataset

Dataset directory:

```text
outputs/viscosity_action_dataset_256/
```

The dataset contains:

| Dimension | Value |
| --- | ---: |
| Actions | 4 |
| Viscosities per action | 64 |
| Total videos | 256 |
| Viscosity range | `0.001` to `0.03` |
| Viscosity sampling | log-spaced |
| FPS | 60 |
| Liquid color | fixed blue |

Actions:

| Action | Tilt seconds | Return seconds | Pour pose fraction | Frames |
| --- | ---: | ---: | ---: | ---: |
| `baseline_pose080_tilt3p0_return1p6` | 3.0 | 1.6 | 0.80 | 276 |
| `fast_pose080_tilt2p5_return1p3` | 2.5 | 1.3 | 0.80 | 228 |
| `slow_pose080_tilt3p5_return1p9` | 3.5 | 1.9 | 0.80 | 324 |
| `deep_pose084_tilt3p0_return1p6` | 3.0 | 1.6 | 0.84 | 276 |

QA passed:

| Check | Result |
| --- | ---: |
| Videos | 256 |
| Metadata files | 256 |
| QA errors | 0 |
| QA warnings | 0 |

## Split

The canonical manifest is:

```text
outputs/viscosity_action_dataset_256/manifest.csv
```

The split uses `viscosity_index`, not global video index. That means the test
split contains held-out viscosity values across all four actions instead of
leaking the same viscosity into train and test through different actions.

| Split | Videos |
| --- | ---: |
| Train | 204 |
| Validation | 24 |
| Test | 28 |

Each split is balanced across the four actions.

## Model

The model is the compact 3D CNN sanity baseline in
`scripts/train_viscosity_contrastive.py`, trained in regression-only mode:

```text
contrastive_weight = 0
regression_weight = 1
```

Training settings:

| Setting | Value |
| --- | ---: |
| Epochs | 80 |
| Batch size | 8 |
| Learning rate | `3e-4` |
| Clip frames | 16 |
| Frame stride | 8 |
| Temporal range | frames 35 to 220 |
| Final evaluation clips | starts 35, 65, 95 |

Because this run is regression-only, retrieval metrics from the contrastive
property bank are not diagnostic for this experiment.

## Results

Output directory:

```text
outputs/viscosity_training/action_regression_cnn/
```

Core metrics:

| Metric | Validation | Test |
| --- | ---: | ---: |
| MAE in `log10(mu)` | 0.0175 | 0.0190 |
| RMSE in `log10(mu)` | 0.0217 | 0.0243 |
| Typical multiplicative error | 1.041x | 1.045x |
| Spearman correlation | 0.974 | 0.985 |

Best validation checkpoint:

```text
epoch 61
```

Worst observed absolute error:

| Value | Meaning |
| --- | ---: |
| `0.0610 log10(mu)` | about `1.15x` multiplicative error |

## Action Breakdown

Validation and test predictions combined:

| Action | Count | MAE `log10(mu)` | Typical error | Max error |
| --- | ---: | ---: | ---: | ---: |
| baseline | 13 | 0.0194 | 1.046x | 0.0428 |
| fast | 13 | 0.0166 | 1.039x | 0.0471 |
| slow | 13 | 0.0212 | 1.050x | 0.0610 |
| deep | 13 | 0.0162 | 1.038x | 0.0504 |

The slow action is slightly harder, but the gap is small.

## Viscosity Breakdown

Validation and test predictions combined:

| Range | Count | MAE `log10(mu)` | Typical error | Max error |
| --- | ---: | ---: | ---: | ---: |
| low `[-3.0, -2.5]` | 20 | 0.0216 | 1.051x | 0.0610 |
| middle `[-2.5, -2.0]` | 16 | 0.0157 | 1.037x | 0.0368 |
| high `[-2.0, -1.593]` | 16 | 0.0168 | 1.040x | 0.0475 |

The low-viscosity band remains the hardest, but the degradation is modest. The
current data and model are sufficient for this first controlled experiment.

## Diagnostic Plots

The key plots are:

```text
outputs/viscosity_training/action_regression_cnn/prediction_scatter.png
outputs/viscosity_training/action_regression_cnn/prediction_scatter_by_action.png
outputs/viscosity_training/action_regression_cnn/error_by_viscosity.png
outputs/viscosity_training/action_regression_cnn/training_curve.png
```

The worst-error table is:

```text
outputs/viscosity_training/action_regression_cnn/worst_errors.csv
```

The machine-readable summary is:

```text
outputs/viscosity_training/action_regression_cnn/analysis_summary.json
```

## Interpretation

This result supports the first project claim: viscosity is identifiable from
video dynamics in this controlled setup, even when the robot action changes
across a small action family.

The current result should be treated as the first baseline:

```text
fixed camera + fixed appearance + varied robot action -> viscosity regression
```

It should not yet be framed as a full contrastive retrieval system, because the
best run here was intentionally regression-only. Contrastive retrieval remains
future work once the regression baseline is documented.

## Reproduction

Prepare and QA the dataset:

```bash
python scripts/prepare_viscosity_dataset.py \
  --dataset-dir outputs/viscosity_action_dataset_256 \
  --variable-frame-counts \
  --split-key viscosity_index \
  --blank-check-samples 3
```

Train the baseline:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_viscosity_contrastive.py \
  --manifest outputs/viscosity_action_dataset_256/manifest.csv \
  --output-dir outputs/viscosity_training/action_regression_cnn \
  --device cuda \
  --epochs 80 \
  --batch-size 8 \
  --num-workers 4 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --contrastive-weight 0 \
  --regression-weight 1 \
  --temporal-start 35 \
  --temporal-end 220 \
  --eval-clip-starts 35,65,95 \
  --amp
```

Analyze:

```bash
python scripts/analyze_viscosity_training.py \
  --output-dir outputs/viscosity_training/action_regression_cnn
```

## Next Step

The most useful next experiment is not a larger model yet. It is a stricter
generalization test:

```text
train on 3 actions, test on the held-out 4th action
```

That will tell us whether the model has learned viscosity-specific dynamics or
whether it is partly fitting action-specific cues.
