# Methodology and evaluation notes

This document separates the experiment design from the demo application so that the repository does not imply stronger conclusions than the data supports.

## Task definition

The target is binary classification:

- `0`: non-VPN flow
- `1`: VPN flow

Inputs are flow-level statistical features derived from ISCXVPN2016. No packet payload inspection is performed by the model in this repository.

## Data windows

Cleaned datasets are included for four aggregation windows:

- 15 seconds
- 30 seconds
- 60 seconds
- 120 seconds

`train.py` selects the matching VPN and non-VPN CSV pair and constructs the binary label at training time.

## Preprocessing

The current reproducible pipeline performs:

- median imputation for numeric columns;
- most-frequent imputation for categorical columns;
- one-hot encoding for categorical columns with unknown-category handling;
- XGBoost binary classification.

Preprocessing and the classifier are saved together in one scikit-learn pipeline so inference cannot accidentally skip training-time transformations.

## Hold-out evaluation

By default, `train.py` uses a stratified random 70/30 split with a fixed random seed. It reports:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- confusion matrix
- full classification report

This is useful for regression testing and reproducing the original style of experiment, but it is not the strongest estimate of deployment performance.

## Cross-validation

`--cv-folds N` adds stratified cross-validation. Example:

```bash
python train.py --window 15 --cv-folds 5
```

Cross-validation is still row-based. If multiple rows come from the same capture/session, the folds may share correlated traffic.

## Main validity risk: correlated flows

Network-flow datasets often contain many observations from the same capture, user activity, application, or session. A random row split can place closely related flows in both training and test sets. The model may therefore learn environment-specific structure that does not transfer to a different network.

For stronger validation, future experiments should preserve a grouping variable and split by one of the following:

1. capture/session;
2. scenario;
3. application category;
4. collection period or host.

A useful benchmark should report both the historical random-split result and at least one grouped or out-of-distribution result.

## Protocol generalization

The ISCXVPN2016 VPN captures were generated with OpenVPN. A high score on this dataset is not evidence that the same model detects WireGuard, IPsec, proprietary VPN transports, or obfuscated tunnels.

Protocol-transfer evaluation should therefore be treated as a separate experiment rather than inferred from the current numbers.

## Reproducibility metadata

Every model bundle written by `train.py` includes:

- flow window;
- random seed;
- train/test sizes;
- class counts;
- ordered feature schema;
- hold-out metrics;
- optional cross-validation metrics;
- UTC creation timestamp.

A human-readable copy is also written to `*.metrics.json`.

## Historical notebooks

The `main-*.ipynb` notebooks are retained as experiment history. They may include exploratory cells, visualizations, and comparisons that are not part of the clean training path. New reproducibility work should use `train.py` and add any new evaluation protocol explicitly rather than silently modifying old notebook results.
