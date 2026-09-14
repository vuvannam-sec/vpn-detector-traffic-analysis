# VPN Traffic Detection from Flow Statistics

A small research project for classifying **VPN vs. non-VPN network flows** using time-based traffic features from the ISCXVPN2016 dataset.

The repository contains the original experiment notebooks, several classical ML baselines, an XGBoost training path, and a Gradio demo for scoring CSV flow records.

## Scope

This project answers a narrow question: given flow-level statistical features collected in the same style as ISCXVPN2016, can a supervised classifier separate VPN from non-VPN traffic?

It is **not** a general-purpose VPN detector. In particular:

- the model does not inspect packet payloads;
- the dataset was collected in a controlled environment and includes OpenVPN traffic;
- performance on other VPN protocols, networks, capture tools, or newer traffic patterns has not been established;
- row-wise train/test splits can overestimate real-world generalization when related flows appear in both sets.

Those constraints are intentional and documented so the reported numbers are not presented as production-grade detection rates.

## Repository layout

```text
.
├── README.md
├── LICENSE
└── VPN-Detector/
    ├── clean_data/              # cleaned 15/30/60/120-second flow datasets
    ├── data/                    # source ARFF data used in the original experiments
    ├── main-*.ipynb             # original exploratory notebooks
    ├── *_classification.py      # original model-comparison scripts
    ├── train.py                 # reproducible XGBoost training entrypoint
    ├── predict.py               # batch CSV inference
    ├── gradio_app.py            # local web demo
    ├── Dockerfile
    ├── requirements.txt
    └── requirements-notebooks.txt
```

The notebooks and older scripts are retained because they document the experiment history. New usage should start with `train.py`, `predict.py`, or `gradio_app.py`.

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/vuvannam-sec/vpn-detector-traffic-analysis.git
cd vpn-detector-traffic-analysis/VPN-Detector

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python train.py --window 15
python gradio_app.py
```

Then open `http://127.0.0.1:7860`.

The default training command writes:

```text
models/xgb_15s.joblib
models/xgb_15s.metrics.json
```

The `.joblib` file contains both the fitted preprocessing/model pipeline and metadata describing the expected feature schema.

## Batch prediction

```bash
python predict.py \
  --model models/xgb_15s.joblib \
  --input clean_data/novpn15s_cleaned.csv \
  --output predictions.csv
```

The output contains the original input columns plus:

- `prediction`: `0` for non-VPN, `1` for VPN;
- `vpn_probability`: model probability for the VPN class.

## Reproducing training

Supported windows are `15`, `30`, `60`, and `120` seconds.

```bash
python train.py --window 60
```

To additionally run stratified cross-validation:

```bash
python train.py --window 60 --cv-folds 5
```

The training script:

1. loads the VPN and non-VPN cleaned CSV files for the requested window;
2. assigns binary labels;
3. creates a stratified hold-out split;
4. fits preprocessing and XGBoost as a single scikit-learn pipeline;
5. reports accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix;
6. stores the fitted pipeline together with feature-schema and run metadata.

See [`VPN-Detector/METHODOLOGY.md`](VPN-Detector/METHODOLOGY.md) for evaluation notes and known limitations.

## Historical experiment results

The original notebooks reported the following best cross-validation results:

| Flow window | Best model | F1 | ROC-AUC |
| --- | --- | ---: | ---: |
| 15 s | XGBoost | 0.94 | 0.99 |
| 30 s | XGBoost | 0.93 | 0.98 |
| 60 s | XGBoost | 0.91 | 0.98 |
| 120 s | XGBoost | 0.92 | 0.98 |

These values are kept as **historical baselines from the original project**. They should not be interpreted as an independently reproduced production benchmark. For a stronger estimate of deployment performance, evaluation should split by capture/session/scenario rather than only by randomly sampled rows.

## Docker

The image trains the 15-second model during the build and then starts the Gradio app.

```bash
cd VPN-Detector

docker build -t vpn-detector .
docker run --rm -p 7860:7860 vpn-detector
```

The Docker build intentionally fails if training fails. This avoids producing an image that starts without a valid model artifact.

## Dataset

The experiments use the **ISCX VPN-nonVPN 2016 (ISCXVPN2016)** dataset from the Canadian Institute for Cybersecurity, University of New Brunswick.

The original dataset contains labelled regular and VPN traffic across multiple application categories. The provider asks users of the dataset to cite the associated paper:

> Gerard Draper-Gil, Arash Habibi Lashkari, Mohammad Mamun, and Ali A. Ghorbani, “Characterization of Encrypted and VPN Traffic Using Time-Related Features,” ICISSP 2016.

Dataset page: https://www.unb.ca/cic/datasets/vpn.html

The repository's code license does **not** relicense the dataset. Dataset files and derived copies remain subject to the original provider's terms and citation requirements.

## Limitations and next steps

The main technical gap is generalization. A useful next phase would evaluate:

- capture/session-aware or scenario-aware train/test splits;
- cross-protocol transfer, especially beyond OpenVPN;
- calibration and threshold selection rather than a fixed 0.5 decision rule;
- robustness to missing or shifted flow features;
- latency and throughput on streaming flow records.

Until those tests are run, this repository is best treated as a reproducible traffic-classification study and demo rather than a production network-control component.

## License

Source code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).

Dataset licensing and citation requirements are separate; see the Dataset section above.
