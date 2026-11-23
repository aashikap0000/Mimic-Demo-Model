# ICU Length-of-Stay Prediction (MIMIC-IV Demo)

A simple ML model using only the **first 24 hours** of an ICU stay
(vital signs, lab results, demographics) to predict whether the patient will stay in the
ICU **longer than 3 days**.

## Data

Uses the open [MIMIC-IV Clinical Database Demo](https://doi.org/10.13026/07hj-2a80)
(100 patients). The data is **not** included in this repo. Download it from PhysioNet and place it like this:

```
data/
├── hosp/   (admissions.csv.gz, patients.csv.gz, labevents.csv.gz, ...)
└── icu/    (icustays.csv.gz, chartevents.csv.gz, d_items.csv.gz, ...)
```

## How to run

```bash
pip install -r requirements.txt
python src/build_cohort.py     # 1. one row per ICU stay + label
python src/build_features.py   # 2. min/max/mean of vitals + labs in first 24h
python src/train.py            # 3. grouped 5-fold CV, saves plots + models
```

## Results

| Model | AUROC |
|---|---|
| Logistic regression | 0.54 |
| Random forest | 0.61 |

With only 117 ICU stays, these numbers are noisy. The point is the pipeline, not the score.
See `results/` for ROC curves and feature importance.

## Project structure

```
src/
├── config.py           # paths, item IDs, settings
├── build_cohort.py     # select ICU stays, create label
├── build_features.py   # extract first-24h vitals + labs
└── train.py            # train, cross-validate, plot, save
```

## License / attribution

Contains information from the [MIMIC-IV Clinical Database Demo](https://doi.org/10.13026/07hj-2a80),
which is made available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).