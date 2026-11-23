"""Step 3: train + evaluate models with patient-grouped cross-validation.

Why grouped CV? Some patients have several ICU stays. If one stay is in the
training fold and another in the test fold, the model can "recognize" the
patient and look better than it really is. GroupKFold keeps each patient in
a single fold.
"""
import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

TARGET = "label_long_stay"
CATEGORICAL = ["gender", "careunit", "admission_type", "insurance"]
# Columns that are IDs, timestamps, or leak the answer (los IS the label!)
DROP = ["subject_id", "hadm_id", "stay_id", "intime", "los", TARGET]


def make_preprocessor(numeric_cols):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # fill missing labs with median
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, CATEGORICAL),
    ])


def make_models(numeric_cols):
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(numeric_cols)),
            ("clf", LogisticRegression(C=0.1, max_iter=1000)),  # small C = strong regularization for tiny data
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(numeric_cols)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=4,
                                           min_samples_leaf=3, random_state=RANDOM_STATE)),
        ]),
    }


def main():
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    X = df.drop(columns=DROP)
    y = df[TARGET].values
    groups = df["subject_id"].values
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]

    print(f"{len(df)} stays, {X.shape[1]} input columns, {y.mean():.0%} positive\n")

    cv = GroupKFold(n_splits=5)
    models = make_models(numeric_cols)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    summary = []
    plt.figure(figsize=(6, 6))
    for name, model in models.items():
        # Out-of-fold predictions: every stay gets a prediction from a model that never saw that patient
        oof = np.zeros(len(y))
        fold_aucs = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[test_idx], oof[test_idx]))

        auc = roc_auc_score(y, oof)
        acc = accuracy_score(y, oof >= 0.5)
        print(f"{name:22s} AUROC {auc:.3f}  (folds: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f})  accuracy {acc:.3f}")
        summary.append({"model": name, "auroc": round(auc, 3),
                        "auroc_fold_std": round(np.std(fold_aucs), 3), "accuracy": round(acc, 3)})

        fpr, tpr, _ = roc_curve(y, oof)
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")

        # Refit on ALL data and save -- this is the model you'd actually use
        model.fit(X, y)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    plt.plot([0, 1], [0, 1], "k--", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Predicting ICU stay > 3 days\n(first 24h data, grouped 5-fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)

    pd.DataFrame(summary).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    # Which features does the random forest rely on most?
    rf = models["random_forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    top = importances.sort_values().tail(15)
    plt.figure(figsize=(7, 6))
    top.plot.barh()
    plt.title("Random forest: top 15 features")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)

    print(f"\nTop features:\n{top.sort_values(ascending=False).head(8).round(3).to_string()}")
    print(f"\nSaved metrics + plots to {RESULTS_DIR}/ and models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()