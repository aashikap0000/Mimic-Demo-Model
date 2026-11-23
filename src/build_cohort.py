"""Step 1: build the cohort -- one row per ICU stay, with the label.

Label: did the ICU stay last longer than LOS_THRESHOLD_DAYS?
"""
import pandas as pd

from config import HOSP_DIR, ICU_DIR, PROCESSED_DIR, OBSERVATION_HOURS, LOS_THRESHOLD_DAYS


def main():
    icustays = pd.read_csv(ICU_DIR / "icustays.csv.gz", parse_dates=["intime", "outtime"])
    patients = pd.read_csv(HOSP_DIR / "patients.csv.gz")
    admissions = pd.read_csv(HOSP_DIR / "admissions.csv.gz")

    cohort = (
        icustays
        .merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")
        .merge(admissions[["hadm_id", "admission_type", "insurance"]], on="hadm_id", how="left")
    )

    # Drop stays shorter than the observation window: we couldn't collect 24h of data for them
    before = len(cohort)
    cohort = cohort[cohort["los"] * 24 >= OBSERVATION_HOURS].copy()
    print(f"Dropped {before - len(cohort)} stays shorter than {OBSERVATION_HOURS}h")

    cohort["label_long_stay"] = (cohort["los"] > LOS_THRESHOLD_DAYS).astype(int)
    cohort = cohort.rename(columns={"anchor_age": "age", "first_careunit": "careunit"})

    keep = ["subject_id", "hadm_id", "stay_id", "intime", "careunit",
            "gender", "age", "admission_type", "insurance", "los", "label_long_stay"]
    cohort = cohort[keep]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "cohort.csv"
    cohort.to_csv(out, index=False)

    print(f"Saved {len(cohort)} ICU stays from {cohort.subject_id.nunique()} patients -> {out}")
    print("Label balance:\n", cohort["label_long_stay"].value_counts().to_string())


if __name__ == "__main__":
    main()