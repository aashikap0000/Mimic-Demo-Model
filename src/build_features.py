"""Step 2: turn raw vitals + labs from the first 24h of each ICU stay into features.

For every vital/lab we compute min, max and mean over the window,
giving one wide row per ICU stay.
"""
import pandas as pd

from config import HOSP_DIR, ICU_DIR, PROCESSED_DIR, OBSERVATION_HOURS, VITAL_ITEMS, LAB_ITEMS


def window_filter(events, cohort, id_col):
    """Keep only events between ICU intime and intime + OBSERVATION_HOURS.

    id_col is the column used to link events to ICU stays:
    "stay_id" for chartevents, "hadm_id" for labevents.
    """
    if id_col == "stay_id":
        events = events.merge(cohort[["stay_id", "intime"]], on="stay_id", how="inner")
    else:
        events = events.merge(cohort[[id_col, "stay_id", "intime"]], on=id_col, how="inner")
    end = events["intime"] + pd.Timedelta(hours=OBSERVATION_HOURS)
    in_window = (events["charttime"] >= events["intime"]) & (events["charttime"] < end)
    return events[in_window]


def summarize(events, names):
    """Pivot long events -> wide features: <name>_min, <name>_max, <name>_mean."""
    events = events.assign(name=events["itemid"].map(names))
    agg = events.groupby(["stay_id", "name"])["valuenum"].agg(["min", "max", "mean"])
    wide = agg.unstack("name")
    wide.columns = [f"{name}_{stat}" for stat, name in wide.columns]
    return wide


def main():
    cohort = pd.read_csv(PROCESSED_DIR / "cohort.csv", parse_dates=["intime"])

    # --- Vitals (chartevents is big, so only read the columns + rows we need) ---
    print("Loading chartevents...")
    vitals = pd.read_csv(
        ICU_DIR / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    vitals = vitals[vitals["itemid"].isin(VITAL_ITEMS)].dropna(subset=["valuenum"])
    vitals = window_filter(vitals, cohort, "stay_id")
    vital_feats = summarize(vitals, VITAL_ITEMS)

    # --- Labs (labevents has hadm_id, not stay_id, so join on hadm_id) ---
    print("Loading labevents...")
    labs = pd.read_csv(
        HOSP_DIR / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    labs = labs[labs["itemid"].isin(LAB_ITEMS)].dropna(subset=["valuenum", "hadm_id"])
    labs["hadm_id"] = labs["hadm_id"].astype(int)
    labs = window_filter(labs, cohort, "hadm_id")
    lab_feats = summarize(labs, LAB_ITEMS)

    features = (
        cohort.set_index("stay_id")
        .join(vital_feats, how="left")
        .join(lab_feats, how="left")
        .reset_index()
    )

    out = PROCESSED_DIR / "features.csv"
    features.to_csv(out, index=False)
    n_feat = vital_feats.shape[1] + lab_feats.shape[1]
    print(f"Saved {features.shape[0]} rows x {n_feat} vital/lab features -> {out}")
    missing = features[list(vital_feats.columns) + list(lab_feats.columns)].isna().mean()
    print("Most-missing features:\n", missing.sort_values(ascending=False).head(5).round(2).to_string())


if __name__ == "__main__":
    main()