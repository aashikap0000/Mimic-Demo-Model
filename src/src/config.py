"""Central place for file paths and settings used by every script."""
from pathlib import Path

# Repo root = the folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
HOSP_DIR = DATA_DIR / "hosp"        # put the hosp/*.csv.gz files here
ICU_DIR = DATA_DIR / "icu"          # put the icu/*.csv.gz files here
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Prediction setup
OBSERVATION_HOURS = 24   # only use data from the first 24h of the ICU stay
LOS_THRESHOLD_DAYS = 3   # label = 1 if ICU stay lasts longer than this
RANDOM_STATE = 42

# Vital signs from icu/chartevents (itemid -> short name)
VITAL_ITEMS = {
    220045: "heart_rate",
    220179: "sbp",          # non-invasive systolic BP
    220180: "dbp",          # non-invasive diastolic BP
    220181: "mbp",          # non-invasive mean BP
    220210: "resp_rate",
    220277: "spo2",
    223761: "temp_f",
}

# Labs from hosp/labevents (itemid -> short name)
LAB_ITEMS = {
    50912: "creatinine",
    51006: "bun",
    50983: "sodium",
    50971: "potassium",
    50882: "bicarbonate",
    50931: "glucose",
    50813: "lactate",
    51301: "wbc",
    51222: "hemoglobin",
    51265: "platelets",
}