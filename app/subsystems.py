"""
Loads the Door, SHM, ACV, and Rail Corrugation subsystems' code from
Optional_Items/<Subsystem>/code/ and exposes one predict_<subsystem>(file_path)
function per subsystem, plus METHODOLOGY and model-parameter info for the app UI.

Deliberately loads from Optional_Items/ rather than src/: Optional_Items/ is
the one location that ships in both the dev repo and a clean hackathon
submission folder (which only contains app/, predictions.zip, Optional_Items/,
and the demo video) -- so the app is self-contained wherever it's copied,
with no dependency on the team's separate src/ + models/ dev workspace.

Door, SHM, and Rail Corrugation each ship their own same-named `features.py`
(different contents per subsystem). To import all three into one process
without one clobbering another in sys.modules, each subsystem's directory is
added to sys.path, its modules imported under their real (bare) names so
their own internal `from features import ...`-style relative imports resolve
correctly, then immediately re-bound to a unique alias and removed from
sys.modules before the next subsystem's same-named files are loaded.
"""

import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_ITEMS = REPO_ROOT / "Optional_Items"


def _load_bare(dirpath, module_names):
    sys.path.insert(0, str(dirpath))
    try:
        loaded = {}
        for name in module_names:
            sys.modules.pop(name, None)
            loaded[name] = importlib.import_module(name)
        return loaded
    finally:
        sys.path.remove(str(dirpath))
        for name in module_names:
            sys.modules.pop(name, None)


_door_mods = _load_bare(OPTIONAL_ITEMS / "Door" / "code", ["segment", "features"])
door_segment, door_features = _door_mods["segment"], _door_mods["features"]

_shm_mods = _load_bare(OPTIONAL_ITEMS / "SHM" / "code", ["rainflow", "features"])
shm_rainflow, shm_features = _shm_mods["rainflow"], _shm_mods["features"]

_rail_mods = _load_bare(OPTIONAL_ITEMS / "Rail Corrugation" / "code", ["features"])
rail_features = _rail_mods["features"]

_acv_mods = _load_bare(OPTIONAL_ITEMS / "ACV" / "code" / "src", ["ingestion", "preprocessing", "pipeline"])
predict_acv = _acv_mods["pipeline"].predict_acv


DOOR_MODEL_PATH = OPTIONAL_ITEMS / "Door" / "model" / "door_model.json"
SHM_MODEL_PATH = OPTIONAL_ITEMS / "SHM" / "model" / "shm_model.json"
RAIL_MODEL_PATH = OPTIONAL_ITEMS / "Rail Corrugation" / "model" / "rail_model.joblib"


METHODOLOGY = {
    "door": {
        "summary": "The uploaded stream is first split into individual door cycles by finding "
                    "gaps in the timestamp — the logger only records rows while a cycle is moving, "
                    "so any gap over 1 second marks a new cycle. Each cycle is then classified Normal "
                    "vs. Abnormal-resistance using a Logistic Regression on its mean motor current "
                    "(one model per Open/Close operation, since their current baselines differ). "
                    "Sustained current, not peak current, is what reveals resistance. Validated with "
                    "Leave-One-Out Cross-Validation across all 110 labelled training cycles.",
        "validation_result": "LOO score: 0.9909",
    },
    "shm": {
        "summary": "Applies rainflow cycle counting (ASTM E1049) to the raw stress signal to extract "
                    "discrete stress cycles, then estimates cumulative fatigue damage with a "
                    "Miner's-rule physics model: damage is proportional to the sum of cycle counts "
                    "times amplitude raised to a material exponent. That exponent wasn't assumed — it "
                    "was grid-searched, and m=5 won decisively over every neighbouring value, evidence "
                    "it recovers the dataset's true underlying fatigue exponent. Validated with "
                    "Leave-One-Out Cross-Validation across the 64 labelled training files.",
        "validation_result": "LOO score: 0.9738",
    },
    "corrugation": {
        "summary": "Extracts 9 physically-motivated features from the 129-channel axle-box "
                    "vibration/shock recording — overall vibration/shock magnitude and Side I vs. "
                    "Side II asymmetry, across RMS energy, kurtosis, and crest factor, plus train "
                    "speed — pooled across all 8 cars for a cleaner signal than any single car gives. "
                    "A class-weighted RandomForest then classifies Normal / Side I / Side II, chosen "
                    "over a simpler Logistic Regression by head-to-head comparison. Validated with "
                    "full Leave-One-Out Cross-Validation (272 refits) scored on macro F1.",
        "validation_result": "LOO macro F1: 0.7048",
    },
}


def get_model_params(subsystem):
    if subsystem == "door":
        with open(DOOR_MODEL_PATH) as f:
            return json.load(f)
    if subsystem == "shm":
        with open(SHM_MODEL_PATH) as f:
            return json.load(f)
    if subsystem == "corrugation":
        bundle = joblib.load(RAIL_MODEL_PATH)
        return {
            "model_type": type(bundle["model"]).__name__,
            "feature_cols": bundle["feature_cols"],
            "loo_macro_f1": bundle["loo_macro_f1"],
        }
    raise ValueError(f"Unknown subsystem: {subsystem}")


def predict_door(file_path, display_name=None):
    with open(DOOR_MODEL_PATH) as f:
        model_info = json.load(f)

    df = pd.read_csv(file_path)
    segments = door_segment.find_segments(df, gap_threshold_ms=model_info["gap_threshold_ms"])

    rows = []
    for s, e in segments:
        chunk = df.iloc[s:e]
        feats = door_features.segment_features(chunk)
        op_model = model_info["operations"][feats["operation"]]
        z = op_model["coef"] * feats["current_mean"] + op_model["intercept"]
        prob_abnormal = float(1 / (1 + np.exp(-z)))
        prediction = "Abnormal resistance" if prob_abnormal >= 0.5 else "Normal"
        rows.append({
            "start_time": str(chunk.iloc[0]["Datetime"]),
            "end_time": str(chunk.iloc[-1]["Datetime"]),
            "operation": feats["operation"],
            "current_mean_mA": round(float(feats["current_mean"]), 2),
            "probability_abnormal": round(prob_abnormal, 4),
            "prediction": prediction,
        })

    csv_df = pd.DataFrame(rows)[["start_time", "end_time", "prediction"]]
    return {
        "segments": rows,
        "n_segments": len(rows),
        "n_abnormal": sum(1 for r in rows if r["prediction"] == "Abnormal resistance"),
        "csv_data": csv_df.to_csv(index=False),
    }


def predict_shm(file_path, display_name=None):
    with open(SHM_MODEL_PATH) as f:
        model_info = json.load(f)

    x = pd.read_csv(file_path, header=None).iloc[:, 0].to_numpy()

    if model_info["type"] == "power_law":
        m = model_info["m"]
        feats = shm_features.extract_features(x, m_values=[m])
        damage = model_info["a"] * feats[f"F_{m}"]
    else:
        raise ValueError(f"Unsupported SHM model type: {model_info['type']}")

    file_id = display_name or Path(file_path).name
    csv_df = pd.DataFrame([{"file_id": file_id, "prediction": damage}])
    return {
        "predicted_damage": float(damage),
        "exponent_m": model_info["m"],
        "csv_data": csv_df.to_csv(index=False),
    }


def predict_rail(file_path, display_name=None):
    bundle = joblib.load(RAIL_MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df = pd.read_csv(file_path)
    groups = rail_features.get_column_groups(df.columns)
    feats = rail_features.extract_features(df, groups)
    X = pd.DataFrame([feats])[feature_cols].to_numpy()
    prediction = model.predict(X)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_.tolist(), model.predict_proba(X)[0].tolist()))

    file_id = display_name or Path(file_path).name
    csv_df = pd.DataFrame([{"file_id": file_id, "prediction": prediction}])
    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "features": {k: round(float(v), 5) for k, v in feats.items()},
        "csv_data": csv_df.to_csv(index=False),
    }
