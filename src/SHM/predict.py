import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from features import extract_features
from train_baseline import load_stress_series

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "SHM"


def predict_file(x, model_info, rf_model=None):
    if model_info["type"] == "power_law":
        feats = extract_features(x, m_values=[model_info["m"]])
        return model_info["a"] * feats[f"F_{model_info['m']}"]
    feats = extract_features(x)
    row = pd.DataFrame([feats])[model_info["feature_cols"]]
    return float(rf_model.predict(row)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of test CSV files")
    parser.add_argument("--output", required=True, help="Path to write shm_predictions.csv")
    args = parser.parse_args()

    with open(MODEL_DIR / "shm_model.json") as f:
        model_info = json.load(f)

    rf_model = None
    if model_info["type"] == "random_forest":
        rf_model = joblib.load(MODEL_DIR / "shm_random_forest.joblib")

    input_dir = Path(args.input)
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        x = load_stress_series(path)
        pred = predict_file(x, model_info, rf_model)
        rows.append({"file_id": path.name, "prediction": pred})
        print(f"{path.name}: {pred:.6f}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
