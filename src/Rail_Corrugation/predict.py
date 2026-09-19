import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import extract_features, get_column_groups

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "Rail_Corrugation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of Rail_Corrugation test CSV files")
    parser.add_argument("--output", required=True, help="Path to write rail_predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_DIR / "rail_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    input_dir = Path(args.input)
    groups = None
    rows = []
    for path in sorted(input_dir.glob("*.csv"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)):
        df = pd.read_csv(path)
        if groups is None:
            groups = get_column_groups(df.columns)
        feats = extract_features(df, groups)
        X = pd.DataFrame([feats])[feature_cols].to_numpy()
        prediction = model.predict(X)[0]
        rows.append({"file_id": path.name, "prediction": prediction})
        print(f"{path.name}: {prediction}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
