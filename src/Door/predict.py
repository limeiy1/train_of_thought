import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import segment_features
from segment import find_segments

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "Door"


def classify(feats, model_info):
    op_model = model_info["operations"][feats["operation"]]
    z = op_model["coef"] * feats["current_mean"] + op_model["intercept"]
    prob_abnormal = 1 / (1 + np.exp(-z))
    return "Abnormal resistance" if prob_abnormal >= 0.5 else "Normal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a continuous Door stream CSV")
    parser.add_argument("--output", required=True, help="Path to write door_predictions.csv")
    args = parser.parse_args()

    with open(MODEL_DIR / "door_model.json") as f:
        model_info = json.load(f)

    df = pd.read_csv(args.input)
    segments = find_segments(df, gap_threshold_ms=model_info["gap_threshold_ms"])

    rows = []
    for s, e in segments:
        chunk = df.iloc[s:e]
        feats = segment_features(chunk)
        prediction = classify(feats, model_info)
        rows.append({
            "start_time": chunk.iloc[0]["Datetime"],
            "end_time": chunk.iloc[-1]["Datetime"],
            "prediction": prediction,
        })
        print(f"{chunk.iloc[0]['Datetime']} -> {chunk.iloc[-1]['Datetime']}: {feats['operation']}, {prediction}")

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} predicted segments to {out_path}")


if __name__ == "__main__":
    main()
