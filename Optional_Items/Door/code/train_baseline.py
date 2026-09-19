import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import segment_features
from metric import score
from segment import find_segments, parse_ts_ms

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Door"
MODEL_DIR = ROOT / "models" / "Door"


def build_table(train_df, labels_df):
    ts_ms = parse_ts_ms(train_df["Datetime"])
    detected = find_segments(train_df)

    mismatches = 0
    rows = []
    for (s, e), (_, true_row) in zip(detected, labels_df.iterrows()):
        chunk = train_df.iloc[s:e]
        if chunk.iloc[0]["Datetime"] != true_row["start_time"] or chunk.iloc[-1]["Datetime"] != true_row["end_time"]:
            mismatches += 1
        feats = segment_features(chunk)
        rows.append({
            **feats,
            "status": true_row["status"],
            "start_ms": ts_ms.iloc[s],
            "end_ms": ts_ms.iloc[e - 1],
        })
    print(f"Segmentation check: {len(detected)} detected vs {len(labels_df)} labelled, {mismatches} mismatches")
    return pd.DataFrame(rows)


def loo_predict(sub):
    X = sub["current_mean"].to_numpy().reshape(-1, 1)
    y = (sub["status"] == "Abnormal resistance").astype(int).to_numpy()
    n = len(y)
    preds = np.empty(n, dtype=int)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = LogisticRegression().fit(X[mask], y[mask])
        preds[i] = model.predict(X[[i]])[0]
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    train_df = pd.read_csv(data_dir / "Train.csv")
    labels_df = pd.read_csv(data_dir / "Train_Segments_Answer.csv")
    table = build_table(train_df, labels_df)

    true_segs, loo_pred_segs = [], []
    models = {}
    for op in ["Open", "Close"]:
        sub = table[table.operation == op].reset_index(drop=True)
        preds = loo_predict(sub)
        y_true = (sub["status"] == "Abnormal resistance").astype(int).to_numpy()
        acc = (preds == y_true).mean()
        print(f"{op}: n={len(sub)}  LOO accuracy={acc:.4f}  mismatches={int((preds != y_true).sum())}")

        for row, pred in zip(sub.itertuples(), preds):
            label = "Abnormal resistance" if pred == 1 else "Normal"
            true_segs.append((row.start_ms, row.end_ms, row.status))
            loo_pred_segs.append((row.start_ms, row.end_ms, label))

        X_full = sub["current_mean"].to_numpy().reshape(-1, 1)
        final_model = LogisticRegression().fit(X_full, y_true)
        models[op] = {
            "coef": float(final_model.coef_[0][0]),
            "intercept": float(final_model.intercept_[0]),
        }

    pipeline_score = score(true_segs, loo_pred_segs)
    print(f"\nEnd-to-end LOO pipeline score (metric.score, segmentation+classification): {pipeline_score:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_info = {"gap_threshold_ms": 1000, "operations": models}
    with open(MODEL_DIR / "door_model.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model config to {MODEL_DIR / 'door_model.json'}")


if __name__ == "__main__":
    main()
