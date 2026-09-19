import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features import M_GRID, extract_features

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "SHM"
MODEL_DIR = ROOT / "models" / "SHM"


def competition_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    return max(0.0, 1.0 - mape)


def load_stress_series(path):
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy()


def build_feature_table(file_dir, filenames):
    rows = []
    for name in filenames:
        x = load_stress_series(file_dir / name)
        rows.append(extract_features(x))
    return pd.DataFrame(rows, index=filenames)


def loo_score_through_origin(F, y): #actual fxn call
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        a = np.sum(F[mask] * y[mask]) / np.sum(F[mask] ** 2)
        preds[i] = a * F[i]
    return competition_score(y, preds), preds


def loo_score_random_forest(X, y, **rf_kwargs):
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = RandomForestRegressor(random_state=0, **rf_kwargs)
        model.fit(X[mask], y[mask])
        preds[i] = model.predict(X[[i]])[0]
    return competition_score(y, preds), preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    filenames = labels["filename"].tolist()
    y = labels["damage"].to_numpy()

    print(f"Loading {len(filenames)} training files and extracting features...")
    feat_table = build_feature_table(data_dir / "Train", filenames)

    print("\n--- Physics-informed single-feature model: D = a * sum(n_i * amplitude_i^m) ---")
    best_m, best_score, best_a = None, -np.inf, None
    for m in M_GRID:
        F = feat_table[f"F_{m}"].to_numpy()
        score, _ = loo_score_through_origin(F, y)
        a = np.sum(F * y) / np.sum(F**2)
        print(f"  m={m:2d}  LOO score={score:.4f}  a={a:.6g}")
        if score > best_score:
            best_m, best_score, best_a = m, score, a

    print(f"\nBest physics-informed model: m={best_m}, LOO score={best_score:.4f}")

    print("\n--- RandomForest baseline on the full feature set (for comparison) ---")
    feature_cols = [c for c in feat_table.columns]
    X = feat_table[feature_cols].to_numpy()
    rf_score, _ = loo_score_random_forest(X, y, n_estimators=200, max_depth=4)
    print(f"  LOO score={rf_score:.4f}")

    if best_score >= rf_score:
        print(f"\nSelecting physics-informed model (m={best_m}) as the final model.")
        F_full = feat_table[f"F_{best_m}"].to_numpy()
        a_final = np.sum(F_full * y) / np.sum(F_full**2)
        model_info = {"type": "power_law", "m": best_m, "a": a_final, "loo_score": best_score}
    else:
        print("\nSelecting RandomForest model as the final model.")
        model = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=4)
        model.fit(X, y)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "shm_random_forest.joblib")
        model_info = {"type": "random_forest", "feature_cols": feature_cols, "loo_score": rf_score}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "shm_model.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"\nSaved model config to {MODEL_DIR / 'shm_model.json'}")


if __name__ == "__main__":
    main()
