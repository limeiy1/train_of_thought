import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from features import extract_features, get_column_groups

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Rail_Corrugation"
MODEL_DIR = ROOT / "models" / "Rail_Corrugation"
CACHE_PATH = ROOT / "outputs" / "Rail_Corrugation" / "train_features_cache.csv"

FEATURE_COLS = [
    "speed", "overall_vib_rms", "log_vib_ratio", "overall_shock_rms", "log_shock_ratio",
    "overall_vib_kurtosis", "log_kurtosis_ratio", "overall_vib_crest", "log_crest_ratio",
]


def build_feature_table(data_dir, use_cache=True):
    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached features from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    labels = pd.read_csv(data_dir / "Train_Labels.csv")
    groups = None
    rows = []
    t0 = time.time()
    for i, r in labels.iterrows():
        df = pd.read_csv(data_dir / "Train" / r["filename"])
        if groups is None:
            groups = get_column_groups(df.columns)
        feats = extract_features(df, groups)
        rows.append({"filename": r["filename"], "label": r["label"], **feats})
        if i % 50 == 0:
            print(f"  {i}/{len(labels)} files processed ({time.time()-t0:.0f}s)")
    table = pd.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CACHE_PATH, index=False)
    return table


def loo_macro_f1(model_fn, X, y):
    n = len(y)
    preds = np.empty(n, dtype=object)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = model_fn()
        model.fit(X[mask], y[mask])
        preds[i] = model.predict(X[[i]])[0]
    return f1_score(y, preds, average="macro"), preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    table = build_feature_table(Path(args.data_dir), use_cache=not args.no_cache)
    X = table[FEATURE_COLS].to_numpy()
    y = table["label"].to_numpy()

    print(f"\nClass distribution: {dict(pd.Series(y).value_counts())}")

    print("\n--- Leave-one-out cross-validation (macro F1) ---")
    logreg_fn = lambda: LogisticRegression(class_weight="balanced", max_iter=2000)
    rf_fn = lambda: RandomForestClassifier(class_weight="balanced", n_estimators=300, max_depth=5, random_state=0)

    logreg_score, logreg_preds = loo_macro_f1(logreg_fn, X, y)
    print(f"LogisticRegression (balanced):  macro F1 = {logreg_score:.4f}")
    rf_score, rf_preds = loo_macro_f1(rf_fn, X, y)
    print(f"RandomForest (balanced):        macro F1 = {rf_score:.4f}")

    from sklearn.metrics import classification_report
    if rf_score >= logreg_score:
        print("\nSelecting RandomForest as the final model.")
        print(classification_report(y, rf_preds))
        final_model = rf_fn()
        chosen_score = rf_score
    else:
        print("\nSelecting LogisticRegression as the final model.")
        print(classification_report(y, logreg_preds))
        final_model = logreg_fn()
        chosen_score = logreg_score

    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_cols": FEATURE_COLS, "loo_macro_f1": chosen_score},
                MODEL_DIR / "rail_model.joblib")
    print(f"\nSaved final model to {MODEL_DIR / 'rail_model.joblib'} (LOO macro F1 = {chosen_score:.4f})")


if __name__ == "__main__":
    main()
