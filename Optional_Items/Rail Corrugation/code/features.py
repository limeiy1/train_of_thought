import numpy as np
from scipy.stats import kurtosis

SIDE1_POSITIONS = [1, 3, 5, 7]
SIDE2_POSITIONS = [2, 4, 6, 8]


def _side_columns(columns, kind, positions):
    return [c for c in columns if c.startswith(kind) and any(f"position {p} " in c for p in positions)]


def get_column_groups(columns):
    return {
        "s1_vib": _side_columns(columns, "Vibration", SIDE1_POSITIONS),
        "s2_vib": _side_columns(columns, "Vibration", SIDE2_POSITIONS),
        "s1_shock": _side_columns(columns, "Shock", SIDE1_POSITIONS),
        "s2_shock": _side_columns(columns, "Shock", SIDE2_POSITIONS),
    }


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


def crest_factor(x):
    r = rms(x)
    return float(np.max(np.abs(x)) / r) if r > 0 else 0.0


def extract_features(df, groups):
    s1v = df[groups["s1_vib"]].to_numpy().ravel()
    s2v = df[groups["s2_vib"]].to_numpy().ravel()
    s1s = df[groups["s1_shock"]].to_numpy().ravel()
    s2s = df[groups["s2_shock"]].to_numpy().ravel()

    rms_s1v, rms_s2v = rms(s1v), rms(s2v)
    rms_s1s, rms_s2s = rms(s1s), rms(s2s)
    kurt_s1v, kurt_s2v = float(kurtosis(s1v)), float(kurtosis(s2v))
    crest_s1v, crest_s2v = crest_factor(s1v), crest_factor(s2v)

    return {
        "speed": float(df["Rotating speed"].mean()),
        "overall_vib_rms": (rms_s1v + rms_s2v) / 2,
        "log_vib_ratio": float(np.log(rms_s1v / rms_s2v)),
        "overall_shock_rms": (rms_s1s + rms_s2s) / 2,
        "log_shock_ratio": float(np.log(rms_s1s / rms_s2s)),
        "overall_vib_kurtosis": (kurt_s1v + kurt_s2v) / 2,
        "log_kurtosis_ratio": float(np.log(kurt_s1v / kurt_s2v)) if kurt_s1v > 0 and kurt_s2v > 0 else 0.0,
        "overall_vib_crest": (crest_s1v + crest_s2v) / 2,
        "log_crest_ratio": float(np.log(crest_s1v / crest_s2v)),
    }
