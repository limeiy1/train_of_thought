import numpy as np

from rainflow import count_cycles

M_GRID = list(range(2, 15))


def rainflow_features(x, m_values=M_GRID):
    cycles = count_cycles(x)
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[2] for c in cycles])
    amps = ranges / 2.0

    feats = {
        "n_cycles": float(counts.sum()),
        "max_amplitude": float(amps.max()) if len(amps) else 0.0,
        "mean_amplitude": float(np.average(amps, weights=counts)) if len(amps) else 0.0,
    }
    for m in m_values:
        feats[f"F_{m}"] = float(np.sum(counts * amps**m))
    return feats


def signal_stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "std": float(x.std()),
        "rms": float(np.sqrt(np.mean(x**2))),
        "peak_to_peak": float(x.max() - x.min()),
        "abs_mean": float(np.abs(x).mean()),
    }


def extract_features(x, m_values=M_GRID):
    feats = rainflow_features(x, m_values)
    feats.update(signal_stats(x))
    return feats
