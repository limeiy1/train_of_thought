import pandas as pd

GAP_THRESHOLD_MS = 1000


def parse_ts_ms(datetime_series):
    parts = datetime_series.str.split("-", expand=True).astype(int)
    parts.columns = ["Y", "Mo", "D", "H", "Mi", "S", "Ms"]
    return (((parts["D"] * 24 + parts["H"]) * 60 + parts["Mi"]) * 60 + parts["S"]) * 1000 + parts["Ms"]


def find_segments(df, gap_threshold_ms=GAP_THRESHOLD_MS):
    """Split a continuous Door stream into cycles.

    The recorder only logs rows while a cycle is in motion: consecutive rows
    within a cycle are always exactly 20ms apart, and the gap between cycles
    is always >= ~10 seconds. Any threshold between those two extremes finds
    the same boundaries.
    """
    ts = parse_ts_ms(df["Datetime"])
    deltas = ts.diff()
    boundaries = [0] + list(deltas[deltas > gap_threshold_ms].index) + [len(df)]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def get_operation(chunk):
    return "Close" if (chunk["Close command"] == 1).mean() > (chunk["Open command"] == 1).mean() else "Open"
