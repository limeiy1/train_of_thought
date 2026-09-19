import numpy as np


def turning_points(x):
    x = np.asarray(x, dtype=np.float64)
    diffs = np.diff(x)
    nonzero_idx = np.flatnonzero(diffs != 0)
    if len(nonzero_idx) < 2:
        return x[[0, -1]]
    d = diffs[nonzero_idx]
    sign_change = np.where(d[:-1] * d[1:] < 0)[0] + 1
    keep = np.concatenate(([nonzero_idx[0]], nonzero_idx[sign_change], [nonzero_idx[-1] + 1]))
    return x[keep]


def count_cycles(x):
    """ASTM E1049-85 four-point rainflow counting.

    Returns a list of (range, mean, count) tuples, count is 1.0 for a closed
    cycle and 0.5 for a leftover half-cycle in the residue.
    """
    points = turning_points(x)
    stack = []
    cycles = []
    for p in points:
        stack.append(p)
        while len(stack) >= 4:
            x1, x2, x3, x4 = stack[-4], stack[-3], stack[-2], stack[-1]
            y1, y2, y3 = abs(x1 - x2), abs(x2 - x3), abs(x3 - x4)
            if y2 <= y1 and y2 <= y3:
                cycles.append((y2, (x2 + x3) / 2.0, 1.0))
                del stack[-3:-1]
            else:
                break
    for j in range(len(stack) - 1):
        rng = abs(stack[j] - stack[j + 1])
        mean = (stack[j] + stack[j + 1]) / 2.0
        cycles.append((rng, mean, 0.5))
    return cycles
