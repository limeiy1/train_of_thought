def iou(t0, t1, p0, p1):
    inter = max(0, min(t1, p1) - max(t0, p0))
    union = (t1 - t0) + (p1 - p0) - inter
    return inter / union if union > 0 else 0.0


def score(true_segs, pred_segs):
    """IoU-weighted F1, per the Door Info Kit Section 4.

    true_segs / pred_segs: list of (start, end, label), start/end comparable
    with `<`/`-` (numeric ms or anything else with the same arithmetic).
    """
    candidates = []
    for ti, (t0, t1, tl) in enumerate(true_segs):
        for pi, (p0, p1, pl) in enumerate(pred_segs):
            if tl != pl:
                continue
            v = iou(t0, t1, p0, p1)
            if v > 0:
                candidates.append((v, ti, pi))
    candidates.sort(key=lambda x: -x[0])

    used_t, used_p = set(), set()
    matches = []
    for v, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matches.append(v)

    soft_recall = sum(matches) / len(true_segs) if true_segs else 0.0
    soft_precision = sum(matches) / len(pred_segs) if pred_segs else 0.0
    if soft_recall + soft_precision == 0:
        return 0.0
    return 2 * soft_recall * soft_precision / (soft_recall + soft_precision)
