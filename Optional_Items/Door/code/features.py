from segment import get_operation


def segment_features(chunk):
    return {
        "operation": get_operation(chunk),
        "current_mean": chunk["Motor current(mA)"].mean(),
    }
