import pandas as pd


def extract_car_features(df):
    """
    Extract car-level ACV fault-localisation features from one case.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed long-format telemetry for one ACV case.

    Returns
    -------
    pandas.DataFrame
        One row per car containing engineered features.
    """

    valid_df = df[
        df["ACV Information Valid"] == "Valid"
    ].copy()

    valid_df["cooling_error"] = (
        valid_df["Indoor Average Temperature"]
        - valid_df["ACV Control Temperature (Cooling)"]
    )

    feature_rows = []

    for car_id in sorted(valid_df["car_id"].unique()):

        car_data = valid_df[
            valid_df["car_id"] == car_id
        ]

        feature_rows.append({
            "car_id": car_id,

            # Indoor-temperature behaviour
            "indoor_temp_mean":
                car_data["Indoor Average Temperature"].mean(),

            "indoor_temp_std":
                car_data["Indoor Average Temperature"].std(),

            # Cooling-performance behaviour
            "cooling_error_mean":
                car_data["cooling_error"].mean(),

            "cooling_error_median":
                car_data["cooling_error"].median(),

            "cooling_error_std":
                car_data["cooling_error"].std(),
        })

    features = pd.DataFrame(feature_rows)

    # Peer-relative indoor temperature
    features["indoor_temp_peer_diff"] = features.apply(
        lambda row:
            row["indoor_temp_mean"]
            - features.loc[
                features["car_id"] != row["car_id"],
                "indoor_temp_mean"
            ].mean(),
        axis=1
    )

    # Peer-relative cooling error
    features["cooling_error_peer_diff"] = features.apply(
        lambda row:
            row["cooling_error_mean"]
            - features.loc[
                features["car_id"] != row["car_id"],
                "cooling_error_mean"
            ].mean(),
        axis=1
    )

    error_matrix = valid_df.pivot(
        index="Time",
        columns="car_id",
        values="cooling_error"
    )

    highest_error_car = error_matrix.idxmax(
        axis=1,
        skipna=True
    )

    persistence = (
        highest_error_car
        .value_counts(normalize=True)
    )

    features["highest_error_fraction"] = (
        features["car_id"]
        .map(persistence)
        .fillna(0)
    )

    return features