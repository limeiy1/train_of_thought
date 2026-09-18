from src.ingestion import load_acv_file, wide_to_long, get_car_ids
from src.preprocessing import preprocess_acv


def predict_acv(file_path):
    """
    Run the ACV refrigerant-leak fault-localisation pipeline.

    Parameters
    ----------
    file_path : str or Path
        Path to an ACV telemetry Excel file.

    Returns
    -------
    ranked_cars : list[str]
        Car IDs ranked from most to least suspicious.

    scores : pandas.DataFrame
        Car-level mean cooling-error scores and ranks.
    """

    # Load input file and detect car IDs from its column headers
    raw_df = load_acv_file(file_path)
    expected_cars = get_car_ids(raw_df)

    # ACV cases are expected to contain 8 cars
    if len(expected_cars) != 8:
        raise ValueError(
            f"Expected 8 cars in the input file, found {len(expected_cars)}."
        )

    # Convert telemetry to long format and preprocess it
    long_df = wide_to_long(raw_df)
    processed_df = preprocess_acv(long_df)

    # Only use telemetry marked as valid
    valid_df = processed_df[
        processed_df["ACV Information Valid"] == "Valid"
    ].copy()

    # Calculate cooling error for each observation
    valid_df["cooling_error"] = (
        valid_df["Indoor Average Temperature"]
        - valid_df["ACV Control Temperature (Cooling)"]
    )

    # Calculate the mean cooling error for each car
    scores = (
        valid_df
        .groupby("car_id")["cooling_error"]
        .mean()
        .reset_index(name="fault_score")
        .sort_values("fault_score", ascending=False)
        .reset_index(drop=True)
    )

    # Assign ranking
    scores["rank"] = range(1, len(scores) + 1)

    # Extract ordered car IDs
    ranked_cars = scores["car_id"].tolist()

    # Validate that every input car appears exactly once in the ranking
    if len(ranked_cars) != len(expected_cars):
        raise ValueError(
            f"Expected predictions for {len(expected_cars)} cars, "
            f"found {len(ranked_cars)}."
        )

    if set(ranked_cars) != set(expected_cars):
        raise ValueError(
            f"Predicted car IDs do not match input car IDs. "
            f"Expected {expected_cars}, got {ranked_cars}."
        )

    return ranked_cars, scores