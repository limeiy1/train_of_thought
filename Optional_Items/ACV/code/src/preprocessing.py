import pandas as pd


PARAMETER_RENAME_MAP = {
    "Outside Temperature Sensor Reading": "Outdoor Average Temperature"
}


def standardize_parameter_names(df):
    """
    Standardize equivalent ACV parameter names across case files.
    """
    return df.rename(columns=PARAMETER_RENAME_MAP)


def remove_empty_telemetry_rows(df):
    """
    Remove car-timestamp rows where all ACV telemetry values are missing.

    Metadata columns and car_id are excluded when determining whether
    telemetry is completely missing.
    """
    metadata_columns = {
        "Car model",
        "Train number",
        "Time",
        "car_id"
    }

    telemetry_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    return df.dropna(
        subset=telemetry_columns,
        how="all"
    ).reset_index(drop=True)


def preprocess_acv(df):
    """
    Apply the common preprocessing steps required before ACV analysis.
    """
    df = standardize_parameter_names(df)
    df = remove_empty_telemetry_rows(df)

    return df