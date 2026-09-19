from pathlib import Path
import pandas as pd


def load_acv_file(file_path):
    """
    Load an ACV Excel telemetry file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ACV file not found: {file_path}")

    if file_path.suffix.lower() != ".xlsx":
        raise ValueError("ACV input file must be an .xlsx file.")

    return pd.read_excel(file_path)


def get_car_ids(df):
    """
    Extract car identifiers from ACV telemetry column names.

    Example:
        'Car 03 - ACV Running Mode' -> '03'
    """
    car_ids = set()

    for column in df.columns:
        if column.startswith("Car ") and " - " in column:
            car_part = column.split(" - ", 1)[0]
            car_id = car_part.replace("Car ", "")
            car_ids.add(car_id)

    return sorted(car_ids)


def get_parameters(df):
    """
    Extract the unique ACV parameter names from telemetry columns.
    """
    parameters = set()

    for column in df.columns:
        if column.startswith("Car ") and " - " in column:
            parameter = column.split(" - ", 1)[1]
            parameters.add(parameter)

    return sorted(parameters)

def wide_to_long(df):
    """
    Convert wide ACV telemetry into one row per timestamp and car.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ACV telemetry in wide format.

    Returns
    -------
    pandas.DataFrame
        Long-format telemetry containing metadata, car_id,
        and the parameters available for each car.
    """
    metadata_columns = [
        col for col in ["Car model", "Train number", "Time"]
        if col in df.columns
    ]

    car_ids = get_car_ids(df)

    car_frames = []

    for car_id in car_ids:
        prefix = f"Car {car_id} - "

        car_columns = [
            col for col in df.columns
            if col.startswith(prefix)
        ]

        car_df = df[metadata_columns + car_columns].copy()

        car_df = car_df.rename(
            columns={
                col: col.removeprefix(prefix)
                for col in car_columns
            }
        )

        car_df["car_id"] = car_id

        car_frames.append(car_df)

    long_df = pd.concat(
        car_frames,
        ignore_index=True
    )

    return long_df