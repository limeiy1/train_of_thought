import argparse
from pathlib import Path

import pandas as pd

from pipeline import predict_acv

def main():
    parser = argparse.ArgumentParser(
        description="ACV refrigerant leakage fault localisation"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input ACV .xlsx file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output acv_predictions.csv file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Run ACV fault-localisation pipeline
    ranked_cars, scores = predict_acv(input_path)

    # Competition-required ranking format
    ranking_string = "|".join(ranked_cars)

    prediction_df = pd.DataFrame([
        {
            "file_id": input_path.name,
            "ranked_cars": ranking_string,
        }
    ])

    # Create output directory if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df.to_csv(
        output_path,
        index=False
    )

    print("ACV prediction complete.")
    print(f"Input: {input_path}")
    print(f"Ranking: {ranking_string}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()