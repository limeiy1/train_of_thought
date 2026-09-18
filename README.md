# Train of Thought — NebulaX Hackathon 2026

This repository contains our solution for the NebulaX Hackathon 2026 predictive maintenance challenge.

The project consists of independent predictive-maintenance subsystems that analyse equipment telemetry and produce fault detection or localisation results through a common application interface.

## Project Structure

```text
train_of_thought/
├── app/                          # Common user interface
├── data/                         # Local datasets (not committed)
├── predictions/                  # Generated subsystem prediction CSVs
├── Optional_Items/
│   └── ACV/
│       └── code/
│           ├── predict.py        # ACV command-line inference entry point
│           ├── src/
│           │   ├── ingestion.py
│           │   ├── preprocessing.py
│           │   ├── features.py
│           │   └── pipeline.py
│           └── notebooks/
│               ├── 01_data_inspection.ipynb
│               ├── 02_eda.ipynb
│               └── 03_feature_experiments.ipynb
├── requirements.txt
└── README.md
```

## Subsystems

### Air-Conditioning & Ventilation (ACV) — Refrigerant Leakage Localisation

The ACV subsystem identifies which car in an eight-car train is most likely affected by a refrigerant leakage fault.

Each input Excel file contains multivariate ACV telemetry for all eight cars. The subsystem analyses the telemetry and returns all car IDs ranked from most to least likely faulty.

#### Approach

Exploratory analysis identified sustained cooling underperformance as the most consistent fault-localisation signal across the schema-compatible labelled training cases.

For each valid telemetry observation, a cooling-error signal is calculated as:

```text
Cooling Error =
Indoor Average Temperature
- ACV Control Temperature (Cooling)
```

The final fault score for each car is the mean cooling error across the complete recording:

```text
Fault Score = mean(Cooling Error)
```

Cars are ranked in descending order of fault score, with a higher score indicating greater sustained cooling underperformance.

Only observations where:

```text
ACV Information Valid == "Valid"
```

are used when calculating the fault score.

#### Method Selection

Several approaches were investigated using the labelled training cases, including:

- Mean cooling error
- Peer-relative temperature features
- Temporal persistence features
- Logistic Regression
- Random Forest
- RBF Support Vector Machine

Mean cooling error provided the most consistent ranking across the five schema-compatible labelled training cases.

Additional temporal-segment experiments were performed to determine whether the ranking remained stable across different portions of each recording. A trimmed-mean experiment was also performed to test whether the result depended on a small number of extreme cooling-error observations.

These experiments supported the use of full-recording mean cooling error as the final fault-localisation score.

One training case contained a substantially different and much richer telemetry schema. This case was analysed separately rather than forcing incompatible parameters into the final scoring method.

The observed performance on the labelled training cases is treated as supporting evidence for method selection rather than an unbiased estimate of performance on unseen data, since the final feature was selected through exploratory analysis of the available labelled cases.

#### Running ACV Inference

From the repository root:

```bash
python Optional_Items/ACV/code/predict.py --input data/ACV/Test/acv_test_case.xlsx --output predictions/acv_predictions.csv
```

Example console output:

```text
ACV prediction complete.
Input: data/ACV/Test/acv_test_case.xlsx
Ranking: 01|03|07|04|08|06|02|05
Output: predictions/acv_predictions.csv
```

#### Prediction Output

The generated `acv_predictions.csv` follows the required ACV ranking format:

```csv
file_id,ranked_cars
acv_test_case.xlsx,01|03|07|04|08|06|02|05
```

`ranked_cars` contains every car ID exactly once, ordered from most to least likely faulty.

The generated prediction file is stored in:

```text
predictions/acv_predictions.csv
```

The subsystem prediction CSVs can subsequently be packaged into the team's final `predictions.zip`.

#### Python Integration

The reusable ACV inference function is implemented in:

```text
Optional_Items/ACV/code/src/pipeline.py
```

Its main interface is:

```python
ranked_cars, scores = predict_acv(uploaded_file_path)
```

The function returns:

- `ranked_cars` — ordered list of car IDs from most to least suspicious.
- `scores` — car-level DataFrame containing `car_id`, `fault_score`, and `rank`.

This interface allows the application layer to display the most likely faulty car, complete ranking, fault scores, and downloadable prediction output without reimplementing the ACV analysis.

#### ACV Development Notebooks

The ACV development process is documented in:

```text
Optional_Items/ACV/code/notebooks/
├── 01_data_inspection.ipynb
├── 02_eda.ipynb
└── 03_feature_experiments.ipynb
```

The notebooks cover:

- Dataset structure and schema inspection
- Data quality and missingness analysis
- Exploratory analysis of faulty and healthy cars
- Cooling-error feature investigation
- Feature and method comparison
- Alternative supervised models
- Schema compatibility analysis
- Temporal robustness analysis
- Outlier robustness analysis
- Final method selection and test inference

The production implementation in `src/` reproduces the final fault-localisation method selected through these experiments.

---

### Other Subsystems

Additional subsystem implementations will be documented here as they are integrated.

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Application

The application provides a common interface for selecting a subsystem, uploading telemetry data, running the corresponding inference pipeline, viewing the prediction result, and downloading the generated output.

Further application instructions will be added as the team interface is integrated.