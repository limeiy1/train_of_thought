# Rail Corrugation — Normal / Side I / Side II Classification

## 1. Problem

Classify each 1-second, multi-channel axle-box vibration/shock recording as
**Normal**, **Side I** (corrugation on the odd-numbered axle-box positions'
rail), or **Side II** (even-numbered positions' rail). 272 labelled training
files, 68 held-out test files. Severely imbalanced: 234 Normal / 14 Side I /
24 Side II. Scored on **macro F1** — the unweighted average of each class's
own F1, which does not let a model coast by always predicting "Normal."

**Headline difference from SHM and Door: there is no single clean feature
here.** Both of those subsystems had one physically-grounded number that
separated classes almost perfectly. Rail Corrugation doesn't — every feature
investigated here has real signal but substantial overlap, so this subsystem
needed an actual multivariate model, honestly validated, rather than a
threshold.

## 2. Data format

Each file: column 1 is `Rotating speed`; columns 2–129 are Vibration/Shock
pairs for 8 cars × 8 axle-box positions (10,000 rows = 1 second @ 10kHz).
Odd positions (1,3,5,7) are the **Side I** rail; even positions (2,4,6,8) are
**Side II**. Confirmed `Test.csv`'s header is byte-identical to `Train.csv`'s.

## 3. Investigation — what actually separates the classes (and what doesn't)

This took several rounds of checking single examples, then verifying against
the full 272-file training set — worth walking through, since the single-file
checks were actively misleading twice.

**Side-asymmetry ratio (vibration RMS, Side I ÷ Side II).** A single example
per class looked clean (Normal ratio ≈1.10, Side I ≈1.22, Side II ≈0.94 — the
right direction for each). Checked across all 272 files: real signal in the
right direction (Side II's median log-ratio is clearly more negative than
Normal's, Side I's is more positive), but **heavy overlap** between all three
classes' full ranges. Not usable alone.

**Overall vibration magnitude (RMS, averaged both sides).** Normal
mean=0.304, Side I mean=0.576, Side II mean=0.719 across all 272 files — a
much clearer trend. But every fault file's `Rotating speed` sits in a narrow
0.503–0.512 band, while Normal spans 0.4996–1.000 — so before trusting this,
checked whether it was just a speed artifact. Restricting to only the
low-speed Normal files (same speed band as every fault file, n=190): their
overall vibration RMS is still meaningfully lower (mean 0.357, median 0.334)
than Side I (median 0.567) or Side II (median 0.721) — so the effect
survives controlling for speed, though the full-range Normal distribution
still overlaps both fault classes at the edges.

**Vibration kurtosis.** The single-example check looked dramatic — Normal
≈3.2–3.4 (near-Gaussian), Side I ≈22–37, Side II ≈39–46 (10x+ higher,
suggesting impulsive/spiky vibration from periodic impact loading). Checked
across all 272 files: **this did not hold up** — Normal's full range is
0.87–307 with a median of 13.1, overlapping both fault classes almost
entirely. This is the second time a 1-2 example spot-check looked like a
clean discriminator and wasn't once checked against the full dataset — a
direct argument for why every "clean separation" claim in this project was
verified against the complete training set, not a handful of examples,
before being trusted (see `SHM/README.md` and `Door/README.md` for the two
cases where it *did* hold up).

**A genuine, disclosed confound: speed correlates with fault status in this
dataset.** All 38 fault files sit at speed ≈0.503–0.512; only 44/234 (19%) of
Normal files do. This is very likely an artifact of how the dataset was
collected (e.g. corrugation testing done under a controlled/reduced speed
regime) rather than a causal relationship, but it's real, available signal in
both Train and (presumably) Test, so it's included as a feature — flagged
here rather than silently relied upon.

## 4. Final feature set (9 features, `features.py`)

Given the imbalance (only 14 Side I examples), the feature set stays
deliberately small and physically motivated rather than exploding into
per-channel or per-car features (which per-car inspection showed to be noisy
and inconsistent in sign — pooling all 8 cars per side gave a cleaner signal
than any individual car):

`speed`, `overall_vib_rms`, `log_vib_ratio`, `overall_shock_rms`,
`log_shock_ratio`, `overall_vib_kurtosis`, `log_kurtosis_ratio`,
`overall_vib_crest`, `log_crest_ratio` — each an "overall" (both-sides-pooled)
magnitude paired with a "log-ratio" (Side I vs Side II) asymmetry, across
three different signal characterizations (RMS energy, kurtosis/impulsiveness,
crest factor), plus speed.

## 5. File architecture

```
src/Rail_Corrugation/
├── features.py        # column-group identification (Side I/II × vib/shock),
│                         RMS/crest-factor helpers, extract_features()
├── train_baseline.py  # builds the feature table (cached), compares
│                         LogisticRegression vs RandomForest via full
│                         leave-one-out CV scored on macro F1, saves the winner
└── predict.py          # loads the saved model, scores new files, writes
                          rail_predictions.csv
```

## 6. Methodology

- **Full leave-one-out cross-validation** (272 refits) — affordable at this
  N, and the most honest generalization estimate available, same philosophy
  as SHM/Door.
- **`class_weight="balanced"`** on both candidate models — without it, a
  classifier could get ~86% accuracy by predicting "Normal" for everything,
  scoring near-zero on the metric that actually matters (macro F1).
- **Two models compared**, not assumed: a regularized multinomial
  `LogisticRegression` and a depth-limited `RandomForestClassifier` (`max_depth=5`,
  to control overfitting risk given only 14–24 minority examples).

## 7. Results

| Model | LOO macro F1 |
|---|---|
| LogisticRegression (balanced) | 0.6737 |
| **RandomForest (balanced)** | **0.7048** |

Per-class breakdown (RandomForest, LOO predictions):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.97 | 0.89 | 0.93 | 234 |
| Side I | 0.31 | 0.64 | 0.42 | 14 |
| Side II | 0.71 | 0.83 | 0.77 | 24 |

**Honest read of this**: Normal is detected reliably. Side II is decent.
**Side I is the weak point** — recall is reasonable (0.64, catches most real
Side I cases) but precision is poor (0.31, meaning roughly 2 false alarms for
every real Side I case flagged). This is the direct, expected consequence of
having only 14 Side I training examples to define that class's boundary in a
9-dimensional feature space with real class overlap — not a bug, and not
hidden: this is a materially harder subsystem than SHM (0.9738) or Door
(0.9909), and the results say so honestly.

Run against the real held-out `Test/` (68 files): 52 Normal / 11 Side I / 5
Side II predicted — proportionally more Side I than training's ~5% rate,
consistent with the known precision issue above (the model over-flags Side I
somewhat).

## 8. Outstanding for submission

- [ ] Wrap `predict.py` (alongside SHM's and Door's) in the shared app.
- [ ] Record the demo video.
- [ ] Add `rail_predictions.csv` to `predictions.zip`.
- [ ] Consider, if time allows: frequency-domain (FFT) features tuned to the
      speed/wavelength relationship the Info Kit describes — the current
      feature set is entirely time-domain; a spectral approach targeting the
      actual corrugation-excited frequency band could plausibly improve Side
      I precision specifically, but wasn't attempted here given time and the
      added risk of tuning a frequency-selective feature on only 14 examples.
