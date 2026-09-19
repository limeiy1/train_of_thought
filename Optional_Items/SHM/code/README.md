# SHM (Structural Health Monitoring) — Cumulative Fatigue Damage Estimation

## 1. Problem

Predict a single numeric **cumulative fatigue damage** value (0 = no damage, 1 = failure)
for each raw dynamic-stress sensor recording from a rail vehicle's carbody/bogie frame.

- 64 labelled training files (`Train_Labels.csv`), 16 unlabelled held-out test files.
- Each file: one column, ~581,000 raw stress readings, no header, no timestamp.
- Scored on `max(0, 1 − MAPE)` against the true damage value (organiser-held).

## 2. Physics background

Fatigue damage from a random vibration signal is traditionally computed with three
combined concepts:

1. **Stress cycles** — a signal doesn't damage a material by its absolute level, but by
   how far it swings up and down. Each swing (peak → valley) is a *cycle*, described by
   its **amplitude** (half the swing size) and **mean**.
2. **S-N curve** — an empirical material law: a fixed-amplitude cycle of size `σₐ` causes
   failure after `N` cycles, following a power law `σₐ^m · N = C`. `m` and `C` are
   material constants; `m` controls how much *more* damaging a bigger swing is (e.g. `m=5`
   means doubling the amplitude causes failure in `2⁵ = 32×` fewer cycles).
3. **Miner's rule** — for a real signal with a mix of cycle sizes, group cycles by
   amplitude level `i`. Damage is the sum of "life fraction used" at each level:

   ```
   D = Σᵢ (nᵢ / Nᵢ) = Σᵢ nᵢ · σᵢ^m / C = (1/C) · Σᵢ (nᵢ · σᵢ^m)
   ```

4. **Rainflow counting** (ASTM E1049) — the standard algorithm for turning a messy,
   continuous stress trace into a clean list of `(amplitude, mean, count)` cycles,
   correctly handling nested swings.

**The key trick used here:** damage `D` is *linear* in `F_m := Σᵢ(nᵢ · σᵢ^m)` for the
*true* material exponent `m`. `F_m` is computable directly from the raw signal via
rainflow counting. Since the true `m` isn't given to us, we grid-search over plausible
values, fitting only the one remaining scale constant (`a = 1/C`) per candidate `m`, and
pick whichever `m` generalises best on held-out validation folds.

## 3. File architecture

```
src/SHM/
├── rainflow.py         # ASTM E1049 four-point rainflow counting, from raw signal
│                         → list of (range, mean, count) cycles
├── features.py          # cycles → F_m for m = 2..14 (the Miner's-rule quantity),
│                         # plus generic signal stats (RMS, std, peak-to-peak) as fallback
├── train_baseline.py    # loads Train data + labels, extracts features, grid-searches m
│                         # with leave-one-out CV, selects + saves the best model
└── predict.py           # loads the saved model, scores new files, writes
                          # shm_predictions.csv in the required file_id,prediction schema
```

Data flow:

```
raw stress CSV (581k rows)
      │
      ▼
rainflow.turning_points()   — strip to local peaks/valleys only
      │
      ▼
rainflow.count_cycles()     — four-point algorithm → (range, mean, count) per cycle
      │
      ▼
features.rainflow_features() — amplitude = range/2; F_m = Σ(count · amplitude^m), m=2..14
      │
      ▼
train_baseline.py           — fit D ≈ a · F_m per candidate m, leave-one-out CV,
      │                        pick best m, refit on all 64 files
      ▼
models/SHM/shm_model.json   — {"type": "power_law", "m": 5, "a": <fitted value>}
      │
      ▼
predict.py                  — apply saved model to Test/*.csv
      │
      ▼
outputs/SHM/shm_predictions.csv   — file_id, prediction
```

## 4. Training / validation process

- **Feature**: for each candidate exponent `m` (2 to 14, then refined in 0.1 steps),
  compute `F_m` per training file.
- **Fit**: least-squares line *forced through the origin* (`D = a·F_m`, no intercept —
  physically, zero cycles must mean zero damage): `a = Σ(F·y) / Σ(F²)`.
- **Validate**: leave-one-out cross-validation (justified by the small N = 64 — a single
  train/test split would be noisy and wasteful). For each file, fit `a` on the other 63,
  predict the held-out file, repeat for all 64, then score with the exact competition
  metric (`max(0, 1 − MAPE)`).
- **Compare**: a generic RandomForest on all features (every `F_m` plus RMS/std/etc.),
  same leave-one-out protocol, as a sanity check against the physics model.
- **Select**: whichever approach scores higher on leave-one-out is refit on all 64 files
  and saved.

## 5. Results

| Model | Leave-one-out score |
|---|---|
| Physics model, `m=4` | 0.5177 |
| **Physics model, `m=5`** | **0.9738** |
| Physics model, `m=6` | 0.6822 |
| RandomForest (all features) | 0.9094 |

Fine search (0.1 steps, 3.5–6.5) confirmed the peak sits at **`m = 5.0` exactly** — a
sharp spike, not a broad trend, which is strong evidence the dataset's reference damage
values were generated with a true fatigue exponent of 5. The single physics-derived
feature beat the fully generic ML model, despite the ML model having access to every
feature including all `F_m` columns.

## 6. Prediction (inference)

`predict.py --input <dir> --output <path>`:

1. Loads the saved model config (`m` and `a` for the physics model).
2. For each file in the input directory: loads the raw stress column, runs rainflow
   counting, computes `F_m` for the saved `m`, predicts `D = a · F_m`.
3. Writes one row per file: `file_id, prediction`.

Run against the real held-out `Test/` folder (16 files) — output confirmed to match the
exact schema in `04_Example_Submission/shm_predictions.csv`.

## 7. Outstanding for submission

- [ ] Wrap `predict.py` in a simple web app (upload a file → see the prediction →
      download) — compulsory deliverable, not yet built.
- [ ] Record a ≤3-minute demo video of the app in use.
- [ ] Zip `shm_predictions.csv` directly (no subfolder) into `predictions.zip`.
- [ ] Optional: copy this file into `Optional_Items/SHM/` as the write-up.
