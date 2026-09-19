# Train Condition Monitoring — Write-Up (All Subsystems)

This covers all four subsystems attempted. Each subsystem's `Optional_Items/<Subsystem>/code/`
folder additionally includes a `README.md` (architecture + results) and `EXPLAINER.md`
(comprehensive function-by-function walkthrough) for deeper reference.

## Door — Cycle segmentation + Abnormal-resistance classification

**Approach**: The raw stream turned out to contain zero idle rows — the logger only records
while a cycle is moving, so segmentation reduces to splitting wherever the timestamp gap
exceeds 1 second (real gaps are always 10+ seconds; within a cycle, always exactly 20ms).
Classification uses a per-operation (Open/Close) Logistic Regression on a single feature: mean
motor current across the cycle — sustained current (not peak) cleanly separates Normal from
Abnormal-resistance, since the controller compensates for friction by drawing more torque
throughout the motion rather than spiking briefly.

**Validation**: Leave-one-out cross-validation across all 110 labelled cycles.

**Result**: End-to-end pipeline score (IoU-weighted F1) = **0.9909**.

## ACV — Refrigerant leak localisation

**Approach**: Compares thermodynamic pull-down behaviour across all cars in a file. The car
with the largest cooling deficit relative to target temperature, despite sustained compressor
operation, is ranked most likely to have a refrigerant leak.

## SHM — Cumulative fatigue damage regression

**Approach**: Applies ASTM E1049 rainflow cycle counting to the raw stress signal, then
estimates damage via a Miner's-rule physics model: `D = a · Σ(cycles × amplitude^m)`. Rather
than assuming a material exponent `m`, it was grid-searched — `m = 5` won decisively over
every neighbouring value, evidence it recovers the dataset's true underlying fatigue exponent
rather than fitting noise.

**Validation**: Leave-one-out cross-validation across the 64 labelled training files.

**Result**: LOO score = **0.9738** (vs. 0.9094 for a generic RandomForest baseline on the same
data — the one-parameter physics model outperformed a much more flexible model).

## Rail Corrugation — Normal / Side I / Side II classification

**Problem**: classify each 1-second, 129-channel axle-box vibration/shock recording as Normal,
Side I, or Side II corrugation (odd axle-box positions 1,3,5,7 ride the Side I rail; even
positions 2,4,6,8 ride Side II). 272 training files, severely imbalanced (234 Normal / 14 Side I
/ 24 Side II). Scored on macro F1, so the model can't coast by always predicting Normal.

**Approach**: unlike Door and SHM, no single feature cleanly separates the classes here — two
candidates that looked strong on a few examples (a Side I/II vibration ratio, then a dramatic
kurtosis jump) both fell apart once checked against all 272 files instead of a handful, which is
exactly the kind of premature conclusion full-dataset checks exist to catch. The final model uses
9 physically-motivated features pooled across all 8 cars (per-car analysis was noisy and
inconsistent in sign): for vibration and shock, an "overall" magnitude (both sides averaged) plus
a "log-ratio" asymmetry (Side I vs II) across RMS energy, kurtosis, and crest factor, plus train
speed — a genuine confound (every fault file sits at one narrow speed) that was checked to still
hold after controlling for speed, then kept as a feature anyway and disclosed rather than hidden.

**Model selection**: a class-weighted LogisticRegression and a class-weighted, depth-limited
RandomForest were compared head-to-head under full leave-one-out cross-validation (272 refits).

**Result**: LOO macro F1 = **0.7048** (RandomForest, vs. 0.6737 for LogisticRegression). Per-class:
Normal F1=0.93, Side II F1=0.77, **Side I F1=0.42** — recall is workable (0.64) but precision is
poor (0.31, ~2 false alarms per real case), a direct and expected consequence of defining that
class's boundary from only 14 training examples. Not hidden — the honest weak point of this
subsystem, given the data available.

## Common methodology across all subsystems

- **Leave-one-out cross-validation**, not k-fold: with as few as 14 examples in a class,
  k-fold would leave some folds with zero minority examples, making per-class metrics
  meaningless. LOOCV tests every example exactly once without ever training on its own answer,
  giving the most stable estimate available from limited data.
- **Simplest model the data actually supports**, not defaulted-to complexity: SHM and Door
  each needed only a handful of fitted numbers because a single feature (or one per operation)
  already separated classes cleanly; Rail Corrugation genuinely required a full model because no
  such single feature existed — verified by comparing simpler and more complex models
  head-to-head under identical validation, every time.
- **No test-set leakage**: model thresholds/parameters are fit only from labelled training
  data; where the unlabelled held-out test files' feature distributions were inspected (e.g.
  Door's Close-current range), it was to sanity-check generalisation, never to retune the model.
