# Rail Corrugation — Comprehensive Walkthrough (Concepts + Every Function)

Deep-dive companion to `README.md`. This explains the physical reasoning
behind every feature, then walks through every function in every script.

## Part 1 — What rail corrugation is, physically

Rail corrugation is a wavy wear pattern that develops on the rail's running
surface from repeated wheel-rail contact — imagine a washboard texture worn
into the rail over time, with a fairly consistent wavelength (centimeters to
tens of centimeters). As a wheel rolls over it, that periodic bump pattern
excites vibration in the axle box (the housing around the wheel's bearing) at
a frequency related to the corrugation's wavelength and the train's speed
(`frequency = speed / wavelength`). Accelerometers mounted on the axle boxes
pick this up directly.

The train has 8 cars, each with 8 axle boxes — positions 1,3,5,7 physically
ride on one rail ("Side I"), positions 2,4,6,8 ride on the other ("Side II").
Corrugation can develop on just one rail while the other stays healthy, so
the two sides have to be judged independently from the same recording, not
treated as one combined "is something wrong" signal.

## Part 2 — Why this problem doesn't have a "Door-style" clean feature

Door had one feature (mean motor current) with a wide, zero-overlap gap
between classes. Rail Corrugation was investigated the same rigorous way —
check a feature, then verify it against every training file rather than a
handful of examples — and here's what that process actually found:

**Side-to-side vibration ratio** (does Side I or Side II show more energy?)
has real signal in the physically expected direction but overlaps heavily
between classes when checked across all 272 files, not just one example per
class.

**Overall vibration magnitude** (how much energy overall, regardless of
side) shows a real, meaningful upward trend from Normal → Side I → Side II,
and — critically — that trend survives even after restricting the Normal
comparison group to only the files recorded at the same low speed every
fault file happens to sit at. That's the check that matters: if the
"Normal is quieter" pattern had vanished once speed was controlled for, it
would have meant the whole signal was just measuring speed, not corrugation.
It didn't vanish — it weakened a little, but stayed real.

**Vibration kurtosis** (a measure of how "spiky"/impulsive a signal is,
vs. smooth Gaussian-like noise) looked like the strongest signal of all on a
single example per class — a 10x+ jump from Normal to either fault class.
This one is worth dwelling on, because it's a **direct lesson in why every
finding in this project got checked against the full dataset before being
trusted**: across all 272 files, Normal's kurtosis ranges from 0.87 all the
way to 307, with a median of 13.1 — completely swamping the apparent
separation the single-example check suggested. A model built trusting that
first impression would have looked great in a demo on 3 files and then failed
badly on real held-out data. This is exactly the kind of mistake leave-one-out
validation (Part 4) exists to catch even if a spot-check doesn't.

**The speed confound, stated plainly.** Every one of the 38 fault files in
training has `Rotating speed` between 0.503 and 0.512. Only 44 of the 234
Normal files (19%) fall in that same narrow band — the rest span all the way
up to 1.000. This is almost certainly an artifact of how the dataset was
assembled (plausibly: corrugation test runs were conducted at a fixed,
controlled speed, while "normal" background data was collected across
whatever speeds the train normally runs at) rather than corrugation somehow
causing low speed. It's used as a feature anyway, because it's real, honestly
available signal in the data we have and — presumably — in the held-out test
data too, generated the same way. But it's disclosed here rather than quietly
relied on, because it's a property of *this dataset's collection process*,
not a property of corrugation itself, and wouldn't necessarily transfer to a
different real-world deployment collected differently.

**Why pool all 8 cars instead of analyzing per-car?** Checked this directly:
looking at Side I vs Side II vibration RMS car-by-car for a known Side I fault
file, the ratio flips sign between cars (some cars show Side I louder, others
show Side II louder, inconsistently) — a lot of car-to-car noise. Pooling all
8 cars' worth of channels together before computing RMS/kurtosis/crest factor
per side gave a visibly cleaner, more consistent number than any single car
did. This matches the physical framing in the Info Kit — the task is judging
the *rail's* condition (a track-level property), not any one car's individual
axle box.

## Part 3 — `features.py`: turning 129 raw columns into 9 numbers

**`_side_columns(columns, kind, positions)`** ([features.py:8-9](features.py#L8-L9))
— a small string-matching helper: given the full column list, a `kind`
("Vibration" or "Shock"), and a list of axle-box `positions`, returns every
column belonging to those positions. The column names are verbose and
descriptive (e.g. `"Vibration of bearing in position 3 of car 5"`), so this
just filters by `startswith(kind)` and checks whether `"position {p} "`
appears in the name for any wanted position `p` — the trailing space in the
match string matters, so `"position 1 "` doesn't accidentally also match
`"position 10 "` if this dataset ever had double-digit positions (it doesn't
here, but it's a cheap safety margin).

**`get_column_groups(columns)`** ([features.py:12-18](features.py#L12-L18))
— calls `_side_columns` four times to produce the four channel groups this
whole subsystem is built on: `s1_vib` (32 columns: 8 cars × Side I's 4
positions), `s2_vib`, `s1_shock`, `s2_shock`. Computed once per file (actually
once total, since every file shares the same header) and reused, rather than
re-deriving which columns belong where on every single feature extraction
call.

**`rms(x)`** ([features.py:21-22](features.py#L21-L22)) — root-mean-square,
`sqrt(mean(x²))`. The standard way to summarize a vibration signal's overall
energy in one number, since a vibration signal oscillates around zero and a
plain mean would just cancel out to ~0.

**`crest_factor(x)`** ([features.py:25-27](features.py#L25-L27)) — peak
amplitude divided by RMS. Two signals can have identical RMS energy while
looking very different: one smooth and evenly oscillating, one mostly quiet
with rare sharp spikes. Crest factor distinguishes them — a higher value
means the signal is more "spiky" relative to its average energy, which is a
classic indicator used in vibration-based fault diagnosis generally (bearing
faults, gear faults, and — the hypothesis here — corrugation's periodic
impact loading).

**`extract_features(df, groups)`** ([features.py:30-51](features.py#L30-L51))
— given one file's raw dataframe and the precomputed column groups:
1. Pulls out all four channel groups as flat 1D arrays (`.to_numpy().ravel()`
   collapses "32 columns × 10,000 rows" into one long array per side/signal
   type, since for these summary statistics it doesn't matter which specific
   channel or timestamp a value came from within its group).
2. Computes RMS, kurtosis (`scipy.stats.kurtosis`, Fisher's definition —
   checked empirically that every observed value here is comfortably
   positive, so the log-ratio below is safe), and crest factor, separately
   for Side I and Side II vibration (kurtosis/crest only needed for
   vibration, not shock, in the final feature set).
3. Returns 9 features, each following one of two patterns:
   - **`overall_X`**: the *average* of Side I's and Side II's value for that
     statistic — "how much is going on overall, regardless of which side."
   - **`log_X_ratio`**: the *log* of Side I's value divided by Side II's —
     "which side is more affected, and by how much." Using the log of the
     ratio rather than the raw ratio makes the feature symmetric: a file
     where Side I is 2x Side II gives the same *magnitude* (opposite sign) as
     one where Side II is 2x Side I, which is a friendlier scale for a linear
     model to work with than the raw ratio (where "2x" and "0.5x" aren't
     symmetric around 1.0).

## Part 4 — `train_baseline.py`: comparing two models honestly

**`build_feature_table(data_dir, use_cache)`** ([train_baseline.py:25-45](train_baseline.py#L25-L45))
— reads `Train_Labels.csv`, then for each of the 272 listed files: loads the
raw CSV (10,000 rows × 129 columns), computes the column groups once (`if
groups is None`, [line 36](train_baseline.py#L36) — since every file shares
an identical header, there's no need to redo this detection 272 times), and
calls `extract_features`. Progress is printed every 50 files since this loop
takes over a minute (reading and processing ~272 × 1.3M raw values). Results
are cached to `outputs/Rail_Corrugation/train_features_cache.csv` — on
subsequent runs, `build_feature_table` just loads that cache instantly
instead of reprocessing every raw file, which matters a lot when iterating on
model choices without needing to change the underlying features.

**`loo_macro_f1(model_fn, X, y)`** ([train_baseline.py:48-57](train_baseline.py#L48-L57))
— the same leave-one-out pattern used for SHM and Door, generalized to
multi-class: for each of the 272 examples, fit a fresh model (`model_fn()`,
a zero-argument constructor closure, so the *same* function works for both
candidate model types) on the other 271, predict the held-out one, collect
all 272 "honest" predictions, then score them with `f1_score(..., average="macro")`
— scikit-learn's macro F1, matching the Info Kit's exact metric definition.

**`main()`** ([train_baseline.py:60-98](train_baseline.py#L60-L98)):
1. Builds (or loads cached) features, splits into `X` (9 feature columns) and
   `y` (string labels).
2. Prints the class distribution up front ([line 70](train_baseline.py#L70))
   — a small but deliberate step: with this much imbalance, you want the
   imbalance visible in the output every time, not just remembered from the
   Info Kit.
3. Defines two model constructors ([lines 73-74](train_baseline.py#L73-L74)):
   - `LogisticRegression(class_weight="balanced", ...)` — `class_weight="balanced"`
     reweights the loss function so the 14 Side I examples count for as much
     in total as the 234 Normal examples during fitting, which is essential
     here — without it, a model can nearly ignore the minority classes and
     still look accurate.
   - `RandomForestClassifier(class_weight="balanced", n_estimators=300, max_depth=5, ...)` —
     `max_depth=5` deliberately limits how deep each tree can grow, which
     caps model complexity given how few minority-class examples exist to
     validate a more complex tree structure against.
4. Runs `loo_macro_f1` for both ([lines 76-79](train_baseline.py#L76-L79)),
   prints both scores side by side.
5. Picks whichever scored higher (RandomForest, 0.7048 vs. 0.6737), prints a
   full per-class `classification_report` for transparency (this is where
   the Side I precision=0.31 weakness in `README.md`'s results table comes
   from — printed and kept, not filtered out), then **refits that model type
   on all 272 examples** ([line 93](train_baseline.py#L93)) — same principle
   as SHM/Door: leave-one-out is for honestly *measuring* quality; the
   *deployed* model uses every available labeled example once validation is
   done.
6. Saves the fitted model object, the feature column order (needed so
   `predict.py` builds its feature vector in the exact order the model
   expects), and the achieved LOO score into one `joblib` bundle
   ([lines 96-97](train_baseline.py#L96-L97)) — unlike SHM/Door's plain-JSON
   models, a `RandomForestClassifier` genuinely has hundreds of decision
   trees' worth of internal parameters, so it needs `joblib`'s
   pickle-based serialization rather than a handful of numbers in JSON.

## Part 5 — `predict.py`: scoring new files

([predict.py:13-39](predict.py#L13-L39)) Loads the saved `joblib` bundle
(model + the exact feature-column order used during training), then for
every CSV file in the input directory (sorted numerically by the digits in
the filename — `Test1.csv, Test2.csv, ..., Test10.csv` rather than the
alphabetical `Test1.csv, Test10.csv, Test2.csv`, purely for readable output
ordering): computes the column groups once, extracts the same 9 features,
arranges them into a `DataFrame` with columns in the model's expected order
(`[feature_cols]`, [line 30](predict.py#L30) — this ordering step matters,
since a `RandomForestClassifier` has no idea which column means what, only
that column 3 always means whatever it meant during training), predicts, and
writes `file_id, prediction` rows.

## Part 6 — Tying it together: why this model looks different from SHM/Door's

SHM's model is 2 numbers. Door's is 4. This subsystem's saved model is an
entire serialized `RandomForestClassifier` — 300 decision trees. That's not
inconsistent with the project's general philosophy of "use the simplest model
the data actually supports" — it's the honest consequence of what the
investigation in Part 2 found: **there is no single feature, or even simple
linear combination of a couple of features, that cleanly separates these
three classes.** A simpler model (the `LogisticRegression` alternative) was
tried first and honestly compared, and it scored measurably worse (0.6737 vs
0.7048) under the exact same leave-one-out protocol used everywhere else in
this project — so the more complex model earned its place here rather than
being reached for by default.

The headline number, **0.7048 LOO macro F1**, and its very unequal per-class
breakdown (Normal 0.93, Side II 0.77, Side I 0.42) is the honest picture of a
genuinely harder problem: 14 training examples is very little to define a
class's boundary in a 9-feature space with real overlap against the other two
classes, and the leave-one-out protocol is what surfaces that limitation
clearly instead of hiding it behind a single seemingly-good number. The
model still adds real value — it catches most Side I cases (recall 0.64) and
most Side II cases (recall 0.83) — it just also raises more false alarms on
Side I specifically than a confident deployment would want, and that's stated
directly rather than glossed over.
