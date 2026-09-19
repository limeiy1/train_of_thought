# Door — Comprehensive Walkthrough (Concepts + Every Function)

This is the deep-dive companion to `README.md` (which covers architecture and
results for a presentation). This file explains **why** the approach works
from first principles, then walks through **every function in every script**
in the order data actually flows through them.

## Part 1 — The problem, restated in plain terms

You're handed one giant CSV: a continuous log of a door controller's sensors
(motor current, voltage, back-EMF, position, and a bunch of switch/command
flags), sampled roughly every 20 milliseconds, running for over an hour.
Somewhere in that log, the door opened and closed roughly 110 times. Nobody
tells you where any single cycle starts or stops — you have to find that
yourself — and *then*, for each cycle you find, decide whether the door
struggled (abnormal resistance, e.g. from a jammed rail or worn rubber strip)
or moved normally.

This is genuinely two separate problems stacked on top of each other:

1. **Segmentation** — where does each cycle begin and end?
2. **Classification** — given a correctly-identified cycle, was it Normal or
   Abnormal?

Getting segmentation wrong ruins classification too, because the scoring
metric requires *both* correct timing *and* correct label for a prediction to
count as a match at all (see Part 3 for the exact rule) — so it's worth being
extremely careful about part 1 before ever touching part 2.

## Part 2 — The two discoveries that make this tractable

### Discovery A: the log has no idle data in it at all

The obvious way to think about segmentation is "look for periods of motor
activity surrounded by quiet." That's what you'd expect from a real deployed
system logging continuously. But when I actually measured the data, that's
**not what's here**:

- `Train.csv` contains exactly 18,036 rows.
- The answer key (`Train_Segments_Answer.csv`) lists 110 segments, and their
  `n_rows` column — how many rows each segment is supposed to span — **sums
  to exactly 18,036.**
- When I computed the time gap between every pair of consecutive rows, the
  result was strictly bimodal: **17,926 gaps of exactly 20ms**, and **109
  gaps of several seconds or more** — nothing in between, no in-between
  "settling" behavior, no noise blurring the line.

Put those three facts together and there's only one explanation: **the
logger simply did not record anything while the door was sitting still.**
Every row in the file belongs to some cycle; the "gaps between cycles" the
documentation warns about are gaps in the *timestamp*, not gaps filled with
flat idle readings. This means segmentation isn't a signal-processing problem
at all — it's an index-slicing problem: **cut the stream wherever the time
jump is unnaturally large**, and you've found every boundary, exactly.

I didn't take this on faith — I verified it by slicing `Train.csv`
sequentially using each segment's declared `n_rows` and confirming the
resulting chunk's first and last timestamps matched the answer key's
`start_time`/`end_time` for all 110 segments, with zero mismatches. I then
checked `Test.csv` independently (since that's the file we actually have to
run inference on) and found the identical structure: 6,253 rows, 37 large
gaps → 38 segments, minimum gap 11.1 seconds vs. a maximum in-cycle delta of
exactly 20ms. Because the smallest possible "real" gap (10+ seconds) is
*five hundred times* larger than the largest possible "still mid-cycle" delta
(20ms), the exact threshold value barely matters — anything from 100ms to
5000ms finds the same boundaries.

### Discovery B: sustained current, not peak current, reveals resistance

Once you can isolate a cycle, how do you tell if it was struggling? The
naive guess — "abnormal cycles draw a higher peak current" — turned out to
be **false**: peak current, total cycle duration, and total distance the
door leaf travelled are all nearly identical between Normal and Abnormal
cycles. What differs dramatically is the **average** current sustained
across the *whole* cycle:

| Operation | Normal mean-current range | Abnormal mean-current range |
|---|---|---|
| Open | 630.3 – 666.5 mA | 738.3 – 972.8 mA |
| Close | 418.6 – 456.2 mA | 484.5 – 735.4 mA |

Zero overlap, in either operation, across all 110 training examples. The
physical story: this door's motor controller behaves like a speed-regulated
servo. It's designed to complete the door's travel in a fixed amount of time
regardless of friction — so when something resists the motion, the
controller compensates by commanding *more torque* (which means more
current) throughout the entire motion to hold the same speed profile, rather
than just spiking briefly and stalling. A brief current spike happens in
*every* cycle anyway (motor start-up/stop transients look similar whether or
not there's resistance) — it's the sustained elevated average that's the
tell.

This also explains why the classifier needs to know **which operation**
(Open vs Close) a segment is: Close cycles run a mechanically different
motion than Open cycles, so they have a completely different baseline
current level (~440 mA vs. ~650 mA). Trying to use one global threshold for
both would blur two well-separated clusters into an overlapping mess.
Conveniently, the raw data already tells you the operation directly via the
`Close command`/`Open command` flag columns — no inference needed.

## Part 3 — `metric.py`: the exact scoring rule (read this before the model code)

Understanding this file first makes everything downstream make sense, since
`train_baseline.py` uses it to actually score the whole pipeline.

**`iou(t0, t1, p0, p1)`** ([metric.py:1-4](metric.py#L1-L4)) — the classic
Intersection-over-Union for two time intervals: how much the true segment
`[t0,t1]` and a predicted segment `[p0,p1]` overlap, divided by how much
territory they cover between them combined. 1.0 means identical intervals;
0.0 means no overlap at all; `union <= 0` (a degenerate zero-length interval)
is treated as 0 rather than dividing by zero.

**`score(true_segs, pred_segs)`** ([metric.py:7-36](metric.py#L7-L36)) — this
is a direct reimplementation of the exact rule spelled out in
`Door_Subsystem_Info_Kit.md` §4:

1. **Build the candidate list** ([lines 13-20](metric.py#L13-L20)): for every
   (true segment, predicted segment) pair, skip it entirely if the labels
   don't match (`if tl != pl: continue` — a segment with perfect timing but
   the wrong label literally cannot match, full stop). For same-label pairs,
   compute IoU; keep the pair only if IoU is actually positive (some overlap
   exists).
2. **Sort candidates by IoU, descending** ([line 21](metric.py#L21)) — this
   is what makes the matching "greedy": the single best-overlapping pair in
   the whole list gets first claim on being matched.
3. **Walk the sorted list, claiming pairs one-to-one** ([lines 23-30](metric.py#L23-L30)):
   `used_t`/`used_p` track which true/predicted segments have already been
   claimed. Walking in descending-IoU order and skipping any pair where
   either side is already used means the best-overlapping pairs always get
   matched first, and no segment can be claimed twice.
4. **Turn the matches into recall/precision** ([lines 32-33](metric.py#L32-L33)):
   `soft_recall` is the *sum of the matched IoU values* divided by how many
   true segments existed — so a sloppy-but-matched segment contributes less
   than a perfectly-timed one, and an unmatched true segment (a miss)
   contributes zero. `soft_precision` is the same sum divided by how many
   segments *you* predicted — so a spurious extra prediction that never
   matches anything drags this down without needing to be explicitly
   penalized elsewhere.
5. **Combine into the final score via harmonic mean** ([lines 34-36](metric.py#L34-L36))
   — standard F1 pattern, just fed "soft" (IoU-weighted) recall/precision
   instead of the usual hard 0/1 recall/precision.

Before trusting this against real data, I hand-worked three scenarios and
checked the code reproduced them exactly: (a) one perfect match plus one
partial-overlap match plus one spurious extra prediction → verified
`soft_recall=0.9`, `soft_precision=0.6`, `score=0.72` by hand first; (b) a
perfectly-timed but wrong-label prediction → verified it scores exactly 0.0,
not some partial credit; (c) two overlapping predictions competing for one
true segment → verified the higher-IoU one wins the match and the other
becomes a free-floating false positive. All three matched hand calculations
exactly before I used this function for anything real.

## Part 4 — `segment.py`: turning the raw stream into cycle boundaries

**`parse_ts_ms(datetime_series)`** ([segment.py:6-9](segment.py#L6-L9)) — the
`Datetime` column looks like `2023-7-5-0-11-17-664`
(Year-Month-Date-Hour-Minute-Second-Millisecond, all hyphen-separated, not
zero-padded) — a nonstandard format pandas can't parse with its usual
datetime tools. This function splits every timestamp string on `-` into 7
integer columns, then converts the whole thing into a single **milliseconds
count** using the arithmetic `((days*24 + hours)*60 + minutes)*60 + seconds)
* 1000 + milliseconds`. Year and month are read but unused in the arithmetic
since every row in this dataset falls on the same day. The result is one
plain number per row that increases monotonically, which is all that's
needed for gap detection.

**`find_segments(df, gap_threshold_ms)`** ([segment.py:12-23](segment.py#L12-L23))
— this is Discovery A turned into code:
1. Convert every row's timestamp to milliseconds (`parse_ts_ms`).
2. `ts.diff()` — the time elapsed since the *previous* row, for every row.
   The first row gets `NaN` (no previous row), which is fine since it's never
   compared against the threshold, only used as a boundary marker directly.
3. `boundaries = [0] + list(deltas[deltas > gap_threshold_ms].index) + [len(df)]`
   — find every row-index where the gap to the previous row exceeds the
   threshold (a cycle boundary), then bookend that list with `0` (the very
   first row starts the first segment) and `len(df)` (one past the last row,
   so the final segment reaches the end of the file).
4. Turn that list of boundary indices into a list of `(start, end)` pairs by
   pairing up consecutive entries — e.g. boundaries `[0, 187, 330, ..., N]`
   become segments `(0,187), (187,330), ...`.

**`get_operation(chunk)`** ([segment.py:26-27](segment.py#L26-L27)) — given
all the rows belonging to one already-identified segment, decides Open vs.
Close by checking which of the `Close command`/`Open command` columns was
active more often across the segment (a majority vote, which is robust to
any single-row noise at the very edges of the cycle). This is possible only
because the raw sensor data already contains this flag directly — nothing
had to be inferred from duration or shape.

## Part 5 — `features.py`: what gets computed per segment

**`segment_features(chunk)`** ([features.py:4-8](features.py#L4-L8)) — takes
one segment's rows and returns exactly two things: `operation` (from
`get_operation`, needed to pick the right classifier) and `current_mean`
(the mean of the `Motor current(mA)` column across every row in the
segment — Discovery B's key feature). Deliberately minimal: since one
feature already separates the classes with a wide, clean margin (see Part
2), adding more features would only add complexity and overfitting risk on a
dataset with just 55 examples per operation, without buying anything.

## Part 6 — `train_baseline.py`: fitting and validating the classifier

**`build_table(train_df, labels_df)`** ([train_baseline.py:18-36](train_baseline.py#L18-L36))
— the bridge between "raw file" and "one row per segment, ready to model":
1. Converts every row's timestamp to milliseconds up front (`parse_ts_ms`,
   [line 19](train_baseline.py#L19)) — needed later to hand real millisecond
   timestamps to `metric.score`.
2. Runs `find_segments` on the *training* file ([line 20](train_baseline.py#L20)) —
   note this doesn't just trust the answer key; it independently re-derives
   the segment boundaries the same way `predict.py` will on unseen data.
3. Walks the detected segments **in lockstep** with the answer key's rows
   ([line 24](train_baseline.py#L24)), and for each one, checks whether the
   detected chunk's first/last timestamp actually equals that row's declared
   `start_time`/`end_time` ([lines 25-27](train_baseline.py#L25-L27)) — this
   is the self-check that proved segmentation is exact (0 mismatches) before
   any modeling was trusted.
4. For each segment, computes features (`segment_features`) and bundles them
   with the true `status` label and the segment's start/end in milliseconds
   ([lines 28-34](train_baseline.py#L28-L34)) — this millisecond timing is
   what makes it possible to later score this table with the *real*
   IoU-weighted metric, not just plain classification accuracy.

**`loo_predict(sub)`** ([train_baseline.py:39-49](train_baseline.py#L39-L49))
— leave-one-out cross-validation for one operation's worth of segments
(`sub` is already filtered to just Open or just Close):
1. `X` is the `current_mean` feature reshaped into the 2D column shape
   scikit-learn expects for a single feature; `y` is `1` for Abnormal
   resistance, `0` for Normal.
2. For every segment `i`: build a boolean `mask` that's `True` everywhere
   except position `i`, fit a fresh `LogisticRegression` on everyone *but*
   segment `i`, then predict segment `i`'s label using a model that never
   saw its true answer. Repeat for every segment.
3. Return the full array of "honest" predictions — one per segment, each
   made by a model that excluded that exact segment during fitting.

This is the same leave-one-out philosophy used for SHM, for the same reason:
with only 55 examples per operation, a single train/test split would be
noisy, but LOO uses every example as its own held-out test exactly once.

**`main()`** ([train_baseline.py:52-94](train_baseline.py#L52-L94)) —
orchestrates everything:
1. Loads `Train.csv` and the answer key, builds the per-segment feature
   table ([lines 58-60](train_baseline.py#L58-L60)).
2. For each operation ([lines 64-81](train_baseline.py#L64-L81)):
   - Runs leave-one-out prediction and prints accuracy ([lines 66-69](train_baseline.py#L66-L69)).
   - **Reconstructs the full segment list with LOO-predicted labels**
     ([lines 71-74](train_baseline.py#L71-L74)) — for every segment, appends
     its true `(start_ms, end_ms, true_status)` to `true_segs` and its
     `(start_ms, end_ms, LOO_predicted_status)` to `loo_pred_segs`. Since
     segmentation was already proven exact, the *timing* in both lists is
     identical — only the *label* can differ where the classifier's
     leave-one-out prediction was wrong. This means feeding these two lists
     into `metric.score` produces an honest, real end-to-end estimate of
     what the actual IoU-weighted F1 would be on genuinely new data, not
     just a plain accuracy number.
   - Refits one final `LogisticRegression` using **all** segments of that
     operation (no held-out this time, since validation is done and we want
     the strongest possible deployed model) and stores its two numbers —
     `coef` and `intercept` — in a plain dict ([lines 76-81](train_baseline.py#L76-L81)).
3. Scores the reconstructed full-dataset prediction against the truth using
   the real metric ([lines 83-84](train_baseline.py#L83-L84)) — this produced
   the reported **0.9909**.
4. Saves the two per-operation `{coef, intercept}` pairs plus the
   segmentation gap threshold to `models/Door/door_model.json`
   ([lines 86-90](train_baseline.py#L86-L90)) — deliberately plain JSON
   rather than a pickled/joblib model object, since the entire "model" is
   just 4 numbers total (2 per operation) and JSON keeps it trivially
   portable and human-readable.

## Part 7 — `predict.py`: turning a new stream into `door_predictions.csv`

**`classify(feats, model_info)`** ([predict.py:15-19](predict.py#L15-L19)) —
manually recomputes what `LogisticRegression.predict` would do, using just
the two saved numbers for the segment's operation:
1. `z = coef * current_mean + intercept` — the linear score a logistic
   regression computes internally.
2. `prob_abnormal = 1 / (1 + exp(-z))` — the sigmoid function, converting
   that linear score into a probability between 0 and 1.
3. Threshold at 0.5, same convention scikit-learn's own `.predict()` uses.

This is done by hand instead of loading a saved scikit-learn model object
specifically because the model *is* just two numbers — reimplementing the
sigmoid avoids adding a joblib/pickle round-trip (and its version-compat
fragility) for something this simple.

**`main()`** ([predict.py:22-54](predict.py#L22-L54)):
1. Parses `--input`/`--output` command-line arguments.
2. Loads the saved `door_model.json`.
3. Reads the input CSV and runs the *exact same* `find_segments` used during
   training (with the same saved gap threshold, [line 32](predict.py#L32)) —
   critically, this is the same function, not a reimplementation, so
   whatever was validated on `Train.csv` behaves identically here.
4. For each detected segment: computes features, classifies it, and records
   its native-format start/end timestamps (taken directly from the first and
   last row's `Datetime` string — no reformatting needed, since the Info Kit
   accepts the dataset's native timestamp format directly).
5. Writes everything to a CSV with exactly the required
   `start_time,end_time,prediction` columns.

## Part 8 — Tying it together: what was actually "trained," and why the numbers came out the way they did

There's no deep learning here, and nothing overwrought — the entire trained
model is **four numbers** (`coef` and `intercept`, once for Open, once for
Close). What made that enough:

1. Segmentation isn't learned at all — it's a deterministic rule
   (`find_segments`) that exploits a structural fact discovered by directly
   measuring the data (Discovery A), verified to be exact on every training
   segment and structurally identical on the real test file.
2. Classification is a single-feature logistic regression per operation,
   because a single feature (`current_mean`) already separates the two
   classes with a wide, clean, physically-explained gap (Discovery B) — more
   features would add overfitting risk on 55 examples without adding
   separating power.
3. **Leave-one-out cross-validation** ([train_baseline.py:39-49](train_baseline.py#L39-L49))
   is what turns "the training data looks separable" into an honest
   generalization estimate: each of the 110 segments gets predicted by a
   model that never saw its own answer.
4. Feeding the LOO predictions through the *actual* competition metric
   (`metric.score`, independently validated against hand-worked examples in
   Part 3) rather than plain accuracy is what produced the credible
   end-to-end number: **0.9909**, decomposed as segmentation being exact (so
   it contributes no error) and classification missing exactly one segment
   out of 110 (`train_seg_005`, sitting at 484.5 mA — precisely the closest
   point to the Close operation's class boundary, which is the tightest of
   the two gaps at only 28.3 mA wide).

That one miss, and the separately-flagged risk that `Test.csv`'s *unlabeled*
Close-current distribution runs a little higher than training's Normal
range, are both left visible rather than patched over — patching them would
require looking at the actual evaluation file's data shape to retune the
threshold, which is exactly the kind of test-set leakage the grading
criteria explicitly penalize.
