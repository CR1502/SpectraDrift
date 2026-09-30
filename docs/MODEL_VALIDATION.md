# Real-data detector validation

The following results come from code run on the public Harvard TEP release on
2026-09-30 during `feat/two-signal-detector`. This checkpoint validates the model
and a small held-out normal subset. Fault latency has not been evaluated yet.
The defaults were fixed before inspecting held-out test flags; no thresholds or
hyperparameters were changed in response to this check.

## Fitting and calibration

Normal training runs 1–10 were selected from the publisher-checksummed file. The
seed-42, 20% calibration split is unchanged from the preprocessing checkpoint:

- Fitting runs: 1, 2, 3, 4, 5, 8, 9, 10; 3,848 windows.
- Calibration runs: 6, 7; 962 windows.
- Window size 20, stride 1, and 104 mean/std features.

PCA retained 55 components and an observed variance fraction of
`0.9530099772357787`, satisfying the requested 0.95 fraction. KS used 512
reference readings sampled from fitting data only, with reference seed 42.
Target calibration flag allowance was 0.01, split into 0.005 per component.
Both thresholds used quantile 0.995, the `higher` method, and strict `>` comparisons.

| Signal | Threshold | Calibration flagged windows | Eligible calibration windows |
| --- | --- | --- | --- |
| PCA reconstruction MSE | `0.18376411807555154` | 4 | 962 |
| Maximum per-channel KS distance | `0.970703125` | 3 | 962 |
| Combined OR | Either component threshold exceeded | 7 | 962 |

Combined empirical calibration flag rate: `7 / 962 = 0.007276507276507277`
(approximately 0.73%). Calibration is not held-out false-positive evaluation.

## Held-out normal check

The saved detector was loaded and applied to normal test runs 1–3 from the
separately checksummed normal testing file. Its normalization, feature scaling,
PCA, KS reference, and thresholds remained unchanged. Each run has 960 samples,
with endpoints 20–960: 941 eligible windows and 19 warm-up samples.

| Normal test run | Eligible windows | PCA flags | KS flags | OR flags |
| --- | --- | --- | --- | --- |
| 1 | 941 | 1 | 0 | 1 |
| 2 | 941 | 3 | 0 | 3 |
| 3 | 941 | 1 | 10 | 11 |
| Total | 2,823 | 5 | 10 | 15 |

Window-level false-positive rate: `15 / 2823 = 0.005313496280552604`
(approximately **0.53%**). All selected observations are fault-free.
This is a three-run subset check, not evaluation of all 500 normal test runs.
Each selected simulation had at least one flag, so the window-level rate must
not be interpreted as the probability of a false alarm per simulation.
Overlapping windows are dependent; no independent-window confidence interval
is asserted.

Local ignored artifacts:

- `artifacts/detector/model.npz` and `manifest.json`
- `artifacts/normal-check/scores.npz` and `summary.json`

The recorded model NPZ SHA-256 is
`b3a48ae21469838273896e8248879104b40f2ae4671378b82f0a67459e17158d`.
The manifests include the original data checksums and actual software versions.
The scoring artifact preserves all per-window scores, channel statistics, flags,
and timestamps for audit and later plotting.

## Tests

All 54 offline tests passed. New coverage includes reconstruction of a known
rank-one fixture, scoring deviations outside its PCA subspace, frozen feature
scaling, constant features, exact KS agreement with SciPy on continuous and
tied values, single-channel shift detection, frozen/reproducible references,
normal-only and disjoint calibration guards, strict threshold ties, the OR rule,
causal scores, eligible-window denominators, safe model serialization, and
rejection of corrupted models or reordered features.

Synthetic fixtures test software behavior only. The numerical thresholds and
held-out false-positive counts above come from real downloaded TEP observations.

## Reproduce

```bash
python -m spectradrift download-data
python -m spectradrift train-detector --output artifacts/detector-reproduction
python -m spectradrift score-data --model artifacts/detector-reproduction \
  --file normal-testing --runs 1 2 3 --output artifacts/normal-reproduction
python -m unittest discover -s tests -v
```

Choose unused output directories. The next benchmark branch will obtain the
faulty test source and measure onset-aware latency and missed-fault coverage,
alongside an explicitly selected normal false-positive evaluation set.
