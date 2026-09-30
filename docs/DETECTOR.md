# Two independent signals, one flag

`spectradrift.models` implements independent reconstruction and distribution
scorers. `spectradrift.detector` trains them using normal operation, calibrates
thresholds on other normal runs, applies the decision rule, and saves models and
test scores. Neither scorer takes fault labels as an optimization target.

## PCA reconstruction scorer

The input is the 104-dimensional mean/std window vector from preprocessing.
Another `StandardScaler`, fitted only on fitting windows, puts feature means and
feature standard deviations on comparable scales. PCA uses full SVD and selects
enough components to retain at least the requested fraction of fitting variance
(default 0.95). There is no supervised fault classifier or hyperparameter search
using test conditions. See the [scikit-learn PCA documentation](https://scikit-learn.org/1.6/modules/generated/sklearn.decomposition.PCA.html).

For a standardized vector `z`, PCA center `m`, and orthonormal component rows `C`:

```text
centered       = z - m
reconstruction = centered @ C.T @ C
anomaly_score  = mean((centered - reconstruction) ** 2)
```

All input scaling, centering, and components remain frozen. Constant feature
columns use scale 1, so they remain finite and a subsequent change can still
produce reconstruction error. Entirely constant fitting features are rejected
because there is no variation for PCA to learn.

## Independent rolling KS statistic

The reference candidates are normalized raw readings from fitting simulations
only. The scorer samples up to 512 candidate rows without replacement using the
configured seed (default 42), then sorts each channel's reference values. It
does not use calibration or test readings to adapt the reference.

At each window endpoint, for each channel `j`, it computes the exact empirical
two-sample Kolmogorov–Smirnov distance:

```text
D_j = sup_x |F_reference,j(x) - F_current_window,j(x)|
drift_score = max_j D_j
```

The pooled reference is a fixed normal empirical distribution, not a moving
reference that could absorb a fault. Current windows are the same causal trailing
sensor windows used by preprocessing. The maximum channel distance makes a
large shift in one sensor sufficient to raise the statistical signal.

The implementation evaluates empirical CDF differences at the sorted current
observations, just before and after jumps. It uses binary searches into the
frozen reference and handles ties. Numerical tests compare every channel with
[`scipy.stats.ks_2samp(...).statistic`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
on continuous and tied/discrete fixtures. Computing distances directly avoids
calculating a p-value at every sensor/window combination.

Process samples and overlapping windows are autocorrelated. The scorer does not
claim nominal KS significance or use p-values to set its threshold. The threshold
is estimated empirically for the already-aggregated maximum statistic, which
includes the effect of considering all channels.

## Normal calibration and decision rule

Fitting and calibration inputs must both be normal training runs with disjoint
simulation identities. Normalization, feature scaling, PCA, and KS references
use fitting runs only. Calibration affects only the two decision thresholds.

Given target calibration flag allowance `alpha` (default 0.01), each component
uses quantile `1 - alpha/2` (default 0.995), computed with `method="higher"`:

```text
anomaly_flag = anomaly_score > anomaly_threshold
drift_flag   = drift_score   > drift_threshold
flag         = anomaly_flag OR drift_flag
```

Strict comparisons handle repeated/tied calibration scores conservatively.
For these empirical calibration windows, allocating half the allowance to each
component limits the OR rate to their combined allowance. This is not an
out-of-sample guarantee; normal test windows are evaluated separately. Window
rates count all eligible windows, not independent samples or alarm events.

No smoothing, debounce, or updating reference is used in this baseline. Both
individual flags and the combined flag are retained so evaluation can explain
which component raised a deviation.

## CLI and artifacts

```bash
python -m spectradrift train-detector \
  --runs 1 2 3 4 5 6 7 8 9 10 \
  --window-size 20 --stride 1 --calibration-fraction 0.2 --seed 42 \
  --variance-retained 0.95 --reference-size 512 --target-flag-rate 0.01 \
  --output artifacts/my-detector

python -m spectradrift score-data --model artifacts/my-detector \
  --file normal-testing --runs 1 2 3 --output artifacts/my-normal-check
```

`train-detector` always reads the checksum-verified normal training source.
`score-data` accepts only testing sources. Once the faulty test file is obtained,
it can score selected faults with `--file faulty-testing --faults 1 2 --runs 1`;
this command saves scores, not onset-aware detection latency.

| Model file | Contents |
| --- | --- |
| `model.npz` | Feature scaling, PCA center/components/variance fractions, sorted KS reference |
| `manifest.json` | NPZ SHA-256, sensor/feature order, normalizer, window configuration, thresholds, software versions, fitting/calibration identities and counts, dataset checksum |

Loading checks the NPZ hash and canonical sensor/feature order, validates shapes
and parameters, and uses `allow_pickle=False`. The saved representation consists
of numerical arrays and JSON rather than executable estimator pickles.

Scoring writes `scores.npz` with source/fault/run identities, endpoint samples,
both scores, per-channel KS distances, both component flags, and combined flags.
`summary.json` records denominators, component/combined rates and counts, per-run
counts, source checksum, and model hash. Warm-up samples are excluded because
they have no complete window. Models and scoring outputs are ignored by Git;
the measured validation record is tracked. Both commands preserve existing
artifact filenames and require a new output directory for another run.
