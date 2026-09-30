# Run-preserving preprocessing

The package now implements dataset acquisition and preprocessing. The detector
and latency/FPR benchmark are separate subsequent parts.

## Loading and validation

`spectradrift.data.load_runs` first compares file byte size and MD5 with
`data/source.json`, then loads the expected R dataframe through `pyreadr`.
It requires exactly the three metadata columns and 52 sensor columns. Sensor
order is canonicalized to `xmeas_1`–`xmeas_41`, then `xmv_1`–`xmv_11`.

Metadata must be finite, integral, and in range. Each selected simulation must
have every expected sample exactly once: 1–500 for training or 1–960 for testing.
Missing runs, duplicate/gapped samples, nonnumeric sensors, and nonfinite sensor
values are errors. Rows may arrive out of order and are sorted by sample.
Metadata checks cover the loaded table; sensor and continuity checks cover the
requested complete runs. Checksummed files are the exact published release.

A run's identity is `(source, faultNumber, simulationRun)`. The same numeric
run ID in training and testing represents a different simulation. A fault ID
identifies a complete test condition, including its normal prefix; the loader
does not treat all rows in a faulty source as post-onset observations.

The CLI selects runs 1–10 by default. A Python API call without an explicit run
selection expects all 500 release runs; missing selections are never silently
dropped. For faulty testing, the default fault selection includes all 20 types.
For example, after obtaining that file:

```bash
python -m spectradrift inspect-data --file faulty-testing --runs 1 --faults 1 2
```

`pyreadr` parses an entire RData object before row selection. Selecting fewer runs
reduces retained arrays, not initial parsing memory. The large faulty test table
contains 9.6 million rows, so allow several GB of available memory for that stage.

## Splitting and normalization

`split_normal_runs` sorts simulation identities, uses a seeded NumPy permutation,
and reserves `ceil(number_of_runs * calibration_fraction)` complete simulations
for calibration. Both partitions must be nonempty. Input order does not change
the split. Calibration simulations share no samples or overlapping windows with
fitting simulations.

`ChannelNormalizer.fit` accepts only normal training simulations. It incrementally
fits a `StandardScaler` on raw fitting samples without concatenating all runs.
Population mean and variance are estimated per channel. Transforming any other
run reuses these frozen statistics; neither calibration nor test observations
change them. Constant channels use scale 1 and remain finite. Saved statistics
include the sensor order and fitting run identities; `from_dict` validates them
when reloading.

Fault labels are metadata used to select the normal source and eventually
evaluate test conditions. They are not inputs to the learned feature representation
or objectives. No faulty training source is consumed.

## Causal windows

`window_features` operates on one `TEPRun` at a time. With window size 20 and
stride 1, the first window uses samples 1–20 and is timestamped at sample 20.
The next uses samples 2–21 and is timestamped at sample 21. Samples 1–19 are
warm-up; there are 481 eligible windows in a 500-sample training run.

At each endpoint, the 104-dimensional feature vector contains all 52 normalized
channel means followed by all 52 normalized channel population standard
deviations. Feature names are recorded explicitly. The batch also exposes its
sensor-window tensor so a statistical drift test can operate independently of
PCA reconstruction features. No window crosses a simulation or source boundary.

Changing future sensor observations cannot change features timestamped before
that change. Stride affects eligibility, while timestamps stay in the original
sample coordinate system for future latency calculations.

## Saved preparation artifacts

`prepare-data` consumes selected normal training simulations only and writes:

| File | Contents |
| --- | --- |
| `fitting.npz` | Fitting run IDs, normalized raw values, window features, endpoints, source |
| `calibration.npz` | Same arrays for disjoint calibration runs using the fitting normalizer |
| `manifest.json` | Source checksum, split/configuration, software versions, feature names, normalization statistics and counts |

For each NPZ, arrays preserve a leading simulation axis:

- `run_ids`: `(number_of_runs,)`
- `normalized_values`: `(number_of_runs, 500, 52)`
- `features`: `(number_of_runs, number_of_eligible_windows, 104)`
- `end_samples`: `(number_of_runs, number_of_eligible_windows)`
- `source`: scalar Unicode string, `normal-training`

Read these arrays with `np.load(path, allow_pickle=False)`. There are no Python
object arrays or executable pickles. Preparation refuses to overwrite any existing
artifact filenames in its output directory. Choose a new output directory for a
new configuration.

```bash
python -m spectradrift prepare-data \
  --runs 1 2 3 4 5 6 7 8 9 10 \
  --window-size 20 --stride 1 --calibration-fraction 0.2 --seed 42 \
  --output artifacts/my-preprocessing
```
