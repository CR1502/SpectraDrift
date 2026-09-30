# SpectraDrift

A Python project for self-supervised drift detection in multivariate industrial
sensor time series, benchmarked on the public Tennessee Eastman Process (TEP).

**Current status: the two-signal detector works.** Verified Harvard TEP data
loading, normal-only normalization, causal windows, PCA reconstruction scoring,
and independent KS drift scoring are implemented. A preliminary held-out normal
check is recorded below. Fault-onset latency and the full benchmark are next.

## Local setup

Use Python 3.11–3.13 (Python 3.12 is the development environment).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m spectradrift
```

The last command (or `python -m spectradrift doctor`) imports each runtime
dependency and prints installed versions, any import errors, and whether the
environment is ready. It exits with a failure code when dependencies cannot be
imported. `spectradrift` is the equivalent installed command.

## Dataset

Use the [Harvard Dataverse TEP release](https://doi.org/10.7910/DVN/6C3JR1),
Rieth et al. (2017), version 1.0. It contains normal operation and 20 fault
conditions across 52 process channels. Acquisition instructions and verified
publisher checksums are in [data/README.md](data/README.md) and
[data/source.json](data/source.json).

Raw data stays out of Git. The dataset is a process simulation benchmark;
detecting its faults does not by itself demonstrate a measured lead time before
output-quality deterioration in a real factory.

## Download and preprocess

Run commands from the repository root:

```bash
# Fetch both normal sources, or reuse existing checksum-verified files.
python -m spectradrift download-data

# Inspect selected complete training simulations.
python -m spectradrift inspect-data --file normal-training --runs 1 2 3

# Prepare runs 1-10, split by simulation with seed 42.
python -m spectradrift prepare-data --output artifacts/preprocessing

# Run the offline data, leakage, and window-boundary tests.
python -m unittest discover -s tests -v
```

The preparation command saves fitting/calibration NPZs plus a JSON manifest
containing frozen normalization statistics, selected run IDs, parameters, and
software versions. Existing artifacts are preserved; choose another `--output`
directory for a new run. Raw data and generated artifacts are ignored by Git.

The actual first preparation run used normal training simulations 1–10, with
20-sample windows and stride 1. It produced **3,848 fitting windows** from eight
simulations and **962 calibration windows** from two simulations. Each feature
vector has 104 entries: 52 channel means followed by 52 population standard
deviations. These are preprocessing counts, not detection metrics.

See [docs/PREPROCESSING.md](docs/PREPROCESSING.md) for the artifact format and
[docs/DATA_VALIDATION.md](docs/DATA_VALIDATION.md) for the actual validation run.

## Two-signal baseline

1. Split normal training simulations into fitting and calibration runs. Fit
   channel standardization on fitting data only. Create causal sliding-window
   features within each run, using channel means and standard deviations.
2. Fit a lightweight PCA reconstruction model on fitting windows only. At
   inference, feature reconstruction error supplies the anomaly score.
3. Independently compare each current sensor window with a fixed normal
   reference using the two-sample Kolmogorov–Smirnov statistic. Aggregate sensor
   statistics into a distribution-shift score.
4. Set each score threshold from separate normal calibration runs. Flag when
   either score exceeds its threshold. Calibrate and report the combined flag's
   behavior; the OR rule can raise more false positives than either component.

PCA standardizes the 104 window features using fitting data only and retains at
least 95% of their variance. Its anomaly score is reconstruction mean squared
error in that feature space. KS uses up to 512 reference readings sampled from
normal fitting runs; its drift score is the maximum empirical KS distance across
the 52 channels. Both components and normalization stay fixed at inference.

The default target calibration flag rate is 1%. Each component threshold is the
99.5th percentile of its normal calibration scores, using NumPy's `higher`
quantile method. Comparisons are strict: a score equal to its threshold does not
flag. The target controls the observed calibration allowance; it is not a
guarantee for unseen runs.

```bash
# Fit on normal training data and calibrate on disjoint normal training runs.
python -m spectradrift train-detector --output artifacts/detector

# Apply the saved model to three independent normal test simulations.
python -m spectradrift score-data --model artifacts/detector \
  --file normal-testing --runs 1 2 3 --output artifacts/normal-check
```

Both commands preserve existing artifacts; use new output directories for
repeated runs. Training loads the verified normal training source directly and
recreates the documented split, so a prior `prepare-data` run is optional.

The actual first model used **55 PCA components**. On normal test simulations
1–3, it flagged **15 of 2,823 eligible windows**, giving a preliminary
window-level false-positive rate of **0.53%**. This checks three complete normal
simulations, not all normal test runs. Each of the three simulations had at least
one flag. Overlapping windows mean this percentage is not a per-simulation false
alarm probability. No faulty data was used to choose or adjust thresholds.

See [docs/DETECTOR.md](docs/DETECTOR.md) for the scoring rule and saved model
format, and [docs/MODEL_VALIDATION.md](docs/MODEL_VALIDATION.md) for measured
thresholds, component counts, and per-run results. Detection latency has not yet
been measured.

The KS statistic is an empirical shift score, with thresholds calibrated
on normal time series. Nominal KS p-values assume independent observations and
are not presented as valid significance levels for autocorrelated sensors.
Fault labels are reserved for evaluating results, never fitting normalization,
PCA, reference distributions, thresholds, or hyperparameters.

## Planned evaluation

For each complete faulty test simulation, latency will be the elapsed sample
count from onset to the first post-onset flag, timestamped at the end of its
causal window. An onset-time flag has latency zero. Missed faults will remain
explicitly undetected; they will not silently become zero-latency detections.
Mean/median latency among detected runs and detection coverage will be reported
per fault type and overall, along with an equal-fault-weighted summary.

False-positive rate will be flagged eligible windows divided by all eligible
windows on held-out normal test runs. Warm-up windows will be excluded explicitly.
Normal prefixes of faulty runs will be reported separately. Windows overlap, so
window-level false-positive rate is not a probability of at least one false alarm
per simulation.

The benchmark branch will save `results.json`, `results.md`, configuration,
dataset provenance, run selections, denominators, and software versions. The
final README metrics will be populated from those actual run artifacts.

## Development and GitHub workflow

See [docs/WORKFLOW.md](docs/WORKFLOW.md). Start with a foundation commit on
`main`, then develop data, models, evaluation, and final documentation on
separate branches. GitHub repository creation, staging, committing, pushing,
and PR commands are supplied for the owner to execute.
