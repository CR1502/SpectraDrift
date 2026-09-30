# SpectraDrift

A Python project for self-supervised drift detection in multivariate industrial
sensor time series, benchmarked on the public Tennessee Eastman Process (TEP).

The working pipeline combines normal-only PCA reconstruction scoring with an
independent rolling Kolmogorov–Smirnov (KS) distribution-shift statistic. An
actual benchmark covers all 20 fault types (10 simulations per type) and all
500 held-out normal simulations. Measured results and limitations are below.

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
# Fetch the three required sources, or reuse checksum-verified files.
python -m spectradrift download-data --files normal-training normal-testing faulty-testing

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

The frozen model uses 55 PCA components. No faulty data was used to choose or
adjust thresholds. The preliminary three-run check is retained as historical
validation; the wider benchmark below is the reported evaluation.

See [docs/DETECTOR.md](docs/DETECTOR.md) for the scoring rule and saved model
format, and [docs/MODEL_VALIDATION.md](docs/MODEL_VALIDATION.md) for training
thresholds and the historical model validation.

The KS statistic is an empirical shift score, with thresholds calibrated
on normal time series. Nominal KS p-values assume independent observations and
are not presented as valid significance levels for autocorrelated sensors.
Fault labels are reserved for evaluating results, never fitting normalization,
PCA, reference distributions, thresholds, or hyperparameters.

## Measured benchmark results

These numbers come from the actual code run saved in [results.json](results.json)
and [results.md](results.md), using the frozen model above:

| Metric | Measured value |
| --- | --- |
| Mean first-flag latency | 55.255 steps (165.765 minutes) |
| Median first-flag latency | 17 steps (51 minutes) |
| Held-out normal window false-positive rate | 1.54% (7,236 / 470,500 eligible windows) |
| Fault runs with a post-onset flag | 200 / 200; zero misses under the first-flag definition |
| Normal runs with at least one flag | 437 / 500 (87.4%) |

Evaluation uses faulty test simulations 1–10 for each type 1–20, and normal test
simulations 1–500. This covers every fault type, but not the full faulty release.
All selected runs are complete. Labels are used only for evaluation.

Latency is the first flagged causal window endpoint at or after onset minus
sample 161 (one-based). Each step represents three minutes. Means and medians
are pooled across detected runs; misses have null latency and explicit counts.
The equal-fault-weighted mean is also 55.255 steps; the median of per-fault
medians is 19.25 steps. See the generated breakdown for every fault type and
the individual PCA/KS results.

FPR counts flagged eligible normal windows, excluding the incomplete-window
warm-up. Overlapping windows are dependent: 1.54% is **not** the probability of
any false alarm during a simulation, as the 87.4% run-level result demonstrates.
Normal prefixes of faulty runs are reported separately.

An eventual first flag does not prove fault attribution. Fault types 3, 9, and
15 have long delays and sparse post-onset flags (1.88–2.49%), close to the normal
window FPR. Consequently, 200/200 first flags should not be interpreted as
reliable diagnosis of all faults. This is an honest baseline, not a validated
industrial early-warning system; quality-loss lead time is not measured.

## Reproduce the benchmark

After setup and the three-file download above:

```bash
# Optional if the same frozen model already exists locally.
python -m spectradrift train-detector --output artifacts/detector

# Write a new report without replacing the checked-in measured results.
python -m spectradrift benchmark --model artifacts/detector \
  --output artifacts/reproduction/results \
  --scores-output artifacts/reproduction/scores

python -m unittest discover -s tests -v
```

Existing model/result/score files are never overwritten; choose fresh output
directories for repeats. The benchmark rechecks dataset checksums, saves both
score streams and flags, and records the exact selection, denominators, model
hash, source-code hashes, and software versions. Training is deterministic for
the documented configuration; small numerical differences across dependency
versions or platforms are possible.

The faulty RData download is approximately 837 MB. Parsing its entire table
requires several GB of available memory even when selecting a run subset.
Manual download instructions are in [data/README.md](data/README.md); detailed
metric definitions, missed-fault handling, and larger-run commands are in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Development and GitHub workflow

See [docs/WORKFLOW.md](docs/WORKFLOW.md). Start with a foundation commit on
`main`, then develop data, models, evaluation, and final documentation on
separate branches. GitHub repository creation, staging, committing, pushing,
and PR commands are supplied for the owner to execute.
