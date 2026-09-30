# TEP benchmark protocol

`python -m spectradrift benchmark` applies an already trained, frozen detector.
It never fits a model, normalizer, reference, feature scaler, or threshold. The
default evaluation selection was fixed before inspecting fault results:

- Normal testing: all simulation IDs 1–500 from the published release.
- Faulty testing: simulation IDs 1–10 for each of fault types 1–20.
- Fault onset: sample 161, using samples 1–160 as the normal prefix.
- Sampling period: three minutes per step.

This covers all 20 fault types and the full normal test release, while selecting
200 of the 10,000 faulty simulations. It does not claim to evaluate every faulty
run. No difficult fault types are excluded.

The separate normal training release supplies the existing model: fitting runs
1, 2, 3, 4, 5, 8, 9, 10 and calibration runs 6, 7. The benchmark rechecks the
normal training file against its publisher checksum and validates the model's
recorded training provenance. Both testing files are independently checksummed
before parsing selected complete simulations.

## Detection latency and misses

Each eligible window is timestamped at its last sample. For a faulty run, the
first flag at an endpoint greater than or equal to 161 supplies the latency:

```text
latency_steps = first_post_onset_flag_sample - 161
latency_minutes = latency_steps * 3
```

At-onset flags have latency zero. Pre-onset flags do not count as detections.
The calculation uses actual sample coordinates, including when stride skips
samples; it never uses a window's array index as the elapsed time.

If no eligible post-onset window is flagged, the result is `detected=false`,
with null first-flag sample and null latency. The last observed eligible sample
and observation horizon are retained. Misses are neither assigned zero nor
silently removed from coverage calculations.

Means and medians are conditional on detection and are paired with detected and
undetected counts. Results include pooled summaries and a breakdown for every
selected fault type. The macro mean averages each detected fault type's mean
with equal weight; the macro median is the median of per-fault medians. Completely
undetected fault types remain in the breakdown with null latency statistics,
and the number of fault types with any detection is explicit.

The same latency and coverage calculations are performed separately for PCA,
KS, and the combined OR flag. Post-onset flagged-window fractions are also saved.

This is a first-flag definition, not an alarm-event transition definition. A
flag already present before onset can continue into the first post-onset window;
its continuation is recorded explicitly. A late first flag can also be a
background false alarm. These metrics do not establish that every detection is
attributable to the injected fault. Faults with sparse post-onset flags should
be interpreted alongside normal false alarms and the per-component results.

## False-positive rate

Every normal test observation has no true fault. Window-level false-positive
rate is:

```text
FPR = flagged eligible normal windows / all eligible normal windows
```

For the frozen 20-sample, stride-1 model, a complete normal test simulation has
941 eligible windows, ending at samples 20–960. Samples 1–19 are warm-up and
are excluded from both numerator and denominator. The complete normal test
release therefore contributes 470,500 eligible windows.

The report saves counts/rates for PCA, KS, and OR, with per-simulation counts.
It also records how many normal simulations had any flag and the number of
episodes of consecutive flagged eligible windows. These prevent a small
window-level rate from being mistaken for a low probability of any alarm during
an entire simulation. Episodes do not cross simulation boundaries.

The normal prefixes of faulty simulations are evaluated separately, using
only eligible endpoints strictly before 161. They are not pooled into the
held-out normal test FPR. If no prefix window is eligible under another window
configuration, its rate is null rather than a fabricated zero.

Overlapping windows and sensor observations are dependent. No nominal KS
p-values or independent-window confidence intervals are used. The onset is
derived from the dataset's documented injection timing, not from changing
`faultNumber` values within a run.

## Run and reproduce

From a clean checkout with dependencies installed:

```bash
python -m spectradrift download-data --files normal-training normal-testing faulty-testing
python -m spectradrift train-detector --output artifacts/detector
python -m spectradrift benchmark --model artifacts/detector \
  --output artifacts/reproduction/results \
  --scores-output artifacts/reproduction/scores
python -m unittest discover -s tests -v
```

The benchmark command prints progress while scoring. `pyreadr` must parse the
entire 9.6-million-row faulty RData table before selecting the 200 default runs.
A run subset reduces retained arrays but not that initial parsing requirement.
Allow several GB of available memory and use the original RData file; compressed
download size is not the in-memory size.

The root `results.json` and `results.md` already contain the checked-in actual
run. The reproduction command uses separate directories to preserve them.
Existing result and score archive filenames are preserved. For a repeat run:

```bash
python -m spectradrift benchmark --model artifacts/detector \
  --output artifacts/benchmark-repeat/results \
  --scores-output artifacts/benchmark-repeat/scores
```

To evaluate all faulty simulations instead, explicitly pass IDs 1–500 using a
shell expansion such as `--fault-runs {1..500}` in bash/zsh. The report records
the exact IDs and whether the full faulty release was evaluated. Other explicit
selections are supported through `--fault-runs`, `--normal-runs`, and `--faults`;
results must be described with their actual scope.

## Saved outputs

| File | Purpose |
| --- | --- |
| `results.json` | Exact metrics, per-fault/per-run latencies and misses, component rates, selected IDs, model manifest, source checksums, software versions, code provenance, score archive hashes |
| `results.md` | Generated readable summary and full selected-fault breakdown |
| `<scores-output>/normal_scores.npz` | Normal test identities, endpoint samples, both scores and flags, combined flags |
| `<scores-output>/fault_scores.npz` | The same arrays for selected faulty test simulations |

NPZs contain numeric/Unicode arrays and are read with `allow_pickle=False`.
Their leading axis is simulation, followed by eligible window. The JSON report
contains their SHA-256 hashes and the hash of the frozen model. Source code
file hashes are recorded because measurements may be made before the owner
commits the benchmark branch; the Git base commit and dirty status are also
recorded explicitly.

Only the compact JSON/Markdown results are intended for Git. Raw data, the
regenerable model, and score archives remain ignored. Dataset provenance and
run-selection scope accompany the results, rather than relying on an undocumented
local subset. This simulation benchmark does not measure output-quality loss
or industrial lead time before quality degradation.
