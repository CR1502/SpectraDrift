# Measured TEP monitoring figure

[tep-monitoring.png](tep-monitoring.png) is rendered by
[scripts/plot_results.py](../../scripts/plot_results.py) using the actual saved
score archives referenced in [results.json](../../results.json). No synthetic
signals, fitting, threshold adjustment, or score smoothing are involved.

The fixed selections are simulation 1 of normal testing, fault 1, and fault 9.
Normal operation illustrates false alarms; fault 1 shows a strong sustained
response; difficult fault 9 shows sparse flags despite a long injected-fault
interval. These illustrations are not representative estimates of aggregate
latency or false-positive rate; the report evaluates the entire documented
selection, including every fault type.

| Example | Measurement from the saved run |
| --- | --- |
| Normal test run 1 | 1 flagged window out of 941 eligible windows |
| Fault 1, run 1 | First post-onset OR flag at sample 166; latency 5 steps |
| Fault 9, run 1 | First post-onset OR flag at sample 778; latency 617 steps; 2 post-onset windows flagged |

Each score is divided by its own frozen threshold. The dashed line at 1 shows
the strict decision boundary: equality does not flag. Pink bottom ticks mark
the combined OR flag. Orange shading begins at one-based sample 161 and ends
with observation at sample 960. The x coordinate is the causal window endpoint.

The y scale is logarithmic with independently scaled panels. Scores are clipped
at a ratio of 0.000001 for log display only; that does not change flags or
measurements. Cross-panel heights should not be compared without reading the
axis ticks. No warm-up scores are fabricated before the first eligible endpoint.

## Recreate

With the original local archives still present, from the repository root:

```bash
python scripts/plot_results.py --results results.json \
  --output artifacts/tep-monitoring-recreated.png
```

For a fresh checkout, first follow the README's download, training, and benchmark
commands. Those save a new report and matching scores outside the tracked files:

```bash
python scripts/plot_results.py --results artifacts/reproduction/results/results.json \
  --output artifacts/reproduction/tep-monitoring.png
```

The script verifies archive SHA-256 hashes against the supplied report, requires
the three exact identities, and checks that saved flags follow the frozen
thresholds and OR rule. If archives were moved, `--scores-dir PATH` looks for the
recorded filenames there, still requiring their original hashes. Regenerated
scores should use their own new report, particularly across platforms with small
floating-point differences.

Existing image files are not overwritten. The renderer uses Matplotlib's Agg
backend and does not require a desktop display, raw RData, or a fitted model
once the report and score archives exist. Missing local archives produce a
regeneration hint, not a placeholder figure.
