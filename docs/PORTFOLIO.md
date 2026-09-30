# Evidence-backed portfolio description

SpectraDrift is a self-supervised drift-detection baseline for multivariate
industrial sensor time series. It combines a normal-only PCA reconstruction
anomaly scorer with an independent rolling per-sensor Kolmogorov–Smirnov
distribution-shift statistic and a calibrated OR decision rule.

Suggested resume wording:

> Built a self-supervised industrial sensor drift-detection pipeline combining
> PCA reconstruction errors and rolling distribution-shift statistics.
> Benchmarked on Tennessee Eastman Process simulations across all 20 fault types:
> mean/median first-flag latency of 55.255/17 steps and a 1.54% held-out normal
> window false-positive rate.

The [actual report](../results.md) and [machine-readable results](../results.json)
support these numbers. Evaluation selects 10 complete faulty simulations per
type and all 500 normal test simulations, not all published faulty runs. The
model is trained and calibrated on disjoint normal training runs only; no test
labels tune its settings. Each TEP sample represents three minutes.

For interviews, make the limits explicit: latency is conditional on a first
post-onset flag, and a late background false alarm can satisfy that definition.
Faults 3, 9, and 15 have sparse responses. The window FPR is not a run-level
false-alarm probability: 437 of 500 normal runs had at least one flag. The
separate component results, explicit misses, and per-fault breakdown are saved
so those limitations are inspectable rather than hidden by an aggregate.

The original project objective is early warning of process deviations before
output quality deteriorates. This simulation benchmark measures fault-onset
latency and normal false alarms, not lead time before quality loss. Present
early warning as an objective, not a demonstrated deployment outcome.
