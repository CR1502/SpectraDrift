# Data/preprocessing validation

This records actual runs during development of `feat/data-preprocessing` on
2026-09-30. These are data integrity and preprocessing checks, not an anomaly
detection benchmark. No latency, detection coverage, or false-positive rate has
been measured in this branch.

## Public source and acquisition

Both normal-operation RData files were downloaded directly from Harvard Dataverse
and matched the publisher byte sizes and MD5 checksums in `data/source.json`:

| File | Bytes | Publisher MD5 |
| --- | --- | --- |
| `TEP_FaultFree_Training.RData` | 24,678,017 | `ec126484534331f85001d8c4ebce6d17` |
| `TEP_FaultFree_Testing.RData` | 47,327,663 | `38ad9810fc871026157086ae2c2f0ee9` |

The downloader also successfully reused both verified files on a subsequent run.
The faulty test file was not downloaded for this checkpoint.

## Real TEP parsing and preparation

The loader checked normal training simulations 1–10: 5,000 samples, with all 52
channels and complete 500-sample runs. It separately checked normal test
simulations 1–3: 2,880 samples in complete 960-sample runs. Both source checksums
were verified before parsing.

Preparation used seed 42, calibration fraction 0.2, window size 20, and stride 1:

| Partition | Simulation IDs | Raw samples | Eligible windows | Feature array shape |
| --- | --- | --- | --- | --- |
| Fitting | 1, 2, 3, 4, 5, 8, 9, 10 | 4,000 | 3,848 | `(8, 481, 104)` |
| Calibration | 6, 7 | 1,000 | 962 | `(2, 481, 104)` |

The saved arrays were reloaded with `allow_pickle=False` and checked for disjoint
run IDs, finite features, expected shapes, and exact endpoint samples 20–500.
Across all fitting samples, normalized channel means matched zero and population
standard deviations matched one within absolute tolerance `1e-9`. No channel
had exactly zero fitting variance in this selected real-data run.

The current-format preparation artifacts are local and ignored by Git:
`artifacts/preprocessing-verified/{fitting.npz,calibration.npz,manifest.json}`.
The manifest records actual normalization statistics, publisher checksum,
configuration, partitions, and software versions.

## Offline tests and environment

All 33 offline tests passed, including an RData parser/writer roundtrip and
complete preparation-stage fixture. They check checksum corruption, transfer
cleanup/retries, schema errors, missing/duplicate samples, source-aware run
identities, normal-only fitting, split disjointness, constant channels, saved
normalizer sensor order, causal windows, stride timestamps, and artifact preservation.
Synthetic fixtures are used solely for these software tests.

Validation environment: Python 3.12.14, NumPy 2.2.6, pandas 2.2.3,
scikit-learn 1.6.1, and pyreadr 0.5.3. Runtime dependencies are pinned in
`requirements.txt`; the environment check and dependency compatibility check pass.

## Reproduce this checkpoint

```bash
python -m spectradrift download-data
python -m spectradrift inspect-data --file normal-training --runs 1 2 3 4 5 6 7 8 9 10
python -m spectradrift inspect-data --file normal-testing --runs 1 2 3
python -m spectradrift prepare-data --output artifacts/preprocessing-reproduction
python -m unittest discover -s tests -v
```

Use an unused output directory. CI runs the offline tests and environment check;
it does not download the benchmark dataset.
