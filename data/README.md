# Obtaining the TEP data

Use Rieth et al. (2017), *Additional Tennessee Eastman Process Simulation Data for
Anomaly Detection Evaluation*, Harvard Dataverse, version 1.0:
[dataset and terms](https://doi.org/10.7910/DVN/6C3JR1).

The publisher's API metadata was checked when creating [source.json](source.json).
It records download URLs, exact byte sizes, and publisher MD5 checksums. MD5 here
checks transfer integrity against the published files; it is not a security guarantee.

Download these files into this directory, keeping their original filenames:

| File | Use |
| --- | --- |
| `TEP_FaultFree_Training.RData` | Train and calibrate on separate normal runs |
| `TEP_FaultFree_Testing.RData` | Held-out normal false-positive evaluation |
| `TEP_Faulty_Testing.RData` | Evaluate faults only |

The three files occupy 908,887,717 bytes in total (about 909 MB). The faulty test
file expands substantially in memory; it contains 9.6 million rows. The measured
benchmark uses a documented subset of complete faulty simulations, not the full
faulty test release. Subsetting after loading
the RData file does not reduce the initial parsing memory requirement.

The preferred command verifies both byte size and publisher MD5 before loading:

```bash
# All sources needed for training, calibration, and evaluation.
python -m spectradrift download-data --files normal-training normal-testing faulty-testing
```

The downloader identifies itself as SpectraDrift, requests the original RData
file, retries transient network failures, and verifies a temporary download before
publishing the final filename. It reuses verified existing files and refuses to
overwrite existing files that fail validation. No Kaggle login or R installation
is required. [Dataverse's Data Access API](https://guides.dataverse.org/en/latest/api/dataaccess.html)
documents the original-file parameter used here.

If an automated download is unavailable, use the dataset page, or these commands
from the repository root. The loader still checks the published checksums:

```bash
curl --fail --location --retry 3 --user-agent 'SpectraDrift/0.1' \
  'https://dataverse.harvard.edu/api/access/datafile/3031241?format=original' \
  --output data/TEP_FaultFree_Training.RData
curl --fail --location --retry 3 --user-agent 'SpectraDrift/0.1' \
  'https://dataverse.harvard.edu/api/access/datafile/3031240?format=original' \
  --output data/TEP_FaultFree_Testing.RData
curl --fail --location --retry 3 --user-agent 'SpectraDrift/0.1' \
  'https://dataverse.harvard.edu/api/access/datafile/3031243?format=original' \
  --output data/TEP_Faulty_Testing.RData
```

All three required files have been downloaded and checksum-verified, including
the faulty testing source for the benchmark. The evaluation selects complete
simulations as documented in [the benchmark protocol](../docs/EVALUATION.md).
Synthetic test fixtures verify software behavior but are never described as
TEP benchmark results.

Each table has three metadata columns (`faultNumber`, `simulationRun`, `sample`)
and 52 process channels. Samples arrive every three minutes. Each test run has
960 samples. The publisher specifies fault injection eight hours into faulty
test runs. Evaluation uses samples 1–160 as the normal prefix and sample
161 (zero-based index 160) as the first fault sample. `faultNumber` identifies
the run's condition, including its normal prefix; it is not an instantaneous
fault-presence label.

The publisher documents distinct random seeds for training and testing, even
when `simulationRun` has the same numeric value. Run identity therefore includes
the source file. Windows must never cross run or source-file boundaries.

Do not download `TEP_Faulty_Training.RData` for this baseline: fitting and
threshold calibration use normal operation only. Keep all 20 fault types
in evaluation, including difficult-to-detect faults.

Raw RData, converted tables, and caches are ignored by Git. Cite the dataset and
review its published public-domain dedication and terms when using it.
