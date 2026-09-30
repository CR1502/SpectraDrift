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
file expands substantially in memory; it contains 9.6 million rows. We will use a
documented subset of complete simulation runs for the first benchmark rather
than describe a subset run as a full-dataset evaluation. Subsetting after loading
the RData file does not reduce the initial parsing memory requirement.

You can download through the dataset page, or from the repository root with:

```bash
curl --fail --location --retry 3 \
  https://dataverse.harvard.edu/api/access/datafile/3031241 \
  --output data/TEP_FaultFree_Training.RData
curl --fail --location --retry 3 \
  https://dataverse.harvard.edu/api/access/datafile/3031240 \
  --output data/TEP_FaultFree_Testing.RData
curl --fail --location --retry 3 \
  https://dataverse.harvard.edu/api/access/datafile/3031243 \
  --output data/TEP_Faulty_Testing.RData
```

These commands obtain data only. The loader and automated validation belong to
the next feature branch. No raw data has been downloaded or evaluated in the
foundation commit.

Each table has three metadata columns (`faultNumber`, `simulationRun`, `sample`)
and 52 process channels. Samples arrive every three minutes. Each test run has
960 samples. The publisher specifies fault injection eight hours into faulty
test runs. We will evaluate with samples 1–160 as the normal prefix and sample
161 (zero-based index 160) as the first fault sample. `faultNumber` identifies
the run's condition, including its normal prefix; it is not an instantaneous
fault-presence label.

The publisher documents distinct random seeds for training and testing, even
when `simulationRun` has the same numeric value. Run identity therefore includes
the source file. Windows must never cross run or source-file boundaries.

Do not download `TEP_Faulty_Training.RData` for this baseline: fitting and
threshold calibration will use normal operation only. Keep all 20 fault types
in evaluation, including difficult-to-detect faults.

Raw RData, converted tables, and caches are ignored by Git. Cite the dataset and
review its published public-domain dedication and terms when using it.
