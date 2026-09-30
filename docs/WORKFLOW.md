# Reviewable development parts

The initial foundation commit is on `main`, and the GitHub repository is
[CR1502/SpectraDrift](https://github.com/CR1502/SpectraDrift). The project owner
executes staging, commit, push, PR, and merge commands. The assistant creates
local feature branches and implements and verifies each part.

## First checkpoint: foundation on main

Run these commands from the repository root after reviewing the foundation:

```bash
git add .gitignore README.md requirements.txt pyproject.toml src/spectradrift data/README.md data/source.json docs/WORKFLOW.md .github/workflows/ci.yml
git commit -m "chore: initialize SpectraDrift project foundation"
gh repo create SpectraDrift --public --source=. --remote=origin
git push -u origin main
```

The GitHub command creates a public portfolio repository for the currently
authenticated GitHub account and adds the remote. It does not push until the
following explicit command. Choose `--private` instead if desired. If a repository
named SpectraDrift already exists, inspect it before using or replacing a remote.

The foundation checkpoint is complete. These setup commands are historical;
do not rerun repository creation for the existing repository.

## Subsequent parts

| Branch | Review scope |
| --- | --- |
| `feat/data-preprocessing` | Dataset acquisition/checksums, schema and run validation, normal-only scaling, causal sliding windows |
| `feat/two-signal-detector` | PCA reconstruction score, independent rolling KS statistic, normal calibration, documented OR rule |
| `feat/benchmark-evaluation` | Onset-aware latency, missed-fault coverage, held-out FPR, real benchmark and saved results |
| `docs/portfolio-results` | Measured README metrics, reproduction instructions, plot and limitations |

The assistant can create and switch local branches. The owner executes commits,
pushes, and PR creation. Complete each PR and merge it into `main` before starting
the next dependent branch. This keeps every PR based on reviewed code.

## Current checkpoint: data and preprocessing

The implementation is on `feat/data-preprocessing`. Both normal-operation files
have been downloaded and checksum-verified, the actual preparation run is
recorded in `docs/DATA_VALIDATION.md`, and 33 offline tests pass.

Review the diff, then execute:

```bash
git diff --stat
git add README.md data/README.md data/source.json docs/WORKFLOW.md docs/PREPROCESSING.md docs/DATA_VALIDATION.md src/spectradrift/__main__.py src/spectradrift/data.py src/spectradrift/preprocessing.py tests .github/workflows/ci.yml
git commit -m "feat: load TEP data and build causal window features"
git push -u origin feat/data-preprocessing
gh pr create --base main --head feat/data-preprocessing \
  --title "Load TEP data and build causal window features" \
  --body "Add checksum-verified Harvard TEP downloads, complete-run validation, normal-only normalization, and causal mean/std window features. Validate on real normal-operation TEP data: 3,848 fitting and 962 calibration windows from disjoint simulations. All 33 offline tests pass. Detection metrics will be measured in the benchmark PR."
```

After reviewing and merging this PR into `main`, tell the assistant to continue.
The next branch is `feat/two-signal-detector`. Do not stage raw data, `.venv`, or
local artifacts. Later checkpoints will provide exact commands and actual
validation details.
