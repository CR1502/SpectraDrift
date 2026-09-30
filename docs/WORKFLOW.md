# Reviewable development parts

Git has been initialized with `main` as its initial branch. An initial commit
must exist before Git can create other branches pointing to it. The project
owner executes staging, commit, GitHub creation, push, and PR commands.

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

No feature branch has been created yet because no initial commit exists.

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

For example, once the data/preprocessing part is implemented and verified:

```bash
git add <explicit-files-listed-at-that-checkpoint>
git commit -m "feat: load TEP data and build causal window features"
git push -u origin feat/data-preprocessing
gh pr create --base main --head feat/data-preprocessing \
  --title "Load TEP data and build causal window features" \
  --body "Add verified TEP ingestion, normal-only normalization, and windows that preserve simulation boundaries. Validation details accompany the implementation."
```

The placeholder in `git add` will be replaced with exact paths at that checkpoint;
it is not a command to run now. Do not stage raw data, `.venv`, or local artifacts.
Later checkpoints will provide exact commands and actual validation details.
