"""Onset-aware TEP latency, explicit misses, and held-out false-positive rates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np

from spectradrift.data import load_manifest, load_runs, verify_file
from spectradrift.detector import ScoreBatch, TwoSignalDetector, _sha256, load_detector


SIGNALS = {"anomaly": "anomaly_flags", "drift": "drift_flags", "combined": "flags"}


def _validate_batches(batches: Sequence[ScoreBatch], source: str) -> None:
    if not batches or len({batch.run_key for batch in batches}) != len(batches):
        raise ValueError("Evaluation needs nonempty, unique simulation identities")
    for batch in batches:
        if batch.run_key[0] != source:
            raise ValueError(f"Expected held-out {source} simulations")
        fault = batch.run_key[1]
        if (source == "normal-testing" and fault != 0) or (source == "faulty-testing" and fault not in range(1, 21)):
            raise ValueError("Fault condition does not match the evaluation source")
        ends = np.asarray(batch.end_samples)
        if ends.ndim != 1 or not len(ends) or not np.isfinite(ends).all():
            raise ValueError("Evaluation endpoints must be a nonempty finite vector")
        if not np.equal(ends, np.floor(ends)).all() or not ((ends >= 1) & (ends <= 960)).all() or not (np.diff(ends) > 0).all():
            raise ValueError("Endpoints must be strictly increasing integer TEP sample numbers")
        for attribute in SIGNALS.values():
            flags = np.asarray(getattr(batch, attribute))
            if flags.shape != ends.shape or flags.dtype.kind != "b":
                raise ValueError("Boolean flags must align exactly with endpoint samples")
        if not np.array_equal(batch.flags, batch.anomaly_flags | batch.drift_flags):
            raise ValueError("Combined flags must follow the detector's OR rule")


def _nonfault_summary(batches: Sequence[ScoreBatch], *, before_sample: int | None = None) -> dict:
    per_run = []
    for batch in batches:
        eligible = np.ones(len(batch.end_samples), dtype=bool) if before_sample is None else batch.end_samples < before_sample
        record = {"key": list(batch.run_key), "eligible_windows": int(eligible.sum()), "signals": {}}
        for name, attribute in SIGNALS.items():
            flags = getattr(batch, attribute)[eligible]
            episodes = int(np.count_nonzero(flags & ~np.r_[False, flags[:-1]])) if len(flags) else 0
            record["signals"][name] = {"flagged_windows": int(flags.sum()), "flag_episodes": episodes}
        per_run.append(record)
    total = sum(record["eligible_windows"] for record in per_run)
    signals = {}
    for name in SIGNALS:
        flagged = sum(record["signals"][name]["flagged_windows"] for record in per_run)
        episodes = sum(record["signals"][name]["flag_episodes"] for record in per_run)
        runs_with_flags = sum(record["signals"][name]["flagged_windows"] > 0 for record in per_run)
        signals[name] = {
            "flagged_windows": flagged,
            "false_positive_rate": flagged / total if total else None,
            "flag_episodes": episodes,
            "runs_with_any_flag": runs_with_flags,
            "fraction_runs_with_any_flag": runs_with_flags / len(per_run),
        }
    return {"simulations": len(per_run), "eligible_windows": total, "signals": signals, "per_run": per_run}


def fault_run_record(batch: ScoreBatch, *, onset_sample: int = 161, sample_period_minutes: int = 3) -> dict:
    """First flagged eligible endpoint at/after onset; misses remain None.

    A pre-existing flag at onset counts as latency zero under this definition,
    but is also identified as continuing a pre-onset episode for interpretation.
    """
    _validate_batches([batch], "faulty-testing")
    if not isinstance(onset_sample, int) or isinstance(onset_sample, bool) or not 1 <= onset_sample <= 960:
        raise ValueError("onset_sample must be an integer in 1..960")
    if not isinstance(sample_period_minutes, int) or sample_period_minutes <= 0:
        raise ValueError("sample_period_minutes must be a positive integer")
    post = batch.end_samples >= onset_sample
    if not post.any():
        raise ValueError("No eligible observations at or after fault onset")
    first_post = int(np.flatnonzero(post)[0])
    record = {
        "key": list(batch.run_key), "onset_sample": onset_sample,
        "first_eligible_post_onset_sample": int(batch.end_samples[first_post]),
        "last_observed_sample": int(batch.end_samples[-1]),
        "observation_horizon_steps": int(batch.end_samples[-1]) - onset_sample,
        "eligible_post_onset_windows": int(post.sum()),
        "signals": {},
    }
    for name, attribute in SIGNALS.items():
        flags = getattr(batch, attribute)
        hits = np.flatnonzero(flags & post)
        first_index = int(hits[0]) if len(hits) else None
        first_sample = int(batch.end_samples[first_index]) if first_index is not None else None
        latency = first_sample - onset_sample if first_sample is not None else None
        continues = bool(first_index == first_post and first_post > 0 and flags[first_post - 1])
        record["signals"][name] = {
            "detected": first_sample is not None,
            "first_flag_sample": first_sample,
            "latency_steps": latency,
            "latency_minutes": latency * sample_period_minutes if latency is not None else None,
            "post_onset_flagged_windows": int(flags[post].sum()),
            "continues_pre_onset_flag_episode": continues,
        }
    return record


def _latency_summary(records: Sequence[dict], signal: str, sample_period_minutes: int) -> dict:
    latencies = [record["signals"][signal]["latency_steps"] for record in records
                 if record["signals"][signal]["detected"]]
    mean = float(np.mean(latencies)) if latencies else None
    median = float(np.median(latencies)) if latencies else None
    windows = sum(record["eligible_post_onset_windows"] for record in records)
    flagged = sum(record["signals"][signal]["post_onset_flagged_windows"] for record in records)
    return {
        "simulations": len(records), "detected_simulations": len(latencies),
        "undetected_simulations": len(records) - len(latencies),
        "detection_rate": len(latencies) / len(records),
        "mean_latency_steps_detected": mean, "median_latency_steps_detected": median,
        "mean_latency_minutes_detected": mean * sample_period_minutes if mean is not None else None,
        "median_latency_minutes_detected": median * sample_period_minutes if median is not None else None,
        "eligible_post_onset_windows": windows, "post_onset_flagged_windows": flagged,
        "post_onset_flag_rate": flagged / windows,
        "detections_continuing_pre_onset_episode": sum(
            record["signals"][signal]["continues_pre_onset_flag_episode"] for record in records
        ),
    }


def evaluate_batches(
    normal_batches: Sequence[ScoreBatch], faulty_batches: Sequence[ScoreBatch], *,
    onset_sample: int = 161, sample_period_minutes: int = 3,
    expected_faults: Sequence[int] = tuple(range(1, 21)),
) -> dict:
    """Evaluate all selected faults, including completely undetected fault types."""
    _validate_batches(normal_batches, "normal-testing")
    _validate_batches(faulty_batches, "faulty-testing")
    expected = list(expected_faults)
    if not expected or len(set(expected)) != len(expected) or any(fault not in range(1, 21) for fault in expected):
        raise ValueError("Expected fault IDs must be unique and in 1..20")
    if {batch.run_key[1] for batch in faulty_batches} != set(expected):
        raise ValueError("Scored fault types do not match the requested evaluation set")
    records = [fault_run_record(batch, onset_sample=onset_sample, sample_period_minutes=sample_period_minutes)
               for batch in sorted(faulty_batches, key=lambda batch: batch.run_key)]
    per_fault = {}
    for fault in sorted(expected):
        subset = [record for record in records if record["key"][1] == fault]
        per_fault[str(fault)] = {
            "fault_number": fault,
            "signals": {signal: _latency_summary(subset, signal, sample_period_minutes) for signal in SIGNALS},
            "runs": subset,
        }
    overall = {signal: _latency_summary(records, signal, sample_period_minutes) for signal in SIGNALS}
    macro = {}
    for signal in SIGNALS:
        summaries = [entry["signals"][signal] for entry in per_fault.values()]
        means = [entry["mean_latency_steps_detected"] for entry in summaries if entry["detected_simulations"]]
        medians = [entry["median_latency_steps_detected"] for entry in summaries if entry["detected_simulations"]]
        macro[signal] = {
            "fault_types": len(summaries), "fault_types_with_any_detection": len(means),
            "mean_of_fault_mean_latency_steps_detected": math.fsum(means) / len(means) if means else None,
            "median_of_fault_median_latency_steps_detected": float(np.median(medians)) if medians else None,
            "mean_fault_detection_rate": float(np.mean([entry["detection_rate"] for entry in summaries])),
        }
    return {
        "onset_sample": onset_sample, "sample_period_minutes": sample_period_minutes,
        "latency_definition": "first flagged eligible endpoint at/after onset minus onset; means/medians exclude misses",
        "miss_representation": "null latency and detected=false; explicit coverage and right-censoring horizon",
        "normal_operation": _nonfault_summary(normal_batches),
        "fault_normal_prefix": _nonfault_summary(faulty_batches, before_sample=onset_sample),
        "overall": overall, "macro_across_fault_types": macro, "per_fault": per_fault,
    }


def _save_scores(path: Path, batches: Sequence[ScoreBatch]) -> None:
    np.savez_compressed(
        path, sources=np.array([batch.run_key[0] for batch in batches]),
        fault_numbers=np.array([batch.run_key[1] for batch in batches]),
        simulation_runs=np.array([batch.run_key[2] for batch in batches]),
        **{name: np.stack([getattr(batch, name) for batch in batches])
           for name in ("end_samples", "anomaly_scores", "drift_scores", "anomaly_flags", "drift_flags", "flags")},
    )


def _score_runs(detector: TwoSignalDetector, runs: Sequence, label: str) -> list[ScoreBatch]:
    batches = []
    for index, run in enumerate(runs, 1):
        batches.append(detector.score_run(run))
        if index % 25 == 0 or index == len(runs):
            print(f"Scored {index}/{len(runs)} {label} simulations", file=sys.stderr, flush=True)
    return batches


def _code_provenance() -> dict:
    files = {path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {"package_file_sha256": files, "package_code_sha256": digest,
            "git_head": head, "git_worktree_dirty": dirty}


def results_markdown(report: dict) -> str:
    def number(value):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value is not None else "undetected"

    metrics = report["metrics"]
    combined = metrics["overall"]["combined"]
    normal = metrics["normal_operation"]
    fpr = normal["signals"]["combined"]
    lines = [
        "# SpectraDrift TEP benchmark results", "",
        f"Generated by an actual benchmark run: {report['generated_at_utc']}.", "",
        f"Scope: {combined['simulations']} faulty simulations across {len(metrics['per_fault'])} fault types; "
        f"{normal['simulations']} held-out normal simulations. This selects complete faulty runs, not the full faulty release.", "",
        f"Combined first-flag detection: {combined['detected_simulations']}/{combined['simulations']} simulations "
        f"({combined['detection_rate']:.2%}); {combined['undetected_simulations']} undetected.", "",
        f"Mean latency among detected runs: **{number(combined['mean_latency_steps_detected'])} steps** "
        f"({number(combined['mean_latency_minutes_detected'])} minutes). "
        f"Median: **{number(combined['median_latency_steps_detected'])} steps** "
        f"({number(combined['median_latency_minutes_detected'])} minutes).", "",
        f"Held-out normal window false-positive rate: **{fpr['false_positive_rate']:.2%}** "
        f"({fpr['flagged_windows']:,}/{normal['eligible_windows']:,} eligible windows). "
        f"{fpr['runs_with_any_flag']}/{normal['simulations']} normal runs had at least one flag; "
        f"there were {fpr['flag_episodes']} runs of consecutive flagged eligible windows.", "",
        "## Per-fault breakdown", "",
        "Latency uses the first flagged window endpoint at/after onset, not an alarm-event transition. "
        "A flag already present at onset can give zero latency; continuation counts are saved separately. "
        "All means and medians below are conditional on detection; undetected runs have null latency in JSON.", "",
        "| Fault | OR detected / runs | Mean steps | Median steps | PCA detected | KS detected | Post-onset OR flag rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fault in metrics["per_fault"].values():
        signal = fault["signals"]
        row = signal["combined"]
        lines.append(f"| {fault['fault_number']} | {row['detected_simulations']}/{row['simulations']} | "
                     f"{number(row['mean_latency_steps_detected'])} | {number(row['median_latency_steps_detected'])} | "
                     f"{signal['anomaly']['detected_simulations']} | {signal['drift']['detected_simulations']} | "
                     f"{row['post_onset_flag_rate']:.2%} |")
    macro = metrics["macro_across_fault_types"]["combined"]
    prefix = metrics["fault_normal_prefix"]
    prefix_rate = prefix["signals"]["combined"]["false_positive_rate"]
    lines += [
        "", "## Definitions and provenance", "",
        f"Onset is sample {metrics['onset_sample']} (one-based); samples 1–{metrics['onset_sample'] - 1} are the normal prefix. "
        f"Sampling period: {metrics['sample_period_minutes']} minutes. Warm-up excludes incomplete windows. "
        "Flags use the frozen PCA OR KS rule, with no post-test threshold tuning.", "",
        f"Equal-fault-weighted mean of detected-run latency means: "
        f"{number(macro['mean_of_fault_mean_latency_steps_detected'])} steps. "
        f"Median of per-fault detected-run medians: {number(macro['median_of_fault_median_latency_steps_detected'])} steps. "
        f"{macro['fault_types_with_any_detection']}/{macro['fault_types']} types had a detection.", "",
        f"Normal-prefix OR false-positive rate: {prefix_rate:.2%}." if prefix_rate is not None else "No eligible normal-prefix windows.", "",
        "First flags in long fault runs can also be background false alarms; a first-flag detection "
        "does not establish attribution to the injected fault. Post-onset flag rates and per-component "
        "latencies are included for interpretation. Overlapping windows are dependent, and a window "
        "false-positive rate is not a probability of a false alarm per run.", "",
        "The test fault condition labels identify whole runs, including their normal prefixes. "
        "The fault onset is the documented release convention, not a per-row label. "
        "Output-quality deterioration or lead time before quality loss is not measured by this benchmark.", "",
        f"Model NPZ SHA-256: `{report['model']['model_sha256']}`.", "",
        "Exact configuration, selection IDs, dataset checksums, software versions, code fingerprints, "
        "per-run latencies, misses, and score archive hashes are in [results.json](results.json). "
        "Ignored per-window score archives can be regenerated using the README commands.", "",
    ]
    return "\n".join(lines)


def run_benchmark(
    *, model_dir: Path | str = "artifacts/detector", data_dir: Path | str = "data",
    manifest_path: Path | str = "data/source.json",
    normal_run_ids: Sequence[int] = tuple(range(1, 501)),
    faulty_run_ids: Sequence[int] = tuple(range(1, 11)), fault_ids: Sequence[int] = tuple(range(1, 21)),
    output_dir: Path | str = ".", scores_dir: Path | str = "artifacts/benchmark",
) -> dict:
    """Benchmark a frozen model; no fitting or calibration occurs here."""
    output = Path(output_dir)
    scores = Path(scores_dir)
    targets = [output / "results.json", output / "results.md", scores / "normal_scores.npz", scores / "fault_scores.npz"]
    if any(path.exists() for path in targets):
        raise ValueError("Benchmark output files already exist; choose new --output and --scores-output directories")
    specs = load_manifest(manifest_path)
    dataset = json.loads(Path(manifest_path).read_text())
    onset = dataset["first_fault_sample_one_based"]
    period = dataset["sample_period_minutes"]
    if onset != dataset["test_normal_prefix_samples"] + 1:
        raise ValueError("Inconsistent dataset fault-onset convention")
    detector = load_detector(model_dir)
    model_record = json.loads((Path(model_dir) / "manifest.json").read_text())
    training = model_record["training"]
    if not training.get("source_checksum_verified") or training.get("publisher_md5") != specs["normal-training"].md5:
        raise ValueError("Benchmark requires a model trained on the verified normal TEP release")
    verify_file(Path(data_dir) / specs["normal-training"].filename, specs["normal-training"])
    print("Loading and validating held-out normal simulations", file=sys.stderr, flush=True)
    normal_runs = load_runs(Path(data_dir) / specs["normal-testing"].filename, specs["normal-testing"], run_ids=normal_run_ids)
    normal_batches = _score_runs(detector, normal_runs, "normal")
    del normal_runs
    print("Parsing the full faulty RData table, then selecting requested complete runs", file=sys.stderr, flush=True)
    faulty_runs = load_runs(Path(data_dir) / specs["faulty-testing"].filename, specs["faulty-testing"],
                           run_ids=faulty_run_ids, fault_ids=fault_ids)
    faulty_batches = _score_runs(detector, faulty_runs, "faulty")
    del faulty_runs
    metrics = evaluate_batches(normal_batches, faulty_batches, onset_sample=onset,
                               sample_period_minutes=period, expected_faults=fault_ids)
    scores.mkdir(parents=True, exist_ok=True)
    _save_scores(scores / "normal_scores.npz", normal_batches)
    _save_scores(scores / "fault_scores.npz", faulty_batches)
    # This guards accidental file mutation during the run; all detector operations
    # themselves are read-only, and no labels feed back into model configuration.
    if _sha256(Path(model_dir) / "model.npz") != model_record["model_sha256"]:
        raise ValueError("Model changed during benchmark evaluation")
    report = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"doi": dataset["doi"], "version": dataset["dataset_version"],
                    "onset_convention": dataset["onset_convention"],
                    "files": [{"filename": spec.filename, "publisher_md5": spec.md5,
                               "size_bytes": spec.size_bytes, "verified": True} for spec in specs.values()]},
        "configuration": {
            "normal_run_ids": sorted(normal_run_ids), "faulty_run_ids": sorted(faulty_run_ids),
            "fault_ids": sorted(fault_ids), "all_fault_types_included": set(fault_ids) == set(range(1, 21)),
            "full_normal_test_release": set(normal_run_ids) == set(range(1, 501)),
            "full_faulty_test_release": set(faulty_run_ids) == set(range(1, 501)) and set(fault_ids) == set(range(1, 21)),
            "selection_rule": "Explicit complete simulation IDs, selected before viewing fault results",
            "model_directory": str(model_dir), "window_size": detector.window_size, "stride": detector.stride,
            "thresholds_frozen": True, "test_label_tuning": False,
        },
        "model": model_record, "metrics": metrics,
        "score_archives": {name: {"path": str(path), "sha256": _sha256(path)}
                           for name, path in (("normal", scores / "normal_scores.npz"), ("faulty", scores / "fault_scores.npz"))},
        "software": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("spectradrift", "numpy", "pandas", "scipy", "scikit-learn", "pyreadr")
        }},
        "code_provenance": _code_provenance(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "results.md").write_text(results_markdown(report))
    return report
