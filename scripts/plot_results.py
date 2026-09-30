"""Plot real, checksum-verified benchmark scores; never fit or change a model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class PlotRun:
    source: str
    fault: int
    run: int
    end_samples: np.ndarray
    anomaly_ratio: np.ndarray
    drift_ratio: np.ndarray
    flags: np.ndarray

    def title(self, onset: int) -> str:
        if self.fault == 0:
            return f"Normal operation · run {self.run} · {int(self.flags.sum())}/{len(self.flags)} windows flagged"
        hits = self.end_samples[self.flags & (self.end_samples >= onset)]
        latency = f"{int(hits[0]) - onset} steps" if len(hits) else "undetected"
        return f"Fault {self.fault} · run {self.run} · first-flag latency: {latency}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(results: Path | str, scores_dir: Path | str | None = None) -> tuple[list[PlotRun], int]:
    """Fixed examples: run 1 of normal operation, fault 1, and difficult fault 9.

    Archive paths are relative to the repository working directory, just as in
    the benchmark report. An explicit scores directory supports relocated files,
    but the original report's checksums must still match.
    """
    report = json.loads(Path(results).read_text())
    thresholds = np.array([report["model"]["anomaly_threshold"], report["model"]["drift_threshold"]])
    if not np.isfinite(thresholds).all() or not (thresholds > 0).all():
        raise ValueError("Plotting score ratios requires positive, finite thresholds")
    onset = report["metrics"]["onset_sample"]
    if not isinstance(onset, int) or isinstance(onset, bool) or not 1 <= onset <= 960:
        raise ValueError("Invalid one-based fault onset")
    examples = []
    for kind, source, faults in (("normal", "normal-testing", (0,)), ("faulty", "faulty-testing", (1, 9))):
        record = report["score_archives"][kind]
        original = Path(record["path"])
        path = Path(scores_dir) / original.name if scores_dir is not None else original
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; regenerate scores with the README benchmark command")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Score archive checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            for fault in faults:
                selected = np.flatnonzero((data["sources"] == source) & (data["fault_numbers"] == fault)
                                          & (data["simulation_runs"] == 1))
                if len(selected) != 1:
                    raise ValueError(f"Expected exactly one {source}, fault {fault}, run 1")
                index = int(selected[0])
                ends, anomaly, drift = [data[name][index] for name in ("end_samples", "anomaly_scores", "drift_scores")]
                a_flags, d_flags, flags = [data[name][index] for name in ("anomaly_flags", "drift_flags", "flags")]
                if ends.ndim != 1 or not len(ends) or not np.isfinite(ends).all() or not (np.diff(ends) > 0).all():
                    raise ValueError("Invalid score window endpoints")
                if not np.equal(ends, np.floor(ends)).all() or not ((ends >= 1) & (ends <= 960)).all():
                    raise ValueError("Endpoints must be one-based TEP sample numbers")
                if any(array.shape != ends.shape for array in (anomaly, drift, a_flags, d_flags, flags)):
                    raise ValueError("Scores and flags must align with endpoints")
                if not np.isfinite(anomaly).all() or not np.isfinite(drift).all() or (anomaly < 0).any() or ((drift < 0) | (drift > 1)).any():
                    raise ValueError("Invalid PCA or empirical KS scores")
                if any(array.dtype.kind != "b" for array in (a_flags, d_flags, flags)):
                    raise ValueError("Flags must be boolean arrays")
                if not (np.array_equal(a_flags, anomaly > thresholds[0])
                        and np.array_equal(d_flags, drift > thresholds[1])
                        and np.array_equal(flags, a_flags | d_flags)):
                    raise ValueError("Archive flags do not follow the frozen strict-threshold OR rule")
                examples.append(PlotRun(source, fault, 1, ends, anomaly / thresholds[0], drift / thresholds[1], flags))
    return examples, onset


def render_plot(results: Path | str, output: Path | str, scores_dir: Path | str | None = None) -> Path:
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}; choose a new --output filename")
    examples, onset = load_examples(results, scores_dir)
    # Headless rendering works locally and in CI. Nothing writes to the scores.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colors = {"anomaly": "#2563eb", "drift": "#0d9488", "flag": "#be185d"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, layout="constrained")
        figure.suptitle("SpectraDrift · frozen PCA + rolling KS on TEP", fontsize=17, fontweight="bold")
        for axis, example in zip(axes, examples):
            if example.fault:
                axis.axvspan(onset, 960, color="#f59e0b", alpha=0.12)
                axis.axvline(onset, color="#b45309", linewidth=1, alpha=0.65)
            # The floor affects log-scale display only, never flags or metrics.
            axis.plot(example.end_samples, np.maximum(example.anomaly_ratio, 1e-6), color=colors["anomaly"], linewidth=1.3)
            axis.plot(example.end_samples, np.maximum(example.drift_ratio, 1e-6), color=colors["drift"], linewidth=1.3)
            axis.axhline(1, color="#475569", linestyle="--", linewidth=1)
            axis.scatter(example.end_samples[example.flags], np.full(int(example.flags.sum()), 0.035),
                         transform=axis.get_xaxis_transform(), color=colors["flag"], marker="|", s=45, linewidths=1)
            axis.set_yscale("log")
            axis.set_ylabel("Score / threshold\n(log scale)")
            axis.set_title(example.title(onset), loc="left", fontsize=11, pad=9)
            axis.grid(axis="y", alpha=0.15)
            axis.set_xlim(1, 960)
        axes[-1].set_xlabel("Window endpoint sample (one-based) · fault injection begins at sample 161")
        handles = [Line2D([], [], color=colors["anomaly"], label="PCA reconstruction error / threshold"),
                   Line2D([], [], color=colors["drift"], label="Max sensor KS distance / threshold"),
                   Line2D([], [], color="#475569", linestyle="--", label="Threshold = 1; flags use strict >"),
                   Line2D([], [], color=colors["flag"], marker="|", linestyle="none", label="Combined OR flag (bottom ticks)"),
                   Patch(facecolor="#f59e0b", alpha=0.2, label="Injected fault interval")]
        figure.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(output, dpi=160, facecolor="white")
        finally:
            plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    parser.add_argument("--scores-dir", type=Path, help="Relocated score archives; hashes must match the report")
    parser.add_argument("--output", type=Path, default=Path("docs/figures/tep-monitoring.png"))
    args = parser.parse_args()
    try:
        path = render_plot(args.results, args.output, args.scores_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"plot-results: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {path}; verified score archive checksums, frozen thresholds, and OR flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
