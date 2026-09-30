"""Environment, dataset acquisition, and preprocessing commands."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from urllib.error import URLError

from spectradrift import __version__


def environment_check() -> int:
    dependencies = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "pyreadr": "pyreadr",
    }
    versions = {}
    errors = {}
    for distribution, module in dependencies.items():
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
        except (ImportError, OSError, importlib.metadata.PackageNotFoundError) as exc:
            errors[distribution] = str(exc)
    print(
        json.dumps(
            {
                "project": "SpectraDrift",
                "version": __version__,
                "python": platform.python_version(),
                "dependencies": versions,
                "errors": errors,
                "environment_ready": not errors,
                "benchmark_status": "not yet implemented",
            },
            indent=2,
        )
    )
    return int(bool(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectraDrift TEP data pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("doctor", help="Check installed dependencies (also the default command)")
    download = commands.add_parser("download-data", help="Download and verify publisher files")
    download.add_argument("--data-dir", type=Path, default=Path("data"))
    download.add_argument("--manifest", type=Path, default=Path("data/source.json"))
    download.add_argument("--files", nargs="+", choices=["normal-training", "normal-testing", "faulty-testing"], default=["normal-training", "normal-testing"])
    inspect = commands.add_parser("inspect-data", help="Validate selected complete simulations")
    inspect.add_argument("--data-dir", type=Path, default=Path("data"))
    inspect.add_argument("--manifest", type=Path, default=Path("data/source.json"))
    inspect.add_argument("--file", choices=["normal-training", "normal-testing", "faulty-testing"], default="normal-training")
    inspect.add_argument("--runs", nargs="+", type=int, default=list(range(1, 11)))
    inspect.add_argument("--faults", nargs="+", type=int, help="All conditions in the selected file by default")
    prepare = commands.add_parser("prepare-data", help="Create normal-only fitting/calibration window artifacts")
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare.add_argument("--manifest", type=Path, default=Path("data/source.json"))
    prepare.add_argument("--runs", nargs="+", type=int, default=list(range(1, 11)))
    prepare.add_argument("--window-size", type=int, default=20)
    prepare.add_argument("--stride", type=int, default=1)
    prepare.add_argument("--calibration-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--output", type=Path, default=Path("artifacts/preprocessing"))
    args = parser.parse_args()
    if args.command in (None, "doctor"):
        return environment_check()
    try:
        from spectradrift.data import SENSOR_COLUMNS, download_file, load_manifest, load_runs
        if args.command == "download-data":
            specs = load_manifest(args.manifest)
            paths = [str(download_file(specs[source], args.data_dir)) for source in args.files]
            print(json.dumps({"verified_files": paths}, indent=2))
        elif args.command == "inspect-data":
            spec = load_manifest(args.manifest)[args.file]
            runs = load_runs(args.data_dir / spec.filename, spec, run_ids=args.runs, fault_ids=args.faults)
            print(json.dumps({
                "source": args.file,
                "checksum_verified": True,
                "sensors": list(SENSOR_COLUMNS),
                "selected_simulations": len(runs),
                "selected_samples": sum(len(run.samples) for run in runs),
                "runs": [{"key": list(run.key), "samples": len(run.samples)} for run in runs],
                "selection_note": "Only the listed simulations were validated for sample continuity and sensor values",
            }, indent=2))
        elif args.command == "prepare-data":
            from spectradrift.preprocessing import prepare_normal_data
            report = prepare_normal_data(
                data_dir=args.data_dir, manifest_path=args.manifest, run_ids=args.runs,
                window_size=args.window_size, stride=args.stride,
                calibration_fraction=args.calibration_fraction, seed=args.seed,
                output_dir=args.output,
            )
            print(json.dumps({
                "manifest": str(args.output / "manifest.json"),
                "checksum_verified": report["checksum_verified"],
                "partitions": report["partitions"],
                "benchmark_status": report["benchmark_status"],
            }, indent=2))
    except (OSError, ValueError, KeyError, URLError) as exc:
        print(f"spectradrift: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
