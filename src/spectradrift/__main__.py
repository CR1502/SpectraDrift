"""Check the installed foundation before building the benchmark pipeline."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform

from spectradrift import __version__


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SpectraDrift environment check. Benchmark commands come in later PRs."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
