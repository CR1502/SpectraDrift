"""Verified acquisition and run-preserving loading of the Harvard TEP release."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyreadr


SENSOR_COLUMNS = tuple(
    [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
)
METADATA_COLUMNS = ("faultNumber", "simulationRun", "sample")
SOURCES = {
    "normal-training": ("fault_free_training", 500, (0,)),
    "normal-testing": ("fault_free_testing", 960, (0,)),
    "faulty-testing": ("faulty_testing", 960, tuple(range(1, 21))),
}


@dataclass(frozen=True)
class FileSpec:
    source: str
    filename: str
    r_object: str
    size_bytes: int
    md5: str
    download_url: str

    @property
    def samples_per_run(self) -> int:
        return SOURCES[self.source][1]

    @property
    def allowed_faults(self) -> tuple[int, ...]:
        return SOURCES[self.source][2]


def load_manifest(path: Path | str = "data/source.json") -> dict[str, FileSpec]:
    """Read the checked-in release manifest, keyed by a CLI-friendly source name."""
    manifest = json.loads(Path(path).read_text())
    if manifest["doi"] != "10.7910/DVN/6C3JR1" or manifest["dataset_version"] != "1.0":
        raise ValueError("Only Harvard TEP version 1.0 is supported")
    if manifest["process_channels"] != len(SENSOR_COLUMNS):
        raise ValueError("Manifest must describe exactly 52 process channels")
    by_object = {value[0]: source for source, value in SOURCES.items()}
    specs = {}
    for record in manifest["files"]:
        source = by_object[record["r_object"]]
        spec = FileSpec(source=source, **{
            name: record[name]
            for name in ("filename", "r_object", "size_bytes", "md5", "download_url")
        })
        if Path(spec.filename).name != spec.filename or spec.size_bytes <= 0:
            raise ValueError("Invalid source filename or byte size")
        if len(spec.md5) != 32 or any(c not in "0123456789abcdef" for c in spec.md5):
            raise ValueError("Invalid publisher MD5 checksum")
        if not spec.download_url.startswith("https://dataverse.harvard.edu/api/access/datafile/"):
            raise ValueError("Expected a Harvard Dataverse HTTPS download URL")
        if source in specs:
            raise ValueError(f"Duplicate manifest source: {source}")
        specs[source] = spec
    if set(specs) != set(SOURCES):
        raise ValueError("The manifest must contain all three benchmark sources")
    return specs


def verify_file(path: Path | str, spec: FileSpec) -> str:
    """Reject missing, truncated, or modified files before parsing RData."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run the download-data command first")
    if path.stat().st_size != spec.size_bytes:
        raise ValueError(f"Wrong byte size for {path}; expected {spec.size_bytes}")
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != spec.md5:
        raise ValueError(f"Publisher checksum mismatch for {path}")
    return actual


def download_file(
    spec: FileSpec, data_dir: Path | str = "data", *, attempts: int = 3, timeout: float = 30
) -> Path:
    """Download to a temporary file, verify it, then publish the completed file.

    Existing verified files are reused. Existing invalid files are never replaced;
    move them aside explicitly before retrying. Only our temporary files are removed.
    """
    if attempts < 1 or timeout <= 0:
        raise ValueError("attempts and timeout must be positive")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / spec.filename
    if destination.exists():
        verify_file(destination, spec)
        print(f"Verified existing {destination}", file=sys.stderr, flush=True)
        return destination
    for attempt in range(attempts):
        temporary = None
        try:
            print(f"Downloading {spec.filename} (attempt {attempt + 1}/{attempts})", file=sys.stderr, flush=True)
            with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                request = Request(spec.download_url, headers={"User-Agent": "SpectraDrift/0.1"})
                with urlopen(request, timeout=timeout) as response:
                    received = 0
                    last_report = time.monotonic()
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if received > spec.size_bytes:
                            raise ValueError("Downloaded file exceeds the publisher's byte size")
                        if time.monotonic() - last_report >= 5:
                            print(f"  {received:,}/{spec.size_bytes:,} bytes", file=sys.stderr, flush=True)
                            last_report = time.monotonic()
            verify_file(temporary, spec)
            # Recheck before replacing: a user or another download may have finished.
            if destination.exists():
                verify_file(destination, spec)
            else:
                temporary.replace(destination)
            print(f"Verified {destination}", file=sys.stderr, flush=True)
            return destination
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            retryable = not isinstance(exc, HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    raise RuntimeError("Download attempts exhausted")


@dataclass(frozen=True)
class TEPRun:
    """A complete, ordered simulation. Its identity includes the source split."""

    source: str
    fault_number: int
    simulation_run: int
    samples: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"Unknown TEP source: {self.source}")
        if self.fault_number not in SOURCES[self.source][2]:
            raise ValueError("Fault condition does not belong to this source")
        if not 1 <= self.simulation_run <= 500 or int(self.simulation_run) != self.simulation_run:
            raise ValueError("simulationRun must be an integer in 1..500")
        samples = np.array(self.samples, copy=True)
        values = np.array(self.values, dtype=np.float64, copy=True)
        if samples.ndim != 1 or not len(samples):
            raise ValueError("A run must have a nonempty sample vector")
        if not np.array_equal(samples, np.arange(1, len(samples) + 1)):
            raise ValueError("Run samples must be consecutive, unique, and start at 1")
        if values.shape != (len(samples), len(SENSOR_COLUMNS)):
            raise ValueError("Expected one row per sample and exactly 52 sensor channels")
        if not np.isfinite(values).all():
            raise ValueError("Sensor values must be finite")
        samples = samples.astype(np.int64)
        samples.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "values", values)

    @property
    def key(self) -> tuple[str, int, int]:
        return self.source, int(self.fault_number), int(self.simulation_run)


def _selection(values: Iterable[int] | None, allowed: Iterable[int], name: str) -> set[int]:
    allowed = set(allowed)
    if values is None:
        return allowed
    requested = list(values)
    if not requested or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in requested):
        raise ValueError(f"{name} must be a nonempty sequence of integers")
    if len(set(requested)) != len(requested) or not set(requested) <= allowed:
        raise ValueError(f"Duplicate or out-of-range {name}")
    return set(requested)


def runs_from_frame(
    frame: pd.DataFrame,
    source: str,
    *,
    run_ids: Iterable[int] | None = None,
    fault_ids: Iterable[int] | None = None,
) -> list[TEPRun]:
    """Validate a source table and return selected complete runs in stable order.

    Metadata validation covers the full table. Sensor/sample-continuity validation
    covers selected runs. By default all release run IDs and fault types are required.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown TEP source: {source}")
    if frame.columns.has_duplicates or set(frame.columns) != set(METADATA_COLUMNS + SENSOR_COLUMNS):
        raise ValueError("Expected the 3 TEP metadata columns and 52 named sensor columns")
    n_samples = SOURCES[source][1]
    bounds = {"faultNumber": (0, 20), "simulationRun": (1, 500), "sample": (1, n_samples)}
    for column, (low, high) in bounds.items():
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric")
        values = frame[column].to_numpy()
        if not np.isfinite(values).all() or not ((values >= low) & (values <= high)).all():
            raise ValueError(f"Invalid or out-of-range {column}")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integers")
    if not frame["faultNumber"].isin(SOURCES[source][2]).all():
        raise ValueError("Source contains unexpected fault conditions")
    selected_runs = _selection(run_ids, range(1, 501), "run IDs")
    selected_faults = _selection(fault_ids, SOURCES[source][2], "fault IDs")
    selected = frame.loc[
        frame["simulationRun"].isin(selected_runs) & frame["faultNumber"].isin(selected_faults)
    ]
    expected = {(fault, run) for fault in selected_faults for run in selected_runs}
    actual = set(selected.groupby(["faultNumber", "simulationRun"], sort=False).groups)
    if actual != expected:
        raise ValueError(f"Missing requested simulations: {sorted(expected - actual)[:10]}")
    runs = []
    for (fault, run_id), group in selected.groupby(["faultNumber", "simulationRun"], sort=True):
        group = group.sort_values("sample")
        if len(group) != n_samples:
            raise ValueError(f"Incomplete simulation {(source, fault, run_id)}: expected {n_samples} rows")
        for column in SENSOR_COLUMNS:
            if not pd.api.types.is_numeric_dtype(group[column]):
                raise ValueError(f"{column} must contain numeric sensor readings")
        runs.append(TEPRun(
            source=source,
            fault_number=int(fault),
            simulation_run=int(run_id),
            samples=group["sample"].to_numpy(),
            values=group.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float64),
        ))
    return runs


def load_runs(
    path: Path | str,
    spec: FileSpec,
    *,
    run_ids: Iterable[int] | None = None,
    fault_ids: Iterable[int] | None = None,
) -> list[TEPRun]:
    """Verify a publisher file and parse its expected R object.

    pyreadr parses the whole RData table before selection: a small run selection
    reduces retained arrays but does not reduce peak parsing memory.
    """
    verify_file(path, spec)
    objects = pyreadr.read_r(str(path), use_objects=[spec.r_object])
    if spec.r_object not in objects or not isinstance(objects[spec.r_object], pd.DataFrame):
        raise ValueError(f"Missing R dataframe {spec.r_object}")
    return runs_from_frame(objects[spec.r_object], spec.source, run_ids=run_ids, fault_ids=fault_ids)
