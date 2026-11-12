"""Read the synchronized BROAD HDF5 files distributed by the dataset authors."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from tempfile import NamedTemporaryFile
import hashlib
import json
import shutil

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST = DATA_DIR / "sources.json"


@dataclass
class Trial:
    name: str
    hz: float
    time: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    reference: np.ndarray
    movement: np.ndarray
    valid_reference: np.ndarray


def default_trials(data_dir, download=False, manifest=MANIFEST):
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    data_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = data_dir / entry["file"]
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"{path} is missing. Run benchmark.py --download.")
            temp_path = None
            try:
                with urlopen(entry["url"], timeout=60) as response:
                    with NamedTemporaryFile(dir=data_dir, delete=False) as temporary:
                        temp_path = Path(temporary.name)
                        shutil.copyfileobj(response, temporary)
                if hashlib.sha256(temp_path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Download checksum mismatch for {path.name}")
                temp_path.replace(path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
        yield path


def read_trial(path):
    with h5py.File(path, "r") as source:
        hz = float(source.attrs["sampling_rate"])
        gyro = np.asarray(source["imu_gyr"], dtype=float)
        accel = np.asarray(source["imu_acc"], dtype=float)
        reference = np.asarray(source["opt_quat"], dtype=float)
        movement = np.asarray(source["movement"], dtype=bool).reshape(-1)

    count = len(movement)
    if gyro.shape != (count, 3) or accel.shape != (count, 3) or reference.shape != (count, 4):
        raise ValueError(f"Unexpected dataset dimensions in {path.name}")
    if not np.isfinite(hz) or hz <= 0 or not (np.isfinite(gyro).all() and np.isfinite(accel).all()):
        raise ValueError(f"Invalid IMU samples or sampling rate in {path.name}")

    norms = np.linalg.norm(reference, axis=1)
    valid = np.isfinite(norms) & (norms > 0)
    if not np.any(valid & movement):
        raise ValueError(f"{path.name} has no valid optical samples in motion")
    reference[valid] /= norms[valid, None]
    reference[~valid] = np.nan
    # Initialize at the first valid optical pose and retain original trial timestamps.
    start = np.flatnonzero(valid)[0]
    time = np.arange(start, count) / hz
    return Trial(path.stem, hz, time, gyro[start:], accel[start:], reference[start:],
                 movement[start:], valid[start:])
