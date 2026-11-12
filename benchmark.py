"""Replay BROAD IMU streams and compare four filters with optical ground truth."""

import argparse
import csv
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from ahrs.dataset import DATA_DIR, default_trials, read_trial
from ahrs.filters import Complementary, GyroOnly, Madgwick, Mahony
from ahrs.metrics import angular_error, initial_rest, rms, slope_per_minute, tilt_error


ROOT = Path(__file__).resolve().parent


def configured_filters(tau=1.0, kp=1.0, ki=0.10, beta=0.035):
    return {
        "Gyro-only": (GyroOnly, {}),
        "Complementary": (Complementary, {"tau": tau}),
        "Mahony": (Mahony, {"kp": kp, "ki": ki}),
        "Madgwick": (Madgwick, {"beta": beta}),
    }


def evaluate(trial, filters, timing_runs):
    dt = 1 / trial.hz
    samples = list(zip(map(tuple, trial.gyro.tolist()), map(tuple, trial.accel.tolist())))
    initial_pose = tuple(trial.reference[0])
    motion = trial.movement & trial.valid_reference
    rest = initial_rest(trial)
    series = {}
    rows = []
    for name, (filter_type, gains) in filters.items():
        times = []
        for _ in range(timing_runs):
            update = filter_type(initial_pose, **gains).update
            start = perf_counter()
            estimates = [initial_pose, *(update(gyro, accel, dt) for gyro, accel in samples[1:])]
            times.append((perf_counter() - start) * 1e6 / (len(samples) - 1))
        estimates = np.array(estimates)
        total = angular_error(estimates, trial.reference)
        tilt = tilt_error(estimates, trial.reference)
        rows.append({
            "trial": trial.name,
            "filter": name,
            "sample_rate_hz": trial.hz,
            "samples_replayed": len(samples),
            "motion_samples_scored": int(motion.sum()),
            "motion_samples_missing_optical": int((trial.movement & ~trial.valid_reference).sum()),
            "motion_total_rmse_deg": rms(total[motion]),
            "motion_total_max_deg": float(total[motion].max()),
            "motion_tilt_rmse_deg": rms(tilt[motion]),
            "motion_tilt_max_deg": float(tilt[motion].max()),
            "rest_samples_scored": int(rest.sum()),
            "rest_total_slope_deg_per_min": slope_per_minute(trial.time[rest], total[rest]),
            "rest_tilt_slope_deg_per_min": slope_per_minute(trial.time[rest], tilt[rest]),
            "median_replay_us_per_update": median(times),
        })
        series[name] = (estimates, total, tilt)
    return rows, series


def aggregate(rows, names):
    summaries = []
    for name in names:
        selected = [row for row in rows if row["filter"] == name]
        count = sum(row["motion_samples_scored"] for row in selected)
        pooled = {}
        for kind in ("total", "tilt"):
            rms_key = f"motion_{kind}_rmse_deg"
            max_key = f"motion_{kind}_max_deg"
            squared = sum(row[rms_key] ** 2 * row["motion_samples_scored"] for row in selected)
            pooled[rms_key] = (squared / count) ** 0.5
            pooled[max_key] = max(row[max_key] for row in selected)
        summaries.append({"filter": name, "trials": len(selected),
                          "motion_samples_scored": count, **pooled})
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Use a finite positive number.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, action="append", help="Custom BROAD HDF5 file. Repeat for multiple trials.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--download", action="store_true", help="Fetch missing default trials and verify their hashes.")
    parser.add_argument("--tau", type=positive, default=1.0, help="Complementary time constant in seconds.")
    parser.add_argument("--kp", type=positive, default=1.0, help="Mahony proportional gain.")
    parser.add_argument("--ki", type=positive, default=0.10, help="Mahony integral gain.")
    parser.add_argument("--beta", type=positive, default=0.035, help="Madgwick gradient gain.")
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    if args.timing_runs < 1:
        parser.error("--timing-runs must be at least 1")

    filters = configured_filters(args.tau, args.kp, args.ki, args.beta)
    paths = args.trial or list(default_trials(args.data_dir, args.download))
    if len({path.stem for path in paths}) != len(paths):
        parser.error("Trial file stems must be unique.")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    representative = None
    for path in paths:
        trial = read_trial(path)
        trial_rows, series = evaluate(trial, filters, args.timing_runs)
        rows.extend(trial_rows)
        print(f"{trial.name}: {trial_rows[0]['motion_samples_scored']:,} valid motion samples")
        if representative is None:
            representative = (trial, series)

    pooled = aggregate(rows, filters)
    write_csv(args.out / "per_trial.csv", rows)
    write_csv(args.out / "pooled.csv", pooled)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timing_runs": args.timing_runs,
        "filter_gains": {name: gains for name, (_, gains) in filters.items()},
    }
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in pooled:
        print(f"{row['filter']:14}  orientation {row['motion_total_rmse_deg']:7.2f} deg RMS"
              f"  |  tilt {row['motion_tilt_rmse_deg']:6.2f} deg RMS")
    if not args.metrics_only:
        from ahrs.visuals import render

        render(args.out, pooled, *representative)


if __name__ == "__main__":
    main()
