"""Replay two additional BROAD trials and measure sensitivity to dropped samples."""

import argparse
from dataclasses import replace
from pathlib import Path

from ahrs.dataset import DATA_DIR, default_trials, read_trial
from ahrs.metrics import rms
from benchmark import ROOT, aggregate, configured_filters, evaluate, write_csv


def decimate(trial, stride):
    keep = slice(None, None, stride)
    return replace(
        trial, hz=trial.hz / stride, time=trial.time[keep], gyro=trial.gyro[keep],
        accel=trial.accel[keep], reference=trial.reference[keep], movement=trial.movement[keep],
        valid_reference=trial.valid_reference[keep],
    )


def rate_sweep(trial, filters, native_series):
    # Score at the same native indices 0, 4, 8, ... for all three replays.
    common = (trial.movement & trial.valid_reference)[::4]
    for stride in (1, 2, 4):
        replay = decimate(trial, stride)
        series = native_series if stride == 1 else evaluate(replay, filters, 1)[1]
        for name, (_, total, tilt) in series.items():
            select = slice(None, None, 4 // stride)
            yield {
                "trial": trial.name,
                "filter": name,
                "stride": stride,
                "sample_rate_hz": replay.hz,
                "motion_samples_scored": int(common.sum()),
                "motion_total_rmse_deg": rms(total[select][common]),
                "motion_tilt_rmse_deg": rms(tilt[select][common]),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--download", action="store_true", help="Restore missing recordings.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    filters = configured_filters()
    paths = list(default_trials(DATA_DIR, args.download, DATA_DIR / "stress_sources.json"))
    stress_rows, rate_rows = [], []
    for path in paths:
        trial = read_trial(path)
        rows, series = evaluate(trial, filters, 3)
        stress_rows.extend(rows)
        print(f"{trial.name}: {rows[0]['motion_samples_scored']:,} valid motion samples")
        if trial.name.startswith("26_"):
            rate_rows.extend(rate_sweep(trial, filters, series))

    fast = next(path for path in default_trials(DATA_DIR, args.download) if path.name.startswith("06_"))
    trial = read_trial(fast)
    rate_rows.extend(rate_sweep(trial, filters, evaluate(trial, filters, 1)[1]))
    write_csv(args.out / "stress_per_trial.csv", stress_rows)
    write_csv(args.out / "stress_pooled.csv", aggregate(stress_rows, filters))
    write_csv(args.out / "sample_rate_sweep.csv", rate_rows)


if __name__ == "__main__":
    main()
