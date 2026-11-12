# AHRS Sensor-Fusion Benchmark

This Python benchmark compares gyro integration, a quaternion complementary filter, Mahony, and Madgwick against optical motion capture. It uses six recordings from the public BROAD dataset.

On the four undisturbed trials, Madgwick lowered tilt RMS error from 9.45° to 1.67° (82.4%) across 169,828 motion samples. The complementary filter had the lowest full-orientation RMS error at 31.30°. The gyro and accelerometer provide no absolute heading reference, so yaw can drift after initialization.

![Two rotating wireframe cubes showing the optical reference and Madgwick estimate](results/orientation_demo.gif)

*Trial 01, first eight seconds of labeled motion. Optical reference on the left, Madgwick on the right. The orange face marks body +X.*

## Results

| Filter | Full orientation RMS | Full orientation max | Tilt RMS | Tilt max |
| --- | ---: | ---: | ---: | ---: |
| Gyro-only | 32.67° | 72.16° | 9.45° | 28.64° |
| Complementary | **31.30°** | 87.13° | 7.78° | 77.49° |
| Mahony | 38.80° | 164.57° | 9.27° | 59.52° |
| Madgwick | 31.36° | **70.77°** | **1.67°** | **9.27°** |

![Bar charts comparing full-orientation and tilt RMS error for the four filters](results/rms_comparison.svg)

| BROAD trial | Motion samples | Gyro-only tilt RMS | Complementary | Mahony | Madgwick |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 · Slow rotation | 35,855 | 17.31° | 0.62° | **0.56°** | 0.71° |
| 06 · Fast rotation | 34,882 | 2.62° | 1.55° | 1.84° | **1.04°** |
| 15 · Fast translation | 30,140 | 10.45° | 18.24° | 21.79° | **3.34°** |
| 19 · Slow combined motion | 68,951 | 3.59° | 1.45° | 1.43° | **1.06°** |

Fast translation was the hardest tilt case for the complementary and Mahony filters at these gains. An accelerometer measures specific force, so sustained translational acceleration can look like a change in tilt. During the first 25 seconds of rest in trial 01, the fitted tilt-error slope dropped from **4.99°/min** for gyro-only to **0.16°/min** for Madgwick. Its full-orientation error still increased at **28.15°/min** in that window.

On the benchmark machine, median replay cost per update ranged from 2.95–3.06 µs for gyro-only, 5.87–6.00 µs for Mahony, 6.68–6.78 µs for Madgwick, and 8.00–8.10 µs for complementary across the four trials. These figures describe the Python implementations on Windows. An STM32 port would need its own timing measurements.

- [Roll, pitch, and yaw tracking for trial 01](results/tracking.svg)
- [Static drift curves for trial 01](results/static_drift.svg)
- [Per-trial error, drift, and desktop replay timings](results/per_trial.csv)
- [Pooled metrics](results/pooled.csv) and [run configuration](results/run.json)

## Disturbance and sample-rate tests

The same gains were replayed on two additional BROAD recordings with 68,704 scored motion samples. In trial 24 the researchers tapped the IMU, causing acceleration spikes. In trial 26 they attached a vibrating phone to the fixture. Both have optical reference orientation.

| Filter | Tapping tilt RMS | Phone vibration tilt RMS | Pooled tilt RMS | Pooled full-orientation RMS |
| --- | ---: | ---: | ---: | ---: |
| Gyro-only | 45.47° | 37.10° | 41.51° | 42.73° |
| Complementary | 1.26° | **1.66°** | 1.47° | 8.75° |
| Mahony | **0.80°** | 1.74° | **1.35°** | **7.57°** |
| Madgwick | 1.26° | 3.42° | 2.57° | 9.17° |

Madgwick had the lowest pooled tilt error on the four original trials. Mahony led on the two disturbance trials, and complementary led on the phone-vibration trial alone.

A sample-rate sweep kept every first, second, or fourth IMU reading from the fast-rotation and phone-vibration trials. Each replay used its own timestep and was scored at identical optical timestamps. Madgwick tilt RMS on fast rotation was 1.04° at 285.71 Hz, 0.81° at 142.86 Hz, and 1.14° at 71.43 Hz. This experiment measures sensitivity to dropping samples. It applies no anti-aliasing filter, so it cannot predict a physical IMU's behavior at those rates.

A synthetic stationary test injects 0.5°/s of gyro bias. Mahony brings orientation error below 0.05° after 60 seconds with a horizontal-axis bias. With the same bias about the vertical axis, it accumulates 30° of heading error over 60 seconds.

- [Disturbance results by trial](results/stress_per_trial.csv) and [pooled metrics](results/stress_pooled.csv)
- [Sample-rate sweep for all four filters](results/sample_rate_sweep.csv)

## Run it

Use Python 3.10 or newer. All six trial files are included. If `python` opens the Microsoft Store on Windows, use `py` in the commands below.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe benchmark.py
.\.venv\Scripts\python.exe stress_test.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

On macOS or Linux, use `.venv/bin/python` in place of the Windows executable path. Running the benchmark regenerates the CSVs, plots, and GIF in `results/`. Use `--out my-results` to write elsewhere, `--download` to recover missing bundled recordings, or `--trial path/to/trial.hdf5` to replay a different BROAD file. Repeat `--trial` for multiple files.

## Filter updates

| Method | Update |
| --- | --- |
| Gyro-only | Integrate measured body angular rate using the quaternion derivative, then normalize. |
| Complementary | Predict with the gyro and apply a fraction of the shortest rotation that aligns predicted up with measured up. The correction weight is `1 - exp(-dt / tau)`, with `tau = 1.0 s`. |
| Mahony | Use the cross product of measured and predicted up for proportional rate correction and integrate it to estimate gyro bias. Gains are `kp = 1.0` and `ki = 0.10`. |
| Madgwick | Compute the gradient of the gravity-alignment residual and subtract its normalized direction from the gyro quaternion derivative. The gradient gain is `beta = 0.035`. |

The four updates are implemented in [filters.py](ahrs/filters.py) using Python arithmetic. NumPy computes metrics and transforms, and h5py reads the trial files.

## Evaluation protocol

- Replay the gyro in rad/s and accelerometer in m/s² at the source rate of 2000/7 Hz, about 286 Hz. The source authors synchronized and aligned the optical quaternions to the IMU. Each filter starts at the same first valid optical pose, including heading. After initialization, optical data is used for evaluation only. The baseline uses gyro readings, and the three fusion filters also use accelerometer readings.
- Evaluate on samples labeled as motion that also have a valid optical pose. The filters continue through optical gaps. Report the root mean square and maximum over the scored samples. The pooled RMS is sample-weighted across trials. Full-orientation error uses the shortest quaternion distance, and tilt error uses the angle between the two predicted world-up directions expressed in body coordinates.
- Estimate static drift by fitting a straight line to the optical error during up to the first 25 seconds of labeled rest after initialization. The reported slope measures error growth in degrees per minute over that window. Timing is the median of three desktop replays per filter and trial, divided by the number of updates. It includes Python iteration and result collection and is machine-dependent.
- These runs use six selected recordings and one fixed set of gains. Motion and acceleration profiles vary across the trials, and the observed ranking can change with other data or gains. The recordings come from the published BROAD study. This is an offline attitude-estimation benchmark for a future drone project. It contains no MPU-6050 acquisition, STM32 implementation, or flight test.

## Sources

- Laidig, Caruso, Cereatti, and Seel. [BROAD — A Benchmark for Robust Inertial Orientation Estimation](https://doi.org/10.3390/data6070072), 2021. [Original dataset repository](https://github.com/dlaidig/broad).
- Madgwick. [An efficient orientation filter for inertial and inertial/magnetic sensor arrays](https://x-io.co.uk/downloads/madgwick_internal_report.pdf), 2010.
- Mahony, Hamel, and Pflimlin. [Nonlinear Complementary Filters on the Special Orthogonal Group](https://doi.org/10.1109/TAC.2008.923738), 2008.
