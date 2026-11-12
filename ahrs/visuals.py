"""Plot benchmark output as SVG and render a side-by-side cube GIF."""

from html import escape
from itertools import combinations, product
from struct import pack

import numpy as np

from .metrics import initial_rest
from .quaternion import rotation_matrices, rpy_deg


COLORS = {
    "Optical reference": "#172b41",
    "Gyro-only": "#c45f36",
    "Complementary": "#078477",
    "Mahony": "#346ac5",
    "Madgwick": "#8a54b1",
}


class Figure:
    def __init__(self, width, height, title, subtitle):
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
            f' viewBox="0 0 {width} {height}" role="img">',
            f"<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>",
            '<rect width="100%" height="100%" rx="16" fill="#f7faff"/>',
        ]
        self.text(46, 49, title, size=29, weight=700)
        self.text(46, 79, subtitle, size=14)

    def text(self, x, y, value, size=13, anchor="start", weight=400, color="#263c56"):
        self.items.append(
            f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="{size}"'
            f' text-anchor="{anchor}" font-weight="{weight}" fill="{color}">{escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, color="#d9e4ef", width=1):
        self.items.append(f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, x, y, width, height, fill):
        self.items.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"/>')

    def save(self, path):
        path.write_text("\n".join([*self.items, "</svg>"]), encoding="utf-8")


def traces(fig, bounds, time, data, ylim, y_ticks, label, wrap=False):
    left, top, width, height = bounds
    xmin, xmax = time[0], time[-1]
    ymin, ymax = ylim
    fig.rect(left, top, width, height, "#ffffff")
    fig.text(left, top - 10, label, weight=700)
    for tick in y_ticks:
        y = top + height * (ymax - tick) / (ymax - ymin)
        fig.line(left, y, left + width, y)
        fig.text(left - 8, y + 4, f"{tick:g}", anchor="end")
    for tick in np.linspace(xmin, xmax, 6):
        x = left + width * (tick - xmin) / (xmax - xmin)
        fig.line(x, top, x, top + height)
        fig.text(x, top + height + 19, f"{tick:.1f}", anchor="middle")
    # Keep all samples for metrics. Thin display points to bound the SVG size.
    step = max(1, len(time) // 1400)
    indices = np.unique(np.r_[np.arange(0, len(time), step), len(time) - 1])
    xs = left + width * (time[indices] - xmin) / (xmax - xmin)
    for name, values in data.items():
        ys = top + height * (ymax - values[indices]) / (ymax - ymin)
        path = []
        previous = None
        for x, y, sample in zip(xs, ys, indices):
            if not np.isfinite(y):
                previous = None
                continue
            continuous = previous is not None
            if continuous:
                segment = values[previous:sample + 1]
                continuous = np.isfinite(segment).all()
                if wrap:
                    continuous = continuous and np.all(np.abs(np.diff(segment)) < 180)
            command = "L" if continuous else "M"
            path.append(f"{command}{x:.1f} {y:.1f}")
            previous = sample
        dash = ' stroke-dasharray="5 3"' if name == "Optical reference" else ""
        fig.items.append(f'<path d="{" ".join(path)}" fill="none" stroke="{COLORS[name]}"'
                         f' stroke-width="1.5" opacity=".90"{dash}/>')


def legend(fig, names, y):
    for i, name in enumerate(names):
        x = 47 + 181 * i
        fig.line(x, y - 4, x + 22, y - 4, COLORS[name], width=3)
        fig.text(x + 29, y, name)


def plot_tracking(out, trial, series):
    fig = Figure(1000, 770, "Attitude tracking", f"BROAD · {trial.name} · optical reference in dashed navy")
    angles = {name: rpy_deg(values[0]) for name, values in series.items()}
    angles["Optical reference"] = rpy_deg(trial.reference)
    legend(fig, angles, 110)
    for i, label in enumerate(("Roll (°)", "Pitch (°)", "Yaw (°)")):
        limit = 90 if i == 1 else 180
        data = {name: values[:, i] for name, values in angles.items()}
        traces(fig, (74, 151 + 194 * i, 875, 142), trial.time, data,
               (-limit, limit), (-limit, -limit / 2, 0, limit / 2, limit), label, wrap=True)
    fig.text(512, 739, "Trial time (s)", anchor="middle")
    fig.save(out / "tracking.svg")


def plot_summary(out, pooled):
    fig = Figure(1000, 386, "Orientation error across four filters",
                 f"{pooled[0]['motion_samples_scored']:,} scored motion samples across {pooled[0]['trials']} BROAD trials")
    for panel, (label, field) in enumerate((
        ("Full orientation RMS error (°)", "motion_total_rmse_deg"),
        ("Tilt RMS error (°)", "motion_tilt_rmse_deg"),
    )):
        x = 46 + 475 * panel
        fig.text(x, 129, label, size=17, weight=700)
        maximum = max(row[field] for row in pooled)
        for i, row in enumerate(pooled):
            name = row["filter"]
            y = 151 + 52 * i
            fig.text(x, y + 18, name)
            width = 235 * row[field] / max(maximum, 1e-9)
            fig.rect(x + 116, y, width, 26, COLORS[name])
            fig.text(round(x + 125 + width, 2), y + 18, f"{row[field]:.2f}°", weight=700)
    fig.save(out / "rms_comparison.svg")


def plot_rest(out, trial, series):
    rest = initial_rest(trial)
    if rest.sum() < 2:
        return
    fig = Figure(1000, 535, "Drift during initial rest",
                 f"BROAD · {trial.name} · first 25 seconds of labeled rest, when available")
    legend(fig, series, 110)
    for i, label in enumerate(("Full orientation error (°)", "Tilt error (°)")):
        data = {name: values[i + 1][rest] for name, values in series.items()}
        limit = max(0.5, np.ceil(max(values.max() for values in data.values()) * 2) / 2)
        traces(fig, (74, 151 + i * 176, 875, 126), trial.time[rest], data,
               (0, limit), np.linspace(0, limit, 5), label)
    fig.text(512, 510, "Trial time (s)", anchor="middle")
    fig.save(out / "static_drift.svg")


def gif_lzw(pixels):
    """Compress an 8-color GIF image with bounded 4-bit LZW dictionaries."""
    # Frequent clear codes keep the code width fixed and the encoder compact.
    clear, end = 8, 9
    codes = [clear]
    table = {}
    stream = iter(pixels)
    prefix = next(stream)
    for pixel in stream:
        pair = prefix, pixel
        if pair in table:
            prefix = table[pair]
            continue
        codes.append(prefix)
        if len(table) == 5:
            codes.append(clear)
            table.clear()
        else:
            table[pair] = 10 + len(table)
        prefix = pixel
    codes.extend((prefix, end))
    if len(codes) % 2:
        codes.append(0)
    compressed = bytes(a | b << 4 for a, b in zip(codes[::2], codes[1::2]))
    blocks = [compressed[i:i+255] for i in range(0, len(compressed), 255)]
    return b"\x03" + b"".join(bytes([len(block)]) + block for block in blocks) + b"\x00"


def cube_demo(out, trial, series):
    indices = np.flatnonzero(trial.movement & trial.valid_reference)
    # Show a short real motion segment at a consistent, near-real-time cadence.
    start = indices[0]
    stop = min(indices[-1] + 1, start + round(8 * trial.hz))
    frames = np.arange(start, stop, max(1, round(trial.hz / 12)))
    frames = frames[trial.valid_reference[frames]]
    if len(frames) == 0:
        return
    width, height = 320, 144
    vertices = np.array(list(product((-1, 1), repeat=3)))
    edges = [(i, j) for i, j in combinations(range(8), 2)
             if np.count_nonzero(vertices[i] - vertices[j]) == 1]
    palette = bytes.fromhex("f7faff 172b41 346ac5 c45f36 d9e4ef 078477 8a54b1 ffffff")
    camera = np.array([[0.87, -0.50, 0], [0.25, 0.43, -0.87]])
    matrices = [rotation_matrices(trial.reference[frames]),
                rotation_matrices(series["Madgwick"][0][frames])]
    with (out / "orientation_demo.gif").open("wb") as gif:
        gif.write(b"GIF89a" + pack("<HHBBB", width, height, 0xF2, 0, 0) + palette)
        gif.write(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")
        for reference, estimate in zip(*matrices):
            image = np.zeros((height, width), dtype=np.uint8)
            image[:, width // 2 - 1:width // 2 + 1] = 4
            for x, matrix, color in ((80, reference, 1), (240, estimate, 2)):
                points = 27 * (vertices @ matrix.T @ camera.T) + (x, 72)
                for i, j in edges:
                    a, b = points[i], points[j]
                    coords = np.linspace(a, b, int(np.linalg.norm(a - b)) + 2).round().astype(int)
                    stroke = 3 if vertices[i, 0] == vertices[j, 0] == 1 else color
                    image[coords[:, 1], coords[:, 0]] = stroke
                    image[coords[:, 1] + 1, coords[:, 0]] = stroke
            gif.write(b"\x21\xf9\x04\x04" + pack("<H", 8) + b"\x00\x00")
            gif.write(b"\x2c" + pack("<HHHHB", 0, 0, width, height, 0))
            gif.write(gif_lzw(image.tobytes()))
        gif.write(b"\x3b")


def render(out, pooled, trial, series):
    plot_summary(out, pooled)
    plot_tracking(out, trial, series)
    plot_rest(out, trial, series)
    cube_demo(out, trial, series)
