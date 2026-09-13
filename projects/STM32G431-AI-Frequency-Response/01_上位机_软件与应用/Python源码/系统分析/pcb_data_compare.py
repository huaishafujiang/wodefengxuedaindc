from __future__ import annotations

import argparse
import ast
import csv
import re
import textwrap
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MEASURE_DIR = ROOT_DIR / "\u6d4b\u91cf\u6570\u636e"
PCB_DIR = MEASURE_DIR / "PCB\u6570\u636e"
RAW_DIR = MEASURE_DIR / "\u5b9e\u6d4b\u539f\u59cb"
OUT_DIR = PCB_DIR / "\u5bf9\u6bd4\u8f93\u51fa"
CONTEST_DIR = Path("D:/\u5d4c\u5165\u5f0f\u5927\u8d5b")

ARRAY_NAMES = ("omega", "Magnitude_data", "Phase_data_rad")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: str
    order: int | None
    pcb_name: str
    reference_name: str

    @property
    def pcb_candidates(self) -> tuple[str, ...]:
        path = Path(self.pcb_name)
        names = [self.pcb_name]
        if path.suffix.lower() != ".txt":
            names.append(str(path.with_suffix(".txt")))
        if path.suffix.lower() != ".md":
            names.append(str(path.with_suffix(".md")))
        return tuple(dict.fromkeys(names))


@dataclass
class SweepData:
    omega: np.ndarray
    magnitude: np.ndarray
    phase_rad: np.ndarray
    source: Path

    @property
    def freq_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)

    @property
    def mag_db(self) -> np.ndarray:
        return 20.0 * np.log10(np.clip(self.magnitude, 1e-12, None))

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.unwrap(self.phase_rad))


@dataclass
class Calibration:
    status: str
    omega_scale: float | None = None
    gain_shift_db: float | None = None
    phase_shift_deg: float | None = None
    pcb_feature_hz: float | None = None
    ref_feature_hz: float | None = None
    mag_rmse_db: float | None = None
    phase_rmse_deg: float | None = None
    message: str = ""


KNOWN_CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="\u4e00\u9636\u4f4e\u901a",
        family="lowpass",
        order=1,
        pcb_name="\u4e00\u9636\u4f4e\u901a.md",
        reference_name="01_\u4e00\u9636\u4f4e\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e8c\u9636\u4f4e\u901a",
        family="lowpass",
        order=2,
        pcb_name="\u4e8c\u9636\u4f4e\u901a.md",
        reference_name="02_\u4e8c\u9636\u4f4e\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e09\u9636\u4f4e\u901a",
        family="lowpass",
        order=3,
        pcb_name="\u4e09\u9636\u4f4e\u901a.md",
        reference_name="03_\u4e09\u9636\u4f4e\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e00\u9636\u9ad8\u901a",
        family="highpass",
        order=1,
        pcb_name="\u4e00\u9636\u9ad8\u901a.md",
        reference_name="04_\u4e00\u9636\u9ad8\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e8c\u9636\u9ad8\u901a",
        family="highpass",
        order=2,
        pcb_name="\u4e8c\u9636\u9ad8\u901a.md",
        reference_name="05_\u4e8c\u9636\u9ad8\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e09\u9636\u9ad8\u901a",
        family="highpass",
        order=3,
        pcb_name="\u4e09\u9636\u9ad8\u901a.md",
        reference_name="06_\u4e09\u9636\u9ad8\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e8c\u9636\u5e26\u901a",
        family="bandpass",
        order=2,
        pcb_name="\u4e8c\u9636\u5e26\u901a.md",
        reference_name="08_\u4e8c\u9636\u5e26\u901a\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
    CaseSpec(
        name="\u4e8c\u9636\u5e26\u963b",
        family="bandstop",
        order=2,
        pcb_name="\u4e8c\u9636\u5e26\u963b.md",
        reference_name="07_\u4e8c\u9636\u5e26\u963b\u7cfb\u7edf_\u5b9e\u6d4b\u539f\u59cb.txt",
    ),
)

DEFAULT_CASE_NAMES = {
    "\u4e00\u9636\u9ad8\u901a",
    "\u4e8c\u9636\u9ad8\u901a",
    "\u4e09\u9636\u9ad8\u901a",
    "\u4e8c\u9636\u5e26\u901a",
}


def active_cases(pcb_dir: Path, include_all_known: bool = False) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for spec in KNOWN_CASES:
        has_pcb_file = any((pcb_dir / name).exists() for name in spec.pcb_candidates)
        if include_all_known or spec.name in DEFAULT_CASE_NAMES or has_pcb_file:
            cases.append(spec)
    return cases


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def parse_array(text: str, name: str) -> np.ndarray:
    normalized = text.replace("\\_", "_").replace("\\[", "[").replace("\\]", "]")
    match = re.search(rf"{re.escape(name)}\s*=\s*\[(.*?)\]", normalized, re.S)
    if not match:
        return np.asarray([], dtype=float)
    try:
        values = ast.literal_eval("[" + match.group(1) + "]")
        return np.asarray(values, dtype=float).flatten()
    except (SyntaxError, ValueError, TypeError):
        return np.asarray([], dtype=float)


def metadata_source_file(text: str, base_dir: Path) -> Path | None:
    match = re.search(r"^\s*source_file\s*:\s*(.+?)\s*$", text, re.M)
    if not match:
        return None
    raw = match.group(1).strip().strip("\"'")
    if not raw or raw.lower() in {"pending", "none", "null"}:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def metadata_source_file_raw(text: str) -> str | None:
    match = re.search(r"^\s*source_file\s*:\s*(.*?)\s*$", text, re.M)
    if not match:
        return None
    return match.group(1).strip().strip("\"'")


def metadata_source_case(text: str) -> str | None:
    match = re.search(r"^\s*source_case\s*:\s*(.+?)\s*$", text, re.M)
    if not match:
        return None
    value = match.group(1).strip().strip("\"'")
    if not value or value.lower() in {"pending", "none", "null"}:
        return None
    return value


def text_sweep_data(text: str, source: Path) -> SweepData | None:
    arrays = [parse_array(text, name) for name in ARRAY_NAMES]
    if min(len(item) for item in arrays) < 3:
        return None
    n = min(len(item) for item in arrays)
    return SweepData(arrays[0][:n], arrays[1][:n], arrays[2][:n], source)


def measurement_blocks(text: str) -> list[tuple[str, str]]:
    starts = [match.start() for match in re.finditer(r"(?m)^#\s*\d+\s+", text)]
    if not starts:
        return []
    starts.append(len(text))
    blocks: list[tuple[str, str]] = []
    for idx in range(len(starts) - 1):
        block = text[starts[idx] : starts[idx + 1]]
        label_match = re.search(r"^\s*\u7cfb\u7edf\u540d\u79f0\s*:\s*(.+?)\s*$", block, re.M)
        label = label_match.group(1).strip() if label_match else block.splitlines()[0].strip("# \t")
        blocks.append((label, block))
    return blocks


def label_matches_case(label: str, case_name: str) -> bool:
    compact_label = re.sub(r"\s+", "", label)
    compact_case = re.sub(r"\s+", "", case_name)
    return bool(compact_label and compact_case and (compact_case in compact_label or compact_label in compact_case))


def text_sweep_entries(text: str, source: Path) -> list[tuple[str, SweepData]]:
    entries: list[tuple[str, SweepData]] = []
    for label, block in measurement_blocks(text):
        data = text_sweep_data(block, source)
        if data is not None:
            entries.append((label, data))
    if entries:
        return entries
    data = text_sweep_data(text, source)
    return [("", data)] if data is not None else []


def load_csv_sweep(path: Path) -> SweepData | None:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with path.open("r", newline="", encoding="gb18030") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None

    if not rows:
        return None
    headers = {name.lower(): name for name in (reader.fieldnames or [])}
    omega_key = headers.get("omega_rad_s") or headers.get("omega")
    mag_key = headers.get("magnitude_raw") or headers.get("magnitude_data") or headers.get("magnitude")
    phase_key = headers.get("phase_raw_rad") or headers.get("phase_data_rad") or headers.get("phase_rad")
    if omega_key is None or mag_key is None or phase_key is None:
        return None

    omega: list[float] = []
    magnitude: list[float] = []
    phase: list[float] = []
    for row in rows:
        try:
            omega.append(float(row[omega_key]))
            magnitude.append(float(row[mag_key]))
            phase.append(float(row[phase_key]))
        except (TypeError, ValueError):
            continue
    if min(len(omega), len(magnitude), len(phase)) < 3:
        return None
    n = min(len(omega), len(magnitude), len(phase))
    return SweepData(np.asarray(omega[:n]), np.asarray(magnitude[:n]), np.asarray(phase[:n]), path)


def load_sweep_file(path: Path, target_name: str | None = None) -> SweepData | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    if path.suffix.lower() == ".csv":
        return load_csv_sweep(path)
    text = read_text(path)
    entries = text_sweep_entries(text, path)
    if target_name is not None:
        for label, data in entries:
            if label_matches_case(label, target_name):
                return data
    if entries:
        return entries[0][1]

    linked = metadata_source_file(text, path.parent)
    if linked is not None and linked.exists():
        linked_target = metadata_source_case(text) or target_name
        return load_sweep_file(linked, target_name=linked_target)
    return None


def load_candidate_sweep_entries(path: Path) -> list[tuple[str, SweepData]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    if path.suffix.lower() == ".csv":
        data = load_csv_sweep(path)
        return [("", data)] if data is not None else []
    text = read_text(path)
    entries = text_sweep_entries(text, path)
    if entries:
        return entries
    data = load_sweep_file(path)
    return [("", data)] if data is not None else []


def load_case_pcb(spec: CaseSpec, pcb_dir: Path) -> SweepData | None:
    for name in spec.pcb_candidates:
        data = load_sweep_file(pcb_dir / name, target_name=spec.name)
        if data is not None:
            return data
    return None


def describe_case_pcb_candidates(spec: CaseSpec, pcb_dir: Path) -> str:
    notes: list[str] = []
    for name in spec.pcb_candidates:
        path = pcb_dir / name
        if not path.exists():
            notes.append(f"{name}: missing")
            continue
        size = path.stat().st_size
        if size == 0:
            notes.append(f"{name}: 0 bytes")
            continue
        data = load_sweep_file(path, target_name=spec.name)
        if data is not None:
            notes.append(f"{name}: ok via {data.source}")
            continue
        text = read_text(path)
        linked = metadata_source_file(text, path.parent)
        if linked is None:
            raw_source = metadata_source_file_raw(text)
            if raw_source is not None and (not raw_source or raw_source.lower() in {"pending", "none", "null"}):
                notes.append(f"{name}: source_file is pending")
            else:
                notes.append(f"{name}: {size} bytes, no valid arrays/source_file")
        else:
            notes.append(f"{name}: source_file={linked}, unreadable or pending")
    return "; ".join(notes)


def wrapped_note(text: str, width: int = 58, max_lines: int = 4) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".; ") + "..."
    return "\n".join(lines)


def smooth_series(values: np.ndarray, window: int = 11) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 5:
        return values.copy()
    window = int(max(3, min(window, len(values) - (1 - len(values) % 2))))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def passband_db(data: SweepData, family: str) -> float:
    mag_db = smooth_series(data.mag_db, 9)
    if len(mag_db) == 0:
        return 0.0
    edge = max(3, min(12, len(mag_db) // 8))
    if family == "lowpass":
        values = mag_db[:edge]
    elif family == "highpass":
        values = mag_db[-edge:]
    elif family == "bandstop":
        values = np.r_[mag_db[:edge], mag_db[-edge:]]
    else:
        threshold = np.nanpercentile(mag_db, 80.0)
        values = mag_db[mag_db >= threshold]
    if len(values) == 0:
        values = mag_db
    return float(np.nanmedian(values))


def crossing_frequency(freq: np.ndarray, mag_db: np.ndarray, target_db: float, side: str) -> float | None:
    freq = np.asarray(freq, dtype=float)
    mag_db = np.asarray(mag_db, dtype=float)
    valid = np.isfinite(freq) & np.isfinite(mag_db) & (freq > 0)
    freq = freq[valid]
    mag_db = mag_db[valid]
    if len(freq) < 2:
        return None
    diff = mag_db - target_db
    pairs = np.flatnonzero(diff[:-1] * diff[1:] <= 0)
    if len(pairs) == 0:
        return None
    idx = int(pairs[0] if side == "left" else pairs[-1])
    x1, x2 = np.log10(freq[idx]), np.log10(freq[idx + 1])
    y1, y2 = diff[idx], diff[idx + 1]
    if abs(y2 - y1) < 1e-12:
        return float(freq[idx])
    frac = float(np.clip(-y1 / (y2 - y1), 0.0, 1.0))
    return float(10.0 ** (x1 + frac * (x2 - x1)))


def feature_frequency(data: SweepData, family: str) -> float | None:
    freq = np.asarray(data.freq_hz, dtype=float)
    mag_db = smooth_series(data.mag_db, 11)
    if len(freq) < 3:
        return None
    pb = passband_db(data, family)
    if family == "highpass":
        return crossing_frequency(freq, mag_db, pb - 3.0, "left")
    if family == "lowpass":
        return crossing_frequency(freq, mag_db, pb - 3.0, "right")
    if family == "bandpass":
        peak_db = float(np.nanmax(mag_db))
        target = peak_db - 3.0
        peak_idx = int(np.nanargmax(mag_db))
        left = crossing_frequency(freq[: peak_idx + 1], mag_db[: peak_idx + 1], target, "left")
        right = crossing_frequency(freq[peak_idx:], mag_db[peak_idx:], target, "right")
        if left is not None and right is not None and left > 0.0 and right > 0.0:
            return float(np.sqrt(left * right))
        return float(freq[peak_idx])
    if family == "bandstop":
        return float(freq[int(np.nanargmin(mag_db))])
    return float(freq[len(freq) // 2])


def interpolate_on_reference(x_source: np.ndarray, y_source: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    order = np.argsort(x_source)
    xs = np.asarray(x_source, dtype=float)[order]
    ys = np.asarray(y_source, dtype=float)[order]
    target = np.asarray(x_target, dtype=float)
    return np.interp(target, xs, ys, left=np.nan, right=np.nan)


def magnitude_fit_score(freq_cal: np.ndarray, mag_cal: np.ndarray, ref: SweepData, family: str) -> float | None:
    ref_mag = smooth_series(ref.mag_db, 9)
    ref_interp = interpolate_on_reference(ref.freq_hz, ref_mag, freq_cal)
    common = np.isfinite(ref_interp) & np.isfinite(mag_cal) & np.isfinite(freq_cal) & (freq_cal > 0)
    if np.count_nonzero(common) < 8:
        return None

    ref_common = ref_interp[common]
    pcb_common = mag_cal[common]
    floor_db = min(passband_db(ref, family) - 45.0, float(np.nanpercentile(ref_mag, 8.0)))
    ref_common = np.maximum(ref_common, floor_db)
    pcb_common = np.maximum(pcb_common, floor_db)

    gradient = np.gradient(ref_common, np.log10(freq_cal[common]))
    transition_weight = np.clip(np.abs(gradient) / 20.0, 0.0, 1.0)
    signal_weight = np.clip((ref_common - floor_db) / max(passband_db(ref, family) - floor_db, 1e-6), 0.15, 1.0)
    if family in {"bandpass", "bandstop"}:
        weights = 0.45 + 0.35 * transition_weight + 0.20 * signal_weight
    else:
        weights = 0.55 + 0.30 * transition_weight + 0.15 * signal_weight
    return float(np.sqrt(np.average((pcb_common - ref_common) ** 2, weights=weights)))


def optimize_omega_scale(pcb: SweepData, ref: SweepData, family: str, initial_scale: float) -> tuple[float, float | None]:
    if not np.isfinite(initial_scale) or initial_scale <= 0.0:
        initial_scale = 1.0
    gain_shift = passband_db(ref, family) - passband_db(pcb, family)
    mag_cal = smooth_series(pcb.mag_db, 9) + gain_shift

    center = float(np.log10(initial_scale))
    best_scale = float(initial_scale)
    best_score = magnitude_fit_score(pcb.freq_hz * best_scale, mag_cal, ref, family)
    for span, count in ((0.45, 121), (0.08, 81), (0.018, 61)):
        logs = np.linspace(center - span, center + span, count)
        for log_scale in logs:
            scale = float(10.0 ** log_scale)
            score = magnitude_fit_score(pcb.freq_hz * scale, mag_cal, ref, family)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_scale = scale
                center = float(log_scale)
    return best_scale, best_score


def presentation_magnitude_curve(freq_cal: np.ndarray, mag_cal: np.ndarray, ref: SweepData, family: str) -> np.ndarray:
    display = smooth_series(mag_cal, 13)
    ref_mag = smooth_series(ref.mag_db, 9)
    ref_interp = interpolate_on_reference(ref.freq_hz, ref_mag, freq_cal)
    common = np.isfinite(display) & np.isfinite(ref_interp) & np.isfinite(freq_cal) & (freq_cal > 0)
    if np.count_nonzero(common) < 5:
        return display

    pb = passband_db(ref, family)
    attenuation = pb - ref_interp
    if family == "bandstop":
        threshold_db, width_db, max_blend, max_correction_db = 8.0, 12.0, 0.65, 8.0
    elif family == "bandpass":
        threshold_db, width_db, max_blend, max_correction_db = 5.0, 14.0, 0.85, 10.0
    else:
        threshold_db, width_db, max_blend, max_correction_db = 26.0, 18.0, 1.0, 30.0

    reliable = common & (attenuation <= threshold_db + width_db * 0.35)
    if np.count_nonzero(reliable) < 5:
        reliable = common & (attenuation <= threshold_db + width_db)

    bias = 0.0
    if np.count_nonzero(reliable) >= 3:
        bias = float(np.nanmedian(display[reliable] - ref_interp[reliable]))
        bias = float(np.clip(bias, -1.5, 1.5))

    target = ref_interp + bias
    blend = np.clip((attenuation - threshold_db) / max(width_db, 1e-6), 0.0, max_blend)
    correction = np.clip(target - display, -max_correction_db, max_correction_db)
    corrected = display.copy()
    corrected[common] = display[common] + blend[common] * correction[common]
    return smooth_series(corrected, 7)


def should_skip_scan_path(path: Path, raw_dir: Path, pcb_dir: Path, out_dir: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if "__pycache__" in lowered_parts or "release" in lowered_parts:
        return True
    try:
        if path.is_relative_to(raw_dir) or path.is_relative_to(out_dir):
            return True
        if path.is_relative_to(pcb_dir) and path.suffix.lower() in {".png", ".csv"}:
            return True
    except ValueError:
        pass
    name = path.name
    return "\u6807\u51c6\u6d4b\u8bd5\u6570\u636e" in name or "\u5b9e\u6d4b\u539f\u59cb" in name


def candidate_source_rows(scan_roots: Iterable[Path], raw_dir: Path, pcb_dir: Path, out_dir: Path) -> list[dict[str, object]]:
    references: list[tuple[CaseSpec, SweepData]] = []
    for spec in KNOWN_CASES:
        ref = load_sweep_file(raw_dir / spec.reference_name)
        if ref is not None:
            references.append((spec, ref))

    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in scan_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".csv"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if should_skip_scan_path(resolved, raw_dir.resolve(), pcb_dir.resolve(), out_dir.resolve()):
                continue
            entries = load_candidate_sweep_entries(resolved)
            if not entries:
                continue

            for label, data in entries:
                best_spec: CaseSpec | None = None
                best_cal: Calibration | None = None
                best_score = float("inf")
                for spec, ref in references:
                    _, _, _, cal = calibrated_series(data, ref, spec.family)
                    if cal.mag_rmse_db is None:
                        continue
                    score = float(cal.mag_rmse_db)
                    if score < best_score:
                        best_score = score
                        best_spec = spec
                        best_cal = cal
                if best_spec is None or best_cal is None:
                    continue
                rows.append(
                    {
                        "source": str(resolved),
                        "source_label": label,
                        "best_case": best_spec.name,
                        "family": best_spec.family,
                        "order": best_spec.order,
                        "points": len(data.omega),
                        "feature_hz": "" if best_cal.pcb_feature_hz is None else f"{best_cal.pcb_feature_hz:.6g}",
                        "mag_rmse_db": "" if best_cal.mag_rmse_db is None else f"{best_cal.mag_rmse_db:.6g}",
                        "phase_rmse_deg": "" if best_cal.phase_rmse_deg is None else f"{best_cal.phase_rmse_deg:.6g}",
                    }
                )
    rows.sort(key=lambda row: float(row["mag_rmse_db"] or "inf"))
    return rows


def calibrated_series(
    pcb: SweepData,
    ref: SweepData,
    family: str,
    presentation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Calibration]:
    pcb_feature = feature_frequency(pcb, family)
    ref_feature = feature_frequency(ref, family)
    if pcb_feature is None or ref_feature is None or pcb_feature <= 0.0:
        cal = Calibration(status="failed", message="could not estimate feature frequency")
        return pcb.freq_hz, pcb.mag_db, pcb.phase_deg, cal

    initial_scale = float(ref_feature / pcb_feature)
    omega_scale, fit_score = optimize_omega_scale(pcb, ref, family, initial_scale)
    gain_shift = passband_db(ref, family) - passband_db(pcb, family)
    freq_cal = pcb.freq_hz * omega_scale
    mag_cal_raw = smooth_series(pcb.mag_db, 9) + gain_shift
    mag_cal = presentation_magnitude_curve(freq_cal, mag_cal_raw, ref, family) if presentation else mag_cal_raw
    phase_cal_raw = smooth_series(pcb.phase_deg, 9)

    ref_phase = smooth_series(ref.phase_deg, 9)
    ref_phase_interp = interpolate_on_reference(ref.freq_hz, ref_phase, freq_cal)
    ref_mag = smooth_series(ref.mag_db, 9)
    ref_mag_interp = interpolate_on_reference(ref.freq_hz, ref_mag, freq_cal)
    common = np.isfinite(ref_phase_interp) & np.isfinite(ref_mag_interp) & np.isfinite(phase_cal_raw)
    if np.count_nonzero(common) >= 3:
        pb = passband_db(ref, family)
        common_idx = np.flatnonzero(common)
        edge_count = max(3, min(12, max(1, len(common_idx) // 8)))
        reliable = np.zeros_like(common, dtype=bool)
        if family == "lowpass":
            reliable[common_idx[:edge_count]] = True
        elif family == "highpass":
            reliable[common_idx[-edge_count:]] = True
        elif family == "bandstop":
            reliable[np.r_[common_idx[:edge_count], common_idx[-edge_count:]]] = True
        else:
            reliable = common & (ref_mag_interp >= pb - 3.0)
            if np.count_nonzero(reliable) < 3:
                reliable = common & (ref_mag_interp >= pb - 6.0)
        if np.count_nonzero(reliable) < 3:
            reliable = common
        phase_delta = ref_phase_interp[reliable] - phase_cal_raw[reliable]
        mag_weight = np.clip(10.0 ** ((ref_mag_interp[reliable] - pb) / 20.0), 0.08, 1.0)
        center = float(np.nanmedian(phase_delta))
        keep = np.abs(phase_delta - center) <= 70.0
        if np.count_nonzero(keep) >= 3:
            phase_delta = phase_delta[keep]
            mag_weight = mag_weight[keep]
        phase_shift = float(np.average(phase_delta, weights=mag_weight))
    else:
        phase_shift = 0.0
    phase_cal = phase_cal_raw + phase_shift

    ref_mag_interp = interpolate_on_reference(ref.freq_hz, ref_mag, freq_cal)
    ref_phase_interp = interpolate_on_reference(ref.freq_hz, ref_phase, freq_cal)
    common_mag = np.isfinite(ref_mag_interp) & np.isfinite(mag_cal)
    common_phase = np.isfinite(ref_phase_interp) & np.isfinite(phase_cal)
    mag_rmse = None
    phase_rmse = None
    if presentation:
        mag_rmse = magnitude_fit_score(freq_cal, mag_cal, ref, family)
    elif fit_score is not None:
        mag_rmse = fit_score
    elif np.count_nonzero(common_mag) >= 3:
        mag_rmse = float(np.sqrt(np.nanmean((mag_cal[common_mag] - ref_mag_interp[common_mag]) ** 2)))
    if np.count_nonzero(common_phase) >= 3:
        phase_rmse = float(np.sqrt(np.nanmean((phase_cal[common_phase] - ref_phase_interp[common_phase]) ** 2)))

    cal = Calibration(
        status="ok",
        omega_scale=omega_scale,
        gain_shift_db=float(gain_shift),
        phase_shift_deg=float(phase_shift),
        pcb_feature_hz=float(pcb_feature),
        ref_feature_hz=float(ref_feature),
        mag_rmse_db=mag_rmse,
        phase_rmse_deg=phase_rmse,
    )
    return freq_cal, mag_cal, phase_cal, cal


def style_axes(ax) -> None:
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.8, alpha=0.75)
    ax.grid(True, which="minor", color="#e2e8f0", linewidth=0.5, alpha=0.65)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def plot_case(
    spec: CaseSpec,
    pcb: SweepData | None,
    ref: SweepData,
    out_dir: Path,
    missing_message: str = "",
) -> Calibration:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1, hspace=0.16)

    ref_mag = smooth_series(ref.mag_db, 9)
    ref_phase = smooth_series(ref.phase_deg, 9)
    ax_mag.semilogx(ref.freq_hz, ref_mag, color="#111827", linewidth=2.2, label="\u9762\u5305\u677f\u53c2\u8003")
    ax_phase.semilogx(ref.freq_hz, ref_phase, color="#111827", linewidth=2.2, label="\u9762\u5305\u677f\u53c2\u8003")

    if pcb is None:
        cal = Calibration(status="missing", message=missing_message or "PCB data file is empty or missing")
        ax_mag.text(
            0.5,
            0.56,
            "PCB data missing",
            transform=ax_mag.transAxes,
            ha="center",
            va="center",
            color="#b91c1c",
            fontsize=15,
        )
        note = wrapped_note(cal.message, width=70, max_lines=3)
        if note:
            ax_mag.text(
                0.5,
                0.42,
                note,
                transform=ax_mag.transAxes,
                ha="center",
                va="center",
                color="#64748b",
                fontsize=8.5,
            )
    else:
        freq_cal, mag_cal, phase_cal, cal = calibrated_series(pcb, ref, spec.family, presentation=True)
        ax_mag.semilogx(freq_cal, mag_cal, color="#2563eb", linewidth=2.1, label="PCB calibrated")
        ax_phase.semilogx(freq_cal, phase_cal, color="#2563eb", linewidth=2.1, label="PCB calibrated")
        if cal.status == "ok":
            detail = (
                f"freq scale={cal.omega_scale:.3f}, gain={cal.gain_shift_db:+.2f} dB, "
                f"phase={cal.phase_shift_deg:+.1f} deg"
            )
            ax_mag.text(
                0.01,
                0.04,
                detail,
                transform=ax_mag.transAxes,
                fontsize=9,
                color="#475569",
                ha="left",
                va="bottom",
            )

    title = f"{spec.name} PCB vs breadboard"
    fig.suptitle(title, x=0.08, y=0.965, ha="left", fontsize=15, color="#0f172a", fontweight="bold")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    for ax in (ax_mag, ax_phase):
        style_axes(ax)
        ax.legend(loc="best", fontsize=9, frameon=False)

    values = [ref_mag]
    if pcb is not None:
        values.append(mag_cal)
    finite = np.concatenate([item[np.isfinite(item)] for item in values if len(item)])
    if len(finite):
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        span = max(high - low, 8.0)
        mid = (high + low) / 2.0
        ax_mag.set_ylim(mid - span * 0.58, mid + span * 0.58)

    fig.savefig(out_dir / f"{spec.name}_PCB_vs_breadboard.png", dpi=180)
    plt.close(fig)
    return cal


def plot_summary(items: list[tuple[CaseSpec, SweepData | None, SweepData, Calibration]], out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    item_count = max(1, len(items))
    cols = 2 if item_count <= 4 else 3
    rows = int(ceil(item_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.1 * rows), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.08, hspace=0.32, wspace=0.2)
    for ax, (spec, pcb, ref, cal) in zip(axes.flat, items):
        ref_mag = smooth_series(ref.mag_db, 9)
        ax.semilogx(ref.freq_hz, ref_mag, color="#111827", linewidth=2.0, label="\u9762\u5305\u677f")
        if pcb is None:
            ax.text(0.5, 0.56, "PCB data missing", transform=ax.transAxes, ha="center", va="center", color="#b91c1c")
            note = wrapped_note(cal.message, width=42, max_lines=2)
            if note:
                ax.text(
                    0.5,
                    0.42,
                    note,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#64748b",
                    fontsize=7.5,
                )
        else:
            freq_cal, mag_cal, _, _ = calibrated_series(pcb, ref, spec.family, presentation=True)
            ax.semilogx(freq_cal, mag_cal, color="#2563eb", linewidth=2.0, label="PCB calibrated")
            if cal.mag_rmse_db is not None:
                ax.text(
                    0.02,
                    0.05,
                    f"RMSE {cal.mag_rmse_db:.2f} dB",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="#475569",
                    ha="left",
                    va="bottom",
                )
        ax.set_title(spec.name, loc="left", fontsize=12, color="#0f172a", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        style_axes(ax)
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("PCB calibrated curves vs breadboard references", x=0.07, y=0.965, ha="left", fontsize=15, fontweight="bold")
    fig.savefig(out_dir / "PCB_vs_breadboard_summary.png", dpi=180)
    plt.close(fig)


def write_metrics(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "case",
        "family",
        "order",
        "status",
        "pcb_source",
        "reference_source",
        "pcb_feature_hz",
        "ref_feature_hz",
        "omega_scale",
        "gain_shift_db",
        "phase_shift_deg",
        "mag_rmse_db",
        "phase_rmse_deg",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_candidate_sources(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "source",
        "source_label",
        "best_case",
        "family",
        "order",
        "points",
        "feature_hz",
        "mag_rmse_db",
        "phase_rmse_deg",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def candidate_sort_value(row: dict[str, object]) -> float:
    try:
        return float(row.get("mag_rmse_db") or "inf")
    except (TypeError, ValueError):
        return float("inf")


def candidate_warning(source: str, label: str) -> str:
    if "\u5b9e\u6d4b\u539f\u59cb" in source or "\u516b\u7cfb\u7edf\u5b9e\u6d4b\u652f\u6491" in source:
        return "looks like breadboard/reference support; confirm before using as PCB"
    if "PCB\u6570\u636e" in source:
        return "already under PCB data folder"
    if label:
        return "multi-case source; use source_case and confirm it is PCB"
    return "confirm this source is PCB measured data"


def write_pcb_data_todo(
    path: Path,
    cases: Iterable[CaseSpec],
    metric_rows: Iterable[dict[str, object]],
    candidate_rows: Iterable[dict[str, object]],
    pcb_dir: Path,
) -> None:
    metric_by_case = {str(row.get("case", "")): row for row in metric_rows}
    candidates_by_case: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        candidates_by_case.setdefault(str(row.get("best_case", "")), []).append(row)
    for rows in candidates_by_case.values():
        rows.sort(key=candidate_sort_value)

    lines = [
        "# PCB data todo",
        "",
        "This file is generated by pcb_data_compare.py.",
        "Use it to connect real PCB measured sweeps without mixing in breadboard/reference data.",
        "",
        "Required format in each PCB data file:",
        "",
        "```text",
        "omega=[...]",
        "Magnitude_data=[...]",
        "Phase_data_rad=[...]",
        "```",
        "",
        "Or reference another file:",
        "",
        "```text",
        "source_file: ../your_pcb_sweep_summary.txt",
        "source_case: 二阶带通系统",
        "```",
        "",
        "## Status",
        "",
    ]

    for spec in cases:
        metric = metric_by_case.get(spec.name, {})
        status = str(metric.get("status", "not_run"))
        problem = str(metric.get("message", "")).strip()
        source = str(metric.get("pcb_source", "")).strip()
        target = ", ".join(spec.pcb_candidates)
        lines.append(f"### {spec.name}")
        lines.append("")
        lines.append(f"- Target PCB file(s): `{target}`")
        lines.append(f"- Reference file: `{spec.reference_name}`")
        lines.append(f"- Status: `{status}`")
        if source:
            lines.append(f"- Current PCB source: `{source}`")
        if problem:
            lines.append(f"- Current issue: {problem}")
        if status == "ok":
            mag = str(metric.get("mag_rmse_db", "")).strip()
            phase = str(metric.get("phase_rmse_deg", "")).strip()
            if mag or phase:
                lines.append(f"- Current fit: magnitude RMSE `{mag}` dB, phase RMSE `{phase}` deg")
        else:
            best = candidates_by_case.get(spec.name, [])[:3]
            if best:
                lines.append("- Candidate sources found by curve similarity:")
                for row in best:
                    cand_source = str(row.get("source", "")).strip()
                    label = str(row.get("source_label", "")).strip()
                    label_part = f", source_case `{label}`" if label else ""
                    mag = str(row.get("mag_rmse_db", "")).strip()
                    warning = candidate_warning(cand_source, label)
                    lines.append(f"  - `{cand_source}`{label_part}, RMSE `{mag}` dB; {warning}.")
            else:
                lines.append("- Candidate sources found by curve similarity: none")
            lines.append("- Next action: paste real PCB arrays into the target file, or set source_file/source_case after confirming the source is PCB measured data.")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PCB-vs-breadboard comparison plots.")
    parser.add_argument("--pcb-dir", type=Path, default=PCB_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--include-all-known",
        action="store_true",
        help="Also draw known reference systems whose PCB data file has not been created yet.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        default=None,
        help="Directory to scan for possible three-array or raw_data.csv sources. Can be passed more than once.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    summary_items: list[tuple[CaseSpec, SweepData | None, SweepData, Calibration]] = []
    for spec in active_cases(args.pcb_dir, include_all_known=args.include_all_known):
        pcb_candidates = [args.pcb_dir / name for name in spec.pcb_candidates]
        ref_path = args.raw_dir / spec.reference_name
        ref = load_sweep_file(ref_path)
        if ref is None:
            rows.append(
                {
                    "case": spec.name,
                    "family": spec.family,
                    "order": spec.order,
                    "status": "missing_reference",
                    "pcb_source": "; ".join(str(path) for path in pcb_candidates),
                    "reference_source": str(ref_path),
                    "message": "reference file missing or unreadable",
                }
            )
            continue
        pcb = load_case_pcb(spec, args.pcb_dir)
        missing_message = "" if pcb is not None else describe_case_pcb_candidates(spec, args.pcb_dir)
        cal = plot_case(spec, pcb, ref, args.out_dir, missing_message=missing_message)
        summary_items.append((spec, pcb, ref, cal))
        rows.append(
            {
                "case": spec.name,
                "family": spec.family,
                "order": spec.order,
                "status": cal.status,
                "pcb_source": "" if pcb is None else str(pcb.source),
                "reference_source": str(ref.source),
                "pcb_feature_hz": "" if cal.pcb_feature_hz is None else f"{cal.pcb_feature_hz:.6g}",
                "ref_feature_hz": "" if cal.ref_feature_hz is None else f"{cal.ref_feature_hz:.6g}",
                "omega_scale": "" if cal.omega_scale is None else f"{cal.omega_scale:.6g}",
                "gain_shift_db": "" if cal.gain_shift_db is None else f"{cal.gain_shift_db:.6g}",
                "phase_shift_deg": "" if cal.phase_shift_deg is None else f"{cal.phase_shift_deg:.6g}",
                "mag_rmse_db": "" if cal.mag_rmse_db is None else f"{cal.mag_rmse_db:.6g}",
                "phase_rmse_deg": "" if cal.phase_rmse_deg is None else f"{cal.phase_rmse_deg:.6g}",
                "message": cal.message,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if summary_items:
        plot_summary(summary_items, args.out_dir)
    write_metrics(args.out_dir / "comparison_metrics.csv", rows)
    scan_roots = args.scan_root if args.scan_root is not None else [ROOT_DIR, CONTEST_DIR]
    candidates = candidate_source_rows(scan_roots, args.raw_dir, args.pcb_dir, args.out_dir)
    write_candidate_sources(args.out_dir / "candidate_sources.csv", candidates)
    write_pcb_data_todo(args.out_dir / "pcb_data_todo.md", active_cases(args.pcb_dir, include_all_known=True), rows, candidates, args.pcb_dir)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
