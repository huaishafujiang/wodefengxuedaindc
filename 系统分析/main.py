import csv
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import font_manager as fm

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

from serial_protocol import (
    DIAGNOSTIC_PATTERNS,
    NAME_PATTERNS,
    build_sweep_command,
    parse_array_from_line,
    parse_diagnostics_from_text,
    parse_measurement_frame_from_text,
    read_measurement_frame_from_serial as read_protocol_measurement_frame_from_serial,
    read_optional_diagnostics_from_serial,
)
from serial_transport import open_serial_transport, write_ascii_command

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


APP_TITLE = '稳频仪 - STM32G431 通用电路频响与奈奎斯特稳定性分析平台'

FILTER_TYPE_LABELS = {
    'lowpass': '\u4f4e\u901a\u6ee4\u6ce2\u5668',
    'highpass': '\u9ad8\u901a\u6ee4\u6ce2\u5668',
    'bandpass': '\u5e26\u901a\u6ee4\u6ce2\u5668',
    'bandstop': '\u5e26\u963b\u6ee4\u6ce2\u5668',
    'other': '\u672a\u8bc6\u522b\u6ee4\u6ce2\u5668',
}

MIN_RELIABLE_POINTS = 12
MIN_RELIABLE_DECADES_SIMPLE = 1.20
MIN_RELIABLE_DECADES_BAND = 1.50
MIN_EDGE_DECADES = 0.25
MIN_ORDER_SPAN_DB = 6.0

EXPECTED_CIRCUIT_CHOICES = [
    'Auto',
    '一阶低通',
    '二阶低通',
    '三阶低通',
    '一阶高通',
    '二阶高通',
    '三阶高通',
    '带通',
    '带阻',
]

EXPECTED_CIRCUIT_MAP = {
    '一阶低通': ('lowpass', 1),
    '二阶低通': ('lowpass', 2),
    '三阶低通': ('lowpass', 3),
    '一阶高通': ('highpass', 1),
    '二阶高通': ('highpass', 2),
    '三阶高通': ('highpass', 3),
    '带通': ('bandpass', None),
    '带阻': ('bandstop', None),
}

AUTO_COARSE_SWEEP = (100.0, 100000.0, 1000.0, 0.6)
AUTO_SWEEP_SEGMENTS = {
    'lowpass': [
        (100.0, 30000.0, 300.0, 0.6),
        (10000.0, 100000.0, 900.0, 0.6),
    ],
    'highpass': [
        (10.0, 5000.0, 50.0, 0.6),
        (1000.0, 50000.0, 300.0, 0.6),
    ],
    'bandpass': [
        (10.0, 5000.0, 50.0, 0.6),
        (1000.0, 30000.0, 300.0, 0.6),
        (30000.0, 100000.0, 1000.0, 0.6),
    ],
    'bandstop': [
        (10.0, 5000.0, 50.0, 0.6),
        (1000.0, 30000.0, 300.0, 0.6),
        (30000.0, 100000.0, 1000.0, 0.6),
    ],
    'other': [
        (10.0, 5000.0, 50.0, 0.6),
        (1000.0, 30000.0, 300.0, 0.6),
        (30000.0, 100000.0, 1000.0, 0.6),
    ],
}


def configure_plot_fonts():
    candidates = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
        'PingFang SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), 'DejaVu Sans')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return chosen


def list_serial_ports():
    if serial is None:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def summarize_measurement_diagnostics(
    diagnostics: dict[str, np.ndarray] | None,
    expected_len: int,
    filter_type: str,
) -> list[str]:
    if not diagnostics:
        return []

    notes: list[str] = []

    def get_array(name: str) -> np.ndarray | None:
        arr = diagnostics.get(name)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float).flatten()
        if len(arr) != expected_len:
            notes.append(f'Diagnostic array {name} length is {len(arr)}, expected {expected_len}; ignored for strict checks.')
            return None
        return arr

    input_rms = get_array('input_rms_v')
    output_rms = get_array('output_rms_v')
    input_dc = get_array('input_dc_v')
    output_dc = get_array('output_dc_v')
    clip_flags = get_array('clip_flags')
    valid_count = get_array('valid_capture_count')
    actual_freq = get_array('actual_freq_hz')
    adc_sample_rate = get_array('adc_sample_rate_hz')
    dac_sample_rate = get_array('dac_sample_rate_hz')
    mag_repeat_span = get_array('magnitude_repeat_span_db')
    phase_repeat_span = get_array('phase_repeat_span_rad')
    input_pp = get_array('input_pp_v')
    output_pp = get_array('output_pp_v')

    if input_rms is not None and output_rms is not None:
        notes.append(
            f'测量质量: 输入RMS中位数 {float(np.median(input_rms)):.4f} V，输出RMS中位数 {float(np.median(output_rms)):.4f} V。'
        )
        if float(np.median(input_rms)) < 0.03:
            notes.append('输入参考信号 RMS 偏小，幅相比值容易受 ADC 噪声影响；建议检查 PA0/DAC 参考点。')

    if input_pp is not None and output_pp is not None:
        notes.append(
            f'峰峰值: 输入P-P中位数 {float(np.median(input_pp)):.4f} V，输出P-P中位数 {float(np.median(output_pp)):.4f} V。'
        )

    if input_dc is not None and output_dc is not None:
        in_dc_med = float(np.median(input_dc))
        out_dc_med = float(np.median(output_dc))
        notes.append(f'测量偏置: PA0 DC≈{in_dc_med:.3f} V，PA1 DC≈{out_dc_med:.3f} V。')
        if out_dc_med < 0.35 or out_dc_med > 2.95:
            notes.append(
                'PA1 直流偏置不在 ADC 安全中间区，单电源 STM32 很可能测到削顶/半波信号；高通/带通输出节点应偏置到约 1.65 V。'
            )
        elif filter_type in ('highpass', 'bandpass') and abs(out_dc_med - 1.65) > 0.45:
            notes.append('PA1 输出偏置偏离 1.65 V 较多，高通/带通的幅相结果可能被单电源输入范围影响。')

    if clip_flags is not None:
        clipped_points = int(np.count_nonzero(clip_flags))
        if clipped_points:
            notes.append(
                f'ADC 削顶告警: {clipped_points}/{expected_len} 个频点触碰 0V 或 3.3V 边界；幅值和相位判据可能失真。'
            )

    if valid_count is not None and len(valid_count):
        min_valid = int(np.min(valid_count))
        if min_valid < 2:
            notes.append(f'有效采集次数最低为 {min_valid}，部分频点抗噪不足；建议复测或降低步进/幅度。')

    if mag_repeat_span is not None and len(mag_repeat_span):
        notes.append(
            f'重复采集幅值稳定性: 中位跨度 {float(np.median(mag_repeat_span)):.3f} dB，最大跨度 {float(np.max(mag_repeat_span)):.3f} dB。'
        )

    if phase_repeat_span is not None and len(phase_repeat_span):
        notes.append(
            f'重复采集相位稳定性: 中位跨度 {float(np.median(np.abs(phase_repeat_span))):.4f} rad，最大跨度 {float(np.max(np.abs(phase_repeat_span))):.4f} rad。'
        )

    if actual_freq is not None and adc_sample_rate is not None and dac_sample_rate is not None:
        notes.append(
            f'采样信息: 实际频率 {float(np.min(actual_freq)):.1f}-{float(np.max(actual_freq)):.1f} Hz，ADC采样率中位数 {float(np.median(adc_sample_rate)):.1f} Hz，DAC采样率中位数 {float(np.median(dac_sample_rate)):.1f} Hz。'
        )

    return notes


def aligned_diagnostics(
    diagnostics: dict[str, np.ndarray] | None,
    expected_len: int,
    sort_idx: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not diagnostics:
        return {}

    aligned: dict[str, np.ndarray] = {}
    for key, value in diagnostics.items():
        arr = np.asarray(value, dtype=float).flatten()
        if len(arr) < expected_len:
            continue
        arr = arr[:expected_len]
        if sort_idx is not None and len(sort_idx) == expected_len:
            arr = arr[sort_idx]
        aligned[key] = arr
    return aligned


def merge_measurement_frames(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    valid_frames = []
    for omega, magnitude, phase, diagnostics in frames:
        omega = np.asarray(omega, dtype=float).flatten()
        magnitude = np.asarray(magnitude, dtype=float).flatten()
        phase = np.asarray(phase, dtype=float).flatten()
        n = min(len(omega), len(magnitude), len(phase))
        if n < 3:
            continue
        valid_frames.append((omega[:n], magnitude[:n], phase[:n], aligned_diagnostics(diagnostics, n)))

    if not valid_frames:
        raise ValueError('自动扫频没有获得有效数据帧。')

    omega_all = np.concatenate([item[0] for item in valid_frames])
    mag_all = np.concatenate([item[1] for item in valid_frames])
    phase_all = np.concatenate([item[2] for item in valid_frames])
    diag_all: dict[str, list[np.ndarray]] = {key: [] for key in DIAGNOSTIC_PATTERNS}
    for omega, _, _, diagnostics in valid_frames:
        for key in DIAGNOSTIC_PATTERNS:
            arr = diagnostics.get(key)
            if arr is None or len(arr) != len(omega):
                diag_all[key].append(np.full(len(omega), np.nan, dtype=float))
            else:
                diag_all[key].append(np.asarray(arr, dtype=float))

    order = np.argsort(omega_all)
    omega_all = omega_all[order]
    mag_all = mag_all[order]
    phase_all = np.unwrap(phase_all[order])

    keep = np.ones(len(omega_all), dtype=bool)
    if len(omega_all) > 1:
        rel_tol = 2.0e-4
        last_kept = 0
        for i in range(1, len(omega_all)):
            if abs(omega_all[i] - omega_all[last_kept]) <= max(abs(omega_all[last_kept]) * rel_tol, 1e-9):
                keep[i] = False
            else:
                last_kept = i

    merged_diag: dict[str, np.ndarray] = {}
    for key, chunks in diag_all.items():
        if chunks:
            merged_diag[key] = np.concatenate(chunks)[order][keep]

    return omega_all[keep], mag_all[keep], phase_all[keep], merged_diag


def apply_measurement_quality_mask(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    notes: list[str] = []
    diag = aligned_diagnostics(diagnostics, len(omega))

    valid = (
        np.isfinite(omega)
        & np.isfinite(magnitude)
        & np.isfinite(phase_rad)
        & (omega > 0.0)
        & (magnitude > 0.0)
    )

    clip_flags = diag.get('clip_flags')
    if clip_flags is not None and len(clip_flags) == len(valid):
        valid &= clip_flags == 0

    valid_count = diag.get('valid_capture_count')
    if valid_count is not None and len(valid_count) == len(valid):
        valid &= valid_count >= 1

    input_rms = diag.get('input_rms_v')
    if input_rms is not None and len(input_rms) == len(valid):
        # A tiny reference channel makes Vout/Vin and phase very sensitive to ADC noise.
        valid &= input_rms >= 0.02

    removed = int(len(valid) - np.count_nonzero(valid))
    if removed <= 0:
        return omega, magnitude, phase_rad, diag, notes

    if np.count_nonzero(valid) < 5:
        notes.append(
            f'质量筛选发现 {removed} 个无效/削顶/参考过小频点，但剩余点数不足；保留原始数据并将判阶标为低可信。'
        )
        return omega, magnitude, phase_rad, diag, notes

    notes.append(f'质量筛选剔除了 {removed} 个无效/削顶/参考过小频点，再进行类型和阶数判断。')
    filtered_diag = {
        key: value[valid] if len(value) == len(valid) else value
        for key, value in diag.items()
    }
    return omega[valid], magnitude[valid], phase_rad[valid], filtered_diag, notes


def matlab_smooth(y: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y.copy()

    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, len(y) if len(y) % 2 == 1 else len(y) - 1)
    wl = max(wl, 5)

    po = min(int(polyorder), wl - 1)
    po = max(po, 1)

    if HAS_SCIPY:
        return savgol_filter(y, wl, po)

    # 备用：简单移动平均
    k = min(5, len(y))
    if k % 2 == 0:
        k -= 1
    half = k // 2
    padded = np.pad(y, (half, half), mode='edge')
    return np.convolve(padded, np.ones(k, dtype=float) / k, mode='valid')


def suppress_isolated_spikes(
    y: np.ndarray,
    *,
    domain: str = 'linear',
    min_threshold: float = 1.0,
    window_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).flatten()
    cleaned = y.copy()
    mask = np.zeros(len(y), dtype=bool)
    if len(y) < window_radius * 2 + 3:
        return cleaned, mask

    if domain == 'db':
        work = 20.0 * np.log10(np.clip(y, 1e-12, None))
    elif domain == 'deg':
        work = np.degrees(y)
    else:
        work = y.copy()

    clean_work = work.copy()
    for i in range(window_radius, len(work) - window_radius):
        local = np.concatenate((work[i - window_radius:i], work[i + 1:i + window_radius + 1]))
        local = local[np.isfinite(local)]
        if len(local) < 4 or not np.isfinite(work[i]):
            continue
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        threshold = max(float(min_threshold), 6.0 * 1.4826 * mad)
        if abs(float(work[i]) - median) > threshold:
            clean_work[i] = median
            mask[i] = True

    if domain == 'db':
        cleaned = np.power(10.0, clean_work / 20.0)
    elif domain == 'deg':
        cleaned = np.radians(clean_work)
    else:
        cleaned = clean_work

    return cleaned, mask


def nearest_index(x: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(x, dtype=float) - float(value))))


def interpolate_omega_at_level(
    omega: np.ndarray,
    y: np.ndarray,
    target: float,
) -> tuple[float, int]:
    omega = np.asarray(omega, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(y) & (omega > 0.0)
    omega = omega[valid]
    y = y[valid]

    if len(omega) < 2:
        if len(omega) == 1:
            return float(omega[0]), 0
        return 0.0, 0

    err = y - float(target)
    for i in range(len(omega) - 1):
        e0 = float(err[i])
        e1 = float(err[i + 1])
        if e0 == 0.0:
            return float(omega[i]), i
        if e0 * e1 <= 0.0:
            y0 = float(y[i])
            y1 = float(y[i + 1])
            if y1 == y0:
                return float(omega[i]), i
            frac = (float(target) - y0) / (y1 - y0)
            frac = float(np.clip(frac, 0.0, 1.0))
            log_w = np.log(float(omega[i])) + frac * (np.log(float(omega[i + 1])) - np.log(float(omega[i])))
            omega_est = float(np.exp(log_w))
            return omega_est, nearest_index(omega, omega_est)

    idx = nearest_index(err, 0.0)
    return float(omega[idx]), idx


def estimate_first_order_phase_cutoff(
    omega: np.ndarray,
    phase_deg: np.ndarray,
    target_phase_deg: float = -45.0,
) -> tuple[float | None, int | None]:
    omega = np.asarray(omega, dtype=float).flatten()
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(phase_deg) & (omega > 0.0)
    omega = omega[valid]
    phase_deg = phase_deg[valid]

    if len(omega) < 2:
        return None, None

    err = phase_deg - float(target_phase_deg)
    candidates: list[tuple[float, int]] = []
    for i in range(len(omega) - 1):
        e0 = float(err[i])
        e1 = float(err[i + 1])
        if e0 == 0.0:
            candidates.append((float(omega[i]), i))
            continue
        if e0 * e1 <= 0.0:
            p0 = float(phase_deg[i])
            p1 = float(phase_deg[i + 1])
            if p1 == p0:
                candidates.append((float(omega[i]), i))
                continue
            frac = (float(target_phase_deg) - p0) / (p1 - p0)
            frac = float(np.clip(frac, 0.0, 1.0))
            log_w = np.log(float(omega[i])) + frac * (np.log(float(omega[i + 1])) - np.log(float(omega[i])))
            omega_est = float(np.exp(log_w))
            candidates.append((omega_est, nearest_index(omega, omega_est)))

    if not candidates:
        return None, None

    # For a one-pole low-pass response there should be only one crossing.
    # If noise creates several crossings, use the one closest to the raw target phase.
    omega_est, idx = min(candidates, key=lambda item: abs(float(phase_deg[item[1]]) - float(target_phase_deg)))
    return omega_est, idx



def build_closed_nyquist_curve(h_jw: np.ndarray) -> np.ndarray:
    """Build an approximate closed Nyquist contour from positive-frequency data."""
    h_jw = np.asarray(h_jw, dtype=complex).flatten()
    if len(h_jw) <= 1:
        return h_jw

    # For real-coefficient systems: H(-jw) = conj(H(jw)).
    # We mirror the positive branch to approximate the full Nyquist contour.
    mirrored = np.conj(h_jw[-2:0:-1]) if len(h_jw) > 2 else np.array([], dtype=complex)
    return np.concatenate([h_jw, mirrored, h_jw[:1]])


def compute_winding_number_ccw(curve: np.ndarray, center: complex = -1.0 + 0.0j) -> int:
    """Counter-clockwise winding number around center."""
    z = np.asarray(curve, dtype=complex).flatten() - center
    if len(z) < 2:
        return 0

    # Avoid undefined phase exactly at center.
    close = np.isclose(z, 0.0, atol=1e-12)
    if np.any(close):
        z = z.copy()
        z[close] = 1e-12 + 0.0j

    angles = np.unwrap(np.angle(z))
    total_angle = float(angles[-1] - angles[0])
    return int(np.round(total_angle / (2.0 * np.pi)))


def estimate_system_order(phase_deg: np.ndarray, omega: np.ndarray) -> tuple[int, list[str]]:
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    omega = np.asarray(omega, dtype=float).flatten()
    notes: list[str] = []

    if len(phase_deg) < 3:
        return 1, notes

    tail_start = max(0, int(0.75 * len(phase_deg)))
    phase_tail = phase_deg[tail_start:]
    phase_tail_median = float(np.median(phase_tail))

    phase_drop = float(phase_deg[0] - phase_deg[-1])
    order_from_tail = int(np.clip(np.round(abs(phase_tail_median) / 90.0), 1, 12))
    order_from_drop = int(np.clip(np.round(abs(phase_drop) / 90.0), 1, 12))
    order_estimate = max(order_from_tail, order_from_drop)

    if len(omega) >= 6 and np.all(omega > 0):
        log_w = np.log10(omega)
        dphi_dlogw = np.gradient(phase_deg, log_w)
        slope_tail = float(np.median(dphi_dlogw[-max(3, len(dphi_dlogw) // 5):]))
        if slope_tail < -20.0:
            notes.append('高频相位仍在下降，当前估计阶次可能偏低。')

    return order_estimate, notes


def fit_lowpass_order_from_magnitude(
    omega: np.ndarray,
    magnitude: np.ndarray,
    max_order: int = 6,
) -> dict[str, object]:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(magnitude) & (omega > 0.0) & (magnitude > 0.0)
    omega = omega[valid]
    magnitude = magnitude[valid]

    if len(omega) < 5:
        return {
            'order': 1,
            'omega_c': float(omega[0]) if len(omega) else 0.0,
            'rmse_db': 0.0,
            'rankings': [],
        }

    low_count = max(3, min(7, len(magnitude)))
    low_gain = max(float(np.median(magnitude[:low_count])), 1e-12)
    mag_db = 20.0 * np.log10(np.clip(magnitude / low_gain, 1e-12, None))
    weights = np.linspace(0.7, 1.5, len(omega))

    omega_min = max(float(np.min(omega)), 1e-9)
    omega_max = max(float(np.max(omega)), omega_min * 1.01)
    omega_grid = np.geomspace(omega_min / 20.0, omega_max * 20.0, 320)

    rankings: list[dict[str, float]] = []
    for order in range(1, max_order + 1):
        best_rmse = float('inf')
        best_omega_c = float(omega_grid[0])
        for omega_c in omega_grid:
            pred_db = -10.0 * float(order) * np.log10(1.0 + (omega / float(omega_c)) ** 2)
            rmse = float(np.sqrt(np.average((mag_db - pred_db) ** 2, weights=weights)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_omega_c = float(omega_c)
        rankings.append({
            'order': float(order),
            'omega_c': best_omega_c,
            'rmse_db': best_rmse,
        })

    best_rmse = min(item['rmse_db'] for item in rankings)
    tolerance = max(best_rmse * 0.12, 0.08)
    selected = next(item for item in rankings if item['rmse_db'] <= best_rmse + tolerance)
    return {
        'order': int(selected['order']),
        'omega_c': float(selected['omega_c']),
        'rmse_db': float(selected['rmse_db']),
        'rankings': rankings,
    }


def assess_phase_quality(phase_deg: np.ndarray) -> dict[str, float]:
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    if len(phase_deg) < 5:
        return {
            'score': 0.0,
            'positive_ratio': 1.0,
            'roughness_deg': 180.0,
            'span_deg': 0.0,
        }

    diff = np.diff(phase_deg)
    positive_ratio = float(np.mean(diff > 8.0))
    large_positive_ratio = float(np.mean(diff > 18.0))
    roughness_deg = float(np.median(np.abs(np.diff(diff)))) if len(diff) >= 2 else 0.0
    span_deg = float(abs(phase_deg[-1] - phase_deg[0]))

    score = 1.0
    score -= 1.2 * positive_ratio
    score -= 0.8 * large_positive_ratio
    score -= min(roughness_deg / 45.0, 0.7)
    score = float(np.clip(score, 0.0, 1.0))

    return {
        'score': score,
        'positive_ratio': positive_ratio,
        'roughness_deg': roughness_deg,
        'span_deg': span_deg,
    }


def fit_lowpass_order_from_phase(
    omega: np.ndarray,
    phase_deg: np.ndarray,
    max_order: int = 6,
) -> dict[str, object]:
    omega = np.asarray(omega, dtype=float).flatten()
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(phase_deg) & (omega > 0.0)
    omega = omega[valid]
    phase_deg = phase_deg[valid]

    quality = assess_phase_quality(phase_deg)
    if len(omega) < 5:
        return {
            'order': 1,
            'omega_c': float(omega[0]) if len(omega) else 0.0,
            'rmse_deg': 0.0,
            'quality': quality,
            'rankings': [],
            'reliable': False,
        }

    low_count = max(3, min(7, len(phase_deg)))
    phase_offset_deg = float(np.median(phase_deg[:low_count]))
    weights = np.linspace(0.5, 1.6, len(omega))

    omega_min = max(float(np.min(omega)), 1e-9)
    omega_max = max(float(np.max(omega)), omega_min * 1.01)
    omega_grid = np.geomspace(omega_min / 20.0, omega_max * 20.0, 320)

    rankings: list[dict[str, float]] = []
    for order in range(1, max_order + 1):
        best_rmse = float('inf')
        best_omega_c = float(omega_grid[0])
        for omega_c in omega_grid:
            pred_phase_deg = phase_offset_deg - np.degrees(float(order) * np.arctan(omega / float(omega_c)))
            rmse = float(np.sqrt(np.average((phase_deg - pred_phase_deg) ** 2, weights=weights)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_omega_c = float(omega_c)
        rankings.append({
            'order': float(order),
            'omega_c': best_omega_c,
            'rmse_deg': best_rmse,
        })

    best_rmse = min(item['rmse_deg'] for item in rankings)
    tolerance = max(best_rmse * 0.12, 4.0)
    selected = next(item for item in rankings if item['rmse_deg'] <= best_rmse + tolerance)
    reliable = bool(
        quality['score'] >= 0.45
        and quality['span_deg'] >= 35.0
        and float(selected['rmse_deg']) <= 28.0
    )
    return {
        'order': int(selected['order']),
        'omega_c': float(selected['omega_c']),
        'rmse_deg': float(selected['rmse_deg']),
        'quality': quality,
        'rankings': rankings,
        'reliable': reliable,
    }


def fit_lowpass_order_from_phase_with_delay(
    omega: np.ndarray,
    phase_deg: np.ndarray,
    max_order: int = 4,
) -> dict[str, object]:
    omega = np.asarray(omega, dtype=float).flatten()
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(phase_deg) & (omega > 0.0)
    omega = omega[valid]
    phase_deg = phase_deg[valid]

    quality = assess_phase_quality(phase_deg)
    if len(omega) < 5:
        return {
            'order': 1,
            'omega_c': float(omega[0]) if len(omega) else 0.0,
            'rmse_deg': 0.0,
            'delay_us': 0.0,
            'quality': quality,
            'rankings': [],
            'reliable': False,
        }

    weights = np.linspace(0.5, 1.6, len(omega))
    sqrt_weights = np.sqrt(weights)
    omega_min = max(float(np.min(omega)), 1e-9)
    omega_max = max(float(np.max(omega)), omega_min * 1.01)
    omega_grid = np.geomspace(omega_min / 20.0, omega_max * 50.0, 420)

    rankings: list[dict[str, float]] = []
    for order in range(1, max_order + 1):
        best_rmse = float('inf')
        best_omega_c = float(omega_grid[0])
        best_offset_deg = 0.0
        best_delay_s = 0.0
        for omega_c in omega_grid:
            pole_phase = -np.degrees(float(order) * np.arctan(omega / float(omega_c)))
            # Pure measurement delay contributes -omega*tau radians. Fit it
            # separately so channel/timer delay is not counted as an extra pole.
            design = np.column_stack([
                np.ones_like(omega),
                -omega * (180.0 / np.pi),
            ])
            y = phase_deg - pole_phase
            weighted_design = design * sqrt_weights[:, None]
            weighted_y = y * sqrt_weights
            coef, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
            pred_phase_deg = pole_phase + design @ coef
            rmse = float(np.sqrt(np.average((phase_deg - pred_phase_deg) ** 2, weights=weights)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_omega_c = float(omega_c)
                best_offset_deg = float(coef[0])
                best_delay_s = float(coef[1])
        rankings.append({
            'order': float(order),
            'omega_c': best_omega_c,
            'rmse_deg': best_rmse,
            'offset_deg': best_offset_deg,
            'delay_us': best_delay_s * 1e6,
        })

    best_rmse = min(item['rmse_deg'] for item in rankings)
    tolerance = max(best_rmse * 0.12, 1.2)
    selected = next(item for item in rankings if item['rmse_deg'] <= best_rmse + tolerance)
    reliable = bool(
        quality['score'] >= 0.45
        and quality['span_deg'] >= 35.0
        and float(selected['rmse_deg']) <= 10.0
    )
    return {
        'order': int(selected['order']),
        'omega_c': float(selected['omega_c']),
        'rmse_deg': float(selected['rmse_deg']),
        'offset_deg': float(selected['offset_deg']),
        'delay_us': float(selected['delay_us']),
        'quality': quality,
        'rankings': rankings,
        'reliable': reliable,
    }


def estimate_system_order_from_fit(
    magnitude: np.ndarray,
    phase_deg: np.ndarray,
    omega: np.ndarray,
) -> tuple[int, list[str]]:
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    omega = np.asarray(omega, dtype=float).flatten()
    notes: list[str] = []

    if min(len(magnitude), len(phase_deg), len(omega)) < 5:
        return 1, notes

    mag_fit = fit_lowpass_order_from_magnitude(omega, magnitude)
    phase_fit = fit_lowpass_order_from_phase(omega, phase_deg)

    mag_order = int(mag_fit['order'])
    slope = compute_mag_slope_db_per_dec(omega, magnitude)
    slope_tail = float(np.median(slope[-max(3, len(slope) // 5):])) if len(slope) else 0.0
    slope_order = estimate_order_from_slope(slope_tail)
    mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
    mag_span_db = float(np.max(mag_db) - np.min(mag_db)) if len(mag_db) else 0.0
    order_estimate = max(mag_order, slope_order, 1)
    delay_artifact_guarded = False
    if slope_order:
        notes.append(f'低通高频尾部斜率支持 {slope_order} 阶；斜率约 {slope_tail:.1f} dB/dec。')

    if phase_fit['reliable']:
        phase_order = int(phase_fit['order'])
        delay_phase_fit = fit_lowpass_order_from_phase_with_delay(
            omega,
            phase_deg,
            max_order=max(4, phase_order),
        )
        delay_phase_order = int(delay_phase_fit['order'])
        phase_support_order = phase_order
        if delay_phase_fit['reliable'] and delay_phase_order < phase_order:
            phase_support_order = delay_phase_order
            delay_artifact_guarded = True
            notes.append(
                f'原始相位拟合像 {phase_order} 阶，但加入约 {float(delay_phase_fit["delay_us"]):.2f} us 测量延迟后可由 {delay_phase_order} 阶解释；不把这部分额外相位计入物理阶数。'
            )
        if (
            mag_order == 2
            and phase_order >= 2
            and delay_phase_fit['reliable']
            and delay_phase_order == 1
            and (not slope_order or slope_order <= 1)
            and mag_span_db < 18.0
        ):
            order_estimate = 1
            delay_artifact_guarded = True
            notes.append(
                f'相位可由一阶低通叠加 {float(delay_phase_fit["delay_us"]):.2f} us 测量延迟解释；幅值跨度为 {mag_span_db:.1f} dB，二阶模板只是有限扫频范围造成的拟合假象。'
            )
        elif mag_order > phase_support_order and slope_order and slope_order <= phase_support_order:
            order_estimate = phase_support_order
            notes.append(
                f'延迟修正后的相位与尾部斜率都只支持 {phase_support_order} 阶；不用更高的幅值模板阶次。'
            )
        elif phase_support_order > order_estimate:
            notes.append(
                f'原始相位拟合看起来像 {phase_order} 阶，但低通相位很容易被运放/采样延迟额外拉低；物理阶数保持在幅值模板和尾部斜率共同支持的 {order_estimate} 阶。'
            )
        if delay_artifact_guarded:
            notes.append(
                f'未补偿延迟的相位拟合会偏向 {phase_order} 阶，但已被延迟补偿相位拟合否决；最终按 {order_estimate} 阶低通判定。'
            )
        else:
            notes.append(
                f'低通幅值模板拟合优先 {mag_order} 阶；幅值 RMSE = {float(mag_fit["rmse_db"]):.2f} dB。'
            )
            notes.append(
                f'相位拟合支持 {phase_order} 阶；相位 RMSE = {float(phase_fit["rmse_deg"]):.1f}°。'
            )
    else:
        quality = phase_fit['quality']
        notes.append(
            f'低通幅值模板拟合优先 {mag_order} 阶；幅值 RMSE = {float(mag_fit["rmse_db"]):.2f} dB。'
        )
        notes.append('相位曲线噪声较大或不完全单调，因此阶数主要由幅值拟合和尾部斜率判断。')
        if quality['span_deg'] < 170.0 and order_estimate >= 3:
            notes.append('Sweep upper limit may still be too low for the phase tail of a higher-order low-pass response.')

    if len(omega) >= 6 and np.all(omega > 0):
        log_w = np.log10(omega)
        dphi_dlogw = np.gradient(phase_deg, log_w)
        phase_slope_tail = float(np.median(dphi_dlogw[-max(3, len(dphi_dlogw) // 5):]))
        if phase_slope_tail < -20.0 and not delay_artifact_guarded:
            notes.append('High-frequency phase is still falling, which is more consistent with a response above first order.')

    return int(np.clip(order_estimate, 1, 12)), notes


def fit_highpass_order_from_magnitude(
    omega: np.ndarray,
    magnitude: np.ndarray,
    max_order: int = 6,
) -> dict[str, object]:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(magnitude) & (omega > 0.0) & (magnitude > 0.0)
    omega = omega[valid]
    magnitude = magnitude[valid]

    if len(omega) < 5:
        return {
            'order': 1,
            'omega_c': float(omega[-1]) if len(omega) else 0.0,
            'rmse_db': 0.0,
            'rankings': [],
        }

    high_count = max(3, min(7, len(magnitude)))
    high_gain = max(float(np.median(magnitude[-high_count:])), 1e-12)
    mag_db = 20.0 * np.log10(np.clip(magnitude / high_gain, 1e-12, None))
    weights = np.linspace(1.5, 0.7, len(omega))

    omega_min = max(float(np.min(omega)), 1e-9)
    omega_max = max(float(np.max(omega)), omega_min * 1.01)
    omega_grid = np.geomspace(omega_min / 20.0, omega_max * 20.0, 320)

    rankings: list[dict[str, float]] = []
    for order in range(1, max_order + 1):
        best_rmse = float('inf')
        best_omega_c = float(omega_grid[0])
        for omega_c in omega_grid:
            pred_db = -10.0 * float(order) * np.log10(1.0 + (float(omega_c) / omega) ** 2)
            rmse = float(np.sqrt(np.average((mag_db - pred_db) ** 2, weights=weights)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_omega_c = float(omega_c)
        rankings.append({
            'order': float(order),
            'omega_c': best_omega_c,
            'rmse_db': best_rmse,
        })

    best_rmse = min(item['rmse_db'] for item in rankings)
    tolerance = max(best_rmse * 0.12, 0.08)
    selected = next(item for item in rankings if item['rmse_db'] <= best_rmse + tolerance)
    return {
        'order': int(selected['order']),
        'omega_c': float(selected['omega_c']),
        'rmse_db': float(selected['rmse_db']),
        'rankings': rankings,
    }


def frequency_span_decades(omega: np.ndarray) -> float:
    omega = np.asarray(omega, dtype=float).flatten()
    valid = omega[np.isfinite(omega) & (omega > 0.0)]
    if len(valid) < 2:
        return 0.0
    return float(np.log10(float(np.max(valid)) / max(float(np.min(valid)), 1e-12)))


def edge_decades(omega: np.ndarray, pivot_omega: float) -> tuple[float, float]:
    omega = np.asarray(omega, dtype=float).flatten()
    valid = omega[np.isfinite(omega) & (omega > 0.0)]
    if len(valid) < 2 or pivot_omega <= 0.0:
        return 0.0, 0.0
    left = float(np.log10(max(float(pivot_omega), 1e-12) / max(float(np.min(valid)), 1e-12)))
    right = float(np.log10(max(float(np.max(valid)), 1e-12) / max(float(pivot_omega), 1e-12)))
    return max(left, 0.0), max(right, 0.0)


def refine_filter_order_with_template(
    filter_type: str,
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_deg: np.ndarray,
    initial_order: int,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    initial_order = int(np.clip(initial_order, 1, 12))

    if filter_type == 'lowpass':
        order, fit_notes = estimate_system_order_from_fit(magnitude, phase_deg, omega)
        notes.extend(fit_notes)
        return int(np.clip(order, 1, 12)), notes

    if filter_type == 'highpass':
        mag_fit = fit_highpass_order_from_magnitude(omega, magnitude)
        order = int(mag_fit['order'])
        notes.append(
            f'High-pass magnitude fit prefers order {order}; magnitude RMSE = {float(mag_fit["rmse_db"]):.2f} dB.'
        )
        phase_order, phase_reliable = estimate_order_from_phase_span(phase_deg)
        if phase_reliable:
            if phase_order < order:
                order = max(phase_order, 1)
                notes.append(f'Phase span only supports order {phase_order}; using the more conservative high-pass order.')
            elif phase_order > order:
                notes.append(
                    f'原始相位跨度看起来像 {phase_order} 阶，但高通高频相位容易叠加测量延迟；不单靠相位把 {order} 阶上调到更高阶。'
                )
            else:
                notes.append(f'Phase span is consistent with order {phase_order}.')
        else:
            notes.append('Phase trace is not reliable enough to raise the high-pass order.')
        return int(np.clip(order, 1, 12)), notes

    if filter_type == 'bandpass':
        peak_idx = int(np.argmax(magnitude))
        left_order = 1
        right_order = 1
        if peak_idx >= 4:
            left_fit = fit_highpass_order_from_magnitude(omega[:peak_idx + 1], magnitude[:peak_idx + 1], max_order=4)
            left_order = int(left_fit['order'])
        if len(omega) - peak_idx >= 5:
            right_fit = fit_lowpass_order_from_magnitude(omega[peak_idx:], magnitude[peak_idx:], max_order=4)
            right_order = int(right_fit['order'])
        order = int(np.clip(left_order + right_order, 2, 8))
        notes.append(f'Band-pass template fit estimates left edge order {left_order} + right edge order {right_order}.')
        return order, notes

    return initial_order, notes


def assess_identification_reliability(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_deg: np.ndarray,
    filter_info: dict[str, object],
    diagnostics: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()

    filter_type = str(filter_info.get('filter_type', 'other'))
    notes: list[str] = []
    point_count = int(min(len(omega), len(magnitude), len(phase_deg)))
    decades = frequency_span_decades(omega[:point_count])
    mag_db = 20.0 * np.log10(np.clip(magnitude[:point_count], 1e-12, None))
    mag_span_db = float(np.max(mag_db) - np.min(mag_db)) if point_count else 0.0
    phase_quality = assess_phase_quality(phase_deg[:point_count])

    required_decades = MIN_RELIABLE_DECADES_BAND if filter_type in ('bandpass', 'bandstop') else MIN_RELIABLE_DECADES_SIMPLE
    confidence = 0.0
    confidence += 0.18 * min(point_count / 30.0, 1.0)
    confidence += 0.24 * min(decades / required_decades, 1.0) if required_decades > 0 else 0.0
    confidence += 0.18 * min(mag_span_db / 12.0, 1.0)
    confidence += 0.18 * float(phase_quality['score'])

    edge_ok = True
    if filter_type in ('lowpass', 'highpass'):
        left_span, right_span = edge_decades(omega, float(filter_info.get('primary_omega', 0.0)))
        relaxed_single_pole_edge = False
        if filter_type == 'lowpass':
            edge_ok = left_span >= MIN_EDGE_DECADES and right_span >= 0.60
            order_hint = int(filter_info.get('order_estimate', 1) or 1)
            if (
                not edge_ok
                and order_hint <= 1
                and left_span >= MIN_EDGE_DECADES
                and right_span >= 0.45
                and mag_span_db >= 10.0
                and float(phase_quality['score']) >= 0.75
            ):
                edge_ok = True
                relaxed_single_pole_edge = True
        else:
            edge_ok = left_span >= 0.60 and right_span >= MIN_EDGE_DECADES
        confidence += 0.18 if edge_ok else 0.06
        if relaxed_single_pole_edge:
            notes.append(
                f'高频侧覆盖 {right_span:.2f} decade，略低于严格高阶判别标准；但幅值跨度和斜率已足够确认一阶低通，若要更精确排除高阶可把最高频率提高到约 10fc。'
            )
        elif not edge_ok:
            notes.append(
                f'截止点两侧覆盖不足：左侧 {left_span:.2f} decade，右侧 {right_span:.2f} decade；阶数只能作为疑似结果。'
            )
    elif filter_type in ('bandpass', 'bandstop'):
        left_cut = float(filter_info.get('left_cutoff_omega', 0.0) or 0.0)
        right_cut = float(filter_info.get('right_cutoff_omega', 0.0) or 0.0)
        low_side, _ = edge_decades(omega, left_cut)
        _, high_side = edge_decades(omega, right_cut)
        middle = float(np.log10(right_cut / left_cut)) if right_cut > left_cut > 0.0 else 0.0
        edge_ok = low_side >= MIN_EDGE_DECADES and high_side >= MIN_EDGE_DECADES and middle >= 0.15
        confidence += 0.18 if edge_ok else 0.06
        if not edge_ok:
            notes.append(
                f'带型滤波器需要左右边沿都被扫到：低侧 {low_side:.2f} decade，中间 {middle:.2f} decade，高侧 {high_side:.2f} decade。'
            )
    else:
        edge_ok = False
        confidence += 0.02

    diag = aligned_diagnostics(diagnostics, point_count)
    has_diag_problem = False
    clip_flags = diag.get('clip_flags')
    if clip_flags is not None and len(clip_flags):
        clipped = int(np.count_nonzero(clip_flags))
        if clipped:
            has_diag_problem = True
            notes.append(f'仍有 {clipped} 个频点带 ADC 削顶标志，阶数判断降级为疑似。')

    valid_count = diag.get('valid_capture_count')
    if valid_count is not None and len(valid_count) and float(np.min(valid_count)) < 2:
        has_diag_problem = True
        notes.append('部分频点有效重复采集次数少于 2 次，建议复测后再确认阶数。')

    input_rms = diag.get('input_rms_v')
    if input_rms is not None and len(input_rms) and float(np.median(input_rms)) < 0.03:
        has_diag_problem = True
        notes.append('输入参考 RMS 中位数低于 30 mV，Vout/Vin 对噪声过敏，阶数判断降级。')

    mag_repeat_span = diag.get('magnitude_repeat_span_db')
    if mag_repeat_span is not None and len(mag_repeat_span):
        mag_repeat_span = np.asarray(mag_repeat_span[:point_count], dtype=float)
        if float(np.nanmedian(mag_repeat_span)) > 0.8 or float(np.nanmax(mag_repeat_span)) > 2.5:
            has_diag_problem = True
            notes.append('重复采集幅值离散过大，说明该次扫频不稳定；建议降低幅度或复测。')

    phase_repeat_span = diag.get('phase_repeat_span_rad')
    if phase_repeat_span is not None and len(phase_repeat_span):
        phase_repeat_span = np.abs(np.asarray(phase_repeat_span[:point_count], dtype=float))
        if float(np.nanmedian(phase_repeat_span)) > 0.12 or float(np.nanmax(phase_repeat_span)) > 0.45:
            has_diag_problem = True
            notes.append('重复采集相位离散过大，阶数和相位裕度判断降级为疑似。')

    if filter_type == 'highpass':
        primary_omega = float(filter_info.get('primary_omega', 0.0) or 0.0)
        order_hint = int(filter_info.get('order_estimate', 1) or 1)
        if primary_omega > 0.0 and point_count:
            low_mask = omega[:point_count] <= primary_omega * 0.10
            low_point_count = int(np.count_nonzero(low_mask))
            # High-order HPF evidence lives far below cutoff. A linear wide sweep can
            # leave too few low-side points, making a 2nd/3rd-order circuit look 1st-order.
            if order_hint <= 1 and mag_span_db >= 35.0 and low_point_count < 8:
                has_diag_problem = True
                notes.append(
                    f'高通低频侧点数不足：0.1fc 以下只有 {low_point_count} 个点；'
                    '三阶高通建议先用低频加密扫频（例如 50-5000 Hz、步进 50 Hz）再判阶。'
                )

            output_rms = diag.get('output_rms_v')
            if output_rms is not None and len(output_rms):
                output_rms = np.asarray(output_rms[:point_count], dtype=float)
                very_low_mask = omega[:point_count] <= primary_omega * 0.20
                low_output = output_rms[very_low_mask & np.isfinite(output_rms)]
                if len(low_output) and float(np.median(low_output)) < 0.003:
                    has_diag_problem = True
                    notes.append(
                        '高通低频输出 RMS 已接近 STM32 ADC 噪声/量化底，'
                        '高阶斜率可能被噪声地板压平成一阶；请提高幅度、降低截止频率或分段加密扫频。'
                    )

    confidence = float(np.clip(confidence, 0.0, 1.0))
    reliable = bool(
        filter_type != 'other'
        and point_count >= MIN_RELIABLE_POINTS
        and decades >= required_decades
        and mag_span_db >= MIN_ORDER_SPAN_DB
        and edge_ok
        and not has_diag_problem
        and confidence >= 0.60
    )

    notes.append(
        f'判阶质量: 点数 {point_count}，扫频跨度 {decades:.2f} decade，幅值跨度 {mag_span_db:.1f} dB，相位质量 {float(phase_quality["score"]):.2f}。'
    )
    if decades < required_decades:
        notes.append(
            f'扫频跨度不足以可靠判阶：当前 {decades:.2f} decade，建议至少 {required_decades:.2f} decade，并覆盖截止点两侧。'
        )
    if mag_span_db < MIN_ORDER_SPAN_DB:
        notes.append(
            f'幅值变化只有 {mag_span_db:.1f} dB，足够找局部截止点但不足以可靠区分一阶/二阶/高阶。'
        )
    if not reliable:
        notes.append('本次阶数结论已标记为“疑似/需复测”，不会作为确定阶数使用。')

    summary = '可靠' if reliable else '疑似/需复测'
    return {
        'reliable': reliable,
        'confidence': confidence,
        'summary': summary,
        'notes': notes,
    }


def format_system_order_label(order_estimate: int) -> str:
    if order_estimate <= 1:
        return '\u4e00\u9636\u7cfb\u7edf'
    if order_estimate == 2:
        return '\u4e8c\u9636\u7cfb\u7edf'
    if order_estimate == 3:
        return '\u4e09\u9636\u7cfb\u7edf'
    return f'\u9ad8\u9636\u7cfb\u7edf\uff08\u4f30\u8ba1\u9636\u6b21 {order_estimate}\uff09'


def db_ratio(num: float, den: float) -> float:
    num = max(float(num), 1e-12)
    den = max(float(den), 1e-12)
    return float(20.0 * np.log10(num / den))


def compute_mag_slope_db_per_dec(omega: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    if len(omega) != len(magnitude) or len(omega) < 3:
        return np.zeros_like(magnitude, dtype=float)

    omega_safe = np.clip(omega.copy(), 1e-12, None)
    for i in range(1, len(omega_safe)):
        if omega_safe[i] <= omega_safe[i - 1]:
            omega_safe[i] = omega_safe[i - 1] * (1.0 + 1e-9)

    mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
    slope = np.gradient(mag_db, np.log10(omega_safe))
    slope = np.asarray(slope, dtype=float)
    slope[~np.isfinite(slope)] = 0.0
    return slope


def estimate_order_from_slope(slope_db_per_dec: float, max_order: int = 6) -> int:
    slope_abs = abs(float(slope_db_per_dec))
    if slope_abs < 10.0:
        return 0
    return int(np.clip(np.round(slope_abs / 20.0), 1, max_order))


def estimate_order_from_phase_span(phase_deg: np.ndarray, max_order: int = 6) -> tuple[int, bool]:
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()
    if len(phase_deg) < 5:
        return 0, False

    diff = np.diff(phase_deg)
    positive_ratio = float(np.mean(diff > 8.0)) if len(diff) else 1.0
    roughness = float(np.median(np.abs(np.diff(diff)))) if len(diff) >= 2 else 0.0
    phase_span = float(abs(phase_deg[-1] - phase_deg[0]))
    reliable = positive_ratio <= 0.12 and roughness <= 18.0 and phase_span >= 60.0
    if not reliable:
        return 0, False

    return int(np.clip(np.round(phase_span / 90.0), 1, max_order)), True


def estimate_band_order_conservatively(
    head_order: int,
    tail_order: int,
    phase_deg: np.ndarray,
    *,
    max_side_order: int = 2,
    max_total_without_phase: int = 3,
    max_total_order: int = 4,
) -> tuple[int, int, int, list[str]]:
    """Estimate band-pass/stop order without trusting noisy edge slopes too much."""
    raw_left = max(int(head_order), 0)
    raw_right = max(int(tail_order), 0)
    raw_total = max(2, raw_left + raw_right)

    left_order = int(np.clip(raw_left if raw_left else 1, 1, max_side_order))
    right_order = int(np.clip(raw_right if raw_right else 1, 1, max_side_order))
    slope_total = int(np.clip(left_order + right_order, 2, max_total_order))

    phase_order, phase_reliable = estimate_order_from_phase_span(phase_deg, max_order=max_total_order)
    notes: list[str] = [
        f'Band edge slopes: left≈{raw_left or 1} order, right≈{raw_right or 1} order before confidence limiting.'
    ]

    if phase_reliable:
        phase_total = int(np.clip(max(2, phase_order), 2, max_total_order))
        order_estimate = int(np.clip(min(slope_total, phase_total), 2, max_total_order))
        notes.append(
            f'Band order is limited by monotonic phase span to avoid counting breadboard/high-frequency roll-off as real poles; using order {order_estimate}.'
        )
    else:
        order_estimate = int(np.clip(min(slope_total, max_total_without_phase), 2, max_total_order))
        notes.append(
            f'Band phase is not reliable enough for high-order detection; using conservative order {order_estimate}.'
        )

    if raw_total > order_estimate:
        notes.append(
            f'Raw edge slopes could imply order {raw_total}, but this is treated as an upper-bound hint, not the displayed physical order.'
        )

    return order_estimate, left_order, right_order, notes


def find_level_crossings(
    omega: np.ndarray,
    y: np.ndarray,
    target: float,
) -> list[tuple[float, int]]:
    omega = np.asarray(omega, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if len(omega) < 2 or len(y) < 2:
        return []

    crossings: list[tuple[float, int]] = []
    for i in range(len(omega) - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue

        e0 = y0 - float(target)
        e1 = y1 - float(target)
        if e0 == 0.0:
            crossings.append((float(omega[i]), i))
            continue
        if e0 * e1 > 0.0 and e1 != 0.0:
            continue

        if y1 == y0:
            omega_est = float(omega[i])
        else:
            frac = float(np.clip((float(target) - y0) / (y1 - y0), 0.0, 1.0))
            log_w = np.log(max(float(omega[i]), 1e-12)) + frac * (
                np.log(max(float(omega[i + 1]), 1e-12)) - np.log(max(float(omega[i]), 1e-12))
            )
            omega_est = float(np.exp(log_w))
        crossings.append((omega_est, nearest_index(omega, omega_est)))

    return crossings


def format_filter_system_label(filter_type: str, order_estimate: int, reliable: bool = True) -> str:
    order_prefix = {
        1: '\u4e00\u9636',
        2: '\u4e8c\u9636',
        3: '\u4e09\u9636',
    }.get(int(order_estimate), f'{int(order_estimate)}\u9636')
    filter_label = FILTER_TYPE_LABELS.get(filter_type, FILTER_TYPE_LABELS['other'])

    if filter_type == 'other':
        return f'{filter_label}\uff08\u65e0\u6cd5\u53ef\u9760\u5224\u9636\uff09'

    label = f'{order_prefix}{filter_label}'
    if reliable:
        return label
    return f'\u7591\u4f3c{label}\uff08\u9700\u590d\u6d4b\u786e\u8ba4\uff09'


def evaluate_expected_circuit(result: 'SystemAnalysisResult', expected: str) -> str:
    if not expected or expected == 'Auto':
        return '未指定期望电路，按自动识别结果验收。'

    target = EXPECTED_CIRCUIT_MAP.get(expected)
    if target is None:
        return f'期望电路“{expected}”不在内置八类列表中。'

    expected_type, expected_order = target
    type_ok = result.filter_type == expected_type
    order_ok = expected_order is None or result.order_estimate == expected_order
    reliability_text = '可靠' if result.order_reliable else '疑似'
    if type_ok and order_ok:
        return f'期望电路匹配：{expected}；本次识别为{result.system_order}（{reliability_text}）。'

    expected_label = FILTER_TYPE_LABELS.get(expected_type, expected_type)
    if expected_order is not None:
        expected_label = f'{expected_order}阶{expected_label}'
    measured_label = f'{result.order_estimate}阶{FILTER_TYPE_LABELS.get(result.filter_type, result.filter_type)}'
    return f'期望电路不匹配：期望 {expected_label}，实测 {measured_label}；请先检查接线、偏置、扫频覆盖和削顶诊断。'


def build_identification_evidence(result: 'SystemAnalysisResult') -> list[str]:
    omega = np.asarray(result.omega, dtype=float)
    mag_db = np.asarray(result.mag_db, dtype=float)
    phase_deg = np.asarray(result.phase_deg, dtype=float)
    evidence: list[str] = []
    if len(omega) >= 6 and np.all(omega > 0):
        slope = compute_mag_slope_db_per_dec(omega, np.clip(result.mag_smooth, 1e-12, None))
        edge = max(3, min(12, len(slope) // 8 if len(slope) else 3))
        head_slope = float(np.median(slope[:edge])) if len(slope) else 0.0
        tail_slope = float(np.median(slope[-edge:])) if len(slope) else 0.0
        evidence.append(f'幅频斜率证据: 低频端 {head_slope:.1f} dB/dec，高频端 {tail_slope:.1f} dB/dec。')

    if len(mag_db):
        evidence.append(f'幅值跨度: {float(np.max(mag_db) - np.min(mag_db)):.1f} dB。')
    if len(phase_deg):
        evidence.append(f'相位跨度: {float(abs(phase_deg[-1] - phase_deg[0])):.1f}°。')

    delay_compensated = any(('测量延迟' in note or '延迟补偿' in note) for note in result.notes)
    evidence.append(f'相位延迟补偿: {"已使用/已考虑" if delay_compensated else "未触发"}。')
    return evidence


def classify_filter_response(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_deg: np.ndarray,
) -> dict[str, object]:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    phase_deg = np.asarray(phase_deg, dtype=float).flatten()

    notes: list[str] = []
    point_count = min(len(omega), len(magnitude), len(phase_deg))
    if point_count < 5:
        omega_fallback = float(omega[0]) if len(omega) else 0.0
        return {
            'filter_type': 'other',
            'order_estimate': 1,
            'system_label': format_filter_system_label('other', 1),
            'primary_omega': omega_fallback,
            'left_cutoff_omega': omega_fallback,
            'right_cutoff_omega': None,
            'target_magnitude': float(magnitude[0]) if len(magnitude) else 0.0,
            'cutoff_method': 'limited data',
            'bandwidth_omega': None,
            'notes': notes,
        }

    edge = max(3, min(12, point_count // 8))
    low_gain = float(np.median(magnitude[:edge]))
    high_gain = float(np.median(magnitude[-edge:]))
    peak_idx = int(np.argmax(magnitude))
    valley_idx = int(np.argmin(magnitude))
    peak_gain = float(magnitude[peak_idx])
    valley_gain = float(magnitude[valley_idx])
    phase_span = float(abs(phase_deg[-1] - phase_deg[0]))

    slope = compute_mag_slope_db_per_dec(omega, magnitude)
    head_slope = float(np.median(slope[:edge]))
    tail_slope = float(np.median(slope[-edge:]))
    head_order = estimate_order_from_slope(head_slope)
    tail_order = estimate_order_from_slope(tail_slope)

    low_to_high_db = db_ratio(low_gain, high_gain)
    high_to_low_db = db_ratio(high_gain, low_gain)
    edge_similarity_db = abs(low_to_high_db)
    peak_edge_margin_db = min(db_ratio(peak_gain, low_gain), db_ratio(peak_gain, high_gain))
    notch_edge_margin_db = min(db_ratio(low_gain, valley_gain), db_ratio(high_gain, valley_gain))

    interior_lo = max(edge, int(0.2 * (point_count - 1)))
    interior_hi = min(point_count - edge - 1, int(0.8 * (point_count - 1)))
    is_peak_interior = interior_lo <= peak_idx <= interior_hi
    is_valley_interior = interior_lo <= valley_idx <= interior_hi

    filter_type = 'other'
    order_estimate = max(head_order, tail_order, 1)

    if (
        is_peak_interior
        and peak_edge_margin_db >= 3.0
        and head_slope > 8.0
        and tail_slope < -8.0
    ):
        filter_type = 'bandpass'
        order_estimate, _, _, band_order_notes = estimate_band_order_conservatively(
            head_order,
            tail_order,
            phase_deg,
        )
        notes.append(
            f'Band-pass shape detected: peak gain exceeds both edges by {peak_edge_margin_db:.1f} dB.'
        )
        notes.extend(band_order_notes)
        if phase_deg[0] > 30.0 and phase_deg[-1] < -20.0:
            notes.append(
                '曲线低频端像高通、但高频端又明显衰减，因此实测为带通形态；如果你预期是一阶 RC 高通，请检查 PA1 是否接在电容后的电阻输出节点、高通输出电阻是否接到 1.65V 中点偏置而不是直接接地，以及高频端是否被额外电容/连线/面包板寄生拖低。STM32 单电源 ADC 不能测围绕 0V 摆动的高通信号。'
            )
    elif (
        is_valley_interior
        and notch_edge_margin_db >= 3.0
    ):
        filter_type = 'bandstop'
        order_estimate, _, _, band_order_notes = estimate_band_order_conservatively(
            head_order,
            tail_order,
            phase_deg,
        )
        notes.append(
            f'Band-stop shape detected: notch depth is about {notch_edge_margin_db:.1f} dB relative to both edges.'
        )
        notes.extend(band_order_notes)
    elif low_to_high_db >= 3.0 or (peak_idx <= edge and tail_slope < -12.0):
        filter_type = 'lowpass'
        order_estimate = max(1, tail_order)
        if order_estimate == 0:
            order_estimate = max(1, int(np.clip(np.round(phase_span / 75.0), 1, 6)))
        phase_order, phase_order_reliable = estimate_order_from_phase_span(phase_deg)
        if phase_order_reliable:
            if phase_order >= 2 and order_estimate > phase_order:
                notes.append(
                    f'Low-pass tail slope suggests order {order_estimate}, but monotonic phase span supports order {phase_order}; using phase-supported order to avoid high-frequency noise overestimation.'
                )
                order_estimate = phase_order
            elif phase_order > order_estimate:
                notes.append(
                    f'原始相位跨度看起来像 {phase_order} 阶，但低通高频相位可能叠加 LM358/采样链路的测量延迟；不单靠原始相位把 {order_estimate} 阶上调到更高阶。'
                )
        elif order_estimate == 1 and 85.0 <= phase_span <= 140.0 and float(omega[-1]) < 25000.0 * 2.0 * np.pi:
            notes.append(
                '当前最高扫频只看到低通前半段，三阶 RC 可能会被显示成一阶；1k+103 三阶低通建议扫到 60000 Hz、步进 600 Hz 后再判阶。'
            )
        notes.append(
            f'检测到低通滚降：高频端斜率约 {tail_slope:.1f} dB/dec。'
        )
    elif high_to_low_db >= 3.0 or (peak_idx >= point_count - edge - 1 and head_slope > 12.0):
        filter_type = 'highpass'
        order_estimate = max(1, head_order)
        if order_estimate == 0:
            order_estimate = max(1, int(np.clip(np.round(phase_span / 75.0), 1, 6)))
        notes.append(
            f'High-pass rise detected: low-frequency slope is about {head_slope:.1f} dB/dec.'
        )
        if phase_deg[-1] < -20.0:
            notes.append(
                '高通高频端相位仍明显为负，可能存在额外延迟或高频滚降；理想一阶 RC 高通的高频相位应逐渐接近 0°。'
            )
    else:
        notes.append('Response shape is not close to a standard low-pass/high-pass/band-pass/band-stop template.')

    primary_omega = float(omega[peak_idx])
    left_cutoff_omega = float(omega[peak_idx])
    right_cutoff_omega: float | None = None
    bandwidth_omega: float | None = None
    cutoff_method = '-3 dB cutoff'

    if filter_type == 'lowpass':
        passband_gain = max(low_gain, peak_gain if peak_idx <= edge else low_gain)
        target_magnitude = passband_gain / np.sqrt(2.0)
        crossings = [item for item in find_level_crossings(omega, magnitude, target_magnitude) if item[1] >= max(edge - 1, peak_idx // 2)]
        if not crossings:
            crossings = find_level_crossings(omega, magnitude, target_magnitude)
        primary_omega = float(crossings[0][0]) if crossings else float(omega[min(peak_idx, point_count - 1)])
        left_cutoff_omega = primary_omega
        notes.append(
            f'先由高频滚降估计为 {order_estimate} 阶；截止频率采用通带 -3 dB 交点。'
        )
    elif filter_type == 'highpass':
        passband_gain = max(high_gain, peak_gain if peak_idx >= point_count - edge - 1 else high_gain)
        target_magnitude = passband_gain / np.sqrt(2.0)
        crossings = [item for item in find_level_crossings(omega, magnitude, target_magnitude) if item[1] <= min(point_count - edge, peak_idx + edge)]
        if not crossings:
            crossings = find_level_crossings(omega, magnitude, target_magnitude)
        primary_omega = float(crossings[-1][0]) if crossings else float(omega[max(peak_idx, 0)])
        left_cutoff_omega = primary_omega
        notes.append(
            f'Estimated order {order_estimate} from the low-frequency rise; cutoff uses the passband -3 dB crossing.'
        )
    elif filter_type == 'bandpass':
        target_magnitude = peak_gain / np.sqrt(2.0)
        crossings = find_level_crossings(omega, magnitude, target_magnitude)
        left_candidates = [item for item in crossings if item[1] < peak_idx]
        right_candidates = [item for item in crossings if item[1] >= peak_idx]
        left_cutoff_omega = float(left_candidates[-1][0]) if left_candidates else float(omega[max(0, peak_idx - 1)])
        right_cutoff_omega = float(right_candidates[0][0]) if right_candidates else float(omega[min(point_count - 1, peak_idx + 1)])
        primary_omega = float(np.sqrt(max(left_cutoff_omega, 1e-12) * max(right_cutoff_omega, 1e-12)))
        bandwidth_omega = float(max(right_cutoff_omega - left_cutoff_omega, 0.0))
        cutoff_method = 'band-pass -3 dB edges'
        notes.append(
            f'Estimated band-pass order {order_estimate}; center frequency is taken as sqrt(ω1·ω2) from the two -3 dB passband edges.'
        )
    elif filter_type == 'bandstop':
        passband_gain = max(low_gain, high_gain)
        target_magnitude = passband_gain / np.sqrt(2.0)
        crossings = find_level_crossings(omega, magnitude, target_magnitude)
        left_candidates = [item for item in crossings if item[1] < valley_idx]
        right_candidates = [item for item in crossings if item[1] >= valley_idx]
        left_cutoff_omega = float(left_candidates[-1][0]) if left_candidates else float(omega[max(0, valley_idx - 1)])
        right_cutoff_omega = float(right_candidates[0][0]) if right_candidates else float(omega[min(point_count - 1, valley_idx + 1)])
        primary_omega = float(omega[valley_idx])
        bandwidth_omega = float(max(right_cutoff_omega - left_cutoff_omega, 0.0))
        cutoff_method = 'band-stop -3 dB edges'
        notes.append(
            f'Estimated band-stop order {order_estimate}; notch frequency is taken from the minimum-magnitude point between the two -3 dB edges.'
        )
    else:
        target_magnitude = float(magnitude[peak_idx])

    return {
        'filter_type': filter_type,
        'order_estimate': int(np.clip(order_estimate, 1, 12)),
        'system_label': format_filter_system_label(filter_type, int(np.clip(order_estimate, 1, 12))),
        'primary_omega': float(primary_omega),
        'left_cutoff_omega': float(left_cutoff_omega),
        'right_cutoff_omega': right_cutoff_omega,
        'target_magnitude': float(target_magnitude),
        'cutoff_method': cutoff_method,
        'bandwidth_omega': bandwidth_omega,
        'notes': notes,
    }


@dataclass
class SystemAnalysisResult:
    omega: np.ndarray
    mag_raw: np.ndarray
    phase_raw_rad: np.ndarray
    mag_smooth: np.ndarray
    phase_smooth_rad: np.ndarray
    mag_db: np.ndarray
    phase_deg: np.ndarray
    real_part: np.ndarray
    imag_part: np.ndarray

    cutoff_index: int
    omega_c: float
    cutoff_phase_deg: float
    cutoff_method: str
    cutoff_mag_db: float
    magnitude_cutoff_omega: float
    secondary_cutoff_omega: float | None
    phase_cutoff_omega: float | None
    bandwidth_omega: float | None
    max_magnitude: float
    resonant_frequency: float

    min_distance: float
    crossing_points: list
    gain_margin_db: float | None
    phase_margin: float | None

    filter_type: str
    system_order: str
    order_estimate: int
    damping_ratio: float | None
    natural_frequency: float | None
    overshoot_percent: float | None
    settling_time_est: float | None

    nyquist_winding_ccw: int
    nyquist_encirclements_cw: int
    assumed_open_loop_rhp_poles: int
    estimated_closed_loop_rhp_poles: int
    nyquist_stable: bool

    stability_score: int
    stability_text: str
    notes: list[str]
    order_reliable: bool = True
    identification_confidence: float = 1.0
    identification_summary: str = '可靠'
    expected_circuit: str = 'Auto'
    expected_match_text: str = '未指定期望电路，按自动识别结果验收。'


def analyze_system(
    omega, magnitude_data, phase_data_rad,
    smooth=True, window_length=11, polyorder=3, fix_g431_axis=False,
    assumed_open_loop_rhp_poles=0, invert_transfer=False
) -> SystemAnalysisResult:
    notes: list[str] = []

    try:
        assumed_open_loop_rhp_poles = int(assumed_open_loop_rhp_poles)
    except (TypeError, ValueError):
        raise ValueError('开环右半平面极点数 P 必须是非负整数。')
    if assumed_open_loop_rhp_poles < 0:
        raise ValueError('开环右半平面极点数 P 必须是非负整数。')

    omega = np.asarray(omega, dtype=float).flatten()
    magnitude_data = np.asarray(magnitude_data, dtype=float).flatten()
    phase_data_rad = np.asarray(phase_data_rad, dtype=float).flatten()

    min_len = min(len(omega), len(magnitude_data), len(phase_data_rad))
    if min_len < 3:
        raise ValueError('有效数据点太少，至少需要 3 个点。')

    if len({len(omega), len(magnitude_data), len(phase_data_rad)}) != 1:
        notes.append(f'数组长度不一致，已按最短长度 {min_len} 截断。')

    omega = omega[:min_len]
    magnitude_data = magnitude_data[:min_len]
    phase_data_rad = phase_data_rad[:min_len]
    phase_data_rad = np.unwrap(phase_data_rad)

    if invert_transfer:
        magnitude_data = 1.0 / np.clip(magnitude_data, 1e-12, None)
        notes.append('已按当前硬件兼容模式交换幅值方向，并保留相位符号；用于修正幅值通道方向与相位符号不一致的实测数据。')

    # Keep ascending order for robust analysis.
    sort_idx = np.argsort(omega)
    if not np.array_equal(sort_idx, np.arange(len(omega))):
        omega = omega[sort_idx]
        magnitude_data = magnitude_data[sort_idx]
        phase_data_rad = phase_data_rad[sort_idx]
        notes.append('频率轴非递增，已自动按升序重排。')

    if fix_g431_axis and len(omega) >= 3 and np.isclose(omega[1], omega[0]):
        positive = np.diff(omega)
        positive = positive[positive > 0]
        if len(positive):
            step = float(np.median(positive))
            omega = omega[0] + step * np.arange(len(omega), dtype=float)
            notes.append('已应用 G431 首点重复频率轴修正。')

    mag_smooth = np.clip(magnitude_data.copy(), 1e-12, None)
    phase_smooth_rad = phase_data_rad.copy()

    if smooth:
        mag_smooth = np.clip(matlab_smooth(magnitude_data, window_length, polyorder), 1e-12, None)
        phase_smooth_rad = matlab_smooth(phase_data_rad, window_length, polyorder)
        notes.append(f'已进行 Savitzky-Golay 平滑（窗口={window_length}，阶数={polyorder}）。')

    max_magnitude = float(np.max(mag_smooth))

    h_jw = mag_smooth * np.exp(1j * phase_smooth_rad)
    real_part = np.real(h_jw)
    imag_part = np.imag(h_jw)

    mag_db = 20.0 * np.log10(np.clip(mag_smooth, 1e-12, None))
    phase_deg = np.degrees(phase_smooth_rad)

    distance_to_minus_one = np.sqrt((real_part + 1.0) ** 2 + imag_part ** 2)
    min_distance = float(np.min(distance_to_minus_one))

    negative_real_indices = np.where(real_part < -0.95)[0]
    crossing_points = []
    if len(negative_real_indices) >= 2:
        for i in range(len(negative_real_indices) - 1):
            idx1 = int(negative_real_indices[i])
            idx2 = int(negative_real_indices[i + 1])
            if imag_part[idx1] * imag_part[idx2] < 0:
                crossing_points.append((idx1, idx2))

    phase_180_indices = np.where(np.abs(phase_deg + 180.0) < 12.0)[0]
    gain_margin_db = None
    if len(phase_180_indices) > 0:
        min_gain_at_180 = float(np.min(mag_smooth[phase_180_indices]))
        gain_margin_db = float(-20.0 * np.log10(max(min_gain_at_180, 1e-12)))

    unity_gain_indices = np.where(np.abs(mag_smooth - 1.0) < 0.12)[0]
    phase_margin = None
    if len(unity_gain_indices) > 0:
        min_idx = int(np.argmin(np.abs(phase_deg[unity_gain_indices] + 180.0)))
        phase_margin = float(phase_deg[unity_gain_indices[min_idx]] + 180.0)

    max_idx = int(np.argmax(mag_smooth))
    resonant_frequency = float(omega[max_idx])

    order_estimate, order_notes = estimate_system_order_from_fit(mag_smooth, phase_deg, omega)
    notes.extend(order_notes)

    if order_estimate <= 1:
        system_order = '一阶系统'
    elif order_estimate == 2:
        system_order = '二阶系统'
    else:
        system_order = f'高阶系统（估计阶次 {order_estimate}）'
    system_order = format_system_order_label(order_estimate)

    low_freq_count = max(3, min(5, len(mag_smooth)))
    low_freq_gain = float(np.median(mag_smooth[:low_freq_count]))
    if order_estimate <= 1:
        target_magnitude = low_freq_gain / np.sqrt(2.0)
    else:
        target_magnitude = max_magnitude / np.sqrt(2.0)

    magnitude_cutoff_omega, magnitude_cutoff_index = interpolate_omega_at_level(
        omega, mag_smooth, target_magnitude
    )
    omega_c = float(magnitude_cutoff_omega)
    cutoff_index = int(magnitude_cutoff_index)
    phase_cutoff_omega: float | None = None
    cutoff_method = '-3 dB magnitude'

    if order_estimate <= 1:
        phase_cutoff_omega, phase_cutoff_index = estimate_first_order_phase_cutoff(omega, phase_deg)
        if phase_cutoff_omega is not None and phase_cutoff_index is not None:
            phase_mag_delta = abs(float(phase_cutoff_omega) - float(magnitude_cutoff_omega)) / max(float(magnitude_cutoff_omega), 1e-12)
            notes.append('一阶系统已同时计算幅值 -3 dB 和相位 -45° 截止点；主截止角频率采用通用的幅值 -3 dB 定义。')
            if phase_mag_delta > 0.08:
                notes.append(f'幅值 -3 dB 与相位 -45° 截止点相差 {phase_mag_delta * 100.0:.1f}%，建议优先看 -3 dB 幅值结果并检查相位偏差来源。')
        else:
            notes.append('未找到 -45° 相位交点，截止频率保持为幅值 -3 dB 结果。')

    cutoff_index = nearest_index(omega, omega_c)
    cutoff_phase_deg = float(phase_deg[cutoff_index])
    cutoff_mag_db = float(mag_db[cutoff_index])

    damping_ratio = None
    natural_frequency = None
    overshoot_percent = None
    settling_time_est = None

    if order_estimate == 2:
        natural_frequency = float(omega[max_idx]) if max_magnitude > 1.05 else omega_c

        if max_magnitude > 1.05:
            damping_ratio = 1.0 / (2.0 * np.sqrt(max_magnitude**2 - 1.0))
        else:
            phi_at_c = np.abs(phase_deg[cutoff_index])
            if phi_at_c > 70.0:
                damping_ratio = float(np.clip((180.0 - phi_at_c) / 90.0, 0.05, 0.95))
            else:
                damping_ratio = 0.707

        damping_ratio = float(np.clip(damping_ratio, 0.01, 2.0))
        if damping_ratio < 1.0:
            overshoot_percent = float(100.0 * np.exp(-np.pi * damping_ratio / np.sqrt(1.0 - damping_ratio**2)))
        else:
            overshoot_percent = 0.0

        if damping_ratio > 0.01 and natural_frequency is not None:
            settling_time_est = float(4.0 / (damping_ratio * natural_frequency))

    nyquist_closed = build_closed_nyquist_curve(h_jw)
    nyquist_winding_ccw = compute_winding_number_ccw(nyquist_closed, center=-1.0 + 0.0j)
    nyquist_encirclements_cw = -nyquist_winding_ccw
    estimated_closed_loop_rhp_poles = assumed_open_loop_rhp_poles + nyquist_encirclements_cw
    nyquist_stable = (estimated_closed_loop_rhp_poles == 0)

    notes.append(
        f'奈奎斯特判据：N(顺时针包围数)={nyquist_encirclements_cw}，'
        f'P(开环右半平面极点数)={assumed_open_loop_rhp_poles}，'
        f'Z=P+N={estimated_closed_loop_rhp_poles}。'
    )

    score = 0
    if nyquist_stable:
        score += 3
    if gain_margin_db is not None and gain_margin_db > 6.0:
        score += 1
    if phase_margin is not None and phase_margin > 35.0:
        score += 1
    if min_distance > 0.3:
        score += 1

    if not nyquist_stable:
        stability_text = f'奈奎斯特判据显示闭环可能不稳定：Z=P+N={estimated_closed_loop_rhp_poles}。'
    elif score >= 5:
        stability_text = '奈奎斯特判据显示闭环稳定，且稳定裕度较好。'
    elif score >= 3:
        stability_text = '奈奎斯特判据显示闭环稳定，但裕度偏小。'
    else:
        stability_text = '奈奎斯特判据显示闭环稳定，但接近稳定边界，建议扩大扫频范围复测。'

    return SystemAnalysisResult(
        omega=omega,
        mag_raw=magnitude_data,
        phase_raw_rad=phase_data_rad,
        mag_smooth=mag_smooth,
        phase_smooth_rad=phase_smooth_rad,
        mag_db=mag_db,
        phase_deg=phase_deg,
        real_part=real_part,
        imag_part=imag_part,
        cutoff_index=cutoff_index,
        omega_c=omega_c,
        cutoff_phase_deg=cutoff_phase_deg,
        cutoff_method=cutoff_method,
        cutoff_mag_db=cutoff_mag_db,
        magnitude_cutoff_omega=float(magnitude_cutoff_omega),
        phase_cutoff_omega=phase_cutoff_omega,
        max_magnitude=max_magnitude,
        resonant_frequency=resonant_frequency,
        min_distance=min_distance,
        crossing_points=crossing_points,
        gain_margin_db=gain_margin_db,
        phase_margin=phase_margin,
        system_order=system_order,
        order_estimate=order_estimate,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        overshoot_percent=overshoot_percent,
        settling_time_est=settling_time_est,
        nyquist_winding_ccw=nyquist_winding_ccw,
        nyquist_encirclements_cw=nyquist_encirclements_cw,
        assumed_open_loop_rhp_poles=assumed_open_loop_rhp_poles,
        estimated_closed_loop_rhp_poles=estimated_closed_loop_rhp_poles,
        nyquist_stable=nyquist_stable,
        stability_score=score,
        stability_text=stability_text,
        notes=notes,
    )


def analyze_system_v2(
    omega, magnitude_data, phase_data_rad,
    smooth=True, window_length=11, polyorder=3, fix_g431_axis=False,
    assumed_open_loop_rhp_poles=0, invert_transfer=False, diagnostics=None
) -> SystemAnalysisResult:
    notes: list[str] = []

    try:
        assumed_open_loop_rhp_poles = int(assumed_open_loop_rhp_poles)
    except (TypeError, ValueError):
        raise ValueError('\u5f00\u73af\u53f3\u534a\u5e73\u9762\u6781\u70b9\u6570 P \u5fc5\u987b\u662f\u975e\u8d1f\u6574\u6570\u3002')
    if assumed_open_loop_rhp_poles < 0:
        raise ValueError('\u5f00\u73af\u53f3\u534a\u5e73\u9762\u6781\u70b9\u6570 P \u5fc5\u987b\u662f\u975e\u8d1f\u6574\u6570\u3002')

    omega = np.asarray(omega, dtype=float).flatten()
    magnitude_data = np.asarray(magnitude_data, dtype=float).flatten()
    phase_data_rad = np.asarray(phase_data_rad, dtype=float).flatten()

    min_len = min(len(omega), len(magnitude_data), len(phase_data_rad))
    if min_len < 3:
        raise ValueError('\u6709\u6548\u6570\u636e\u70b9\u592a\u5c11\uff0c\u81f3\u5c11\u9700\u8981 3 \u4e2a\u70b9\u3002')

    if len({len(omega), len(magnitude_data), len(phase_data_rad)}) != 1:
        notes.append(f'\u6570\u7ec4\u957f\u5ea6\u4e0d\u4e00\u81f4\uff0c\u5df2\u6309\u6700\u77ed\u957f\u5ea6 {min_len} \u622a\u65ad\u3002')

    omega = omega[:min_len]
    magnitude_data = magnitude_data[:min_len]
    phase_data_rad = np.unwrap(phase_data_rad[:min_len])

    if invert_transfer:
        magnitude_data = 1.0 / np.clip(magnitude_data, 1e-12, None)
        notes.append('\u5df2\u6309\u5f53\u524d\u786c\u4ef6\u517c\u5bb9\u6a21\u5f0f\u4ea4\u6362\u5e45\u503c\u65b9\u5411\uff0c\u5e76\u4fdd\u7559\u76f8\u4f4d\u7b26\u53f7\uff1b\u7528\u4e8e\u4fee\u6b63\u5e45\u503c\u901a\u9053\u65b9\u5411\u4e0e\u76f8\u4f4d\u7b26\u53f7\u4e0d\u4e00\u81f4\u7684\u5b9e\u6d4b\u6570\u636e\u3002')

    sort_idx = np.argsort(omega)
    if not np.array_equal(sort_idx, np.arange(len(omega))):
        omega = omega[sort_idx]
        magnitude_data = magnitude_data[sort_idx]
        phase_data_rad = phase_data_rad[sort_idx]
        notes.append('\u9891\u7387\u8f74\u975e\u9012\u589e\uff0c\u5df2\u81ea\u52a8\u6309\u5347\u5e8f\u91cd\u6392\u3002')
    diagnostics = aligned_diagnostics(diagnostics, min_len, sort_idx)

    if fix_g431_axis and len(omega) >= 3 and np.isclose(omega[1], omega[0]):
        positive = np.diff(omega)
        positive = positive[positive > 0]
        if len(positive):
            step = float(np.median(positive))
            omega = omega[0] + step * np.arange(len(omega), dtype=float)
            notes.append('\u5df2\u5e94\u7528 G431 \u9996\u70b9\u91cd\u590d\u9891\u7387\u8f74\u4fee\u6b63\u3002')

    omega, magnitude_data, phase_data_rad, diagnostics, quality_mask_notes = apply_measurement_quality_mask(
        omega,
        magnitude_data,
        phase_data_rad,
        diagnostics,
    )
    notes.extend(quality_mask_notes)

    magnitude_for_analysis = magnitude_data.copy()
    phase_for_analysis = phase_data_rad.copy()
    if smooth:
        magnitude_for_analysis, mag_spike_mask = suppress_isolated_spikes(
            magnitude_for_analysis,
            domain='db',
            min_threshold=1.0,
            window_radius=3,
        )
        phase_for_analysis, phase_spike_mask = suppress_isolated_spikes(
            phase_for_analysis,
            domain='deg',
            min_threshold=12.0,
            window_radius=3,
        )
        if np.any(mag_spike_mask):
            notes.append(f'已抑制 {int(np.count_nonzero(mag_spike_mask))} 个孤立幅值毛刺点，用邻域中值替代后再平滑。')
        if np.any(phase_spike_mask):
            notes.append(f'已抑制 {int(np.count_nonzero(phase_spike_mask))} 个孤立相位毛刺点，用邻域中值替代后再平滑。')

    mag_smooth = np.clip(magnitude_for_analysis.copy(), 1e-12, None)
    phase_smooth_rad = phase_for_analysis.copy()
    if smooth:
        mag_smooth = np.clip(matlab_smooth(magnitude_for_analysis, window_length, polyorder), 1e-12, None)
        phase_smooth_rad = matlab_smooth(phase_for_analysis, window_length, polyorder)
        notes.append(
            f'\u5df2\u8fdb\u884c Savitzky-Golay \u5e73\u6ed1\uff08\u7a97\u53e3={window_length}\uff0c\u9636\u6570={polyorder}\uff09\u3002'
        )

    max_magnitude = float(np.max(mag_smooth))
    max_idx = int(np.argmax(mag_smooth))
    min_idx = int(np.argmin(mag_smooth))

    h_jw = mag_smooth * np.exp(1j * phase_smooth_rad)
    real_part = np.real(h_jw)
    imag_part = np.imag(h_jw)
    mag_db = 20.0 * np.log10(np.clip(mag_smooth, 1e-12, None))
    phase_deg = np.degrees(phase_smooth_rad)

    distance_to_minus_one = np.sqrt((real_part + 1.0) ** 2 + imag_part ** 2)
    min_distance = float(np.min(distance_to_minus_one))

    negative_real_indices = np.where(real_part < -0.95)[0]
    crossing_points = []
    if len(negative_real_indices) >= 2:
        for i in range(len(negative_real_indices) - 1):
            idx1 = int(negative_real_indices[i])
            idx2 = int(negative_real_indices[i + 1])
            if imag_part[idx1] * imag_part[idx2] < 0:
                crossing_points.append((idx1, idx2))

    phase_180_indices = np.where(np.abs(phase_deg + 180.0) < 12.0)[0]
    gain_margin_db = None
    if len(phase_180_indices) > 0:
        min_gain_at_180 = float(np.min(mag_smooth[phase_180_indices]))
        gain_margin_db = float(-20.0 * np.log10(max(min_gain_at_180, 1e-12)))

    unity_gain_indices = np.where(np.abs(mag_smooth - 1.0) < 0.12)[0]
    phase_margin = None
    if len(unity_gain_indices) > 0:
        min_phase_idx = int(np.argmin(np.abs(phase_deg[unity_gain_indices] + 180.0)))
        phase_margin = float(phase_deg[unity_gain_indices[min_phase_idx]] + 180.0)

    filter_info = classify_filter_response(omega, mag_smooth, phase_deg)
    filter_type = str(filter_info['filter_type'])
    order_estimate = int(filter_info['order_estimate'])
    notes.extend(filter_info['notes'])

    refined_order, model_fit_notes = refine_filter_order_with_template(
        filter_type,
        omega,
        mag_smooth,
        phase_deg,
        order_estimate,
    )
    if refined_order != order_estimate:
        notes.append(f'模型拟合将初始阶次 {order_estimate} 修正为 {refined_order}。')
        order_estimate = refined_order
    notes.extend(model_fit_notes)

    identification = assess_identification_reliability(
        omega,
        mag_smooth,
        phase_deg,
        {**filter_info, 'order_estimate': order_estimate},
        diagnostics,
    )
    order_reliable = bool(identification['reliable'])
    identification_confidence = float(identification['confidence'])
    identification_summary = str(identification['summary'])
    system_order = format_filter_system_label(filter_type, order_estimate, reliable=order_reliable)

    notes.extend(summarize_measurement_diagnostics(diagnostics, len(omega), filter_type))
    notes.extend(identification['notes'])

    omega_c = float(filter_info['primary_omega'])
    magnitude_cutoff_omega = float(filter_info['left_cutoff_omega'])
    secondary_cutoff_omega = (
        None if filter_info['right_cutoff_omega'] is None else float(filter_info['right_cutoff_omega'])
    )
    bandwidth_omega = (
        None if filter_info['bandwidth_omega'] is None else float(filter_info['bandwidth_omega'])
    )
    target_magnitude = float(max(filter_info['target_magnitude'], 1e-12))
    cutoff_method = str(filter_info['cutoff_method'])

    phase_cutoff_omega: float | None = None
    if filter_type == 'lowpass' and order_estimate <= 1:
        phase_cutoff_omega, phase_cutoff_index = estimate_first_order_phase_cutoff(omega, phase_deg)
        if phase_cutoff_omega is not None and phase_cutoff_index is not None:
            phase_mag_delta = abs(float(phase_cutoff_omega) - float(magnitude_cutoff_omega)) / max(float(magnitude_cutoff_omega), 1e-12)
            notes.append(
                '\u4e00\u9636\u4f4e\u901a\u989d\u5916\u8ba1\u7b97\u4e86 -45\u00b0 \u76f8\u4f4d\u622a\u6b62\u70b9\uff0c\u4f18\u5148\u4ee5\u5e45\u9891 -3 dB \u7ed3\u679c\u4e3a\u51c6\u3002'
            )
            if phase_mag_delta > 0.08:
                notes.append(
                    f'\u5e45\u9891 -3 dB \u4e0e\u76f8\u4f4d -45\u00b0 \u622a\u6b62\u70b9\u76f8\u5dee {phase_mag_delta * 100.0:.1f}%\uff0c\u8bf7\u68c0\u67e5\u63a5\u7ebf\u65b9\u5411\u548c\u626b\u9891\u8303\u56f4\u3002'
                )
        else:
            notes.append('\u672a\u627e\u5230 -45\u00b0 \u76f8\u4f4d\u4ea4\u70b9\uff0c\u622a\u6b62\u89d2\u9891\u7387\u4fdd\u6301\u4e3a\u5e45\u9891 -3 dB \u7ed3\u679c\u3002')

    cutoff_index = nearest_index(omega, omega_c)
    cutoff_phase_deg = float(phase_deg[cutoff_index])
    cutoff_mag_db = float(20.0 * np.log10(target_magnitude))

    resonant_frequency = float(omega[min_idx if filter_type == 'bandstop' else max_idx])

    damping_ratio = None
    natural_frequency = None
    overshoot_percent = None
    settling_time_est = None
    if filter_type == 'lowpass' and order_estimate == 2:
        natural_frequency = float(omega[max_idx]) if max_magnitude > 1.05 else omega_c
        if max_magnitude > 1.05:
            damping_ratio = 1.0 / (2.0 * np.sqrt(max(max_magnitude**2 - 1.0, 1e-12)))
        else:
            phi_at_c = np.abs(phase_deg[cutoff_index])
            damping_ratio = float(np.clip((180.0 - phi_at_c) / 90.0, 0.05, 0.95)) if phi_at_c > 70.0 else 0.707

        damping_ratio = float(np.clip(damping_ratio, 0.01, 2.0))
        if damping_ratio < 1.0:
            overshoot_percent = float(100.0 * np.exp(-np.pi * damping_ratio / np.sqrt(1.0 - damping_ratio**2)))
        else:
            overshoot_percent = 0.0

        if damping_ratio > 0.01 and natural_frequency is not None:
            settling_time_est = float(4.0 / (damping_ratio * natural_frequency))

    nyquist_closed = build_closed_nyquist_curve(h_jw)
    nyquist_winding_ccw = compute_winding_number_ccw(nyquist_closed, center=-1.0 + 0.0j)
    nyquist_encirclements_cw = -nyquist_winding_ccw
    estimated_closed_loop_rhp_poles = assumed_open_loop_rhp_poles + nyquist_encirclements_cw
    nyquist_stable = (estimated_closed_loop_rhp_poles == 0)
    notes.append(
        f'Nyquist \u5224\u636e\uff1aN(\u987a\u65f6\u9488\u5305\u56f4\u6570)={nyquist_encirclements_cw}\uff0c'
        f'P(\u5f00\u73af\u53f3\u534a\u5e73\u9762\u6781\u70b9\u6570)={assumed_open_loop_rhp_poles}\uff0c'
        f'Z=P+N={estimated_closed_loop_rhp_poles}\u3002'
    )

    score = 0
    if nyquist_stable:
        score += 3
    if gain_margin_db is not None and gain_margin_db > 6.0:
        score += 1
    if phase_margin is not None and phase_margin > 35.0:
        score += 1
    if min_distance > 0.3:
        score += 1

    if not nyquist_stable:
        stability_text = f'\u5948\u594e\u65af\u7279\u5224\u636e\u663e\u793a\u95ed\u73af\u53ef\u80fd\u4e0d\u7a33\u5b9a\uff1aZ=P+N={estimated_closed_loop_rhp_poles}\u3002'
    elif score >= 5:
        stability_text = '\u5948\u594e\u65af\u7279\u5224\u636e\u663e\u793a\u95ed\u73af\u7a33\u5b9a\uff0c\u4e14\u7a33\u5b9a\u88d5\u5ea6\u8f83\u597d\u3002'
    elif score >= 3:
        stability_text = '\u5948\u594e\u65af\u7279\u5224\u636e\u663e\u793a\u95ed\u73af\u7a33\u5b9a\uff0c\u4f46\u88d5\u5ea6\u504f\u5c0f\u3002'
    else:
        stability_text = '\u5948\u594e\u65af\u7279\u5224\u636e\u663e\u793a\u95ed\u73af\u7a33\u5b9a\uff0c\u4f46\u63a5\u8fd1\u7a33\u5b9a\u8fb9\u754c\uff0c\u5efa\u8bae\u6269\u5927\u626b\u9891\u8303\u56f4\u590d\u6d4b\u3002'

    return SystemAnalysisResult(
        omega=omega,
        mag_raw=magnitude_data,
        phase_raw_rad=phase_data_rad,
        mag_smooth=mag_smooth,
        phase_smooth_rad=phase_smooth_rad,
        mag_db=mag_db,
        phase_deg=phase_deg,
        real_part=real_part,
        imag_part=imag_part,
        cutoff_index=cutoff_index,
        omega_c=omega_c,
        cutoff_phase_deg=cutoff_phase_deg,
        cutoff_method=cutoff_method,
        cutoff_mag_db=cutoff_mag_db,
        magnitude_cutoff_omega=float(magnitude_cutoff_omega),
        secondary_cutoff_omega=secondary_cutoff_omega,
        phase_cutoff_omega=phase_cutoff_omega,
        bandwidth_omega=bandwidth_omega,
        max_magnitude=max_magnitude,
        resonant_frequency=resonant_frequency,
        min_distance=min_distance,
        crossing_points=crossing_points,
        gain_margin_db=gain_margin_db,
        phase_margin=phase_margin,
        filter_type=filter_type,
        system_order=system_order,
        order_estimate=order_estimate,
        damping_ratio=damping_ratio,
        natural_frequency=natural_frequency,
        overshoot_percent=overshoot_percent,
        settling_time_est=settling_time_est,
        nyquist_winding_ccw=nyquist_winding_ccw,
        nyquist_encirclements_cw=nyquist_encirclements_cw,
        assumed_open_loop_rhp_poles=assumed_open_loop_rhp_poles,
        estimated_closed_loop_rhp_poles=estimated_closed_loop_rhp_poles,
        nyquist_stable=nyquist_stable,
        stability_score=score,
        stability_text=stability_text,
        notes=notes,
        order_reliable=order_reliable,
        identification_confidence=identification_confidence,
        identification_summary=identification_summary,
    )


analyze_system = analyze_system_v2


class MatlabExactApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry('1520x980')
        self.root.minsize(1300, 800)

        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.reader = None
        self.result = None
        self.font_name = configure_plot_fonts()

        self._build_ui()
        self.refresh_ports()
        self.root.after(100, self._poll_queue)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill='both', expand=True)

        top = ttk.LabelFrame(main, text='采集控制', padding=10)
        top.pack(fill='x')

        ttk.Label(top, text='串口').grid(row=0, column=0, sticky='w')
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(top, textvariable=self.port_var, width=14, state='readonly')
        self.port_box.grid(row=0, column=1, padx=4)
        ttk.Button(top, text='刷新', command=self.refresh_ports).grid(row=0, column=2, padx=4)

        ttk.Label(top, text='波特率').grid(row=0, column=3, sticky='w', padx=(12, 0))
        self.baud_var = tk.StringVar(value='115200')
        ttk.Entry(top, textvariable=self.baud_var, width=10).grid(row=0, column=4, padx=4)

        ttk.Label(top, text='超时(s)').grid(row=0, column=5, sticky='w', padx=(12, 0))
        self.timeout_var = tk.StringVar(value='60')
        ttk.Entry(top, textvariable=self.timeout_var, width=8).grid(row=0, column=6, padx=4)

        self.smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='按 MATLAB 平滑', variable=self.smooth_var).grid(row=0, column=7, padx=(12, 0))

        self.fix_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='G431 首点重复修正', variable=self.fix_var).grid(row=0, column=8, padx=(8, 0))

        ttk.Label(top, text='窗口').grid(row=0, column=9, sticky='w', padx=(12, 0))
        self.window_var = tk.StringVar(value='11')
        ttk.Entry(top, textvariable=self.window_var, width=6).grid(row=0, column=10, padx=4)

        ttk.Label(top, text='平滑阶数').grid(row=0, column=11, sticky='w', padx=(8, 0))
        self.poly_var = tk.StringVar(value='3')
        ttk.Entry(top, textvariable=self.poly_var, width=6).grid(row=0, column=12, padx=4)

        self.swap_io_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='交换幅值方向', variable=self.swap_io_var).grid(row=0, column=13, padx=(12, 0))

        ttk.Button(top, text='读取一帧', command=self.read_one_frame).grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        ttk.Button(top, text='连续读取', command=self.read_continuous).grid(row=1, column=2, columnspan=2, sticky='ew', pady=(10, 0), padx=(8, 0))
        ttk.Button(top, text='停止', command=self.stop_reading).grid(row=1, column=4, sticky='ew', pady=(10, 0), padx=(8, 0))
        ttk.Button(top, text='导入文本测试', command=self.import_text).grid(row=1, column=5, columnspan=2, sticky='ew', pady=(10, 0), padx=(8, 0))
        ttk.Button(top, text='保存当前CSV', command=self.save_csv).grid(row=1, column=7, columnspan=2, sticky='ew', pady=(10, 0), padx=(8, 0))
        ttk.Button(top, text='保存当前图像', command=self.save_png).grid(row=1, column=9, columnspan=2, sticky='ew', pady=(10, 0), padx=(8, 0))

        ttk.Label(top, text='起始频率 Hz').grid(row=2, column=0, sticky='w', pady=(10, 0))
        self.f_start_var = tk.StringVar(value='100')
        ttk.Entry(top, textvariable=self.f_start_var, width=10).grid(row=2, column=1, padx=4, pady=(10, 0))

        ttk.Label(top, text='终止频率 Hz').grid(row=2, column=2, sticky='w', pady=(10, 0))
        self.f_stop_var = tk.StringVar(value='60000')
        ttk.Entry(top, textvariable=self.f_stop_var, width=10).grid(row=2, column=3, padx=4, pady=(10, 0))

        ttk.Label(top, text='步进频率 Hz').grid(row=2, column=4, sticky='w', pady=(10, 0))
        self.f_step_var = tk.StringVar(value='600')
        ttk.Entry(top, textvariable=self.f_step_var, width=10).grid(row=2, column=5, padx=4, pady=(10, 0))

        ttk.Label(top, text='幅度 Vpp').grid(row=2, column=6, sticky='w', pady=(10, 0))
        self.amp_var = tk.StringVar(value='0.6')
        ttk.Entry(top, textvariable=self.amp_var, width=8).grid(row=2, column=7, padx=4, pady=(10, 0))

        ttk.Button(top, text='发送扫频', command=self.command_sweep_once).grid(row=2, column=8, columnspan=2, sticky='ew', pady=(10, 0), padx=(8, 0))

        ttk.Label(top, text='期望电路').grid(row=3, column=0, sticky='w', pady=(10, 0))
        self.expected_circuit_var = tk.StringVar(value='Auto')
        self.expected_circuit_box = ttk.Combobox(
            top,
            textvariable=self.expected_circuit_var,
            values=EXPECTED_CIRCUIT_CHOICES,
            width=12,
            state='readonly',
        )
        self.expected_circuit_box.grid(row=3, column=1, padx=4, pady=(10, 0))
        ttk.Button(top, text='自动识别八电路', command=self.command_auto_identify).grid(
            row=3, column=2, columnspan=3, sticky='ew', pady=(10, 0), padx=(8, 0)
        )

        ttk.Label(top, text='开环RHP极点P').grid(row=2, column=10, sticky='w', pady=(10, 0), padx=(12, 0))
        self.open_loop_p_var = tk.StringVar(value='0')
        ttk.Entry(top, textvariable=self.open_loop_p_var, width=6).grid(row=2, column=11, padx=4, pady=(10, 0))

        self.system_type_var = tk.StringVar(value='待分析')
        ttk.Label(top, text='系统类型:').grid(row=1, column=13, sticky='e', padx=(20,0))
        ttk.Label(top, textvariable=self.system_type_var, foreground='blue', font=('微软雅黑', 10, 'bold')).grid(row=1, column=14, sticky='w')

        self.status_var = tk.StringVar(value='就绪')
        ttk.Label(top, text='状态:').grid(row=1, column=15, sticky='e', padx=(12, 0), pady=(10, 0))
        ttk.Label(top, textvariable=self.status_var, foreground='blue').grid(row=1, column=16, sticky='w', pady=(10, 0))

        body = ttk.PanedWindow(main, orient='horizontal')
        body.pack(fill='both', expand=True, pady=(10, 0))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.fig = Figure(figsize=(10.5, 7.5), dpi=100)
        self.ax_ny = self.fig.add_subplot(221)
        self.ax_mag = self.fig.add_subplot(222)
        self.ax_phase = self.fig.add_subplot(224)
        self.ax_stab = self.fig.add_subplot(223)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        tb = NavigationToolbar2Tk(self.canvas, left, pack_toolbar=False)
        tb.update()
        tb.pack(fill='x')

        logbox = ttk.LabelFrame(right, text='分析结果 / 日志', padding=8)
        logbox.pack(fill='both', expand=True)

        self.text = tk.Text(logbox, wrap='word')
        self.text.pack(side='left', fill='both', expand=True)

        ys = ttk.Scrollbar(logbox, orient='vertical', command=self.text.yview)
        ys.pack(side='right', fill='y')
        self.text.configure(yscrollcommand=ys.set)

        self.text.insert('end', '稳频仪已就绪\n支持自动扫频、伯德图、奈奎斯特图、N/P/Z 稳定性判据、增益裕度和相位裕度分析。\n\n')

    def log(self, msg: str):
        self.text.insert('end', f'[{time.strftime("%H:%M:%S")}] {msg}\n')
        self.text.see('end')

    def refresh_ports(self):
        ports = list_serial_ports()
        self.port_box['values'] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        self.status_var.set('未检测到串口' if not ports else f'检测到 {len(ports)} 个串口')
        self.log(f'串口列表: {ports if ports else "无"}')

    def _build_sweep_command(self):
        try:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            amp = float(self.amp_var.get().strip())
        except ValueError:
            messagebox.showwarning('提示', '扫频参数格式不正确。')
            return None

        if f_start <= 0 or f_stop < f_start or f_step <= 0:
            messagebox.showwarning('提示', '频率范围不正确。')
            return None
        if amp <= 0 or amp > 3.0:
            messagebox.showwarning('提示', '幅度建议设置在 0 到 3.0 Vpp 之间。')
            return None

        return f'SWEEP {f_start:g} {f_stop:g} {f_step:g} {amp:g}\n'

    def _build_sweep_command_quiet(self):
        try:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            amp = float(self.amp_var.get().strip())
        except ValueError:
            return None

        if f_start <= 0 or f_stop < f_start or f_step <= 0:
            return None
        if amp <= 0 or amp > 3.0:
            return None

        return f'SWEEP {f_start:g} {f_stop:g} {f_step:g} {amp:g}\n'

    def _get_assumed_open_loop_poles(self):
        try:
            p = int(self.open_loop_p_var.get().strip())
        except ValueError:
            messagebox.showwarning('提示', '开环右半平面极点数 P 必须是非负整数。')
            return None
        if p < 0:
            messagebox.showwarning('提示', '开环右半平面极点数 P 必须是非负整数。')
            return None
        return p

    def _start_reader(self, continuous=False, command=None):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo('提示', '读取线程已在运行。')
            return

        if self._get_assumed_open_loop_poles() is None:
            return

        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning('提示', '请先选择串口。')
            return

        try:
            int(self.baud_var.get().strip())
            float(self.timeout_var.get().strip())
            int(self.window_var.get().strip())
            int(self.poly_var.get().strip())
        except ValueError:
            messagebox.showwarning('提示', '参数格式不正确。')
            return

        fallback_command = command if command is not None else self._build_sweep_command_quiet()

        self.stop_event.clear()
        self.reader = ThreeLineReader(
            port=self.port_var.get().strip(),
            baudrate=int(self.baud_var.get().strip()),
            timeout_sec=float(self.timeout_var.get().strip()),
            out_queue=self.queue,
            stop_event=self.stop_event,
            continuous=continuous,
            command=command,
            fallback_command=fallback_command
        )
        self.reader.start()
        self.status_var.set('读取中')

    def read_one_frame(self):
        self._start_reader(continuous=False)

    def command_sweep_once(self):
        command = self._build_sweep_command()
        if command is None:
            return
        self._start_reader(continuous=False, command=command)

    def command_auto_identify(self):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo('提示', '读取线程已在运行。')
            return
        if self._get_assumed_open_loop_poles() is None:
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning('提示', '请先选择串口。')
            return
        try:
            baudrate = int(self.baud_var.get().strip())
            timeout_sec = float(self.timeout_var.get().strip())
            int(self.window_var.get().strip())
            int(self.poly_var.get().strip())
        except ValueError:
            messagebox.showwarning('提示', '参数格式不正确。')
            return

        expected = self.expected_circuit_var.get().strip() or 'Auto'
        self.stop_event.clear()
        self.reader = AutoSweepReader(
            port=port,
            baudrate=baudrate,
            timeout_sec=timeout_sec,
            out_queue=self.queue,
            stop_event=self.stop_event,
            expected_circuit=expected,
            smooth=self.smooth_var.get(),
            window_length=int(self.window_var.get().strip()),
            polyorder=int(self.poly_var.get().strip()),
            fix_g431_axis=self.fix_var.get(),
            assumed_open_loop_rhp_poles=self._get_assumed_open_loop_poles() or 0,
            invert_transfer=self.swap_io_var.get(),
        )
        self.reader.start()
        self.status_var.set('自动识别扫频中')
        self.log(f'开始自动识别八电路；期望电路={expected}。')

    def read_continuous(self):
        self._start_reader(continuous=True)

    def stop_reading(self):
        self.stop_event.set()
        self.status_var.set('已停止')
        self.log('已请求停止读取。')

    def import_text(self):
        path = filedialog.askopenfilename(
            title='选择包含数组的文本文件',
            filetypes=[('Text', '*.txt *.log *.m *.csv'), ('All', '*.*')]
        )
        if not path:
            return

        try:
            text = Path(path).read_text(encoding='utf-8', errors='ignore')
            omega, mag, phase = parse_arrays_from_text(text)
            diagnostics = parse_diagnostics_from_text(text)
            self.handle_frame(omega, mag, phase, source=f'文件导入: {Path(path).name}', diagnostics=diagnostics)
        except Exception as e:
            messagebox.showerror('导入失败', str(e))

    def handle_frame(self, omega, mag, phase, source='串口', diagnostics=None):
        assumed_p = self._get_assumed_open_loop_poles()
        if assumed_p is None:
            return

        result = analyze_system_v2(
            omega, mag, phase,
            smooth=self.smooth_var.get(),
            window_length=int(self.window_var.get().strip()),
            polyorder=int(self.poly_var.get().strip()),
            fix_g431_axis=self.fix_var.get(),
            assumed_open_loop_rhp_poles=assumed_p,
            invert_transfer=self.swap_io_var.get(),
            diagnostics=diagnostics
        )
        expected = getattr(self, 'expected_circuit_var', tk.StringVar(value='Auto')).get().strip() or 'Auto'
        result.expected_circuit = expected
        result.expected_match_text = evaluate_expected_circuit(result, expected)
        result.notes.append(result.expected_match_text)
        self.result = result
        self.system_type_var.set(result.system_order)
        self.status_var.set(f'{source} 解析完成：{result.system_order}，点数 {len(result.omega)}')
        self.log(f'{source} 接收成功：系统类型 {result.system_order}，ωc={result.omega_c:.2f} rad/s，点数 {len(result.omega)}')

        for note in result.notes:
            self.log(note)

        self.append_report(result)
        self.update_plots(result)

    def append_report(self, r: SystemAnalysisResult):
        gm_text = '无法计算' if r.gain_margin_db is None else f'{r.gain_margin_db:.2f} dB'
        pm_text = '无法计算' if r.phase_margin is None else f'{r.phase_margin:.2f} °'
        stable_text = '稳定' if r.nyquist_stable else '不稳定'

        lines = [
            '=== 稳频仪：频响与奈奎斯特稳定性分析报告 ===',
            f'截止判据: {r.cutoff_method}',
            f'幅值 -3 dB 角频率: {r.magnitude_cutoff_omega:.2f} rad/s',
            f'相位 -45° 角频率: {"无法计算" if r.phase_cutoff_omega is None else f"{r.phase_cutoff_omega:.2f} rad/s"}',
            f'系统类型: {r.system_order}',
            f'估计阶次: {r.order_estimate}',
            f'数据点数: {len(r.omega)}',
            f'截止角频率 ωc = {r.omega_c:.2f} rad/s',
            f'截止点相位 = {r.cutoff_phase_deg:.2f} °',
            f'最大幅值 = {r.max_magnitude:.4f} ({20*np.log10(max(r.max_magnitude,1e-12)):.2f} dB)',
            f'谐振频率 ≈ {r.resonant_frequency:.2f} rad/s',
            f'到 (-1,0) 最小距离: {r.min_distance:.4f}',
            f'Nyquist 穿越点数（近似）: {len(r.crossing_points)}',
            f'Nyquist 逆时针包围数: {r.nyquist_winding_ccw}',
            f'Nyquist 顺时针包围数 N: {r.nyquist_encirclements_cw}',
            f'开环右半平面极点数 P: {r.assumed_open_loop_rhp_poles}',
            f'闭环右半平面极点数 Z = P + N: {r.estimated_closed_loop_rhp_poles}',
            f'奈奎斯特判据稳定性: {stable_text}',
            f'增益裕度: {gm_text}',
            f'相位裕度: {pm_text}',
            ''
        ]

        if r.order_estimate == 2 and r.damping_ratio is not None:
            lines += [
                '--- 二阶系统特征参数 ---',
                f'阻尼比 ζ = {r.damping_ratio:.4f}',
                f'自然频率 ωn = {r.natural_frequency:.2f} rad/s',
                f'超调量 = {r.overshoot_percent:.2f}%' if r.overshoot_percent is not None else '超调量 = 无法计算',
                f'调节时间(2%) ≈ {r.settling_time_est:.3f} s' if r.settling_time_est is not None else '调节时间 = 无法计算',
                ''
            ]

        lines.append(f'稳定性综合判定: {r.stability_text} (得分 {r.stability_score}/6)')
        lines.append('')

        self.text.insert('end', '\n'.join(lines))
        self.text.see('end')

    def update_plots(self, r: SystemAnalysisResult):
        self.ax_ny.clear()
        self.ax_mag.clear()
        self.ax_phase.clear()
        self.ax_stab.clear()

        # Nyquist 图
        self.ax_ny.plot(r.real_part, r.imag_part, 'b-', lw=2, label='H(jω) 轨迹')
        self.ax_ny.scatter(r.real_part, r.imag_part, s=30, c='r', label='数据点')
        self.ax_ny.plot(r.real_part[0], r.imag_part[0], 'go', ms=8, label='低频点')
        self.ax_ny.plot(r.real_part[-1], r.imag_part[-1], 'rx', ms=8, label='高频点')
        self.ax_ny.plot(r.real_part[r.cutoff_index], r.imag_part[r.cutoff_index], 'k*', ms=12, label=f'ωc 点 ({r.omega_c:.2f})')
        self.ax_ny.plot(-1, 0, 'ks', ms=8, label='临界点(-1,0)')
        self.ax_ny.axhline(0, color='0.65', linestyle=':', lw=0.8)
        self.ax_ny.axvline(-1, color='0.65', linestyle=':', lw=0.8)
        self.ax_ny.grid(True, alpha=0.35)
        self.ax_ny.axis('equal')

        max_abs = max(float(np.max(np.abs(r.real_part))), float(np.max(np.abs(r.imag_part))), 1.0) * 1.1
        self.ax_ny.set_xlim([-max_abs, max_abs])
        self.ax_ny.set_ylim([-max_abs, max_abs])
        self.ax_ny.set_title('奈奎斯特图 H(jω)')
        self.ax_ny.set_xlabel('实部 Re{H(jω)}')
        self.ax_ny.set_ylabel('虚部 Im{H(jω)}')
        self.ax_ny.legend(loc='best', fontsize=8)

        # 幅频响应
        self.ax_mag.semilogx(r.omega, r.mag_db, 'b-')
        self.ax_mag.axvline(r.omega_c, linestyle='--', color='red', lw=1.2)
        self.ax_mag.axhline(r.cutoff_mag_db, linestyle=':', color='black', lw=1.0)
        self.ax_mag.grid(True, which='both', alpha=0.35)
        self.ax_mag.set_title('幅频响应曲线 |H(jω)|(dB)')
        self.ax_mag.set_ylabel('幅值 (dB)')

        # 相频响应
        self.ax_phase.semilogx(r.omega, r.phase_deg, 'r-')
        self.ax_phase.axvline(r.omega_c, linestyle='--', color='red', lw=1.2)
        self.ax_phase.axhline(r.cutoff_phase_deg, linestyle=':', color='blue', lw=1.0)
        self.ax_phase.grid(True, which='both', alpha=0.35)
        self.ax_phase.set_title('相频响应曲线 φ(ω)')
        self.ax_phase.set_xlabel('角频率 ω (rad/s)')
        self.ax_phase.set_ylabel('相位 φ (度)')

        # 稳定性分析奈奎斯特图
        self.ax_stab.plot(r.real_part, r.imag_part, 'b-', lw=2, label='H(jω)轨迹')
        self.ax_stab.scatter(r.real_part, r.imag_part, s=30, c='r', label='数据点')
        self.ax_stab.plot(-1, 0, 'ks', ms=8, label='临界点(-1,j0)')
        self.ax_stab.axhline(0, color='0.65', linestyle=':', lw=0.8)
        self.ax_stab.axvline(-1, color='0.65', linestyle=':', lw=0.8)

        closest_idx = int(np.argmin(np.sqrt((r.real_part + 1.0)**2 + r.imag_part**2)))
        self.ax_stab.plot(r.real_part[closest_idx], r.imag_part[closest_idx], 'mo', ms=8,
                          label=f'最近点(距离={r.min_distance:.3f})')

        if len(r.crossing_points) > 0:
            for i, (a, _) in enumerate(r.crossing_points):
                self.ax_stab.plot(r.real_part[a], r.imag_part[a], 'g^', ms=8,
                                  label='可能穿越点' if i == 0 else None)

        max_abs2 = max(float(np.max(np.abs(r.real_part))), float(np.max(np.abs(r.imag_part))), 1.0) * 1.2
        self.ax_stab.set_xlim([-max_abs2, max_abs2])
        self.ax_stab.set_ylim([-max_abs2, max_abs2])
        self.ax_stab.grid(True, alpha=0.35)
        self.ax_stab.axis('equal')
        self.ax_stab.set_title('稳定性分析奈奎斯特图')
        self.ax_stab.set_xlabel('实部 Re{H(jω)}')
        self.ax_stab.set_ylabel('虚部 Im{H(jω)}')
        self.ax_stab.legend(loc='best', fontsize=8)

        color = 'green' if r.stability_score >= 5 else 'orange' if r.stability_score >= 3 else 'red'
        self.ax_stab.text(
            0.02, 0.98, r.stability_text,
            transform=self.ax_stab.transAxes,
            va='top', ha='left',
            color=color,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.8')
        )

        self.ax_stab.text(
            0.02, 0.88,
            f'阶次={r.order_estimate}，N={r.nyquist_encirclements_cw}，P={r.assumed_open_loop_rhp_poles}，Z=P+N={r.estimated_closed_loop_rhp_poles}',
            transform=self.ax_stab.transAxes,
            va='top', ha='left',
            color='black',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7)
        )

        if r.order_estimate == 2 and r.damping_ratio is not None:
            self.ax_stab.text(
                0.02, 0.78,
                f'阻尼比ζ={r.damping_ratio:.3f}，自然频率ωn={r.natural_frequency:.1f}',
                transform=self.ax_stab.transAxes,
                va='top', ha='left',
                color='blue',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7)
            )

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def save_csv(self):
        if self.result is None:
            messagebox.showinfo('提示', '当前没有数据可保存。')
            return

        path = filedialog.asksaveasfilename(
            title='保存 CSV',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')]
        )
        if not path:
            return

        r = self.result
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([
                'omega', 'Magnitude_raw', 'Magnitude_smooth',
                'Phase_raw_rad', 'Phase_smooth_rad',
                'Magnitude_dB', 'Phase_deg', 'Real_part', 'Imag_part'
            ])
            for row in zip(
                r.omega, r.mag_raw, r.mag_smooth, r.phase_raw_rad,
                r.phase_smooth_rad, r.mag_db, r.phase_deg,
                r.real_part, r.imag_part
            ):
                w.writerow(row)
        self.log(f'已保存 CSV: {path}')

    def save_png(self):
        if self.result is None:
            messagebox.showinfo('提示', '当前没有图像可保存。')
            return

        path = filedialog.asksaveasfilename(
            title='保存图像',
            defaultextension='.png',
            filetypes=[('PNG', '*.png')]
        )
        if not path:
            return

        self.fig.savefig(path, dpi=180, bbox_inches='tight')
        self.log(f'已保存图像: {path}')

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == 'log':
                    self.log(payload)
                elif kind == 'error':
                    self.status_var.set('发生错误')
                    self.log(f'错误: {payload}')
                    messagebox.showerror('错误', payload)
                elif kind == 'frame':
                    if len(payload) == 4:
                        omega, mag, phase, diagnostics = payload
                    else:
                        omega, mag, phase = payload
                        diagnostics = None
                    self.handle_frame(omega, mag, phase, diagnostics=diagnostics)
                elif kind == 'auto_frame':
                    omega, mag, phase, diagnostics, source, expected = payload
                    self.expected_circuit_var.set(expected)
                    self.handle_frame(omega, mag, phase, source=source, diagnostics=diagnostics)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def on_close(self):
        try:
            self.stop_event.set()
            if self.reader is not None:
                self.reader.join(timeout=1.0)
        except Exception:
            pass
        self.root.destroy()


class ThreeLineReader(threading.Thread):
    def __init__(self, port, baudrate, timeout_sec, out_queue, stop_event, continuous=False, command=None, fallback_command=None):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout_sec = timeout_sec
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.continuous = continuous
        self.command = command
        self.fallback_command = fallback_command

    def run(self):
        if serial is None:
            self.out_queue.put(('error', '未安装 pyserial，请先执行 pip install pyserial'))
            return

        ser = None
        try:
            ser = open_serial_transport(serial, self.port, self.baudrate, self.timeout_sec)

            if self.command:
                write_ascii_command(ser, self.command)
                self.out_queue.put(('log', f'已发送命令: {self.command.strip()}'))

            self.out_queue.put(('log', f'已连接串口 {self.port} @ {self.baudrate}'))
            self.out_queue.put(('log', '按兼容协议读取：必需三行 omega / Magnitude_data / Phase_data_rad，若存在则继续读取诊断数组。'))

            passive_deadline = time.monotonic() + max(0.8, self.timeout_sec * 2.5)
            fallback_sent = bool(self.command)
            if not self.command and self.fallback_command:
                self.out_queue.put(('log', '兼容模式：先等旧版自动三行输出，超时后自动补发 SWEEP。'))

            while not self.stop_event.is_set():
                line1 = ser.readline().decode(errors='ignore').strip()
                if not line1:
                    if (
                        not fallback_sent
                        and self.fallback_command
                        and time.monotonic() >= passive_deadline
                    ):
                        write_ascii_command(ser, self.fallback_command)
                        fallback_sent = True
                        self.out_queue.put(('log', f'Fallback command sent: {self.fallback_command.strip()}'))
                    continue

                if not NAME_PATTERNS['omega'].match(line1):
                    if (
                        not fallback_sent
                        and self.fallback_command
                        and time.monotonic() >= passive_deadline
                    ):
                        write_ascii_command(ser, self.fallback_command)
                        fallback_sent = True
                        self.out_queue.put(('log', f'Fallback command sent after non-frame traffic: {self.fallback_command.strip()}'))
                    continue

                line2 = ser.readline().decode(errors='ignore').strip()
                line3 = ser.readline().decode(errors='ignore').strip()

                if not NAME_PATTERNS['magnitude'].match(line2) or not NAME_PATTERNS['phase'].match(line3):
                    self.out_queue.put(('log', f'格式不匹配，已跳过本次帧'))
                    continue

                try:
                    omega = parse_array_from_line(line1)
                    mag = parse_array_from_line(line2)
                    phase = parse_array_from_line(line3)
                    diagnostics = read_optional_diagnostics_from_serial(ser)
                    if diagnostics:
                        self.out_queue.put(('log', f'已读取测量诊断数组: {", ".join(sorted(diagnostics.keys()))}'))
                    self.out_queue.put(('frame', (omega, mag, phase, diagnostics)))
                    if self.continuous and not self.command:
                        fallback_sent = False
                        passive_deadline = time.monotonic() + max(0.8, self.timeout_sec * 2.5)
                    if not self.continuous:
                        break
                except Exception as e:
                    self.out_queue.put(('log', f'解析失败: {e}'))

        except Exception as e:
            self.out_queue.put(('error', str(e)))
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            self.out_queue.put(('log', '串口读取线程已结束'))


class AutoSweepReader(threading.Thread):
    def __init__(
        self,
        port,
        baudrate,
        timeout_sec,
        out_queue,
        stop_event,
        expected_circuit='Auto',
        smooth=True,
        window_length=11,
        polyorder=3,
        fix_g431_axis=False,
        assumed_open_loop_rhp_poles=0,
        invert_transfer=False,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout_sec = timeout_sec
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.expected_circuit = expected_circuit or 'Auto'
        self.smooth = smooth
        self.window_length = window_length
        self.polyorder = polyorder
        self.fix_g431_axis = fix_g431_axis
        self.assumed_open_loop_rhp_poles = assumed_open_loop_rhp_poles
        self.invert_transfer = invert_transfer

    @staticmethod
    def _command_from_segment(segment):
        f_start, f_stop, f_step, amp = segment
        return build_sweep_command(f_start, f_stop, f_step, amp)

    def _candidate_type_from_coarse(self, frame):
        omega, mag, phase, diagnostics = frame
        expected_target = EXPECTED_CIRCUIT_MAP.get(self.expected_circuit)
        if expected_target is not None:
            return expected_target[0]
        try:
            result = analyze_system_v2(
                omega,
                mag,
                phase,
                smooth=self.smooth,
                window_length=self.window_length,
                polyorder=self.polyorder,
                fix_g431_axis=self.fix_g431_axis,
                assumed_open_loop_rhp_poles=self.assumed_open_loop_rhp_poles,
                invert_transfer=self.invert_transfer,
                diagnostics=diagnostics,
            )
            return result.filter_type if result.filter_type in AUTO_SWEEP_SEGMENTS else 'other'
        except Exception as exc:
            self.out_queue.put(('log', f'粗扫候选类型分析失败，改用全频补扫: {exc}'))
            return 'other'

    def run(self):
        if serial is None:
            self.out_queue.put(('error', '未安装 pyserial，请先执行 pip install pyserial'))
            return

        ser = None
        try:
            ser = open_serial_transport(
                serial,
                self.port,
                self.baudrate,
                timeout=max(0.2, min(self.timeout_sec, 1.0)),
            )

            frames = []
            coarse_command = self._command_from_segment(AUTO_COARSE_SWEEP)
            self.out_queue.put(('log', f'自动识别粗扫: {coarse_command.strip()}'))
            write_ascii_command(ser, coarse_command)
            coarse_frame = read_measurement_frame_from_serial(ser, self.stop_event, self.timeout_sec)
            frames.append(coarse_frame)

            candidate_type = self._candidate_type_from_coarse(coarse_frame)
            segments = AUTO_SWEEP_SEGMENTS.get(candidate_type, AUTO_SWEEP_SEGMENTS['other'])
            self.out_queue.put(('log', f'粗扫候选类型: {FILTER_TYPE_LABELS.get(candidate_type, candidate_type)}；开始补扫 {len(segments)} 段。'))

            sent = {coarse_command.strip()}
            for segment in segments:
                if self.stop_event.is_set():
                    break
                command = self._command_from_segment(segment)
                if command.strip() in sent:
                    continue
                sent.add(command.strip())
                self.out_queue.put(('log', f'自动识别补扫: {command.strip()}'))
                write_ascii_command(ser, command)
                frames.append(read_measurement_frame_from_serial(ser, self.stop_event, self.timeout_sec))

            omega, mag, phase, diagnostics = merge_measurement_frames(frames)
            source = f'自动识别({FILTER_TYPE_LABELS.get(candidate_type, candidate_type)}补扫)'
            self.out_queue.put(('auto_frame', (omega, mag, phase, diagnostics, source, self.expected_circuit)))
        except Exception as exc:
            self.out_queue.put(('error', f'自动识别失败: {exc}'))
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            self.out_queue.put(('log', '自动识别线程已结束'))


def parse_arrays_from_text(text: str):
    frame = parse_measurement_frame_from_text(text)
    return frame.omega, frame.magnitude, frame.phase


def read_measurement_frame_from_serial(ser, stop_event, timeout_sec: float):
    frame = read_protocol_measurement_frame_from_serial(ser, stop_event, timeout_sec)
    return frame.as_legacy_tuple()


def matlab_app_append_report_v2(self, r: SystemAnalysisResult):
    gm_text = '\u65e0\u6cd5\u8ba1\u7b97' if r.gain_margin_db is None else f'{r.gain_margin_db:.2f} dB'
    pm_text = '\u65e0\u6cd5\u8ba1\u7b97' if r.phase_margin is None else f'{r.phase_margin:.2f} \u00b0'
    stable_text = '\u7a33\u5b9a' if r.nyquist_stable else '\u4e0d\u7a33\u5b9a'
    peak_label = '\u9677\u6ce2\u89d2\u9891\u7387' if r.filter_type == 'bandstop' else '\u8c10\u632f\u89d2\u9891\u7387'

    lines = [
        '=== \u7a33\u9891\u4eea\uff1a\u9891\u54cd\u4e0e\u5948\u594e\u65af\u7279\u7a33\u5b9a\u6027\u5206\u6790\u62a5\u544a ===',
        f'\u7cfb\u7edf\u7c7b\u578b: {r.system_order}',
        f'\u6ee4\u6ce2\u5668\u65cf: {FILTER_TYPE_LABELS.get(r.filter_type, FILTER_TYPE_LABELS["other"])}',
        f'\u4f30\u8ba1\u9636\u6b21: {r.order_estimate}',
        f'\u5224\u9636\u53ef\u9760\u6027: {r.identification_summary}\uff08\u7f6e\u4fe1\u5ea6 {r.identification_confidence:.2f}\uff09',
        f'期望电路: {getattr(r, "expected_circuit", "Auto")}',
        f'期望匹配: {getattr(r, "expected_match_text", "未指定期望电路，按自动识别结果验收。")}',
        f'\u6570\u636e\u70b9\u6570: {len(r.omega)}',
        f'\u622a\u6b62/\u7279\u5f81\u5224\u636e: {r.cutoff_method}',
    ]

    lines.append('--- 识别证据 ---')
    lines.extend(build_identification_evidence(r))
    diagnostic_notes = [
        note for note in r.notes
        if any(key in note for key in ('测量质量', '峰峰值', '重复采集', '采样信息', '削顶', '偏置', 'RMS'))
    ]
    if diagnostic_notes:
        lines.append('--- 测量质量诊断 ---')
        lines.extend(diagnostic_notes[:8])

    if r.filter_type in ('lowpass', 'highpass'):
        lines.extend([
            f'-3 dB \u622a\u6b62\u89d2\u9891\u7387 = {r.magnitude_cutoff_omega:.2f} rad/s',
            f'\u76f8\u4f4d -45\u00b0 \u89d2\u9891\u7387 = {"\u65e0\u6cd5\u8ba1\u7b97" if r.phase_cutoff_omega is None else f"{r.phase_cutoff_omega:.2f} rad/s"}',
        ])
    elif r.filter_type == 'bandpass':
        lines.extend([
            f'\u5de6\u4fa7 -3 dB \u89d2\u9891\u7387 = {r.magnitude_cutoff_omega:.2f} rad/s',
            f'\u53f3\u4fa7 -3 dB \u89d2\u9891\u7387 = {"\u65e0\u6cd5\u8ba1\u7b97" if r.secondary_cutoff_omega is None else f"{r.secondary_cutoff_omega:.2f} rad/s"}',
            f'\u4e2d\u5fc3\u89d2\u9891\u7387 \u03c90 = {r.omega_c:.2f} rad/s',
            f'\u5e26\u5bbd = {"\u65e0\u6cd5\u8ba1\u7b97" if r.bandwidth_omega is None else f"{r.bandwidth_omega:.2f} rad/s"}',
        ])
    elif r.filter_type == 'bandstop':
        lines.extend([
            f'\u5de6\u4fa7 -3 dB \u89d2\u9891\u7387 = {r.magnitude_cutoff_omega:.2f} rad/s',
            f'\u53f3\u4fa7 -3 dB \u89d2\u9891\u7387 = {"\u65e0\u6cd5\u8ba1\u7b97" if r.secondary_cutoff_omega is None else f"{r.secondary_cutoff_omega:.2f} rad/s"}',
            f'\u9677\u6ce2\u89d2\u9891\u7387 \u03c90 = {r.omega_c:.2f} rad/s',
            f'\u963b\u5e26\u5bbd\u5ea6 = {"\u65e0\u6cd5\u8ba1\u7b97" if r.bandwidth_omega is None else f"{r.bandwidth_omega:.2f} rad/s"}',
        ])
    else:
        lines.append(f'\u4e3b\u7279\u5f81\u89d2\u9891\u7387 = {r.omega_c:.2f} rad/s')

    lines.extend([
        f'\u7279\u5f81\u9891\u70b9\u76f8\u4f4d = {r.cutoff_phase_deg:.2f} \u00b0',
        f'\u5e45\u9891\u9608\u503c = {r.cutoff_mag_db:.2f} dB',
        f'\u6700\u5927\u5e45\u503c = {r.max_magnitude:.4f} ({20*np.log10(max(r.max_magnitude, 1e-12)):.2f} dB)',
        f'{peak_label} \u2248 {r.resonant_frequency:.2f} rad/s',
        f'\u5230 (-1,0) \u6700\u5c0f\u8ddd\u79bb = {r.min_distance:.4f}',
        f'Nyquist \u7a7f\u8d8a\u70b9\u6570\uff08\u8fd1\u4f3c\uff09: {len(r.crossing_points)}',
        f'Nyquist \u9006\u65f6\u9488\u5305\u56f4\u6570: {r.nyquist_winding_ccw}',
        f'Nyquist \u987a\u65f6\u9488\u5305\u56f4\u6570 N: {r.nyquist_encirclements_cw}',
        f'\u5f00\u73af\u53f3\u534a\u5e73\u9762\u6781\u70b9\u6570 P: {r.assumed_open_loop_rhp_poles}',
        f'\u95ed\u73af\u53f3\u534a\u5e73\u9762\u6781\u70b9\u6570 Z = P + N: {r.estimated_closed_loop_rhp_poles}',
        f'\u5948\u594e\u65af\u7279\u5224\u636e\u7a33\u5b9a\u6027: {stable_text}',
        f'\u589e\u76ca\u88d5\u5ea6: {gm_text}',
        f'\u76f8\u4f4d\u88d5\u5ea6: {pm_text}',
        '',
    ])

    if r.filter_type == 'lowpass' and r.order_estimate == 2 and r.damping_ratio is not None:
        lines += [
            '--- \u4e8c\u9636\u4f4e\u901a\u7279\u5f81\u53c2\u6570 ---',
            f'\u963b\u5c3c\u6bd4 \u03b6 = {r.damping_ratio:.4f}',
            f'\u81ea\u7136\u89d2\u9891\u7387 \u03c9n = {r.natural_frequency:.2f} rad/s',
            f'\u8d85\u8c03\u91cf = {r.overshoot_percent:.2f}%' if r.overshoot_percent is not None else '\u8d85\u8c03\u91cf = \u65e0\u6cd5\u8ba1\u7b97',
            f'\u8c03\u8282\u65f6\u95f4(2%) \u2248 {r.settling_time_est:.3f} s' if r.settling_time_est is not None else '\u8c03\u8282\u65f6\u95f4 = \u65e0\u6cd5\u8ba1\u7b97',
            '',
        ]

    lines.append(f'\u7a33\u5b9a\u6027\u7efc\u5408\u5224\u5b9a: {r.stability_text} (\u5f97\u5206 {r.stability_score}/6)')
    lines.append('')
    self.text.insert('end', '\n'.join(lines))
    self.text.see('end')


def matlab_app_update_plots_v2(self, r: SystemAnalysisResult):
    self.ax_ny.clear()
    self.ax_mag.clear()
    self.ax_phase.clear()
    self.ax_stab.clear()

    extra_markers = []
    if not np.isclose(r.magnitude_cutoff_omega, r.omega_c):
        extra_markers.append(float(r.magnitude_cutoff_omega))
    if r.secondary_cutoff_omega is not None and not np.isclose(r.secondary_cutoff_omega, r.omega_c):
        extra_markers.append(float(r.secondary_cutoff_omega))

    h_positive = r.real_part + 1j * r.imag_part
    h_full = build_closed_nyquist_curve(h_positive)

    self.ax_ny.plot(h_full.real, h_full.imag, color='0.72', linestyle='--', lw=1.2, label='完整Nyquist镜像')
    if r.filter_type == 'lowpass' and r.order_estimate == 1 and r.omega_c > 0.0:
        low_count = max(3, min(8, len(r.mag_smooth)))
        dc_gain = float(np.median(r.mag_smooth[:low_count]))
        omega_ref = np.geomspace(max(float(np.min(r.omega)), 1e-12), max(float(np.max(r.omega)), 1e-12), 300)
        h_ref = dc_gain / (1.0 + 1j * omega_ref / float(r.omega_c))
        self.ax_ny.plot(h_ref.real, h_ref.imag, color='darkorange', linestyle=':', lw=1.8, label='一阶低通理论参考')

    self.ax_ny.plot(r.real_part, r.imag_part, 'b-', lw=2, label='H(j\u03c9) 正频率')
    self.ax_ny.scatter(r.real_part, r.imag_part, s=30, c='r', label='\u6570\u636e\u70b9')
    self.ax_ny.plot(r.real_part[0], r.imag_part[0], 'go', ms=8, label='\u4f4e\u9891\u70b9')
    self.ax_ny.plot(r.real_part[-1], r.imag_part[-1], 'rx', ms=8, label='\u9ad8\u9891\u70b9')
    self.ax_ny.plot(r.real_part[r.cutoff_index], r.imag_part[r.cutoff_index], 'k*', ms=12, label=f'\u7279\u5f81\u70b9 ({r.omega_c:.2f})')
    self.ax_ny.plot(-1, 0, 'ks', ms=8, label='\u4e34\u754c\u70b9 (-1,0)')
    self.ax_ny.axhline(0, color='0.65', linestyle=':', lw=0.8)
    self.ax_ny.axvline(-1, color='0.65', linestyle=':', lw=0.8)
    self.ax_ny.grid(True, alpha=0.35)
    self.ax_ny.set_aspect('equal', adjustable='box')

    max_abs = max(float(np.max(np.abs(h_full.real))), float(np.max(np.abs(h_full.imag))), 1.0) * 1.1
    self.ax_ny.set_xlim([-max_abs, max_abs])
    self.ax_ny.set_ylim([-max_abs, max_abs])
    self.ax_ny.set_title('\u5948\u594e\u65af\u7279\u56fe H(j\u03c9)')
    self.ax_ny.set_xlabel('\u5b9e\u90e8 Re{H(j\u03c9)}')
    self.ax_ny.set_ylabel('\u865a\u90e8 Im{H(j\u03c9)}')
    self.ax_ny.legend(loc='best', fontsize=8)

    self.ax_mag.semilogx(r.omega, r.mag_db, 'b-')
    self.ax_mag.axvline(r.omega_c, linestyle='--', color='red', lw=1.2)
    for marker in extra_markers:
        self.ax_mag.axvline(marker, linestyle=':', color='darkorange', lw=1.0)
    self.ax_mag.axhline(r.cutoff_mag_db, linestyle=':', color='black', lw=1.0)
    self.ax_mag.grid(True, which='both', alpha=0.35)
    self.ax_mag.set_title('\u5e45\u9891\u54cd\u5e94\u66f2\u7ebf |H(j\u03c9)|(dB)')
    self.ax_mag.set_ylabel('\u5e45\u503c (dB)')

    self.ax_phase.semilogx(r.omega, r.phase_deg, 'r-')
    self.ax_phase.axvline(r.omega_c, linestyle='--', color='red', lw=1.2)
    for marker in extra_markers:
        self.ax_phase.axvline(marker, linestyle=':', color='darkorange', lw=1.0)
    self.ax_phase.axhline(r.cutoff_phase_deg, linestyle=':', color='blue', lw=1.0)
    self.ax_phase.grid(True, which='both', alpha=0.35)
    self.ax_phase.set_title('\u76f8\u9891\u54cd\u5e94\u66f2\u7ebf \u03c6(\u03c9)')
    self.ax_phase.set_xlabel('\u89d2\u9891\u7387 \u03c9 (rad/s)')
    self.ax_phase.set_ylabel('\u76f8\u4f4d \u03c6 (\u00b0)')

    self.ax_stab.plot(h_full.real, h_full.imag, color='0.72', linestyle='--', lw=1.2, label='完整Nyquist镜像')
    self.ax_stab.plot(r.real_part, r.imag_part, 'b-', lw=2, label='H(j\u03c9)正频率')
    self.ax_stab.scatter(r.real_part, r.imag_part, s=30, c='r', label='\u6570\u636e\u70b9')
    self.ax_stab.plot(-1, 0, 'ks', ms=8, label='\u4e34\u754c\u70b9 (-1,j0)')
    self.ax_stab.axhline(0, color='0.65', linestyle=':', lw=0.8)
    self.ax_stab.axvline(-1, color='0.65', linestyle=':', lw=0.8)

    closest_idx = int(np.argmin(np.sqrt((r.real_part + 1.0) ** 2 + r.imag_part ** 2)))
    self.ax_stab.plot(
        r.real_part[closest_idx], r.imag_part[closest_idx], 'mo', ms=8,
        label=f'\u6700\u8fd1\u70b9(\u8ddd\u79bb={r.min_distance:.3f})'
    )

    if len(r.crossing_points) > 0:
        for i, (a, _) in enumerate(r.crossing_points):
            self.ax_stab.plot(r.real_part[a], r.imag_part[a], 'g^', ms=8, label='\u53ef\u80fd\u7a7f\u8d8a\u70b9' if i == 0 else None)

    max_abs2 = max(float(np.max(np.abs(h_full.real))), float(np.max(np.abs(h_full.imag))), 1.0) * 1.2
    self.ax_stab.set_xlim([-max_abs2, max_abs2])
    self.ax_stab.set_ylim([-max_abs2, max_abs2])
    self.ax_stab.grid(True, alpha=0.35)
    self.ax_stab.set_aspect('equal', adjustable='box')
    self.ax_stab.set_title('\u7a33\u5b9a\u6027\u5206\u6790\u5948\u594e\u65af\u7279\u56fe')
    self.ax_stab.set_xlabel('\u5b9e\u90e8 Re{H(j\u03c9)}')
    self.ax_stab.set_ylabel('\u865a\u90e8 Im{H(j\u03c9)}')
    self.ax_stab.legend(loc='best', fontsize=8)

    color = 'green' if r.stability_score >= 5 else 'orange' if r.stability_score >= 3 else 'red'
    self.ax_stab.text(
        0.02, 0.98, r.stability_text,
        transform=self.ax_stab.transAxes,
        va='top', ha='left',
        color=color,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.8')
    )
    self.ax_stab.text(
        0.02, 0.88,
        f'\u9636\u6b21={r.order_estimate}\uff08{r.identification_summary}\uff09\uff0cN={r.nyquist_encirclements_cw}\uff0cP={r.assumed_open_loop_rhp_poles}\uff0cZ=P+N={r.estimated_closed_loop_rhp_poles}',
        transform=self.ax_stab.transAxes,
        va='top', ha='left',
        color='black',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    self.ax_stab.text(
        0.02, 0.78,
        FILTER_TYPE_LABELS.get(r.filter_type, FILTER_TYPE_LABELS['other']),
        transform=self.ax_stab.transAxes,
        va='top', ha='left',
        color='blue',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    self.fig.tight_layout()
    self.canvas.draw_idle()


MatlabExactApp.append_report = matlab_app_append_report_v2
MatlabExactApp.update_plots = matlab_app_update_plots_v2


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use('vista')
    except Exception:
        pass

    app = MatlabExactApp(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
