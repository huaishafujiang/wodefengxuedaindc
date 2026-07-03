from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from system_analysis.io.serial_protocol import DIAGNOSTIC_PATTERNS

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


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
MAX_STANDARD_SIMPLE_FILTER_ORDER = 3
COMPACT_DIAGNOSTIC_KEYS = {'adc_code_range', 'clip_point_count'}
PER_POINT_DIAGNOSTIC_KEYS = tuple(
    key for key in DIAGNOSTIC_PATTERNS
    if key not in COMPACT_DIAGNOSTIC_KEYS
)
LOWPASS_OUTPUT_NOISE_FLOOR_RMS_V = 0.004
LOWPASS_NOISE_FLOOR_DROP_DB = 30.0
LOWPASS_NOISE_FLOOR_MIN_KEEP_POINTS = 12
LOWPASS_NOISE_FLOOR_MIN_REMOVE_POINTS = 3
HIGHPASS_OUTPUT_NOISE_FLOOR_RMS_V = 0.004
HIGHPASS_NOISE_FLOOR_DROP_DB = 30.0
HIGHPASS_NOISE_FLOOR_MIN_KEEP_POINTS = 12
HIGHPASS_NOISE_FLOOR_MIN_REMOVE_POINTS = 3
MIN_VALID_ANALYSIS_POINTS = 5
MAX_FREQ_STEPS = 200

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
        (100.0, 3000.0, 50.0, 1.2),
        (3000.0, 20000.0, 300.0, 1.2),
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

DEMO_SWEEP_SEGMENTS = {
    ('lowpass', 1): [
        (50.0, 8000.0, 100.0, 0.6),
        (8000.0, 60000.0, 800.0, 0.6),
    ],
    ('lowpass', 2): [
        (50.0, 8000.0, 100.0, 0.6),
        (8000.0, 60000.0, 800.0, 0.6),
    ],
    ('lowpass', 3): [
        (100.0, 3000.0, 50.0, 1.2),
        (3000.0, 20000.0, 300.0, 1.2),
    ],
    ('highpass', 1): [
        (50.0, 8000.0, 100.0, 0.6),
        (8000.0, 60000.0, 800.0, 0.6),
    ],
    ('highpass', 2): [
        (50.0, 8000.0, 100.0, 0.6),
        (8000.0, 60000.0, 800.0, 0.6),
    ],
    ('highpass', 3): [
        (50.0, 8000.0, 100.0, 0.6),
        (8000.0, 60000.0, 800.0, 0.6),
    ],
    ('bandpass', None): [
        (50.0, 2500.0, 50.0, 0.6),
        (2500.0, 25000.0, 250.0, 0.6),
        (25000.0, 80000.0, 1000.0, 0.6),
    ],
    ('bandstop', None): [
        (100.0, 1000.0, 100.0, 0.6),
        (1000.0, 2400.0, 25.0, 0.6),
        (2400.0, 30000.0, 300.0, 0.6),
        (30000.0, 80000.0, 1000.0, 0.6),
    ],
}


def demo_segments_for_expected(expected: str):
    target = EXPECTED_CIRCUIT_MAP.get(expected)
    if target is None:
        return None
    return DEMO_SWEEP_SEGMENTS.get(target) or DEMO_SWEEP_SEGMENTS.get((target[0], None))


def sweep_segment_point_count(f_start: float, f_stop: float, f_step: float) -> int:
    if f_start <= 0.0 or f_stop < f_start or f_step <= 0.0:
        return 0
    return int((float(f_stop) - float(f_start)) / float(f_step)) + 1


def get_visible_sweep_presets() -> list[dict]:
    presets: list[dict] = []
    for label in EXPECTED_CIRCUIT_CHOICES:
        if label == 'Auto':
            continue
        segments = demo_segments_for_expected(label) or []
        visible_segments = []
        total_points = 0
        for index, segment in enumerate(segments, start=1):
            f_start, f_stop, f_step, amplitude_vpp = segment
            points = sweep_segment_point_count(f_start, f_stop, f_step)
            total_points += points
            visible_segments.append(
                {
                    'index': index,
                    'f_start_hz': float(f_start),
                    'f_stop_hz': float(f_stop),
                    'f_step_hz': float(f_step),
                    'amplitude_vpp': float(amplitude_vpp),
                    'points': points,
                }
            )
        presets.append(
            {
                'label': label,
                'target': EXPECTED_CIRCUIT_MAP.get(label),
                'segments': visible_segments,
                'total_points': total_points,
            }
        )
    return presets


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
    input_min_code = get_array('input_min_code')
    input_max_code = get_array('input_max_code')
    output_min_code = get_array('output_min_code')
    output_max_code = get_array('output_max_code')
    adc_code_range = diagnostics.get('adc_code_range')
    clip_point_count = diagnostics.get('clip_point_count')

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

    if (
        input_min_code is not None and input_max_code is not None
        and output_min_code is not None and output_max_code is not None
    ):
        in_min_values = input_min_code[np.isfinite(input_min_code)]
        in_max_values = input_max_code[np.isfinite(input_max_code)]
        out_min_values = output_min_code[np.isfinite(output_min_code)]
        out_max_values = output_max_code[np.isfinite(output_max_code)]
        if len(in_min_values) and len(in_max_values) and len(out_min_values) and len(out_max_values):
            in_min = int(np.min(in_min_values))
            in_max = int(np.max(in_max_values))
            out_min = int(np.min(out_min_values))
            out_max = int(np.max(out_max_values))
            notes.append(
                f'ADC原始码范围: PA0 {in_min}..{in_max}，PA1 {out_min}..{out_max}。'
            )
            if out_min <= 4 or out_max >= 4091:
                notes.append('PA1 原始码已经触边，串口读数不是解析错，而是输出通道发生削顶或偏置/接线异常。')
            if in_min <= 4 or in_max >= 4091:
                notes.append('PA0 原始码已经触边，请降低 DAC 幅度或检查 DA1/AD1 同点连接。')
    elif adc_code_range is not None:
        adc_code_range = np.asarray(adc_code_range, dtype=float).flatten()
        if len(adc_code_range) >= 4:
            in_min, in_max, out_min, out_max = [int(round(x)) for x in adc_code_range[:4]]
            notes.append(
                f'ADC原始码范围: PA0 {in_min}..{in_max}，PA1 {out_min}..{out_max}。'
            )
            if out_min <= 4 or out_max >= 4091:
                notes.append('PA1 原始码已经触边，串口读数不是解析错，而是输出通道发生削顶或偏置/接线异常。')
            if in_min <= 4 or in_max >= 4091:
                notes.append('PA0 原始码已经触边，请降低 DAC 幅度或检查 DA1/AD1 同点连接。')

    if clip_point_count is not None:
        clip_point_count = np.asarray(clip_point_count, dtype=float).flatten()
        if len(clip_point_count) >= 2 and np.all(np.isfinite(clip_point_count[:2])):
            notes.append(
                f'削顶频点统计: PA0 {int(round(clip_point_count[0]))} 个，PA1 {int(round(clip_point_count[1]))} 个。'
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
        finite_valid_count = np.asarray(valid_count, dtype=float)
        finite_valid_count = finite_valid_count[np.isfinite(finite_valid_count)]
        if len(finite_valid_count):
            min_valid = int(np.min(finite_valid_count))
            if min_valid < 2:
                notes.append(f'有效采集次数最低为 {min_valid}，部分频点抗噪不足；建议复测或降低步进/幅度。')

    if mag_repeat_span is not None and len(mag_repeat_span):
        finite_mag_repeat_span = np.asarray(mag_repeat_span, dtype=float)
        finite_mag_repeat_span = finite_mag_repeat_span[np.isfinite(finite_mag_repeat_span)]
        if len(finite_mag_repeat_span):
            notes.append(
                f'重复采集幅值稳定性: 中位跨度 {float(np.median(finite_mag_repeat_span)):.3f} dB，最大跨度 {float(np.max(finite_mag_repeat_span)):.3f} dB。'
            )

    if phase_repeat_span is not None and len(phase_repeat_span):
        finite_phase_repeat_span = np.asarray(phase_repeat_span, dtype=float)
        finite_phase_repeat_span = finite_phase_repeat_span[np.isfinite(finite_phase_repeat_span)]
        if len(finite_phase_repeat_span):
            notes.append(
                f'重复采集相位稳定性: 中位跨度 {float(np.median(np.abs(finite_phase_repeat_span))):.4f} rad，最大跨度 {float(np.max(np.abs(finite_phase_repeat_span))):.4f} rad。'
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
        if key in COMPACT_DIAGNOSTIC_KEYS:
            aligned[key] = arr
            continue
        if len(arr) < expected_len:
            continue
        arr = arr[:expected_len]
        if sort_idx is not None and len(sort_idx) == expected_len:
            arr = arr[sort_idx]
        aligned[key] = arr
    return aligned


def describe_invalid_measurement_frame(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None,
    min_valid_points: int = MIN_VALID_ANALYSIS_POINTS,
) -> tuple[bool, list[str], int]:
    diag = aligned_diagnostics(diagnostics, len(omega))
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    phase_rad = np.asarray(phase_rad, dtype=float).flatten()
    n = min(len(omega), len(magnitude), len(phase_rad))
    if n == 0:
        return True, ['测量帧为空，无法分析系统类型。'], 0

    omega = omega[:n]
    magnitude = magnitude[:n]
    phase_rad = phase_rad[:n]
    valid = (
        np.isfinite(omega)
        & np.isfinite(magnitude)
        & np.isfinite(phase_rad)
        & (omega > 0.0)
        & (magnitude > 0.0)
    )

    clip_flags = diag.get('clip_flags')
    clipped_points = 0
    output_clipped = 0
    input_clipped = 0
    if clip_flags is not None and len(clip_flags) >= n:
        clip_flags = np.asarray(clip_flags[:n], dtype=float)
        clip_flags = np.where(np.isfinite(clip_flags), clip_flags, 0.0)
        input_clipped = int(np.count_nonzero((clip_flags.astype(int) & 0x01) != 0))
        output_clipped = int(np.count_nonzero((clip_flags.astype(int) & 0x02) != 0))
        clipped_points = int(np.count_nonzero(clip_flags))
        valid &= clip_flags == 0

    valid_count = diag.get('valid_capture_count')
    if valid_count is not None and len(valid_count) >= n:
        valid_count = np.asarray(valid_count[:n], dtype=float)
        valid &= valid_count >= 1

    input_rms = diag.get('input_rms_v')
    if input_rms is not None and len(input_rms) >= n:
        input_rms = np.asarray(input_rms[:n], dtype=float)
        valid &= input_rms >= 0.02

    valid_points = int(np.count_nonzero(valid))
    if valid_points >= int(min_valid_points):
        return False, [], valid_points

    notes = [
        f'有效频点不足：当前只有 {valid_points}/{n} 个点可用于系统识别，至少需要 {int(min_valid_points)} 个。',
    ]
    zero_mag_count = int(np.count_nonzero(np.isfinite(magnitude) & (magnitude <= 0.0)))
    if zero_mag_count:
        notes.append(f'Magnitude_data 中有 {zero_mag_count}/{n} 个点为 0 或负值，无法形成可用的 Bode 幅频曲线。')

    if clipped_points:
        notes.append(f'ADC 削顶频点 {clipped_points}/{n} 个，其中 PA0={input_clipped} 个，PA1={output_clipped} 个。')
    else:
        clip_point_count = diag.get('clip_point_count')
        if clip_point_count is not None:
            clip_point_count = np.asarray(clip_point_count, dtype=float).flatten()
            if len(clip_point_count) >= 2:
                notes.append(
                    f'削顶频点统计：PA0={int(round(clip_point_count[0]))} 个，'
                    f'PA1={int(round(clip_point_count[1]))} 个。'
                )

    adc_code_range = diag.get('adc_code_range')
    if adc_code_range is not None:
        adc_code_range = np.asarray(adc_code_range, dtype=float).flatten()
        if len(adc_code_range) >= 4:
            in_min, in_max, out_min, out_max = [int(round(x)) for x in adc_code_range[:4]]
            notes.append(f'ADC原始码范围：PA0 {in_min}..{in_max}，PA1 {out_min}..{out_max}。')
            if out_min <= 4 or out_max >= 4091:
                notes.append('PA1 输出通道已经贴近 0V/3.3V 边界；这不是三阶识别算法能修正的数据。')

    output_dc = diag.get('output_dc_v')
    if output_dc is not None and len(output_dc) >= n:
        output_dc = np.asarray(output_dc[:n], dtype=float)
        if len(output_dc):
            out_dc_med = float(np.nanmedian(output_dc))
            notes.append(f'PA1 输出 DC 中位数约 {out_dc_med:.3f} V。')
            if out_dc_med < 0.35 or out_dc_med > 2.95:
                notes.append('PA1 偏置不在 STM32 ADC 中间区，建议检查输出节点偏置/接线，并把高通、带通、运放输出偏到约 1.65V。')

    output_rms = diag.get('output_rms_v')
    if output_rms is not None and len(output_rms) >= n:
        output_rms = np.asarray(output_rms[:n], dtype=float)
        notes.append(f'PA1 输出 RMS 中位数约 {float(np.nanmedian(output_rms)):.4f} V。')

    notes.append('本帧已拒绝判阶；请先修正 PA1 输出削顶/接线/偏置后重新扫频。')
    return True, notes, valid_points


def merge_measurement_frames(
    frames: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    valid_frames = []
    for frame_index, frame in enumerate(frames):
        if hasattr(frame, 'as_legacy_tuple'):
            omega, magnitude, phase, diagnostics = frame.as_legacy_tuple()
        else:
            omega, magnitude, phase, diagnostics = frame
        omega = np.asarray(omega, dtype=float).flatten()
        magnitude = np.asarray(magnitude, dtype=float).flatten()
        phase = np.asarray(phase, dtype=float).flatten()
        n = min(len(omega), len(magnitude), len(phase))
        if n < 3:
            continue
        order = np.argsort(omega[:n], kind='mergesort')
        valid_frames.append((
            omega[:n][order],
            magnitude[:n][order],
            np.unwrap(phase[:n][order]),
            aligned_diagnostics(diagnostics, n),
            frame_index,
            order,
        ))

    if not valid_frames:
        raise ValueError('自动扫频没有获得有效数据帧。')

    adjusted_frames = []
    reference_omega = np.array([], dtype=float)
    reference_phase = np.array([], dtype=float)
    for omega, magnitude, phase, diagnostics, frame_index, order in valid_frames:
        adjusted_phase = phase.copy()
        if len(reference_omega) >= 2:
            overlap = (
                np.isfinite(omega)
                & (omega >= float(reference_omega[0]))
                & (omega <= float(reference_omega[-1]))
            )
            if int(np.count_nonzero(overlap)) >= 2:
                ref_phase = np.interp(omega[overlap], reference_omega, reference_phase)
                delta = float(np.nanmedian(ref_phase - adjusted_phase[overlap]))
                if np.isfinite(delta):
                    adjusted_phase += delta
        adjusted_frames.append((omega, magnitude, adjusted_phase, diagnostics, frame_index, order))
        reference_omega = np.concatenate([reference_omega, omega])
        reference_phase = np.concatenate([reference_phase, adjusted_phase])
        ref_order = np.argsort(reference_omega, kind='mergesort')
        reference_omega = reference_omega[ref_order]
        reference_phase = np.unwrap(reference_phase[ref_order])

    omega_all = np.concatenate([item[0] for item in adjusted_frames])
    mag_all = np.concatenate([item[1] for item in adjusted_frames])
    phase_all = np.concatenate([item[2] for item in adjusted_frames])
    frame_index_all = np.concatenate([
        np.full(len(item[0]), int(item[4]), dtype=int) for item in adjusted_frames
    ])
    diag_all: dict[str, list[np.ndarray]] = {key: [] for key in PER_POINT_DIAGNOSTIC_KEYS}
    compact_diag: dict[str, list[np.ndarray]] = {key: [] for key in COMPACT_DIAGNOSTIC_KEYS}
    for omega, _, _, diagnostics, _, order in adjusted_frames:
        for key in PER_POINT_DIAGNOSTIC_KEYS:
            arr = diagnostics.get(key)
            if arr is None or len(arr) != len(omega):
                diag_all[key].append(np.full(len(omega), np.nan, dtype=float))
            else:
                diag_all[key].append(np.asarray(arr, dtype=float)[order])
        for key in COMPACT_DIAGNOSTIC_KEYS:
            arr = diagnostics.get(key)
            if arr is not None:
                compact_diag[key].append(np.asarray(arr, dtype=float).flatten())

    order = np.argsort(omega_all, kind='mergesort')
    omega_all = omega_all[order]
    mag_all = mag_all[order]
    phase_all = np.unwrap(phase_all[order])
    frame_index_all = frame_index_all[order]

    selected_indices: list[int] = []
    if len(omega_all) > 1:
        rel_tol = 2.0e-4
        group_start = 0
        for i in range(1, len(omega_all) + 1):
            in_group = False
            if i < len(omega_all):
                in_group = abs(omega_all[i] - omega_all[group_start]) <= max(
                    abs(omega_all[group_start]) * rel_tol,
                    1e-9,
                )
            if in_group:
                continue
            group = np.arange(group_start, i, dtype=int)
            selected_indices.append(int(group[np.argmax(frame_index_all[group])]))
            group_start = i
    elif len(omega_all) == 1:
        selected_indices.append(0)

    selected = np.asarray(selected_indices, dtype=int)

    merged_diag: dict[str, np.ndarray] = {}
    for key, chunks in diag_all.items():
        if chunks:
            merged_diag[key] = np.concatenate(chunks)[order][selected]
    if compact_diag['adc_code_range']:
        ranges = [arr for arr in compact_diag['adc_code_range'] if len(arr) >= 4]
        if ranges:
            merged_diag['adc_code_range'] = np.array([
                min(float(arr[0]) for arr in ranges),
                max(float(arr[1]) for arr in ranges),
                min(float(arr[2]) for arr in ranges),
                max(float(arr[3]) for arr in ranges),
            ], dtype=float)
    if compact_diag['clip_point_count']:
        counts = [arr for arr in compact_diag['clip_point_count'] if len(arr) >= 2]
        if counts:
            merged_diag['clip_point_count'] = np.array([
                sum(float(arr[0]) for arr in counts),
                sum(float(arr[1]) for arr in counts),
            ], dtype=float)

    return omega_all[selected], mag_all[selected], phase_all[selected], merged_diag


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

    notes.append(
        f'质量筛选发现 {removed} 个无效/削顶/参考过小频点；按当前设置保留全部 {len(valid)} 个频点继续分析和绘图。'
    )
    return omega, magnitude, phase_rad, diag, notes


def trim_lowpass_noise_floor_for_analysis(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    notes: list[str] = []
    diag = aligned_diagnostics(diagnostics, len(omega))
    output_rms = diag.get('output_rms_v')
    if output_rms is None or len(output_rms) != len(omega) or len(omega) < LOWPASS_NOISE_FLOOR_MIN_KEEP_POINTS:
        return omega, magnitude, phase_rad, diag, notes

    output_rms = np.asarray(output_rms, dtype=float)
    mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
    passband_count = max(3, min(8, len(mag_db) // 6))
    passband_db = float(np.median(mag_db[:passband_count]))
    low_output = (
        np.isfinite(output_rms)
        & (
            (
                (output_rms <= LOWPASS_OUTPUT_NOISE_FLOOR_RMS_V)
                & (mag_db <= passband_db - LOWPASS_NOISE_FLOOR_DROP_DB)
            )
            | (
                (output_rms <= 0.020)
                & (mag_db <= passband_db - 24.0)
            )
        )
    )

    if not np.any(low_output):
        return omega, magnitude, phase_rad, diag, notes

    first_low = int(np.argmax(low_output))
    tail_mask = np.zeros(len(omega), dtype=bool)
    tail_mask[first_low:] = low_output[first_low:]

    keep = ~tail_mask
    removed = int(np.count_nonzero(tail_mask))
    if removed < LOWPASS_NOISE_FLOOR_MIN_REMOVE_POINTS:
        return omega, magnitude, phase_rad, diag, notes
    if int(np.count_nonzero(keep)) < LOWPASS_NOISE_FLOOR_MIN_KEEP_POINTS:
        notes.append(
            f'低通高频端有 {removed} 个点的 PA1 输出接近 ADC 噪声底，但保留后点数不足，暂不剔除。'
        )
        return omega, magnitude, phase_rad, diag, notes

    notes.append(
        f'低通高频端发现 {removed} 个 PA1 输出小于 {LOWPASS_OUTPUT_NOISE_FLOOR_RMS_V * 1000:.0f} mVrms '
        '且已低于通带 30 dB 的噪声底点；按当前设置保留这些频点继续分析和绘图。'
    )
    return omega, magnitude, phase_rad, diag, notes


def trim_highpass_noise_floor_for_analysis(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    notes: list[str] = []
    diag = aligned_diagnostics(diagnostics, len(omega))
    output_rms = diag.get('output_rms_v')
    if output_rms is None or len(output_rms) != len(omega) or len(omega) < HIGHPASS_NOISE_FLOOR_MIN_KEEP_POINTS:
        return omega, magnitude, phase_rad, diag, notes

    output_rms = np.asarray(output_rms, dtype=float)
    mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
    passband_count = max(3, min(8, len(mag_db) // 6))
    passband_db = float(np.median(mag_db[-passband_count:]))
    low_output = (
        np.isfinite(output_rms)
        & (output_rms <= HIGHPASS_OUTPUT_NOISE_FLOOR_RMS_V)
        & (mag_db <= passband_db - HIGHPASS_NOISE_FLOOR_DROP_DB)
    )

    if not np.any(low_output):
        return omega, magnitude, phase_rad, diag, notes

    last_low = int(np.max(np.where(low_output)[0]))
    head_mask = np.zeros(len(omega), dtype=bool)
    head_mask[:last_low + 1] = True

    keep = ~head_mask
    removed = int(np.count_nonzero(head_mask))
    if removed < HIGHPASS_NOISE_FLOOR_MIN_REMOVE_POINTS:
        return omega, magnitude, phase_rad, diag, notes
    if int(np.count_nonzero(keep)) < HIGHPASS_NOISE_FLOOR_MIN_KEEP_POINTS:
        notes.append(
            f'高通低频端有 {removed} 个点的 PA1 输出接近 ADC 噪声底，但保留后点数不足，暂不剔除。'
        )
        return omega, magnitude, phase_rad, diag, notes

    notes.append(
        f'高通低频端发现 {removed} 个 PA1 输出小于 {HIGHPASS_OUTPUT_NOISE_FLOOR_RMS_V * 1000:.0f} mVrms '
        '且已低于通带 30 dB 的噪声底点；按当前设置保留这些频点继续分析和绘图。'
    )
    return omega, magnitude, phase_rad, diag, notes


def lowpass_noise_floor_analysis_mask(
    magnitude: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    diag = aligned_diagnostics(diagnostics, len(magnitude))
    output_rms = diag.get('output_rms_v')
    if output_rms is None or len(output_rms) != len(magnitude) or len(magnitude) < LOWPASS_NOISE_FLOOR_MIN_KEEP_POINTS:
        return None

    output_rms = np.asarray(output_rms, dtype=float)
    mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
    passband_count = max(3, min(8, len(mag_db) // 6))
    passband_db = float(np.median(mag_db[:passband_count]))
    low_output = (
        np.isfinite(output_rms)
        & (
            (
                (output_rms <= LOWPASS_OUTPUT_NOISE_FLOOR_RMS_V)
                & (mag_db <= passband_db - LOWPASS_NOISE_FLOOR_DROP_DB)
            )
            | (
                (output_rms <= 0.020)
                & (mag_db <= passband_db - 24.0)
            )
        )
    )
    if not np.any(low_output):
        return None
    first_low = int(np.argmax(low_output))
    tail_mask = np.zeros(len(magnitude), dtype=bool)
    tail_mask[first_low:] = True
    keep = ~tail_mask
    if (
        int(np.count_nonzero(tail_mask)) < LOWPASS_NOISE_FLOOR_MIN_REMOVE_POINTS
        or int(np.count_nonzero(keep)) < LOWPASS_NOISE_FLOOR_MIN_KEEP_POINTS
    ):
        return None
    return keep


def highpass_noise_floor_analysis_mask(
    magnitude: np.ndarray,
    diagnostics: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    diag = aligned_diagnostics(diagnostics, len(magnitude))
    output_rms = diag.get('output_rms_v')
    if output_rms is None or len(output_rms) != len(magnitude) or len(magnitude) < HIGHPASS_NOISE_FLOOR_MIN_KEEP_POINTS:
        return None

    output_rms = np.asarray(output_rms, dtype=float)
    mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
    passband_count = max(3, min(8, len(mag_db) // 6))
    passband_db = float(np.median(mag_db[-passband_count:]))
    low_output = (
        np.isfinite(output_rms)
        & (output_rms <= HIGHPASS_OUTPUT_NOISE_FLOOR_RMS_V)
        & (mag_db <= passband_db - HIGHPASS_NOISE_FLOOR_DROP_DB)
    )
    if not np.any(low_output):
        return None
    last_low = int(np.max(np.where(low_output)[0]))
    head_mask = np.zeros(len(magnitude), dtype=bool)
    head_mask[:last_low + 1] = True
    keep = ~head_mask
    if (
        int(np.count_nonzero(head_mask)) < HIGHPASS_NOISE_FLOOR_MIN_REMOVE_POINTS
        or int(np.count_nonzero(keep)) < HIGHPASS_NOISE_FLOOR_MIN_KEEP_POINTS
    ):
        return None
    return keep


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
    max_order: int = MAX_STANDARD_SIMPLE_FILTER_ORDER,
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
    max_order: int = MAX_STANDARD_SIMPLE_FILTER_ORDER,
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
    order_estimate = max(mag_order, 1)
    delay_artifact_guarded = False
    if slope_order:
        notes.append(f'低通高频尾部斜率支持 {slope_order} 阶；斜率约 {slope_tail:.1f} dB/dec。')

    if phase_fit['reliable']:
        phase_order = int(phase_fit['order'])
        consensus_order = max(mag_order, phase_order, 1)
        mag_rankings = mag_fit.get('rankings', [])
        phase_rankings = phase_fit.get('rankings', [])
        selected_mag_rmse = float(mag_fit.get('rmse_db', 0.0))
        selected_phase_rmse = float(phase_fit.get('rmse_deg', 0.0))
        if slope_order > consensus_order:
            notes.append(
                f'尾部斜率单独像 {slope_order} 阶，但幅值模板和相位拟合共同支持 {consensus_order} 阶；'
                '高频端接近噪声底时不让斜率把阶数抬高。'
            )
        elif slope_order > order_estimate:
            order_estimate = slope_order
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
        lower_mag_rank = next(
            (item for item in mag_rankings if int(item.get('order', 0)) == int(phase_support_order)),
            None,
        )
        lower_phase_rank = next(
            (item for item in phase_rankings if int(item.get('order', 0)) == int(phase_support_order)),
            None,
        )
        lower_mag_rmse = float(lower_mag_rank['rmse_db']) if lower_mag_rank is not None else selected_mag_rmse
        lower_phase_rmse = float(lower_phase_rank['rmse_deg']) if lower_phase_rank is not None else selected_phase_rmse
        strong_higher_order_evidence = bool(
            mag_order >= 3
            and phase_order >= mag_order
            and phase_support_order == mag_order - 1
            and selected_mag_rmse <= 0.70
            and (lower_mag_rmse - selected_mag_rmse) >= max(0.25, selected_mag_rmse * 0.90)
            and selected_phase_rmse <= 10.0
            and (lower_phase_rmse - selected_phase_rmse) >= max(4.0, selected_phase_rmse * 0.45)
            and mag_span_db >= 18.0
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
        elif strong_higher_order_evidence:
            order_estimate = mag_order
            delay_artifact_guarded = False
            notes.append(
                f'幅值模板和原始相位都强支持 {mag_order} 阶；延迟补偿虽然可拟合成 {phase_support_order} 阶，但低阶幅值/相位误差明显更大，因此保持 {mag_order} 阶低通。'
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
        if order_estimate > consensus_order and slope_order >= order_estimate:
            order_estimate = consensus_order
            notes.append(
                f'最终低通阶数按幅值/相位一致证据限制为 {consensus_order} 阶；'
                '尾部斜率只作为提示，不单独显示更高阶。'
            )
        if not delay_artifact_guarded:
            notes.append(
                f'低通幅值模板拟合优先 {mag_order} 阶；幅值 RMSE = {float(mag_fit["rmse_db"]):.2f} dB。'
            )
            notes.append(
                f'相位拟合支持 {phase_order} 阶；相位 RMSE = {float(phase_fit["rmse_deg"]):.1f}°。'
            )
    else:
        quality = phase_fit['quality']
        if slope_order > order_estimate and slope_order <= mag_order + 1 and mag_span_db >= 18.0:
            order_estimate = slope_order
        elif slope_order > mag_order + 1:
            notes.append(
                f'尾部斜率单独像 {slope_order} 阶，但相位质量不足且幅值模板只支持 {mag_order} 阶；'
                '不把噪声底/运放高频滚降当作额外物理阶数。'
            )
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

    if (
        mag_order > 3
        and slope_order == 3
        and phase_fit['reliable']
        and int(phase_fit['order']) >= 3
        and mag_span_db >= 18.0
    ):
        order_estimate = min(order_estimate, 3)
        notes.append('低通有效尾段斜率和相位共同支持三阶；有限扫频/噪声底截尾下不把四阶模板近似拟合当成真实四阶。')

    if delay_artifact_guarded:
        notes.append(
            f'未补偿延迟的相位拟合会偏向 {int(phase_fit["order"])} 阶，但已结合幅值、斜率和延迟补偿重新约束；最终按 {order_estimate} 阶低通判定。'
        )

    return int(np.clip(order_estimate, 1, 12)), notes


def fit_highpass_order_from_magnitude(
    omega: np.ndarray,
    magnitude: np.ndarray,
    max_order: int = MAX_STANDARD_SIMPLE_FILTER_ORDER,
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


def fit_highpass_order_from_transition_band(
    omega: np.ndarray,
    magnitude: np.ndarray,
    max_order: int = MAX_STANDARD_SIMPLE_FILTER_ORDER,
) -> dict[str, object] | None:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    valid = np.isfinite(omega) & np.isfinite(magnitude) & (omega > 0.0) & (magnitude > 0.0)
    omega = omega[valid]
    magnitude = magnitude[valid]

    if len(omega) < 24:
        return None

    high_count = max(3, min(8, len(magnitude) // 6))
    high_gain = max(float(np.median(magnitude[-high_count:])), 1e-12)
    relative_db = 20.0 * np.log10(np.clip(magnitude / high_gain, 1e-12, None))

    best: dict[str, object] | None = None
    for floor_db in (-32.0, -30.0, -28.0, -25.0, -22.0, -20.0):
        keep = relative_db >= floor_db
        kept = int(np.count_nonzero(keep))
        removed = int(len(keep) - kept)
        if removed < 2 or kept < 20:
            continue
        if frequency_span_decades(omega[keep]) < 0.75:
            continue

        fit = fit_highpass_order_from_magnitude(omega[keep], magnitude[keep], max_order=max_order)
        order = int(fit['order'])
        rmse_db = float(fit['rmse_db'])
        if order <= 1 or rmse_db > 0.90:
            continue

        candidate = {
            'order': order,
            'omega_c': float(fit['omega_c']),
            'rmse_db': rmse_db,
            'rankings': fit.get('rankings', []),
            'floor_db': float(floor_db),
            'kept_points': kept,
            'removed_points': removed,
        }
        if best is None:
            best = candidate
            continue
        if order > int(best['order']) or (
            order == int(best['order']) and rmse_db < float(best['rmse_db'])
        ):
            best = candidate

    return best


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
        capped_order = cap_standard_filter_order(filter_type, order)
        if capped_order != order:
            notes.append(
                f'低通模板拟合给出 {order} 阶，但本项目标准低通电路最高为三阶；'
                f'最终按 {capped_order} 阶输出，并把更高阶视为噪声/寄生/有限扫频提示。'
            )
        return capped_order, notes

    if filter_type == 'highpass':
        mag_fit = fit_highpass_order_from_magnitude(
            omega,
            magnitude,
            max_order=MAX_STANDARD_SIMPLE_FILTER_ORDER,
        )
        order = int(mag_fit['order'])
        notes.append(
            f'High-pass magnitude fit prefers order {order}; magnitude RMSE = {float(mag_fit["rmse_db"]):.2f} dB.'
        )
        transition_fit = fit_highpass_order_from_transition_band(
            omega,
            magnitude,
            max_order=MAX_STANDARD_SIMPLE_FILTER_ORDER,
        )
        transition_order = int(transition_fit['order']) if transition_fit is not None else 0
        if transition_fit is not None:
            notes.append(
                f'High-pass transition-band fit prefers order {transition_order}; '
                f'ignored {int(transition_fit["removed_points"])} points below {float(transition_fit["floor_db"]):.0f} dB '
                f'and got RMSE = {float(transition_fit["rmse_db"]):.2f} dB.'
            )
            if transition_order > order:
                order = transition_order
        slope = compute_mag_slope_db_per_dec(omega, magnitude)
        edge = max(3, min(12, len(slope) // 8)) if len(slope) else 0
        head_slope = float(np.median(slope[:edge])) if edge else 0.0
        slope_order = estimate_order_from_slope(head_slope, max_order=MAX_STANDARD_SIMPLE_FILTER_ORDER)
        phase_order, phase_reliable = estimate_order_from_phase_span(
            phase_deg,
            max_order=MAX_STANDARD_SIMPLE_FILTER_ORDER,
        )
        mag_db = 20.0 * np.log10(np.clip(magnitude, 1e-12, None))
        mag_span_db = float(np.max(mag_db) - np.min(mag_db)) if len(mag_db) else 0.0
        phase_span_deg = float(abs(phase_deg[-1] - phase_deg[0])) if len(phase_deg) else 0.0
        rankings = mag_fit.get('rankings', [])
        second_order_rank = next(
            (item for item in rankings if int(item.get('order', 0)) == 2),
            None,
        )
        finite_sweep_second_order_evidence = False
        if order == 1 and second_order_rank is not None:
            selected_rmse = float(mag_fit.get('rmse_db', 0.0))
            second_rmse = float(second_order_rank['rmse_db'])
            rmse_gap = second_rmse - selected_rmse
            second_order_near_tie = bool(
                rmse_gap <= max(0.35, selected_rmse * 0.25)
                and second_rmse <= 1.60
            )
            finite_sweep_second_order_evidence = bool(
                second_order_near_tie
                and mag_span_db >= 28.0
                and head_slope >= 24.0
                and phase_span_deg >= 95.0
            )
            if finite_sweep_second_order_evidence:
                order = 2
                notes.append(
                    'High-pass first/second-order magnitude fits are nearly tied; '
                    f'rising-edge slope {head_slope:.1f} dB/dec, magnitude span {mag_span_db:.1f} dB, '
                    f'and phase span {phase_span_deg:.1f} deg support a finite-sweep second-order high-pass.'
                )
        if phase_reliable:
            if phase_order < order:
                phase_rank = next(
                    (item for item in rankings if int(item.get('order', 0)) == int(phase_order)),
                    None,
                )
                selected_rmse = float(mag_fit.get('rmse_db', 0.0))
                phase_rmse = float(phase_rank['rmse_db']) if phase_rank is not None else selected_rmse
                rmse_gap = phase_rmse - selected_rmse
                strong_magnitude_evidence = rmse_gap >= max(0.60, selected_rmse * 1.5)
                slope_supports_order = bool(
                    slope_order >= order
                    or (order >= 3 and head_slope >= 20.0 * (float(order) - 0.60))
                )
                transition_supports_order = bool(
                    transition_fit is not None
                    and transition_order >= order
                    and float(transition_fit['rmse_db']) <= 0.55
                    and int(transition_fit['kept_points']) >= 24
                )
                if (
                    phase_order == order - 1
                    and strong_magnitude_evidence
                    and slope_supports_order
                ):
                    notes.append(
                        f'Phase span only reaches {phase_order} 阶 after low-frequency noise-floor trimming, '
                        f'but high-pass magnitude fit is stronger by {rmse_gap:.2f} dB RMSE and the rising-edge slope is {head_slope:.1f} dB/dec; keeping order {order}.'
                    )
                elif finite_sweep_second_order_evidence and order == 2 and phase_order == 1:
                    notes.append(
                        'Phase span alone reaches only first order, but the high-pass rising edge and near-tied second-order fit keep the circuit as second order.'
                    )
                elif transition_supports_order:
                    notes.append(
                        f'Phase span only supports order {phase_order}, but high-pass phase wraps at high order '
                        f'and the transition-band magnitude fit strongly supports order {order}; keeping order {order}.'
                    )
                else:
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
        capped_order = cap_standard_filter_order(filter_type, order)
        if capped_order != order:
            notes.append(
                f'高通局部拟合给出 {order} 阶，但本项目标准高通电路最高为三阶；'
                f'最终按 {capped_order} 阶输出，避免智能补扫把三阶高通误升为更高阶。'
            )
        return capped_order, notes

    if filter_type == 'bandpass':
        peak_idx = int(np.argmax(magnitude))
        left_order = 1
        right_order = 1
        left_fit = None
        right_fit = None
        if peak_idx >= 4:
            left_fit = fit_highpass_order_from_magnitude(omega[:peak_idx + 1], magnitude[:peak_idx + 1], max_order=4)
            left_order = int(left_fit['order'])
        if len(omega) - peak_idx >= 5:
            right_fit = fit_lowpass_order_from_magnitude(omega[peak_idx:], magnitude[peak_idx:], max_order=4)
            right_order = int(right_fit['order'])
        if left_order == 1 and right_order > 1 and right_fit is not None:
            rankings = right_fit.get('rankings', [])
            first_order_rank = next(
                (item for item in rankings if int(item.get('order', 0)) == 1),
                None,
            )
            selected_rmse = float(right_fit.get('rmse_db', 0.0))
            first_rmse = float(first_order_rank['rmse_db']) if first_order_rank is not None else selected_rmse
            rmse_gap = first_rmse - selected_rmse
            slope = compute_mag_slope_db_per_dec(omega, magnitude)
            edge = max(3, min(12, len(slope) // 8)) if len(slope) else 0
            tail_slope = float(np.median(slope[-edge:])) if edge else 0.0
            finite_right_edge = bool(peak_idx >= int(0.25 * len(magnitude)))
            if (
                first_order_rank is not None
                and rmse_gap <= max(0.45, selected_rmse * 0.25)
                and tail_slope < -18.0
                and finite_right_edge
            ):
                notes.append(
                    'Band-pass right edge is distorted or only partly covered; '
                    f'first-order low-pass edge is within {rmse_gap:.2f} dB RMSE of the selected higher-order fit, '
                    'so the cascaded 1st-order HP + 1st-order LP band-pass is kept as order 2.'
                )
                right_order = 1
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
                f'截止点两侧覆盖不足：左侧 {left_span:.2f} decade，右侧 {right_span:.2f} decade；建议复测确认阶数。'
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
        finite_clip_flags = np.asarray(clip_flags, dtype=float)
        finite_clip_flags = finite_clip_flags[np.isfinite(finite_clip_flags)]
        clipped = int(np.count_nonzero(finite_clip_flags)) if len(finite_clip_flags) else 0
        if clipped:
            has_diag_problem = True
            notes.append(f'仍有 {clipped} 个频点带 ADC 削顶标志，建议复测确认阶数。')

    valid_count = diag.get('valid_capture_count')
    if valid_count is not None and len(valid_count):
        finite_valid_count = np.asarray(valid_count, dtype=float)
        finite_valid_count = finite_valid_count[np.isfinite(finite_valid_count)]
        if len(finite_valid_count) and float(np.min(finite_valid_count)) < 2:
            has_diag_problem = True
            notes.append('部分频点有效重复采集次数少于 2 次，建议复测后再确认阶数。')

    input_rms = diag.get('input_rms_v')
    if input_rms is not None and len(input_rms):
        finite_input_rms = np.asarray(input_rms, dtype=float)
        finite_input_rms = finite_input_rms[np.isfinite(finite_input_rms)]
        if len(finite_input_rms) and float(np.median(finite_input_rms)) < 0.03:
            has_diag_problem = True
            notes.append('输入参考 RMS 中位数低于 30 mV，Vout/Vin 对噪声过敏，阶数判断降级。')

    mag_repeat_span = diag.get('magnitude_repeat_span_db')
    if mag_repeat_span is not None and len(mag_repeat_span):
        mag_repeat_span = np.asarray(mag_repeat_span[:point_count], dtype=float)
        mag_repeat_span = mag_repeat_span[np.isfinite(mag_repeat_span)]
        if len(mag_repeat_span) and (float(np.nanmedian(mag_repeat_span)) > 0.8 or float(np.nanmax(mag_repeat_span)) > 2.5):
            has_diag_problem = True
            notes.append('重复采集幅值离散过大，说明该次扫频不稳定；建议降低幅度或复测。')

    phase_repeat_span = diag.get('phase_repeat_span_rad')
    if phase_repeat_span is not None and len(phase_repeat_span):
        phase_repeat_span = np.abs(np.asarray(phase_repeat_span[:point_count], dtype=float))
        phase_repeat_span = phase_repeat_span[np.isfinite(phase_repeat_span)]
        if len(phase_repeat_span) and (float(np.nanmedian(phase_repeat_span)) > 0.12 or float(np.nanmax(phase_repeat_span)) > 0.45):
            has_diag_problem = True
            notes.append('重复采集相位离散过大，建议复测确认阶数。')

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
                    notes.append(
                        '高通低频输出 RMS 已接近 STM32 ADC 噪声/量化底，'
                        '该项仅作为测量质量提醒；若斜率、模板拟合和相位跨度已支持当前阶次，不降级系统类型。'
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
        notes.append('测量质量提醒：本次数据存在覆盖或诊断限制，建议复测确认阶数。')

    summary = '可靠' if reliable else '需复测确认'
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


def cap_standard_filter_order(filter_type: str, order_estimate: int) -> int:
    order_estimate = int(np.clip(order_estimate, 1, 12))
    if filter_type in ('lowpass', 'highpass'):
        return int(np.clip(order_estimate, 1, MAX_STANDARD_SIMPLE_FILTER_ORDER))
    if filter_type in ('bandpass', 'bandstop'):
        return max(2, order_estimate)
    return order_estimate


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

    return f'{order_prefix}{filter_label}'


def evaluate_expected_circuit(result: 'SystemAnalysisResult', expected: str) -> str:
    if not expected or expected == 'Auto':
        return '未指定期望电路，按自动识别结果验收。'

    target = EXPECTED_CIRCUIT_MAP.get(expected)
    if target is None:
        return f'期望电路“{expected}”不在内置八类列表中。'

    expected_type, expected_order = target
    type_ok = result.filter_type == expected_type
    order_ok = expected_order is None or result.order_estimate == expected_order
    reliability_text = '可靠' if result.order_reliable else '需复测确认'
    if type_ok and order_ok:
        return f'期望电路匹配：{expected}；本次识别为{result.system_order}（{reliability_text}）。'

    expected_label = FILTER_TYPE_LABELS.get(expected_type, expected_type)
    if expected_order is not None:
        expected_label = f'{expected_order}阶{expected_label}'
    measured_label = f'{result.order_estimate}阶{FILTER_TYPE_LABELS.get(result.filter_type, result.filter_type)}'
    if expected_type == 'highpass' and result.filter_type == 'bandpass':
        return (
            f'期望电路不匹配：期望 {expected_label}，实测 {measured_label}。'
            '曲线低频端像高通、但高频端又明显衰减，因此实测为带通形态；'
            '请检查高通输出节点、1.65V 中点偏置以及高频端额外电容/连线/面包板寄生。'
        )
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


def normalize_phase_for_display(phase_deg: float, filter_type: str) -> float:
    phase = float(phase_deg)
    if not np.isfinite(phase):
        return phase
    return float(((phase + 180.0) % 360.0) - 180.0)


def phase_array_for_display(phase_deg: np.ndarray, filter_type: str) -> np.ndarray:
    phase = np.asarray(phase_deg, dtype=float)
    if filter_type == 'highpass':
        shifted = phase.copy()
        finite = shifted[np.isfinite(shifted)]
        if len(finite) and float(np.nanmedian(finite)) < -45.0:
            shifted = shifted + 360.0
        return shifted
    return phase


def _format_optional_omega(value: float | None) -> str:
    return '无法计算' if value is None else f'{float(value):.2f} rad/s'


def build_filter_report_lines(r: 'SystemAnalysisResult') -> list[str]:
    cutoff_phase = normalize_phase_for_display(r.cutoff_phase_deg, r.filter_type)
    max_mag_db = 20.0 * np.log10(max(r.max_magnitude, 1e-12))

    lines = [
        '=== 稳频仪：滤波器频响识别分析报告 ===',
        f'系统类型: {r.system_order}',
        f'滤波器族: {FILTER_TYPE_LABELS.get(r.filter_type, FILTER_TYPE_LABELS["other"])}',
        f'估计阶次: {r.order_estimate}',
        f'判阶可靠性: {r.identification_summary}（置信度 {r.identification_confidence:.2f}）',
        f'期望电路: {getattr(r, "expected_circuit", "Auto")}',
        f'期望匹配: {getattr(r, "expected_match_text", "未指定期望电路，按自动识别结果验收。")}',
        f'数据点数: {len(r.omega)}',
        f'截止/特征判据: {r.cutoff_method}',
        '--- 识别证据 ---',
    ]
    lines.extend(build_identification_evidence(r))

    diagnostic_notes = [
        note for note in r.notes
        if any(key in note for key in ('测量质量', '测量质量提醒', '峰峰值', '重复采集', '采样信息', '削顶', '偏置', 'RMS'))
    ]
    if diagnostic_notes:
        lines.append('--- 测量质量诊断 ---')
        lines.extend(diagnostic_notes[:8])

    if r.filter_type in ('lowpass', 'highpass'):
        lines.extend([
            f'-3 dB 截止角频率 = {r.magnitude_cutoff_omega:.2f} rad/s',
            f'特征频点相位 = {cutoff_phase:.2f} °',
            f'幅频阈值 = {r.cutoff_mag_db:.2f} dB',
            f'最大/通带幅值 = {r.max_magnitude:.4f} ({max_mag_db:.2f} dB)',
        ])
        if r.order_estimate == 1:
            phase_target = '-45°' if r.filter_type == 'lowpass' else '+45°'
            lines.append(
                f'一阶相位 {phase_target} 参考角频率 = {_format_optional_omega(r.phase_cutoff_omega)}'
            )
    elif r.filter_type == 'bandpass':
        lines.extend([
            f'左侧 -3 dB 角频率 = {r.magnitude_cutoff_omega:.2f} rad/s',
            f'右侧 -3 dB 角频率 = {_format_optional_omega(r.secondary_cutoff_omega)}',
            f'中心角频率 ω0 = {r.omega_c:.2f} rad/s',
            f'带宽 = {_format_optional_omega(r.bandwidth_omega)}',
            f'峰值角频率 = {r.resonant_frequency:.2f} rad/s',
            f'峰值幅值 = {r.max_magnitude:.4f} ({max_mag_db:.2f} dB)',
            f'特征频点相位 = {cutoff_phase:.2f} °',
        ])
    elif r.filter_type == 'bandstop':
        notch_mag = float(np.min(np.clip(r.mag_smooth, 1e-12, None)))
        notch_depth_db = db_ratio(r.max_magnitude, notch_mag)
        lines.extend([
            f'左侧 -3 dB 角频率 = {r.magnitude_cutoff_omega:.2f} rad/s',
            f'右侧 -3 dB 角频率 = {_format_optional_omega(r.secondary_cutoff_omega)}',
            f'陷波角频率 ω0 = {r.omega_c:.2f} rad/s',
            f'阻带宽度 = {_format_optional_omega(r.bandwidth_omega)}',
            f'陷波深度 = {notch_depth_db:.1f} dB',
            f'特征频点相位 = {cutoff_phase:.2f} °',
        ])
    else:
        lines.extend([
            f'主特征角频率 = {r.omega_c:.2f} rad/s',
            f'特征频点相位 = {cutoff_phase:.2f} °',
        ])

    lines.append('')
    return lines


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
    head_order = estimate_order_from_slope(head_slope, max_order=MAX_STANDARD_SIMPLE_FILTER_ORDER)
    tail_order = estimate_order_from_slope(tail_slope, max_order=MAX_STANDARD_SIMPLE_FILTER_ORDER)

    low_to_high_db = db_ratio(low_gain, high_gain)
    high_to_low_db = db_ratio(high_gain, low_gain)
    edge_similarity_db = abs(low_to_high_db)
    peak_edge_margin_db = min(db_ratio(peak_gain, low_gain), db_ratio(peak_gain, high_gain))
    notch_edge_margin_db = min(db_ratio(low_gain, valley_gain), db_ratio(high_gain, valley_gain))

    interior_lo = max(edge, int(0.2 * (point_count - 1)))
    interior_hi = min(point_count - edge - 1, int(0.8 * (point_count - 1)))
    is_peak_index_interior = interior_lo <= peak_idx <= interior_hi
    is_valley_index_interior = interior_lo <= valley_idx <= interior_hi
    peak_left_span, peak_right_span = edge_decades(omega, float(omega[peak_idx]))
    valley_left_span, valley_right_span = edge_decades(omega, float(omega[valley_idx]))
    # Linear sweeps put low-frequency notches early in the array even when they
    # are centered on a logarithmic frequency axis, so use frequency coverage
    # as well as point index to decide whether a peak/valley is truly internal.
    is_peak_interior = bool(
        is_peak_index_interior
        or (peak_left_span >= MIN_EDGE_DECADES and peak_right_span >= MIN_EDGE_DECADES)
    )
    is_valley_interior = bool(
        is_valley_index_interior
        or (valley_left_span >= MIN_EDGE_DECADES and valley_right_span >= MIN_EDGE_DECADES)
    )

    filter_type = 'other'
    order_estimate = max(head_order, tail_order, 1)
    peak_target_magnitude = peak_gain / np.sqrt(2.0) if peak_gain > 0.0 else 0.0
    peak_level_crossings = find_level_crossings(omega, magnitude, peak_target_magnitude)
    peak_left_crossings = [item for item in peak_level_crossings if item[1] < peak_idx]
    peak_right_crossings = [item for item in peak_level_crossings if item[1] >= peak_idx]
    low_edge_floor = float(np.min(magnitude[:edge])) if edge else low_gain
    high_edge_floor = float(np.min(magnitude[-edge:])) if edge else high_gain
    peak_endpoint_margin_db = min(
        db_ratio(peak_gain, low_edge_floor),
        db_ratio(peak_gain, high_edge_floor),
    )
    has_two_sided_peak_crossings = bool(
        peak_left_crossings
        and peak_right_crossings
        and peak_endpoint_margin_db >= 2.5
        and peak_left_crossings[-1][0] < float(omega[peak_idx]) < peak_right_crossings[0][0]
        and peak_right_crossings[0][0] / max(peak_left_crossings[-1][0], 1e-12) >= 1.25
    )

    if has_two_sided_peak_crossings or (
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
        if has_two_sided_peak_crossings:
            notes.append(
                'Band-pass shape detected from two -3 dB crossings around the measured peak; '
                f'peak exceeds the endpoint floors by at least {peak_endpoint_margin_db:.1f} dB.'
            )
        else:
            notes.append(
                f'Band-pass shape detected: peak gain exceeds both edges by {peak_edge_margin_db:.1f} dB.'
            )
        notes.extend(band_order_notes)
    elif (
        is_valley_interior
        and notch_edge_margin_db >= 3.0
        and (
            edge_similarity_db <= 12.0
            or notch_edge_margin_db >= max(10.0, edge_similarity_db + 3.0)
        )
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
        high_freq_phase_display = normalize_phase_for_display(phase_deg[-1], 'highpass')
        if high_freq_phase_display < -20.0:
            notes.append(
                '高通高频端相位仍明显为负，可能存在额外延迟或高频滚降；理想一阶 RC 高通的高频相位应逐渐接近 0°。'
            )
    else:
        notes.append('Response shape is not close to a standard low-pass/high-pass/band-pass/band-stop template.')

    capped_order = cap_standard_filter_order(filter_type, order_estimate)
    if capped_order != order_estimate and filter_type in ('lowpass', 'highpass'):
        notes.append(
            f'项目内置标准{FILTER_TYPE_LABELS.get(filter_type, "")}最高按三阶验收；'
            f'局部斜率/拟合给出的 {order_estimate} 阶作为异常提示，不作为最终物理阶数。'
        )
        order_estimate = capped_order

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
        'order_estimate': cap_standard_filter_order(filter_type, order_estimate),
        'system_label': format_filter_system_label(filter_type, cap_standard_filter_order(filter_type, order_estimate)),
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
            notes.append('一阶低通已同时计算幅值 -3 dB 和相位 -45° 截止点；主截止角频率采用通用的幅值 -3 dB 定义。')
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
    diagnostics = aligned_diagnostics(diagnostics, min_len)

    invalid_frame, invalid_notes, _ = describe_invalid_measurement_frame(
        omega,
        magnitude_data,
        phase_data_rad,
        diagnostics,
    )
    if invalid_frame:
        raise ValueError('\n'.join(invalid_notes))

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

    lowpass_hint = False
    highpass_hint = False
    if len(magnitude_data) >= 5:
        edge = max(3, min(8, len(magnitude_data) // 8))
        low_gain_hint = float(np.median(magnitude_data[:edge]))
        high_gain_hint = float(np.median(magnitude_data[-edge:]))
        lowpass_hint = db_ratio(low_gain_hint, high_gain_hint) >= 12.0
        highpass_hint = db_ratio(high_gain_hint, low_gain_hint) >= 12.0

    if lowpass_hint:
        omega, magnitude_data, phase_data_rad, diagnostics, noise_floor_notes = trim_lowpass_noise_floor_for_analysis(
            omega,
            magnitude_data,
            phase_data_rad,
            diagnostics,
        )
        notes.extend(noise_floor_notes)
    elif highpass_hint:
        omega, magnitude_data, phase_data_rad, diagnostics, noise_floor_notes = trim_highpass_noise_floor_for_analysis(
            omega,
            magnitude_data,
            phase_data_rad,
            diagnostics,
        )
        notes.extend(noise_floor_notes)

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
    omega_for_identification = omega
    mag_for_identification = mag_smooth
    phase_for_identification_deg = phase_deg
    raw_mag_for_identification = np.clip(magnitude_for_analysis.copy(), 1e-12, None)
    raw_phase_for_identification_deg = np.degrees(phase_for_analysis)
    lowpass_analysis_mask = lowpass_noise_floor_analysis_mask(mag_smooth, diagnostics)
    highpass_analysis_mask = highpass_noise_floor_analysis_mask(mag_smooth, diagnostics)
    if lowpass_hint and lowpass_analysis_mask is not None:
        omega_for_identification = omega[lowpass_analysis_mask]
        mag_for_identification = mag_smooth[lowpass_analysis_mask]
        phase_for_identification_deg = phase_deg[lowpass_analysis_mask]
        raw_mag_for_identification = raw_mag_for_identification[lowpass_analysis_mask]
        raw_phase_for_identification_deg = raw_phase_for_identification_deg[lowpass_analysis_mask]
        notes.append(
            f'低通判阶内部忽略 {len(omega) - len(omega_for_identification)} 个高频噪声底点，但绘图和报告保留全部 {len(omega)} 个频点。'
        )
    elif highpass_hint and highpass_analysis_mask is not None:
        omega_for_identification = omega[highpass_analysis_mask]
        mag_for_identification = mag_smooth[highpass_analysis_mask]
        phase_for_identification_deg = phase_deg[highpass_analysis_mask]
        raw_mag_for_identification = raw_mag_for_identification[highpass_analysis_mask]
        raw_phase_for_identification_deg = raw_phase_for_identification_deg[highpass_analysis_mask]
        notes.append(
            f'高通判阶内部忽略 {len(omega) - len(omega_for_identification)} 个低频噪声底点，但绘图和报告保留全部 {len(omega)} 个频点。'
        )

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

    filter_info = classify_filter_response(omega_for_identification, mag_for_identification, phase_for_identification_deg)
    filter_type = str(filter_info['filter_type'])
    order_estimate = int(filter_info['order_estimate'])
    notes.extend(filter_info['notes'])

    refined_order, model_fit_notes = refine_filter_order_with_template(
        filter_type,
        omega_for_identification,
        mag_for_identification,
        phase_for_identification_deg,
        order_estimate,
    )
    if refined_order != order_estimate:
        notes.append(f'模型拟合将初始阶次 {order_estimate} 修正为 {refined_order}。')
        order_estimate = refined_order
    notes.extend(model_fit_notes)
    if filter_type == 'highpass' and smooth:
        raw_refined_order, raw_model_fit_notes = refine_filter_order_with_template(
            filter_type,
            omega_for_identification,
            raw_mag_for_identification,
            raw_phase_for_identification_deg,
            order_estimate,
        )
        if raw_refined_order > order_estimate:
            notes.append(
                'High-pass order cross-check used the unsmoothed magnitude edge because smoothing can lift '
                f'the low-frequency stopband; order changed from {order_estimate} to {raw_refined_order}.'
            )
            order_estimate = raw_refined_order
            notes.extend(raw_model_fit_notes)
        elif order_estimate <= 1:
            notes.extend(raw_model_fit_notes)
    capped_order = cap_standard_filter_order(filter_type, order_estimate)
    if capped_order != order_estimate:
        notes.append(
            f'标准{FILTER_TYPE_LABELS.get(filter_type, "")}识别结果限制为 {capped_order} 阶；'
            f'{order_estimate} 阶只作为寄生/噪声/有限扫频异常提示。'
        )
        order_estimate = capped_order

    identification = assess_identification_reliability(
        omega_for_identification,
        mag_for_identification,
        phase_for_identification_deg,
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
    elif filter_type == 'highpass' and order_estimate <= 1:
        phase_cutoff_omega, phase_cutoff_index = estimate_first_order_phase_cutoff(
            omega,
            phase_deg,
            target_phase_deg=45.0,
        )
        if phase_cutoff_omega is not None and phase_cutoff_index is not None:
            phase_mag_delta = abs(float(phase_cutoff_omega) - float(magnitude_cutoff_omega)) / max(float(magnitude_cutoff_omega), 1e-12)
            notes.append(
                '一阶高通额外计算了 +45° 相位参考点，优先以幅频 -3 dB 结果为准。'
            )
            if phase_mag_delta > 0.08:
                notes.append(
                    f'幅频 -3 dB 与相位 +45° 参考点相差 {phase_mag_delta * 100.0:.1f}%，请检查接线方向和扫频范围。'
                )

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
