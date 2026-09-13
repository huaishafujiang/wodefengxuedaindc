from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


MIN_REFERENCE_POINTS = 3
MAGNITUDE_FLOOR = 1e-9


def _aligned_arrays(
    omega: Any,
    magnitude: Any,
    phase_rad: Any,
    *,
    require_positive_magnitude: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_arr = np.asarray(omega, dtype=float).flatten()
    magnitude_arr = np.asarray(magnitude, dtype=float).flatten()
    phase_arr = np.asarray(phase_rad, dtype=float).flatten()

    n = min(len(omega_arr), len(magnitude_arr), len(phase_arr))
    if n < MIN_REFERENCE_POINTS:
        raise ValueError("校正参考至少需要 3 个频点。")

    omega_arr = omega_arr[:n]
    magnitude_arr = magnitude_arr[:n]
    phase_arr = phase_arr[:n]

    valid = np.isfinite(omega_arr) & (omega_arr > 0.0)
    valid &= np.isfinite(magnitude_arr)
    valid &= np.isfinite(phase_arr)
    if require_positive_magnitude:
        valid &= magnitude_arr > MAGNITUDE_FLOOR
    if int(np.count_nonzero(valid)) < MIN_REFERENCE_POINTS:
        raise ValueError("校正参考有效频点不足；请确认直通/参考测量的幅值不为 0。")

    omega_arr = omega_arr[valid]
    magnitude_arr = magnitude_arr[valid]
    phase_arr = np.unwrap(phase_arr[valid])

    order = np.argsort(omega_arr)
    omega_arr = omega_arr[order]
    magnitude_arr = magnitude_arr[order]
    phase_arr = phase_arr[order]

    unique_omega, inverse = np.unique(omega_arr, return_inverse=True)
    if len(unique_omega) != len(omega_arr):
        mag_unique = np.empty_like(unique_omega, dtype=float)
        phase_unique = np.empty_like(unique_omega, dtype=float)
        for index in range(len(unique_omega)):
            mask = inverse == index
            mag_unique[index] = float(np.median(magnitude_arr[mask]))
            phase_unique[index] = float(np.median(phase_arr[mask]))
        omega_arr = unique_omega
        magnitude_arr = mag_unique
        phase_arr = phase_unique

    if len(omega_arr) < MIN_REFERENCE_POINTS:
        raise ValueError("校正参考去重后频点不足，至少需要 3 个不同频率。")
    return omega_arr, magnitude_arr, phase_arr


@dataclass(frozen=True)
class CalibrationProfile:
    omega: np.ndarray
    magnitude: np.ndarray
    phase_rad: np.ndarray
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        omega, magnitude, phase = _aligned_arrays(
            self.omega,
            self.magnitude,
            self.phase_rad,
            require_positive_magnitude=True,
        )
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "phase_rad", phase)

    @property
    def point_count(self) -> int:
        return int(len(self.omega))

    @property
    def min_hz(self) -> float:
        return float(self.omega[0] / (2.0 * np.pi))

    @property
    def max_hz(self) -> float:
        return float(self.omega[-1] / (2.0 * np.pi))

    @property
    def label(self) -> str:
        return self.source or self.created_at.strftime("校正参考 %H:%M:%S")

    @property
    def magnitude_db_span(self) -> tuple[float, float]:
        mag_db = 20.0 * np.log10(np.clip(self.magnitude, MAGNITUDE_FLOOR, None))
        return float(np.min(mag_db)), float(np.max(mag_db))

    @property
    def phase_span_deg(self) -> float:
        return float(np.degrees(np.max(self.phase_rad) - np.min(self.phase_rad)))


@dataclass(frozen=True)
class CalibrationApplication:
    omega: np.ndarray
    magnitude: np.ndarray
    phase_rad: np.ndarray
    correction_magnitude: np.ndarray
    correction_phase_rad: np.ndarray
    notes: list[str]
    outside_reference_count: int = 0


def build_calibration_profile(
    omega: Any,
    magnitude: Any,
    phase_rad: Any,
    *,
    source: str = "",
) -> CalibrationProfile:
    return CalibrationProfile(
        omega=np.asarray(omega, dtype=float),
        magnitude=np.asarray(magnitude, dtype=float),
        phase_rad=np.asarray(phase_rad, dtype=float),
        source=source,
    )


def apply_calibration(
    omega: Any,
    magnitude: Any,
    phase_rad: Any,
    profile: CalibrationProfile,
) -> CalibrationApplication:
    target_omega = np.asarray(omega, dtype=float).flatten()
    target_magnitude = np.asarray(magnitude, dtype=float).flatten()
    target_phase = np.asarray(phase_rad, dtype=float).flatten()

    n = min(len(target_omega), len(target_magnitude), len(target_phase))
    if n < MIN_REFERENCE_POINTS:
        raise ValueError("待校正测量至少需要 3 个频点。")

    target_omega = target_omega[:n]
    target_magnitude = target_magnitude[:n]
    target_phase = target_phase[:n]

    valid = np.isfinite(target_omega) & (target_omega > 0.0)
    valid &= np.isfinite(target_magnitude)
    valid &= np.isfinite(target_phase)
    if int(np.count_nonzero(valid)) < MIN_REFERENCE_POINTS:
        raise ValueError("待校正测量有效频点不足。")

    log_ref_omega = np.log(profile.omega)
    log_ref_magnitude = np.log(np.clip(profile.magnitude, MAGNITUDE_FLOOR, None))

    correction_magnitude = np.ones_like(target_magnitude, dtype=float)
    correction_phase = np.zeros_like(target_phase, dtype=float)
    log_target_omega = np.log(target_omega[valid])
    correction_magnitude[valid] = np.exp(
        np.interp(log_target_omega, log_ref_omega, log_ref_magnitude)
    )
    correction_phase[valid] = np.interp(log_target_omega, log_ref_omega, profile.phase_rad)

    corrected_magnitude = target_magnitude.copy()
    corrected_phase = np.unwrap(target_phase.copy())
    corrected_magnitude[valid] = target_magnitude[valid] / np.clip(
        correction_magnitude[valid],
        MAGNITUDE_FLOOR,
        None,
    )
    corrected_phase[valid] = corrected_phase[valid] - correction_phase[valid]

    outside = valid & ((target_omega < profile.omega[0]) | (target_omega > profile.omega[-1]))
    outside_count = int(np.count_nonzero(outside))

    notes = [
        f"已应用参考校正：{profile.label}；幅值除以参考链路， 相位扣除参考链路。"
    ]
    if outside_count:
        notes.append(f"有 {outside_count} 个频点超出校正参考频段，已使用最近参考端点值。")
    return CalibrationApplication(
        omega=target_omega,
        magnitude=corrected_magnitude,
        phase_rad=corrected_phase,
        correction_magnitude=correction_magnitude,
        correction_phase_rad=correction_phase,
        notes=notes,
        outside_reference_count=outside_count,
    )


def profile_summary(profile: CalibrationProfile | None) -> str:
    if profile is None:
        return "未设置校正参考"
    mag_min_db, mag_max_db = profile.magnitude_db_span
    return (
        f"{profile.label} | {profile.point_count} 点 | "
        f"{profile.min_hz:.1f}-{profile.max_hz:.1f} Hz | "
        f"参考幅值 {mag_min_db:.2f}..{mag_max_db:.2f} dB | "
        f"相位跨度 {profile.phase_span_deg:.1f}°"
    )
