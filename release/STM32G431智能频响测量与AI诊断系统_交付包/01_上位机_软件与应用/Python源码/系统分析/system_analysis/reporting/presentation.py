from __future__ import annotations

from typing import Any

import numpy as np

from system_analysis.analysis.filter_analysis import phase_array_for_display


def _series_or_empty(value: Any) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=float).flatten()


def _smooth_series(values: Any, window: int = 9) -> np.ndarray:
    values = _series_or_empty(values)
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


def _highpass_noise_floor_mask(result: Any) -> np.ndarray:
    omega = _series_or_empty(getattr(result, "omega", []))
    mag_db = _series_or_empty(getattr(result, "mag_db", []))
    n = min(len(omega), len(mag_db))
    mask = np.zeros(n, dtype=bool)
    if n < 8 or str(getattr(result, "filter_type", "other")) != "highpass":
        return mask

    omega = omega[:n]
    mag_db = mag_db[:n]
    finite = np.isfinite(omega) & np.isfinite(mag_db) & (omega > 0.0)
    if np.count_nonzero(finite) < 8:
        return mask

    edge = max(3, min(10, n // 8))
    passband_db = float(np.nanmedian(mag_db[-edge:]))
    cutoff_omega = float(getattr(result, "magnitude_cutoff_omega", getattr(result, "omega_c", omega[n // 2])))
    if not np.isfinite(cutoff_omega) or cutoff_omega <= 0.0:
        cutoff_omega = float(omega[n // 2])

    log_omega = np.log10(np.clip(omega, 1e-12, None))
    slope = np.gradient(mag_db, log_omega)
    if n >= 7:
        kernel = np.ones(5, dtype=float) / 5.0
        slope = np.convolve(slope, kernel, mode="same")

    order = max(1, int(getattr(result, "order_estimate", 1) or 1))
    low_side = omega < cutoff_omega * 0.72
    far_below_passband = mag_db < passband_db - max(14.0, 5.0 * order)
    too_flat_for_highpass = slope < max(10.0, 4.0 * order)
    candidate = finite & low_side & far_below_passband & too_flat_for_highpass
    if not np.any(candidate):
        return mask

    candidate_indices = np.flatnonzero(candidate)
    last = int(candidate_indices[0])
    for idx in candidate_indices:
        if idx <= last + 2:
            last = int(idx)
        else:
            break
    mask[: last + 1] = candidate[: last + 1]
    return mask


def presentation_bode_series(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return display-only Bode curves derived from measured data.

    This intentionally does not alter analysis results, filter identification,
    exported raw arrays, or Nyquist data. It only smooths the visible Bode line
    and hides the leading high-pass noise-floor segment that otherwise looks
    like a false flat stopband in presentations.
    """
    omega = _series_or_empty(getattr(result, "omega", []))
    mag_db = _smooth_series(getattr(result, "mag_db", []), 9)
    phase_display = phase_array_for_display(getattr(result, "phase_deg", []), getattr(result, "filter_type", "other"))
    phase_display = _smooth_series(phase_display, 9)
    n = min(len(omega), len(mag_db), len(phase_display))
    if n == 0:
        return omega[:0], mag_db[:0], phase_display[:0]

    omega = omega[:n]
    mag_db = mag_db[:n]
    phase_display = phase_display[:n]
    valid = np.isfinite(omega) & np.isfinite(mag_db) & np.isfinite(phase_display) & (omega > 0.0)
    if np.count_nonzero(valid) < 5:
        return omega, mag_db, phase_display

    if str(getattr(result, "filter_type", "other")) == "highpass":
        noise_mask = _highpass_noise_floor_mask(result)
        if len(noise_mask) < n:
            noise_mask = np.pad(noise_mask, (0, n - len(noise_mask)), constant_values=False)
        else:
            noise_mask = noise_mask[:n]
        if np.any(noise_mask):
            first_good = int(np.argmax(~noise_mask)) if np.any(~noise_mask) else n
            if 0 < first_good < n:
                mag_db[:first_good] = np.nan
                phase_display[:first_good] = np.nan

    return omega, mag_db, phase_display
