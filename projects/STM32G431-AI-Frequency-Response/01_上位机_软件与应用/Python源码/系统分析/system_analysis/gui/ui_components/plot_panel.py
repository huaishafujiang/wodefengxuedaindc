from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotPanel(QWidget):
    """Pure-white Bode/Nyquist panel that consumes MeasurementSession data."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("plotPanel")
        self.setStyleSheet(
            """
            QWidget#plotPanel {
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
            """
        )
        self._setup_ui()
        self.show_empty_state()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.fig = Figure(figsize=(9, 6), dpi=100, facecolor="#FFFFFF", constrained_layout=True)
        grid = self.fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], height_ratios=[1.0, 1.0])
        self.ax_mag = self.fig.add_subplot(grid[0, :2])
        self.ax_phase = self.fig.add_subplot(grid[1, :2], sharex=self.ax_mag)
        self.ax_nyquist = self.fig.add_subplot(grid[:, 2])

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QToolBar { border: none; background: #FFFFFF; spacing: 4px; }")

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def show_empty_state(self) -> None:
        self._clear_axes()
        self.ax_mag.text(
            0.5,
            0.5,
            "等待测量分析",
            transform=self.ax_mag.transAxes,
            ha="center",
            va="center",
            color="#9AA0A6",
            fontsize=13,
        )
        self.ax_phase.text(
            0.5,
            0.5,
            "相频数据将在扫频后显示",
            transform=self.ax_phase.transAxes,
            ha="center",
            va="center",
            color="#9AA0A6",
            fontsize=11,
        )
        self.ax_nyquist.text(
            0.5,
            0.5,
            "Nyquist",
            transform=self.ax_nyquist.transAxes,
            ha="center",
            va="center",
            color="#9AA0A6",
            fontsize=11,
        )
        self.canvas.draw_idle()

    def update_plots(self, session: Any) -> None:
        if not bool(getattr(session, "has_complete_arrays", getattr(session, "is_valid", False))):
            return

        result = getattr(session, "result", None)
        if result is not None:
            omega = np.asarray(getattr(result, "omega", []), dtype=float)
            mag_linear = np.asarray(getattr(result, "mag_smooth", getattr(result, "mag_raw", [])), dtype=float)
            mag_db = np.asarray(getattr(result, "mag_db", []), dtype=float)
            phase_deg = np.asarray(getattr(result, "phase_deg", []), dtype=float)
            real_part = np.asarray(getattr(result, "real_part", []), dtype=float)
            imag_part = np.asarray(getattr(result, "imag_part", []), dtype=float)
        else:
            omega = np.asarray(getattr(session, "omega", []), dtype=float)
            mag_linear = np.asarray(getattr(session, "magnitude_data", []), dtype=float)
            mag_db = 20.0 * np.log10(np.clip(mag_linear, 1e-12, None))
            phase_rad = np.asarray(getattr(session, "phase_data_rad", []), dtype=float)
            phase_deg = np.degrees(phase_rad)
            real_part = mag_linear * np.cos(phase_rad)
            imag_part = mag_linear * np.sin(phase_rad)

        n = min(len(omega), len(mag_db), len(phase_deg), len(real_part), len(imag_part))
        if n <= 0:
            self.show_empty_state()
            return

        omega = omega[:n]
        mag_db = mag_db[:n]
        phase_deg = phase_deg[:n]
        real_part = real_part[:n]
        imag_part = imag_part[:n]
        valid = np.isfinite(omega) & (omega > 0.0)
        valid &= np.isfinite(mag_db) & np.isfinite(phase_deg) & np.isfinite(real_part) & np.isfinite(imag_part)
        if not np.any(valid):
            self.show_empty_state()
            return

        omega = omega[valid]
        freq_hz = omega / (2.0 * np.pi)
        mag_db = mag_db[valid]
        phase_deg = phase_deg[valid]
        real_part = real_part[valid]
        imag_part = imag_part[valid]

        self._clear_axes()
        line_color = "#1A73E8"

        self.ax_mag.semilogx(freq_hz, mag_db, color=line_color, linewidth=2.0)
        self.ax_phase.semilogx(freq_hz, phase_deg, color=line_color, linewidth=2.0)
        self.ax_nyquist.plot(real_part, imag_part, color=line_color, linewidth=2.0)
        self.ax_nyquist.plot(real_part[0], imag_part[0], "o", color="#1E8E3E", markersize=5, label="起点")
        self.ax_nyquist.plot(real_part[-1], imag_part[-1], "x", color="#D93025", markersize=6, label="终点")
        self.ax_nyquist.plot(-1.0, 0.0, marker="+", color="#D93025", markersize=10, mew=2)
        self.ax_nyquist.axhline(0.0, color="#DADCE0", linewidth=1.0)
        self.ax_nyquist.axvline(0.0, color="#DADCE0", linewidth=1.0)
        self.ax_nyquist.set_aspect("equal", adjustable="datalim")
        self.ax_nyquist.legend(loc="best", fontsize=8, frameon=False)
        self.canvas.draw_idle()

    def _clear_axes(self) -> None:
        for axis in (self.ax_mag, self.ax_phase, self.ax_nyquist):
            axis.clear()
        self._format_axes()

    def _format_axes(self) -> None:
        for axis in (self.ax_mag, self.ax_phase, self.ax_nyquist):
            axis.set_facecolor("#FFFFFF")
            axis.grid(True, which="major", color="#EDEDED", linestyle="-", linewidth=0.8)
            axis.grid(True, which="minor", color="#F3F4F4", linestyle="-", linewidth=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_color("#DADCE0")
            axis.spines["left"].set_color("#DADCE0")
            axis.tick_params(colors="#5F6368", labelsize=8)

        self.ax_mag.set_title("Bode 幅频响应", loc="left", fontsize=10, color="#202124")
        self.ax_mag.set_ylabel("Magnitude (dB)", color="#5F6368")
        self.ax_phase.set_title("Bode 相频响应", loc="left", fontsize=10, color="#202124")
        self.ax_phase.set_xlabel("Frequency (Hz)", color="#5F6368")
        self.ax_phase.set_ylabel("Phase (deg)", color="#5F6368")
        self.ax_nyquist.set_title("Nyquist 图", loc="left", fontsize=10, color="#202124")
        self.ax_nyquist.set_xlabel("Real", color="#5F6368")
        self.ax_nyquist.set_ylabel("Imaginary", color="#5F6368")
