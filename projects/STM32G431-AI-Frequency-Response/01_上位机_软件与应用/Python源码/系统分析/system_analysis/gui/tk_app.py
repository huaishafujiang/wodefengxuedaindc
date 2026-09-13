from __future__ import annotations

import csv
import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

from matplotlib import font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from system_analysis.analysis.ai_diagnosis import format_ai_diagnosis_report_lines, run_intelligent_diagnosis
from system_analysis.analysis.control_compensation import settings_from_inputs
from system_analysis.reporting.diagnosis_view import diagnosis_clipboard_text, structured_diagnosis_sections
from system_analysis.analysis.filter_analysis import (
    EXPECTED_CIRCUIT_CHOICES,
    MAX_FREQ_STEPS,
    SystemAnalysisResult,
    analyze_system_v2,
    build_filter_report_lines,
    evaluate_expected_circuit,
    sweep_segment_point_count,
)
from system_analysis.core.measurement_session import MeasurementSession
from system_analysis.reporting.plotting import create_bode_nyquist_axes, plot_sessions
from system_analysis.reporting.report_export import export_html_report
from system_analysis.io.serial_protocol import build_sweep_command, parse_diagnostics_from_text, parse_measurement_frame_from_text
from system_analysis.io.serial_readers import ThreeLineReader, list_serial_ports
from system_analysis.io.serial_transport import open_serial_transport, write_ascii_command
from system_analysis.io.smart_resweep_reader import SmartResweepReader
from system_analysis.analysis.transfer_formula import build_transfer_formula_view, save_formula_png

try:
    import serial
except Exception:
    serial = None


APP_TITLE = "稳频仪 - STM32G431 通用电路频响与稳定性分析平台"


def configure_plot_fonts():
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_arrays_from_text(text: str):
    frame = parse_measurement_frame_from_text(text)
    return frame.omega, frame.magnitude, frame.phase


class MatlabExactApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1580x980")
        self.root.minsize(1320, 820)

        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.reader = None
        self.sessions: list[MeasurementSession] = []
        self.session_by_iid: dict[str, MeasurementSession] = {}
        self.current_session: MeasurementSession | None = None
        self._next_session_id = 1
        self._note_update_job = None

        self.result: SystemAnalysisResult | None = None
        self.current_measurement = None
        self.ai_diagnosis = None
        self.font_name = configure_plot_fonts()

        self._build_ui()
        self.root.after(0, lambda: self._maximize_window(self.root))
        self.refresh_ports()
        self.root.after(100, self._poll_queue)

    def _build_ui(self):
        self._configure_style()
        self._init_ui_state()

        main = ttk.Frame(self.root, padding=(12, 10), style="App.TFrame")
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        toolbar = self._build_top_toolbar(main)
        toolbar.grid(row=0, column=0, sticky="ew")

        sweep_panel = self._build_sweep_panel(main)
        sweep_panel.grid(row=1, column=0, sticky="ew", pady=(10, 10))

        body = ttk.PanedWindow(main, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew")
        self.body_paned = body
        body.bind("<Configure>", lambda _event: self.root.after_idle(self._fit_body_panes))

        plot_host = ttk.Frame(body, padding=(0, 0, 8, 0), style="App.TFrame")
        sidebar_host = ttk.Frame(body, padding=(8, 0, 0, 0), style="App.TFrame")
        body.add(plot_host, weight=5)
        body.add(sidebar_host, weight=3)

        self._build_plot_area(plot_host)
        self._build_diagnosis_sidebar(sidebar_host)

        bottom = ttk.Frame(main, height=245, style="App.TFrame")
        bottom.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        bottom.grid_propagate(False)
        self._build_bottom_notebook(bottom)

        self.log("稳频仪已就绪；接收测量帧后会创建内存 session 并异步分析。")
        self._update_manual_point_count()

    def _fit_body_panes(self) -> None:
        body = getattr(self, "body_paned", None)
        if body is None:
            return
        try:
            width = body.winfo_width()
            if width > 900:
                body.sashpos(0, int(width * 0.62))
        except Exception:
            pass

    def _configure_style(self):
        self.colors = {
            "bg": "#ffffff",
            "surface": "#ffffff",
            "surface_alt": "#ffffff",
            "toolbar": "#ffffff",
            "border": "#d7dee8",
            "ink": "#172033",
            "muted": "#64748b",
            "accent": "#1d4ed8",
            "accent_hover": "#1e40af",
            "danger": "#dc2626",
            "danger_hover": "#b91c1c",
            "soft": "#ffffff",
        }
        bg = self.colors["bg"]
        surface = self.colors["surface"]
        ink = self.colors["ink"]
        muted = self.colors["muted"]

        self.root.configure(bg=bg)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        ui_font = self.font_name or "Microsoft YaHei"
        try:
            style.configure(".", font=(ui_font, 9), background=bg, foreground=ink)
            style.configure("App.TFrame", background=bg)
            style.configure("Toolbar.TFrame", background=self.colors["toolbar"])
            style.configure("Surface.TFrame", background=surface)
            style.configure("Inset.TFrame", background=self.colors["surface_alt"])
            style.configure(
                "Surface.TLabelframe",
                background=surface,
                bordercolor=self.colors["border"],
                relief="solid",
            )
            style.configure(
                "Surface.TLabelframe.Label",
                background=surface,
                foreground=ink,
                font=(ui_font, 10, "bold"),
            )
            style.configure("Toolbar.TLabel", background=self.colors["toolbar"], foreground=ink)
            style.configure(
                "ToolbarTitle.TLabel",
                background=self.colors["toolbar"],
                foreground=ink,
                font=(ui_font, 13, "bold"),
            )
            style.configure("ToolbarMuted.TLabel", background=self.colors["toolbar"], foreground=muted)
            style.configure("Header.TLabel", font=(ui_font, 15, "bold"), foreground=ink, background=bg)
            style.configure("Subtle.TLabel", font=(ui_font, 9), foreground=muted, background=bg)
            style.configure("Surface.TLabel", foreground=ink, background=surface)
            style.configure("SurfaceMuted.TLabel", foreground=muted, background=surface)
            style.configure("Inset.TLabel", foreground=ink, background=self.colors["surface_alt"])
            style.configure("Value.TLabel", font=(ui_font, 10, "bold"), foreground=ink, background=surface)
            style.configure(
                "Result.TLabel",
                font=(ui_font, 12, "bold"),
                foreground=self.colors["accent"],
                background=self.colors["toolbar"],
            )
            style.configure("Badge.TLabel", font=(ui_font, 9, "bold"), foreground=ink, background=self.colors["soft"])
            style.configure(
                "Primary.TButton",
                font=(ui_font, 10, "bold"),
                foreground="#ffffff",
                background=self.colors["accent"],
                bordercolor=self.colors["accent"],
                focusthickness=2,
                focuscolor=self.colors["accent"],
                padding=(12, 6),
            )
            style.map(
                "Primary.TButton",
                background=[("active", self.colors["accent_hover"]), ("disabled", "#94a3b8")],
                bordercolor=[("active", self.colors["accent_hover"])],
            )
            style.configure(
                "Secondary.TButton",
                foreground=ink,
                background=surface,
                bordercolor=self.colors["border"],
                padding=(10, 5),
            )
            style.map("Secondary.TButton", background=[("active", self.colors["surface_alt"])])
            style.configure(
                "Danger.TButton",
                font=(ui_font, 9, "bold"),
                foreground="#ffffff",
                background=self.colors["danger"],
                bordercolor=self.colors["danger"],
                padding=(10, 5),
            )
            style.map(
                "Danger.TButton",
                background=[("active", self.colors["danger_hover"]), ("disabled", "#fca5a5")],
                bordercolor=[("active", self.colors["danger_hover"])],
            )
            style.configure("Treeview", rowheight=26, background=surface, fieldbackground=surface, foreground=ink)
            style.configure("Treeview.Heading", font=(ui_font, 9, "bold"), background=self.colors["surface_alt"])
            style.configure("TNotebook", background=bg, borderwidth=0)
            style.configure("TNotebook.Tab", padding=(14, 6), font=(ui_font, 9))
        except Exception:
            pass

    def _maximize_window(self, window) -> None:
        try:
            window.state("zoomed")
            return
        except Exception:
            pass
        try:
            window.attributes("-zoomed", True)
            return
        except Exception:
            pass
        try:
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            window.geometry(f"{screen_w}x{screen_h}+0+0")
        except Exception:
            pass

    def _init_ui_state(self):
        self.system_type_var = tk.StringVar(value="待分析")
        self.status_var = tk.StringVar(value="就绪")
        self.connection_state_var = tk.StringVar(value="未连接")

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.timeout_var = tk.StringVar(value="60")

        self.f_start_var = tk.StringVar(value="100")
        self.f_stop_var = tk.StringVar(value="20000")
        self.f_step_var = tk.StringVar(value="100")
        self.amp_var = tk.StringVar(value="1.2")
        self.point_count_var = tk.StringVar(value="点数 200")

        self.smooth_var = tk.BooleanVar(value=True)
        self.fix_var = tk.BooleanVar(value=False)
        self.swap_io_var = tk.BooleanVar(value=False)
        self.window_var = tk.StringVar(value="11")
        self.poly_var = tk.StringVar(value="3")
        self.open_loop_p_var = tk.StringVar(value="0")
        self.expected_circuit_var = tk.StringVar(value="Auto")
        self.control_compensation_enabled_var = tk.BooleanVar(value=True)
        self.control_compensation_mode_var = tk.StringVar(value="自动")
        self.control_target_pm_var = tk.StringVar(value="50")
        self.control_target_gm_var = tk.StringVar(value="6")
        self.control_safety_phase_var = tk.StringVar(value="8")
        self.control_target_crossover_var = tk.StringVar(value="")
        self.control_low_freq_boost_var = tk.StringVar(value="0")

    def _build_top_toolbar(self, parent):
        toolbar = ttk.Frame(parent, padding=(12, 8), style="Toolbar.TFrame")
        toolbar.columnconfigure(15, weight=1)

        ttk.Label(toolbar, text="STM32G431 智能频响分析仪", style="ToolbarTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 14)
        )
        ttk.Label(toolbar, text="COM", style="ToolbarMuted.TLabel").grid(row=0, column=1, sticky="w")
        self.port_box = ttk.Combobox(toolbar, textvariable=self.port_var, width=12, state="readonly")
        self.port_box.grid(row=0, column=2, sticky="w", padx=(5, 8))
        ttk.Button(toolbar, text="刷新", command=self.refresh_ports, style="Secondary.TButton").grid(
            row=0, column=3, sticky="w", padx=(0, 8)
        )
        ttk.Label(toolbar, text="波特率", style="ToolbarMuted.TLabel").grid(row=0, column=4, sticky="w")
        ttk.Entry(toolbar, textvariable=self.baud_var, width=9).grid(row=0, column=5, sticky="w", padx=(5, 10))

        ttk.Button(toolbar, text="PING", command=self.command_ping, style="Secondary.TButton").grid(
            row=0, column=6, sticky="w", padx=(0, 4)
        )
        ttk.Button(toolbar, text="HELP", command=self.command_help, style="Secondary.TButton").grid(
            row=0, column=7, sticky="w", padx=(0, 4)
        )
        ttk.Button(toolbar, text="STOP", command=self.stop_reading, style="Danger.TButton").grid(
            row=0, column=8, sticky="w", padx=(2, 14)
        )

        status = ttk.Frame(toolbar, style="Toolbar.TFrame")
        status.grid(row=0, column=15, sticky="e")
        ttk.Label(status, text="连接", style="ToolbarMuted.TLabel").grid(row=0, column=0, sticky="e")
        ttk.Label(status, textvariable=self.connection_state_var, style="Toolbar.TLabel").grid(
            row=0, column=1, sticky="e", padx=(5, 14)
        )
        ttk.Label(status, text="Session", style="ToolbarMuted.TLabel").grid(row=0, column=2, sticky="e")
        ttk.Label(status, textvariable=self.status_var, style="Toolbar.TLabel").grid(
            row=0, column=3, sticky="e", padx=(5, 14)
        )
        ttk.Label(status, text="识别", style="ToolbarMuted.TLabel").grid(row=0, column=4, sticky="e")
        ttk.Label(status, textvariable=self.system_type_var, style="Result.TLabel").grid(
            row=0, column=5, sticky="e", padx=(5, 0)
        )
        return toolbar

    def _build_sweep_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="扫频测量", padding=(12, 9), style="Surface.TLabelframe")
        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(3, weight=1)
        panel.columnconfigure(5, weight=1)
        panel.columnconfigure(7, weight=1)

        self._entry(panel, 0, 0, "起始 Hz", self.f_start_var, width=10, padx=(6, 14))
        self._entry(panel, 0, 2, "终止 Hz", self.f_stop_var, width=10, padx=(6, 14))
        self._entry(panel, 0, 4, "步进 Hz", self.f_step_var, width=10, padx=(6, 14))
        self._entry(panel, 0, 6, "幅值 Vpp", self.amp_var, width=8, padx=(6, 14))
        ttk.Label(panel, textvariable=self.point_count_var, style="SurfaceMuted.TLabel").grid(
            row=0, column=8, sticky="w", padx=(0, 12)
        )
        ttk.Button(panel, text="开始扫频分析", command=self.command_sweep_once, style="Primary.TButton").grid(
            row=0, column=9, sticky="e", ipady=2
        )
        for var in (self.f_start_var, self.f_stop_var, self.f_step_var):
            var.trace_add("write", lambda *_: self._update_manual_point_count())
        return panel

    def _build_plot_area(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        plot_box = ttk.LabelFrame(parent, text="频响曲线", padding=(10, 8), style="Surface.TLabelframe")
        plot_box.grid(row=0, column=0, sticky="nsew")
        plot_box.rowconfigure(1, weight=1)
        plot_box.columnconfigure(0, weight=1)

        header = ttk.Frame(plot_box, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Bode 幅频 / 相频 + Nyquist", style="Value.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="放大 Bode", command=lambda: self._open_plot_zoom("bode"), style="Secondary.TButton").grid(
            row=0, column=1, sticky="e", padx=(0, 6)
        )
        ttk.Button(
            header,
            text="放大 Nyquist",
            command=lambda: self._open_plot_zoom("nyquist"),
            style="Secondary.TButton",
        ).grid(row=0, column=2, sticky="e")

        canvas_frame = ttk.Frame(plot_box, style="Surface.TFrame")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(11.8, 7.2), dpi=100, facecolor="#ffffff", constrained_layout=True)
        self.ax_mag, self.ax_phase, self.ax_ny = create_bode_nyquist_axes(self.fig)
        self.ax_stab = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb = NavigationToolbar2Tk(self.canvas, canvas_frame, pack_toolbar=False)
        tb.update()
        tb.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        plot_sessions(self.fig, (self.ax_mag, self.ax_phase, self.ax_ny), [], None)

    def _build_diagnosis_sidebar(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        diagnosis_box = ttk.LabelFrame(parent, text="当前诊断摘要", padding=10, style="Surface.TLabelframe")
        diagnosis_box.grid(row=0, column=0, sticky="ew")
        diagnosis_box.rowconfigure(0, weight=1)
        diagnosis_box.columnconfigure(1, weight=1)
        self.diagnosis_vars: dict[str, tk.StringVar] = {}
        self.diagnosis_value_labels = []

        summary_canvas = tk.Canvas(
            diagnosis_box,
            height=300,
            bd=0,
            highlightthickness=0,
            background=self.colors["surface"],
        )
        summary_canvas.grid(row=0, column=0, columnspan=2, sticky="ew")
        summary_frame = ttk.Frame(summary_canvas, style="Surface.TFrame")
        summary_window = summary_canvas.create_window((0, 0), window=summary_frame, anchor="nw")
        summary_frame.columnconfigure(1, weight=1)
        self.diagnosis_summary_canvas = summary_canvas

        def _refresh_summary_scrollregion(_event=None):
            summary_canvas.configure(scrollregion=summary_canvas.bbox("all"))

        def _resize_summary_frame(event):
            summary_canvas.itemconfigure(summary_window, width=event.width)
            wraplength = max(320, int(event.width) - 116)
            for label in self.diagnosis_value_labels:
                label.configure(wraplength=wraplength)
            _refresh_summary_scrollregion()

        def _scroll_summary(event):
            if getattr(event, "num", None) == 4:
                summary_canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                summary_canvas.yview_scroll(1, "units")
            elif getattr(event, "delta", 0):
                summary_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def _bind_summary_wheel(widget):
            widget.bind("<MouseWheel>", _scroll_summary)
            widget.bind("<Button-4>", _scroll_summary)
            widget.bind("<Button-5>", _scroll_summary)

        summary_frame.bind("<Configure>", _refresh_summary_scrollregion)
        summary_canvas.bind("<Configure>", _resize_summary_frame)
        _bind_summary_wheel(summary_canvas)
        _bind_summary_wheel(summary_frame)

        summary_keys = (
            "判定",
            "置信度",
            "测量健康",
            "AI结论",
            "关键频率",
            "Top3",
            "测量质量",
            "控制校正",
            "测量补扫",
            "故障证据",
            "下一步建议",
        )
        for row, key in enumerate(summary_keys):
            key_label = ttk.Label(summary_frame, text=key, style="SurfaceMuted.TLabel")
            key_label.grid(row=row, column=0, sticky="nw", pady=3)
            var = tk.StringVar(value="暂无")
            self.diagnosis_vars[key] = var
            value_style = "Value.TLabel" if key in ("判定", "置信度", "测量健康") else "Surface.TLabel"
            value_label = ttk.Label(summary_frame, textvariable=var, style=value_style, wraplength=360)
            value_label.grid(
                row=row, column=1, sticky="ew", padx=(10, 0), pady=3
            )
            self.diagnosis_value_labels.append(value_label)
            _bind_summary_wheel(key_label)
            _bind_summary_wheel(value_label)
        diagnosis_actions = ttk.Frame(diagnosis_box, style="Surface.TFrame")
        diagnosis_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        diagnosis_actions.columnconfigure(0, weight=1)
        diagnosis_actions.columnconfigure(1, weight=1)
        ttk.Button(
            diagnosis_actions,
            text="刷新当前诊断",
            command=self.run_ai_diagnosis_for_current,
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            diagnosis_actions,
            text="智能补扫",
            command=self.command_smart_resweep,
            style="Primary.TButton",
        ).grid(row=0, column=1, sticky="ew")
        ttk.Button(
            diagnosis_actions,
            text="手动补扫",
            command=self.command_manual_resweep,
            style="Secondary.TButton",
        ).grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))

        history_box = ttk.LabelFrame(parent, text="测量历史", padding=10, style="Surface.TLabelframe")
        history_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        history_box.rowconfigure(0, weight=1)
        history_box.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(
            history_box,
            columns=("time", "source", "points", "decision", "confidence"),
            show="headings",
            height=10,
            selectmode="extended",
        )
        history_cols = {
            "time": ("时间", 74),
            "source": ("来源", 96),
            "points": ("点数", 52),
            "decision": ("判定", 150),
            "confidence": ("置信度", 58),
        }
        for col, (label, width) in history_cols.items():
            self.history_tree.heading(col, text=label)
            anchor = "center" if col in ("time", "points", "confidence") else "w"
            self.history_tree.column(col, width=width, anchor=anchor, stretch=col in ("source", "decision"))
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        history_scroll = ttk.Scrollbar(history_box, orient="vertical", command=self.history_tree.yview)
        history_scroll.grid(row=0, column=1, sticky="ns")
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        note_frame = ttk.LabelFrame(parent, text="当前 session 备注", padding=10, style="Surface.TLabelframe")
        note_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        note_frame.columnconfigure(0, weight=1)
        self.remark_var = tk.StringVar()
        self.remark_entry = ttk.Entry(note_frame, textvariable=self.remark_var)
        self.remark_entry.grid(row=0, column=0, sticky="ew")
        self.remark_var.trace_add("write", self._on_remark_change)

    def _build_bottom_notebook(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        log_tab = ttk.Frame(notebook, padding=8, style="Surface.TFrame")
        report_tab = ttk.Frame(notebook, padding=8, style="Surface.TFrame")
        advanced_tab = ttk.Frame(notebook, padding=10, style="Surface.TFrame")
        control_tab = ttk.Frame(notebook, padding=10, style="Surface.TFrame")
        export_tab = ttk.Frame(notebook, padding=10, style="Surface.TFrame")
        notebook.add(log_tab, text="流水日志")
        notebook.add(report_tab, text="详细报告")
        notebook.add(advanced_tab, text="高级扫频参数")
        notebook.add(control_tab, text="自动控制校正")
        notebook.add(export_tab, text="导出设置")

        self._build_log_tab(log_tab)
        self._build_report_tab(report_tab)
        self._build_advanced_tab(advanced_tab)
        self._build_control_compensation_tab(control_tab)
        self._build_export_tab(export_tab)

    def _build_log_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            parent,
            wrap="word",
            font=("Consolas", 9),
            height=7,
            relief="flat",
            padx=10,
            pady=8,
            bg="#ffffff",
            fg=self.colors["ink"],
            insertbackground=self.colors["ink"],
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _build_report_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        tools = ttk.Frame(parent, style="Surface.TFrame")
        tools.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        tools.columnconfigure(4, weight=1)
        ttk.Button(tools, text="复制诊断", command=self.copy_diagnosis, style="Secondary.TButton").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        ttk.Button(tools, text="清空报告视图", command=self.clear_report_view, style="Secondary.TButton").grid(
            row=0, column=1, sticky="w", padx=(0, 6)
        )
        ttk.Button(tools, text="放大报告", command=self.open_report_zoom, style="Secondary.TButton").grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )
        ttk.Button(tools, text="导出 HTML 报告", command=self.export_report, style="Primary.TButton").grid(
            row=0, column=3, sticky="w"
        )
        self.report_text = tk.Text(
            parent,
            wrap="word",
            font=("Consolas", 9),
            height=8,
            relief="flat",
            padx=10,
            pady=8,
            bg="#ffffff",
            fg=self.colors["ink"],
            insertbackground=self.colors["ink"],
        )
        self.report_text.grid(row=1, column=0, sticky="nsew")
        report_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.report_text.yview)
        report_scroll.grid(row=1, column=1, sticky="ns")
        self.report_text.configure(yscrollcommand=report_scroll.set)

    def _build_advanced_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Checkbutton(parent, text="平滑", variable=self.smooth_var).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Checkbutton(parent, text="首点修正", variable=self.fix_var).grid(row=0, column=1, sticky="w", padx=(0, 12))
        ttk.Checkbutton(parent, text="交换输入/输出", variable=self.swap_io_var).grid(row=0, column=2, sticky="w", padx=(0, 12))

        fields = ttk.Frame(parent, style="Surface.TFrame")
        fields.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(12, 0))
        fields.columnconfigure(1, weight=1)
        fields.columnconfigure(3, weight=1)
        fields.columnconfigure(5, weight=1)
        self._entry(fields, 0, 0, "窗口", self.window_var, width=6, padx=(6, 18))
        self._entry(fields, 0, 2, "阶数", self.poly_var, width=6, padx=(6, 18))
        self._entry(fields, 0, 4, "RHP P", self.open_loop_p_var, width=6, padx=(6, 18))
        self._entry(fields, 0, 6, "超时 s", self.timeout_var, width=7, padx=(6, 18))
        ttk.Label(fields, text="期望电路", style="Surface.TLabel").grid(row=0, column=8, sticky="w")
        self.expected_circuit_box = ttk.Combobox(
            fields,
            textvariable=self.expected_circuit_var,
            values=EXPECTED_CIRCUIT_CHOICES,
            width=16,
            state="readonly",
        )
        self.expected_circuit_box.grid(row=0, column=9, sticky="w", padx=(6, 0))

    def _build_control_compensation_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        parent.columnconfigure(5, weight=1)

        ttk.Checkbutton(
            parent,
            text="启用自动控制校正",
            variable=self.control_compensation_enabled_var,
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))

        ttk.Label(parent, text="校正类型", style="Surface.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            parent,
            textvariable=self.control_compensation_mode_var,
            values=("自动", "比例增益", "超前", "滞后", "滞后-超前"),
            width=12,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(6, 18), pady=(0, 8))
        self._entry(parent, 1, 2, "目标 PM °", self.control_target_pm_var, width=8, padx=(6, 18))
        self._entry(parent, 1, 4, "最小 GM dB", self.control_target_gm_var, width=8, padx=(6, 0))

        self._entry(parent, 2, 0, "安全相位 °", self.control_safety_phase_var, width=8, padx=(6, 18))
        self._entry(parent, 2, 2, "目标穿越 Hz", self.control_target_crossover_var, width=10, padx=(6, 18))
        self._entry(parent, 2, 4, "低频增益 dB", self.control_low_freq_boost_var, width=8, padx=(6, 0))

        hint = (
            "这里是自动控制原理里的校正：用实测开环 Bode 数据计算当前相位裕度/增益裕度；"
            "若不满足目标，自动给出 Gc(s)、零极点、alpha/beta/T 和预计校正后裕度。"
        )
        ttk.Label(parent, text=hint, style="SurfaceMuted.TLabel", wraplength=980).grid(
            row=3, column=0, columnspan=6, sticky="ew", pady=(8, 0)
        )

    def _build_export_tab(self, parent):
        parent.columnconfigure(4, weight=1)
        ttk.Button(parent, text="导入文本", command=self.import_text, style="Secondary.TButton").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ttk.Button(parent, text="保存 CSV", command=self.save_csv, style="Secondary.TButton").grid(
            row=0, column=1, sticky="w", padx=(0, 8)
        )
        ttk.Button(parent, text="保存图像", command=self.save_png, style="Secondary.TButton").grid(
            row=0, column=2, sticky="w", padx=(0, 8)
        )
        ttk.Button(parent, text="导出 HTML 报告", command=self.export_report, style="Primary.TButton").grid(
            row=0, column=3, sticky="w", padx=(0, 12)
        )
        ttk.Button(parent, text="智能补扫", command=self.command_smart_resweep, style="Secondary.TButton").grid(
            row=1, column=0, sticky="w", pady=(12, 0), padx=(0, 8)
        )
        ttk.Button(parent, text="手动补扫", command=self.command_manual_resweep, style="Secondary.TButton").grid(
            row=1, column=1, sticky="w", pady=(12, 0), padx=(0, 8)
        )

    def _open_plot_zoom(self, kind: str):
        title = "Bode 曲线放大" if kind == "bode" else "Nyquist 曲线放大"
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1100x760")
        win.minsize(840, 560)
        win.configure(bg=self.colors["bg"])

        host = ttk.Frame(win, padding=10, style="App.TFrame")
        host.pack(fill="both", expand=True)
        host.rowconfigure(1, weight=1)
        host.columnconfigure(0, weight=1)
        ttk.Label(host, text=title, style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        fig = Figure(figsize=(10.5, 6.8), dpi=100, facecolor="#ffffff", constrained_layout=False)
        if kind == "bode":
            grid = fig.add_gridspec(2, 1, left=0.08, right=0.98, top=0.95, bottom=0.09, hspace=0.28)
            ax_mag = fig.add_subplot(grid[0, 0])
            ax_phase = fig.add_subplot(grid[1, 0], sharex=ax_mag)
            ax_ny = fig.add_axes([0.0, 0.0, 0.01, 0.01], frameon=False)
            ax_ny.set_visible(False)
        elif kind == "nyquist":
            ax_ny = fig.add_subplot(111)
            ax_mag = fig.add_axes([0.0, 0.0, 0.01, 0.01], frameon=False)
            ax_phase = fig.add_axes([0.0, 0.0, 0.01, 0.01], frameon=False)
            ax_mag.set_visible(False)
            ax_phase.set_visible(False)
        else:
            ax_mag, ax_phase, ax_ny = create_bode_nyquist_axes(fig)

        canvas = FigureCanvasTkAgg(fig, master=host)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        plot_sessions(fig, (ax_mag, ax_phase, ax_ny), self._selected_sessions(), self.current_session)
        if kind == "bode":
            ax_ny.set_visible(False)
        elif kind == "nyquist":
            ax_mag.set_visible(False)
            ax_phase.set_visible(False)
        canvas.draw_idle()

        tb = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
        tb.update()
        tb.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _current_report_text(self) -> str:
        if hasattr(self, "report_text"):
            text = self.report_text.get("1.0", "end-1c").strip()
            if text:
                return text
        if self.current_session is None:
            return "暂无报告。"
        if self.current_session.error:
            return str(self.current_session.error)
        if self.current_session.report_lines:
            return "\n".join(self.current_session.report_lines)
        if self.current_session.result is None:
            return "分析中..."
        return "暂无报告。"

    def _report_zoom_sections(self) -> dict[str, str]:
        session = self.current_session
        if session is None:
            return {
                "诊断摘要": "暂无测量。",
                "等效传函": "暂无等效传递函数。",
                "完整报告": self._current_report_text(),
            }
        summary_lines = [f"{key}: {value}" for key, value in structured_diagnosis_sections(session).items()]
        formula_view = build_transfer_formula_view(session)
        return {
            "诊断摘要": "\n".join(summary_lines),
            "等效传函": formula_view.copy_text(),
            "完整报告": self._current_report_text(),
        }

    def open_report_zoom(self):
        sections = self._report_zoom_sections()
        text = "\n\n".join(f"### {title}\n{body}" for title, body in sections.items())
        formula_view = build_transfer_formula_view(self.current_session)
        win = tk.Toplevel(self.root)
        win.title("详细报告放大")
        win.geometry("1220x820")
        win.minsize(900, 620)
        win.configure(bg=self.colors["bg"])

        host = ttk.Frame(win, padding=10, style="App.TFrame")
        host.pack(fill="both", expand=True)
        host.rowconfigure(1, weight=1)
        host.columnconfigure(0, weight=1)

        header = ttk.Frame(host, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        title = "详细报告放大"
        if self.current_session is not None:
            title = f"session #{self.current_session.id} 详细报告"
        ttk.Label(header, text=title, style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(
            header,
            text="复制全文",
            command=lambda: self._copy_text_to_clipboard(text, "放大报告全文已复制到剪贴板。"),
            style="Secondary.TButton",
        ).grid(row=0, column=2, sticky="e", padx=(8, 0))

        notebook = ttk.Notebook(host)
        notebook.grid(row=1, column=0, sticky="nsew")

        formula_tab = ttk.Frame(notebook, padding=12, style="Surface.TFrame")
        summary_tab = ttk.Frame(notebook, padding=10, style="Surface.TFrame")
        raw_tab = ttk.Frame(notebook, padding=10, style="Surface.TFrame")
        notebook.add(formula_tab, text="等效传函公式")
        notebook.add(summary_tab, text="诊断摘要")
        notebook.add(raw_tab, text="完整报告")

        self._build_formula_zoom_tab(formula_tab, formula_view)
        self._build_report_text_tab(summary_tab, sections["诊断摘要"], font=("Microsoft YaHei", 11))
        self._build_report_text_tab(raw_tab, sections["完整报告"], font=("Consolas", 11))

    def _build_report_text_tab(self, parent, text: str, font):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        text_widget = tk.Text(
            parent,
            wrap="word",
            font=font,
            relief="flat",
            padx=14,
            pady=12,
            bg="#ffffff",
            fg=self.colors["ink"],
            insertbackground=self.colors["ink"],
        )
        text_widget.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(parent, orient="vertical", command=text_widget.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=scroll.set)
        text_widget.insert("1.0", text)

    def _build_formula_zoom_tab(self, parent, formula_view):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text=formula_view.title, style="Header.TLabel").grid(row=0, column=0, sticky="w")

        card = tk.Frame(parent, bg="#ffffff", highlightthickness=1, highlightbackground=self.colors["border"])
        card.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        card.columnconfigure(0, weight=1)

        image_path = Path(tempfile.gettempdir()) / f"stm32g431_formula_session_{getattr(self.current_session, 'id', 'preview')}.png"
        rendered = save_formula_png(formula_view, image_path)
        if rendered:
            try:
                formula_image = tk.PhotoImage(file=str(image_path))
                image_label = tk.Label(card, image=formula_image, bg="#ffffff")
                image_label.image = formula_image
                image_label.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 6))
            except Exception:
                rendered = False
        if not rendered:
            tk.Label(
                card,
                text=formula_view.plain,
                bg="#ffffff",
                fg=self.colors["ink"],
                font=("Consolas", 18, "bold"),
                wraplength=980,
                justify="center",
            ).grid(row=0, column=0, sticky="ew", padx=18, pady=(20, 10))

        detail_text = "\n".join(f"• {line}" for line in formula_view.details + formula_view.schematic_notes if line)
        tk.Label(
            card,
            text=detail_text or "暂无等效传递函数。请先完成测量和 AI 诊断。",
            bg="#ffffff",
            fg=self.colors["muted"],
            font=("Microsoft YaHei", 11),
            justify="left",
            anchor="w",
            wraplength=980,
        ).grid(row=1, column=0, sticky="ew", padx=22, pady=(10, 18))

        ttk.Button(
            parent,
            text="复制公式说明",
            command=lambda: self._copy_text_to_clipboard(formula_view.copy_text(), "等效传函公式已复制到剪贴板。"),
            style="Secondary.TButton",
        ).grid(row=2, column=0, sticky="e", pady=(10, 0))

    def _entry(self, parent, row, col, label, variable, width=10, padx=(8, 12)):
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(row=row, column=col, sticky="w")
        widget = ttk.Entry(parent, textvariable=variable, width=width)
        widget.grid(row=row, column=col + 1, sticky="ew", padx=padx, pady=(0 if row == 0 else 8, 0))
        return widget

    def log(self, msg: str):
        if not hasattr(self, "log_text"):
            return
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")

    def refresh_ports(self):
        ports = list_serial_ports()
        self.port_box["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        self.connection_state_var.set("未检测到串口" if not ports else f"检测到 {len(ports)} 个串口")
        if self.current_session is None:
            self.status_var.set("就绪")
        self.log(f"串口列表: {ports if ports else '无'}")

    def command_ping(self):
        self._send_short_serial_command("PING", read_lines=6, timeout_sec=1.2)

    def command_help(self):
        self._send_short_serial_command("HELP", read_lines=14, timeout_sec=2.0)

    def _send_short_serial_command(self, command: str, read_lines: int = 8, timeout_sec: float = 1.0):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo("提示", "读取线程正在运行，请先停止或等待完成。")
            return
        if serial is None:
            messagebox.showwarning("提示", "未安装 pyserial，请先执行 pip install pyserial。")
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("提示", "请先选择串口。")
            return
        try:
            baudrate = int(self.baud_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "波特率格式不正确。")
            return
        self.status_var.set(f"发送 {command} 中")
        threading.Thread(
            target=self._short_serial_command_worker,
            args=(command, port, baudrate, read_lines, timeout_sec),
            daemon=True,
        ).start()

    def _short_serial_command_worker(self, command: str, port: str, baudrate: int, read_lines: int, timeout_sec: float):
        ser = None
        try:
            ser = open_serial_transport(serial, port, baudrate, min(max(timeout_sec, 0.2), 1.0))
            self.queue.put(("connection_status", f"{port} @ {baudrate}"))
            write_ascii_command(ser, command if command.endswith("\n") else command + "\n")
            self.queue.put(("log", f"已发送命令: {command}"))

            lines = []
            deadline = time.monotonic() + max(timeout_sec, 0.5)
            while time.monotonic() < deadline and len(lines) < read_lines:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                lines.append(line)
                deadline = max(deadline, time.monotonic() + 0.25)

            if lines:
                for line in lines:
                    self.queue.put(("log", f"{command}> {line}"))
            else:
                self.queue.put(("log", f"{command} 未收到文本响应。"))
            self.queue.put(("status", f"{command} 完成"))
        except Exception as exc:
            self.queue.put(("error", f"{command} 失败: {exc}"))
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass

    def _update_manual_point_count(self):
        try:
            points = sweep_segment_point_count(
                float(self.f_start_var.get().strip()),
                float(self.f_stop_var.get().strip()),
                float(self.f_step_var.get().strip()),
            )
        except ValueError:
            points = 0
        suffix = "" if points <= MAX_FREQ_STEPS else f" > {MAX_FREQ_STEPS}"
        self.point_count_var.set(f"点数 {points}{suffix}")

    def _build_sweep_command(self, show_errors=True):
        try:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            amp = float(self.amp_var.get().strip())
        except ValueError:
            if show_errors:
                messagebox.showwarning("提示", "扫频参数格式不正确。")
            return None

        if f_start <= 0 or f_stop < f_start or f_step <= 0:
            if show_errors:
                messagebox.showwarning("提示", "频率范围不正确。")
            return None
        if amp <= 0 or amp > 3.0:
            if show_errors:
                messagebox.showwarning("提示", "幅值建议设置在 0 到 3.0 Vpp 之间。")
            return None
        points = sweep_segment_point_count(f_start, f_stop, f_step)
        if points > MAX_FREQ_STEPS and show_errors:
            ok = messagebox.askyesno(
                "点数超过固件上限",
                f"本次手动扫频约 {points} 点，固件当前上限为 {MAX_FREQ_STEPS} 点。\n"
                "继续发送时固件可能只保留前 200 点，是否继续？",
            )
            if not ok:
                return None
        return build_sweep_command(f_start, f_stop, f_step, amp)

    def _get_assumed_open_loop_poles(self):
        try:
            p = int(self.open_loop_p_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "开环右半平面极点数 P 必须是非负整数。")
            return None
        if p < 0:
            messagebox.showwarning("提示", "开环右半平面极点数 P 必须是非负整数。")
            return None
        return p

    def _analysis_settings(self):
        assumed_p = self._get_assumed_open_loop_poles()
        if assumed_p is None:
            return None
        control_settings = self._control_compensation_settings_from_ui(show_errors=True)
        if control_settings is None and bool(self.control_compensation_enabled_var.get()):
            return None
        try:
            return {
                "smooth": bool(self.smooth_var.get()),
                "window_length": int(self.window_var.get().strip()),
                "polyorder": int(self.poly_var.get().strip()),
                "fix_g431_axis": bool(self.fix_var.get()),
                "assumed_open_loop_rhp_poles": int(assumed_p),
                "invert_transfer": bool(self.swap_io_var.get()),
                "control_compensation_settings": control_settings,
            }
        except ValueError:
            messagebox.showwarning("提示", "分析参数格式不正确。")
            return None

    def _control_compensation_settings_from_ui(self, show_errors: bool = True):
        try:
            return settings_from_inputs(
                enabled=bool(self.control_compensation_enabled_var.get()),
                mode=self.control_compensation_mode_var.get(),
                target_phase_margin=self.control_target_pm_var.get(),
                target_gain_margin_db=self.control_target_gm_var.get(),
                safety_phase=self.control_safety_phase_var.get(),
                target_crossover_hz=self.control_target_crossover_var.get(),
                low_frequency_gain_boost_db=self.control_low_freq_boost_var.get(),
            )
        except Exception as exc:
            if show_errors:
                messagebox.showwarning("自动控制校正参数错误", str(exc))
            return None

    def _start_reader(self, command):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo("提示", "读取线程已在运行。")
            return
        if self._analysis_settings() is None:
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("提示", "请先选择串口。")
            return
        try:
            baudrate = int(self.baud_var.get().strip())
            timeout_sec = float(self.timeout_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "串口参数格式不正确。")
            return

        self.stop_event.clear()
        self.reader = ThreeLineReader(
            port=port,
            baudrate=baudrate,
            timeout_sec=timeout_sec,
            out_queue=self.queue,
            stop_event=self.stop_event,
            continuous=False,
            command=command,
            fallback_command=command,
        )
        self.reader.start()
        self.connection_state_var.set(f"{port} @ {baudrate}")
        self.status_var.set("扫频读取中")

    def command_sweep_once(self):
        command = self._build_sweep_command()
        if command is None:
            return
        self._start_reader(command=command)

    def stop_reading(self):
        self.stop_event.set()
        self.status_var.set("已请求停止")
        sent = False
        if self.reader is not None and hasattr(self.reader, "send_stop_command"):
            try:
                sent = bool(self.reader.send_stop_command())
            except Exception:
                sent = False
        if not sent:
            self._send_stop_best_effort()
        self.log("已请求停止读取，并尝试向固件发送 STOP。")

    def _send_stop_best_effort(self):
        if serial is None:
            return
        port = self.port_var.get().strip()
        if not port:
            return
        try:
            baudrate = int(self.baud_var.get().strip())
            ser = open_serial_transport(serial, port, baudrate, 0.2)
            try:
                write_ascii_command(ser, "STOP\n")
            finally:
                ser.close()
        except Exception:
            pass

    def command_smart_resweep(self):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo("提示", "读取线程正在运行，请先停止或等待完成。")
            return
        if self.current_session is None or self.ai_diagnosis is None:
            messagebox.showinfo("提示", "请先完成一次测量和智能诊断。")
            return
        sweep_plan = list(getattr(self.ai_diagnosis, "active_sweep_plan", None) or [])
        if not sweep_plan:
            sweep_plan = list(getattr(self.ai_diagnosis, "adaptive_sweep_commands", None) or [])
        if not sweep_plan:
            sweep_plan = [{"command": "DIAGNOSE", "reason": "先完整诊断当前合并数据，再决定是否需要继续自动补扫"}]
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("提示", "请先选择串口。")
            return
        try:
            baudrate = int(self.baud_var.get().strip())
            timeout_sec = float(self.timeout_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "串口参数格式不正确。")
            return

        expected = self.expected_circuit_var.get().strip() or "Auto"
        self.stop_event.clear()
        self.reader = SmartResweepReader(
            port=port,
            baudrate=baudrate,
            timeout_sec=timeout_sec,
            out_queue=self.queue,
            stop_event=self.stop_event,
            base_frame=self.current_session.as_legacy_tuple(),
            sweep_steps=sweep_plan,
            expected_circuit=expected,
            mode_label="智能补扫",
            merged_source="智能补扫合并结果",
            analysis_settings=self.current_session.analysis_settings,
            adaptive=True,
            max_steps=6,
        )
        self.reader.start()
        self.connection_state_var.set(f"{port} @ {baudrate}")
        self.status_var.set("智能补扫中")
        self.log(f"开始智能补扫，共 {len(sweep_plan)} 条 SWEEP 命令。")

    def command_manual_resweep(self):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo("提示", "读取线程正在运行，请先停止或等待完成。")
            return
        if self.current_session is None or not self.current_session.has_complete_arrays:
            messagebox.showinfo("提示", "请先完成一次基础测量，再执行手动补扫。")
            return
        command = self._build_sweep_command()
        if command is None:
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("提示", "请先选择串口。")
            return
        try:
            baudrate = int(self.baud_var.get().strip())
            timeout_sec = float(self.timeout_var.get().strip())
        except ValueError:
            messagebox.showwarning("提示", "串口参数格式不正确。")
            return

        expected = self.expected_circuit_var.get().strip() or "Auto"
        self.stop_event.clear()
        manual_step = {
            "command": command.strip(),
            "reason": "手动补扫: 使用上方扫频参数加密当前 session",
        }
        self.reader = SmartResweepReader(
            port=port,
            baudrate=baudrate,
            timeout_sec=timeout_sec,
            out_queue=self.queue,
            stop_event=self.stop_event,
            base_frame=self.current_session.as_legacy_tuple(),
            sweep_steps=[manual_step],
            expected_circuit=expected,
            mode_label="手动补扫",
            merged_source="手动补扫合并结果",
            analysis_settings=self.current_session.analysis_settings,
            adaptive=False,
            max_steps=1,
        )
        self.reader.start()
        self.connection_state_var.set(f"{port} @ {baudrate}")
        self.status_var.set("手动补扫中")
        self.log(f"开始手动补扫: {command.strip()}，将与当前 session #{self.current_session.id} 合并分析。")

    def run_ai_diagnosis_for_current(self):
        if self.current_session is None or self.current_session.result is None:
            messagebox.showinfo("提示", "当前没有可诊断的测量数据。")
            return
        session = self.current_session
        self.log(f"刷新 session #{session.id} 智能诊断。")
        threading.Thread(target=self._diagnosis_worker, args=(session,), daemon=True).start()

    def _diagnosis_worker(self, session: MeasurementSession):
        try:
            control_settings = session.analysis_settings.get("control_compensation_settings")
            diagnosis = run_intelligent_diagnosis(
                session.omega,
                session.magnitude,
                session.phase,
                diagnostics=session.diagnostics,
                analysis_result=session.result,
                control_compensation_settings=control_settings,
            )
            session.ai_diagnosis = diagnosis
            session.control_compensation_report = getattr(diagnosis, "control_compensation_report", None)
            session.report_lines = self._build_session_report_lines(session, session.result, diagnosis)
            self.queue.put(("diagnosis_done", session.id))
        except Exception as exc:
            self.queue.put(("analysis_error", (session.id, str(exc))))

    def import_text(self):
        path = filedialog.askopenfilename(
            title="选择包含数组的文本文件",
            filetypes=[("Text", "*.txt *.log *.m *.csv"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            omega, mag, phase = parse_arrays_from_text(text)
            diagnostics = parse_diagnostics_from_text(text)
            self.handle_frame(omega, mag, phase, source=f"文件导入: {Path(path).name}", diagnostics=diagnostics)
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def handle_frame(self, omega, mag, phase, source="串口", diagnostics=None):
        settings = self._analysis_settings()
        if settings is None:
            return
        omega = np.asarray(omega, dtype=float).flatten()
        mag = np.asarray(mag, dtype=float).flatten()
        phase = np.asarray(phase, dtype=float).flatten()
        session = MeasurementSession(
            id=self._next_session_id,
            created_at=datetime.now(),
            source=source,
            omega=omega,
            magnitude=mag,
            phase=phase,
            raw_omega=omega,
            raw_magnitude=mag,
            raw_phase=phase,
            diagnostics=diagnostics or {},
            expected_circuit=self.expected_circuit_var.get().strip() or "Auto",
            serial_port=self.port_var.get().strip(),
            baudrate=int(self.baud_var.get().strip()) if self.baud_var.get().strip().isdigit() else None,
            sweep_command=self._build_sweep_command(show_errors=False) or "",
            analysis_settings=settings,
            status="pending",
        )
        self._next_session_id += 1
        self.sessions.append(session)
        self._insert_history_session(session)
        self._select_session(session)
        self.status_var.set(f"{source} 已创建 session，后台分析中")
        self.log(f"{source} 接收成功：session #{session.id}，点数 {session.point_count}，已投递分析线程。")
        threading.Thread(target=self._analysis_worker, args=(session,), daemon=True).start()

    def _analysis_worker(self, session: MeasurementSession):
        try:
            settings = dict(session.analysis_settings)
            control_settings = settings.pop("control_compensation_settings", None)
            result = analyze_system_v2(
                session.omega,
                session.magnitude,
                session.phase,
                diagnostics=session.diagnostics,
                **settings,
            )
            expected = session.expected_circuit or "Auto"
            result.expected_circuit = expected
            result.expected_match_text = evaluate_expected_circuit(result, expected)
            result.notes.append(result.expected_match_text)
            diagnosis = run_intelligent_diagnosis(
                session.omega,
                session.magnitude,
                session.phase,
                diagnostics=session.diagnostics,
                analysis_result=result,
                control_compensation_settings=control_settings,
            )
            session.result = result
            session.ai_diagnosis = diagnosis
            session.control_compensation_report = getattr(diagnosis, "control_compensation_report", None)
            session.report_lines = self._build_session_report_lines(session, result, diagnosis)
            session.status = "done"
            self.queue.put(("analysis_done", session.id))
        except Exception as exc:
            session.error = str(exc)
            session.status = "error"
            self.queue.put(("analysis_done", session.id))

    def _build_session_report_lines(self, session: MeasurementSession, result, diagnosis) -> list[str]:
        return build_filter_report_lines(result) + [""] + format_ai_diagnosis_report_lines(diagnosis)

    def _insert_history_session(self, session: MeasurementSession):
        iid = str(session.id)
        self.session_by_iid[iid] = session
        self.history_tree.insert("", 0, iid=iid, values=self._history_values(session))

    def _history_values(self, session: MeasurementSession):
        return (
            session.created_at.strftime("%H:%M:%S"),
            session.source,
            session.point_count,
            session.decision_label,
            session.confidence_text,
        )

    def _refresh_history_session(self, session: MeasurementSession):
        iid = str(session.id)
        if iid in self.session_by_iid:
            self.history_tree.item(iid, values=self._history_values(session))

    def _select_session(self, session: MeasurementSession):
        iid = str(session.id)
        self.history_tree.selection_set(iid)
        self.history_tree.focus(iid)
        self.history_tree.see(iid)
        self._set_current_session(session)

    def _on_history_select(self, _event=None):
        selection = list(self.history_tree.selection())
        if not selection:
            return
        focus = self.history_tree.focus()
        iid = focus if focus in selection else selection[0]
        session = self.session_by_iid.get(iid)
        if session is not None:
            self._set_current_session(session)
        self._refresh_plot_view()

    def _selected_sessions(self) -> list[MeasurementSession]:
        selected = [self.session_by_iid[iid] for iid in self.history_tree.selection() if iid in self.session_by_iid]
        return selected or ([self.current_session] if self.current_session is not None else [])

    def _set_current_session(self, session: MeasurementSession):
        self.current_session = session
        self.result = session.result
        self.current_measurement = session.as_legacy_tuple()
        self.ai_diagnosis = session.ai_diagnosis
        self.system_type_var.set(session.decision_label)
        self._set_remark_var(session.remark)
        self._update_diagnosis_panel(session)
        self._update_report_view(session)
        if session.result is not None:
            self.status_var.set(f"当前 session #{session.id}: {session.decision_label}，点数 {session.point_count}")
        elif session.error:
            self.status_var.set(f"当前 session #{session.id}: 无效测量")
        else:
            self.status_var.set(f"当前 session #{session.id}: 分析中")
        self._refresh_plot_view()

    def _set_remark_var(self, value: str):
        self._suspend_remark_trace = True
        try:
            self.remark_var.set(value or "")
        finally:
            self._suspend_remark_trace = False

    def _on_remark_change(self, *_):
        if getattr(self, "_suspend_remark_trace", False):
            return
        if self._note_update_job is not None:
            self.root.after_cancel(self._note_update_job)
        self._note_update_job = self.root.after(250, self._commit_remark)

    def _commit_remark(self):
        self._note_update_job = None
        if self.current_session is None:
            return
        self.current_session.remark = self.remark_var.get()
        self._refresh_history_session(self.current_session)

    def _update_diagnosis_panel(self, session: MeasurementSession | None):
        sections = structured_diagnosis_sections(session)
        for key, var in self.diagnosis_vars.items():
            var.set(sections.get(key, "暂无"))
        canvas = getattr(self, "diagnosis_summary_canvas", None)
        if canvas is not None:
            self.root.after_idle(lambda: canvas.configure(scrollregion=canvas.bbox("all")))

    def _update_report_view(self, session: MeasurementSession | None):
        self.report_text.delete("1.0", "end")
        if session is None:
            return
        if session.error:
            self.report_text.insert("end", session.error)
            return
        if session.report_lines:
            self.report_text.insert("end", "\n".join(session.report_lines))
        elif session.result is None:
            self.report_text.insert("end", "分析中...")

    def _refresh_plot_view(self):
        plot_sessions(self.fig, (self.ax_mag, self.ax_phase, self.ax_ny), self._selected_sessions(), self.current_session)

    def _find_session_by_id(self, session_id: int) -> MeasurementSession | None:
        for session in self.sessions:
            if session.id == session_id:
                return session
        return None

    def _handle_session_completed(self, session_id: int):
        session = self._find_session_by_id(session_id)
        if session is None:
            return
        self._refresh_history_session(session)
        if session.error:
            self.log(f"session #{session.id} 分析失败：{session.error}")
            if self.current_session is session:
                self._set_current_session(session)
            messagebox.showwarning("测量帧无效", session.error)
            return
        self.log(
            f"session #{session.id} 分析完成：{session.result.system_order}，"
            f"confidence={session.ai_diagnosis.confidence:.2f}"
        )
        if self.current_session is session:
            self._set_current_session(session)

    def clear_report_view(self):
        self.report_text.delete("1.0", "end")

    def _copy_text_to_clipboard(self, text: str, log_message: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.log(log_message)

    def copy_diagnosis(self):
        text = diagnosis_clipboard_text(self.current_session)
        self._copy_text_to_clipboard(text, "当前诊断已复制到剪贴板。")

    def export_report(self):
        if self.current_session is None or self.current_session.result is None:
            messagebox.showinfo("提示", "当前没有可导出的分析结果。")
            return
        default_name = f"report_session_{self.current_session.id}_{self.current_session.created_at.strftime('%Y%m%d_%H%M%S')}"
        out_dir = filedialog.askdirectory(title="选择报告导出目录")
        if not out_dir:
            return
        target_dir = Path(out_dir) / default_name
        try:
            path = export_html_report(self.current_session, self.fig, target_dir)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.log(f"报告已导出：{path}")
        messagebox.showinfo("导出完成", f"报告已导出到：\n{path}")

    def save_csv(self):
        if self.result is None:
            messagebox.showinfo("提示", "当前没有数据可保存。")
            return
        path = filedialog.asksaveasfilename(title="保存 CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        r = self.result
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "omega",
                    "Magnitude_raw",
                    "Magnitude_smooth",
                    "Phase_raw_rad",
                    "Phase_smooth_rad",
                    "Magnitude_dB",
                    "Phase_deg",
                    "Real_part",
                    "Imag_part",
                ]
            )
            for row in zip(
                r.omega,
                r.mag_raw,
                r.mag_smooth,
                r.phase_raw_rad,
                r.phase_smooth_rad,
                r.mag_db,
                r.phase_deg,
                r.real_part,
                r.imag_part,
            ):
                writer.writerow(row)
        self.log(f"已保存 CSV：{path}")

    def save_png(self):
        if self.result is None:
            messagebox.showinfo("提示", "当前没有图像可保存。")
            return
        path = filedialog.asksaveasfilename(title="保存图像", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        self.fig.savefig(path, dpi=180, bbox_inches="tight")
        self.log(f"已保存图像：{path}")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "connection_status":
                    self.connection_state_var.set(str(payload))
                elif kind == "error":
                    self.status_var.set("发生错误")
                    self.log(f"错误: {payload}")
                    messagebox.showerror("错误", payload)
                elif kind == "frame":
                    if len(payload) == 4:
                        omega, mag, phase, diagnostics = payload
                    else:
                        omega, mag, phase = payload
                        diagnostics = None
                    self.handle_frame(omega, mag, phase, diagnostics=diagnostics)
                elif kind == "auto_frame":
                    omega, mag, phase, diagnostics, source, expected = payload
                    self.expected_circuit_var.set(expected)
                    self.handle_frame(omega, mag, phase, source=source, diagnostics=diagnostics)
                elif kind in ("analysis_done", "diagnosis_done"):
                    self._handle_session_completed(int(payload))
                elif kind == "analysis_error":
                    session_id, message = payload
                    session = self._find_session_by_id(int(session_id))
                    if session is not None:
                        session.error = message
                        session.status = "error"
                    self._handle_session_completed(int(session_id))
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


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    app = MatlabExactApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
