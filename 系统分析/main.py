import csv
import queue
import threading
import time
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')

from matplotlib import font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from serial_protocol import (
    build_sweep_command,
    parse_diagnostics_from_text,
    parse_measurement_frame_from_text,
)
from serial_readers import (
    AutoSweepReader,
    ThreeLineReader,
    list_serial_ports,
    read_measurement_frame_from_serial,
)
from smart_resweep_reader import SmartResweepReader
from ai_diagnosis import format_ai_diagnosis_report_lines, run_intelligent_diagnosis


APP_TITLE = '稳频仪 - STM32G431 通用电路频响与奈奎斯特稳定性分析平台'

from filter_analysis import *  # Re-export analysis API for existing scripts.


def configure_plot_fonts():
    candidates = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
        'PingFang SC', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), 'DejaVu Sans')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return chosen


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
        self.current_measurement = None
        self.ai_diagnosis = None
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

        self.text.insert('end', '稳频仪已就绪\n支持自动扫频、伯德图、滤波器类型/阶次/截止频率识别；Nyquist 图保留用于高级查看。\n\n')

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

    def _build_sweep_command(self, show_errors=True):
        try:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            amp = float(self.amp_var.get().strip())
        except ValueError:
            if show_errors:
                messagebox.showwarning('提示', '扫频参数格式不正确。')
            return None

        if f_start <= 0 or f_stop < f_start or f_step <= 0:
            if show_errors:
                messagebox.showwarning('提示', '频率范围不正确。')
            return None
        if amp <= 0 or amp > 3.0:
            if show_errors:
                messagebox.showwarning('提示', '幅度建议设置在 0 到 3.0 Vpp 之间。')
            return None

        return build_sweep_command(f_start, f_stop, f_step, amp)

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

        fallback_command = command if command is not None else self._build_sweep_command(show_errors=False)

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

    def run_ai_diagnosis_for_current(self):
        if self.current_measurement is None or self.result is None:
            messagebox.showinfo('提示', '当前没有可诊断的测量数据。')
            return
        omega, mag, phase, diagnostics = self.current_measurement
        diagnosis = self._append_ai_diagnosis(omega, mag, phase, diagnostics)
        self.log(f'智能诊断完成: {diagnosis.system_label}, confidence={diagnosis.confidence:.2f}')

    def command_smart_resweep(self):
        if self.reader is not None and self.reader.is_alive():
            messagebox.showinfo('提示', '读取线程正在运行，请先停止或等待完成。')
            return
        if self.current_measurement is None or self.ai_diagnosis is None:
            messagebox.showinfo('提示', '请先完成一次测量和智能诊断。')
            return
        sweep_plan = getattr(self.ai_diagnosis, 'active_sweep_plan', None)
        if not sweep_plan:
            messagebox.showinfo('提示', '当前诊断没有生成智能补扫计划。')
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning('提示', '请先选择串口。')
            return
        try:
            baudrate = int(self.baud_var.get().strip())
            timeout_sec = float(self.timeout_var.get().strip())
        except ValueError:
            messagebox.showwarning('提示', '串口参数格式不正确。')
            return

        expected = getattr(self, 'expected_circuit_var', tk.StringVar(value='Auto')).get().strip() or 'Auto'
        self.stop_event.clear()
        self.reader = SmartResweepReader(
            port=port,
            baudrate=baudrate,
            timeout_sec=timeout_sec,
            out_queue=self.queue,
            stop_event=self.stop_event,
            base_frame=self.current_measurement,
            sweep_steps=sweep_plan,
            expected_circuit=expected,
        )
        self.reader.start()
        self.status_var.set('智能补扫中')
        self.log(f'开始智能补扫，共 {len(sweep_plan)} 条 SWEEP 命令。')

    def _append_ai_diagnosis(self, omega, mag, phase, diagnostics=None):
        diagnosis = run_intelligent_diagnosis(
            omega,
            mag,
            phase,
            diagnostics=diagnostics,
            analysis_result=self.result,
        )
        self.ai_diagnosis = diagnosis
        lines = format_ai_diagnosis_report_lines(diagnosis)
        self.text.insert('end', '\n'.join(lines))
        self.text.see('end')
        return diagnosis

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

        try:
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
        except ValueError as exc:
            self.result = None
            self.system_type_var.set('本帧无效')
            self.status_var.set(f'{source} 无法判阶')
            message = str(exc)
            self.log(f'{source} 无法判阶：{message}')
            messagebox.showwarning('测量帧无效', message)
            return
        expected = getattr(self, 'expected_circuit_var', tk.StringVar(value='Auto')).get().strip() or 'Auto'
        result.expected_circuit = expected
        result.expected_match_text = evaluate_expected_circuit(result, expected)
        result.notes.append(result.expected_match_text)
        self.result = result
        self.current_measurement = (omega, mag, phase, diagnostics)
        self.system_type_var.set(result.system_order)
        self.status_var.set(f'{source} 解析完成：{result.system_order}，点数 {len(result.omega)}')
        self.log(f'{source} 接收成功：系统类型 {result.system_order}，ωc={result.omega_c:.2f} rad/s，点数 {len(result.omega)}')

        for note in result.notes:
            self.log(note)

        self.append_report(result)
        self._append_ai_diagnosis(omega, mag, phase, diagnostics)
        self.update_plots(result)

    def append_report(self, r: SystemAnalysisResult):
        lines = build_filter_report_lines(r)
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


def parse_arrays_from_text(text: str):
    frame = parse_measurement_frame_from_text(text)
    return frame.omega, frame.magnitude, frame.phase


def matlab_app_append_report_v2(self, r: SystemAnalysisResult):
    lines = build_filter_report_lines(r)
    self.text.insert('end', '\n'.join(lines))
    self.text.see('end')


def matlab_app_build_ui_v2(self):
    bg = '#ffffff'
    surface = '#ffffff'
    border = '#e5e7eb'
    ink = '#111827'
    muted = '#6b7280'
    soft = '#f3f4f6'
    action = '#111827'
    action_hover = '#374151'

    self.root.configure(bg=bg)
    style = ttk.Style(self.root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    ui_font = self.font_name if self.font_name else 'Microsoft YaHei'
    try:
        style.configure('.', font=(ui_font, 9))
        style.configure('App.TFrame', background=bg)
        style.configure('Surface.TFrame', background=surface)
        style.configure(
            'Surface.TLabelframe',
            background=surface,
            bordercolor=border,
            relief='solid',
        )
        style.configure(
            'Surface.TLabelframe.Label',
            background=surface,
            foreground=ink,
            font=(ui_font, 10, 'bold'),
        )
        style.configure('Header.TLabel', font=(ui_font, 16, 'bold'), foreground=ink, background=bg)
        style.configure('Subtle.TLabel', font=(ui_font, 9), foreground=muted, background=bg)
        style.configure('Surface.TLabel', foreground=ink, background=surface)
        style.configure('Muted.Surface.TLabel', foreground=muted, background=surface)
        style.configure('Value.TLabel', font=(ui_font, 10, 'bold'), foreground=ink, background=surface)
        style.configure('Badge.TLabel', font=(ui_font, 9, 'bold'), foreground=ink, background=soft)
        style.configure('Primary.TButton', font=(ui_font, 10, 'bold'), foreground='#ffffff', background=action)
        style.map('Primary.TButton', background=[('active', action_hover), ('disabled', '#9ca3af')])
        style.configure('Cta.TButton', font=(ui_font, 10, 'bold'), foreground='#ffffff', background=action)
        style.map('Cta.TButton', background=[('active', action_hover), ('disabled', '#9ca3af')])
        style.configure('Secondary.TButton', font=(ui_font, 9), foreground=ink, background=surface)
        style.map('Secondary.TButton', background=[('active', soft)])
        style.configure('Danger.TButton', font=(ui_font, 9, 'bold'), foreground=ink, background=soft)
        style.map('Danger.TButton', background=[('active', '#e5e7eb')])
        style.configure('Tool.TButton', font=(ui_font, 9), foreground=ink, background=surface)
        style.map('Tool.TButton', background=[('active', soft)])
        style.configure('TEntry', fieldbackground='#ffffff')
    except Exception:
        pass

    def entry(parent, row, col, label, variable, width=10, padx=(8, 12)):
        ttk.Label(parent, text=label, style='Surface.TLabel').grid(row=row, column=col, sticky='w')
        widget = ttk.Entry(parent, textvariable=variable, width=width)
        widget.grid(row=row, column=col + 1, sticky='ew', padx=padx, pady=(0 if row == 0 else 8, 0))
        return widget

    main = ttk.Frame(self.root, padding=(14, 12), style='App.TFrame')
    main.pack(fill='both', expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(3, weight=1)

    header = ttk.Frame(main, style='App.TFrame')
    header.grid(row=0, column=0, sticky='ew')
    header.columnconfigure(2, weight=1)
    ttk.Label(header, text='STM32G431 智能频响分析仪', style='Header.TLabel').grid(row=0, column=0, sticky='w')
    ttk.Label(header, text='AI辅助诊断 v2', style='Badge.TLabel', padding=(10, 4)).grid(row=0, column=1, sticky='w', padx=(14, 0))
    ttk.Label(header, text='基于频响特征、传递函数模板拟合与故障知识库', style='Subtle.TLabel').grid(
        row=1, column=0, columnspan=2, sticky='w', pady=(3, 0)
    )
    self.system_type_var = tk.StringVar(value='待分析')
    self.status_var = tk.StringVar(value='就绪')
    status_panel = ttk.Frame(header, style='App.TFrame')
    status_panel.grid(row=0, column=2, rowspan=2, sticky='e')
    ttk.Label(status_panel, text='识别结果', style='Subtle.TLabel').grid(row=0, column=0, sticky='e')
    ttk.Label(status_panel, textvariable=self.system_type_var, style='Header.TLabel').grid(row=0, column=1, sticky='e', padx=(10, 0))
    ttk.Label(status_panel, text='状态', style='Subtle.TLabel').grid(row=1, column=0, sticky='e', pady=(4, 0))
    ttk.Label(status_panel, textvariable=self.status_var, foreground=ink, background=bg).grid(
        row=1, column=1, sticky='e', padx=(10, 0), pady=(4, 0)
    )

    control = ttk.Frame(main, style='App.TFrame')
    control.grid(row=1, column=0, sticky='ew', pady=(8, 10))
    control.columnconfigure(0, weight=1)
    control.columnconfigure(1, weight=2)
    control.columnconfigure(2, weight=1)

    conn = ttk.LabelFrame(control, text='连接', padding=(12, 10), style='Surface.TLabelframe')
    sweep = ttk.LabelFrame(control, text='扫频测量', padding=(12, 10), style='Surface.TLabelframe')
    ai_panel = ttk.LabelFrame(control, text='智能诊断', padding=(12, 10), style='Surface.TLabelframe')
    conn.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
    sweep.grid(row=0, column=1, sticky='nsew', padx=4)
    ai_panel.grid(row=0, column=2, sticky='nsew', padx=(8, 0))

    self.port_var = tk.StringVar()
    self.port_box = ttk.Combobox(conn, textvariable=self.port_var, width=14, state='readonly')
    ttk.Label(conn, text='串口', style='Surface.TLabel').grid(row=0, column=0, sticky='w')
    self.port_box.grid(row=0, column=1, sticky='ew', padx=(6, 4))
    ttk.Button(conn, text='刷新', command=self.refresh_ports, style='Tool.TButton').grid(row=0, column=2, sticky='ew')

    self.baud_var = tk.StringVar(value='115200')
    self.timeout_var = tk.StringVar(value='60')
    ttk.Label(conn, text='波特率', style='Surface.TLabel').grid(row=1, column=0, sticky='w', pady=(8, 0))
    ttk.Entry(conn, textvariable=self.baud_var, width=10).grid(row=1, column=1, sticky='ew', padx=(6, 4), pady=(8, 0))
    ttk.Label(conn, text='超时', style='Surface.TLabel').grid(row=1, column=2, sticky='w', pady=(8, 0))
    ttk.Entry(conn, textvariable=self.timeout_var, width=6).grid(row=1, column=3, sticky='ew', padx=(4, 0), pady=(8, 0))
    ttk.Button(conn, text='停止当前任务', command=self.stop_reading, style='Danger.TButton').grid(
        row=2, column=0, columnspan=4, sticky='ew', pady=(10, 0)
    )
    conn.columnconfigure(1, weight=1)

    self.f_start_var = tk.StringVar(value='100')
    self.f_stop_var = tk.StringVar(value='20000')
    self.f_step_var = tk.StringVar(value='100')
    self.amp_var = tk.StringVar(value='1.2')
    entry(sweep, 0, 0, '起始 Hz', self.f_start_var)
    entry(sweep, 0, 2, '终止 Hz', self.f_stop_var)
    entry(sweep, 1, 0, '步进 Hz', self.f_step_var)
    entry(sweep, 1, 2, '幅度 Vpp', self.amp_var, width=8)
    ttk.Button(sweep, text='开始扫频分析', command=self.command_sweep_once, style='Cta.TButton').grid(
        row=2, column=0, columnspan=2, sticky='ew', pady=(10, 0), ipady=3
    )
    ttk.Button(sweep, text='八类自动识别', command=self.command_auto_identify, style='Secondary.TButton').grid(
        row=2, column=2, columnspan=2, sticky='ew', pady=(10, 0), ipady=3
    )
    sweep.columnconfigure(1, weight=1)
    sweep.columnconfigure(3, weight=1)

    self.smooth_var = tk.BooleanVar(value=True)
    self.fix_var = tk.BooleanVar(value=False)
    self.swap_io_var = tk.BooleanVar(value=False)
    self.window_var = tk.StringVar(value='11')
    self.poly_var = tk.StringVar(value='3')
    self.open_loop_p_var = tk.StringVar(value='0')
    self.expected_circuit_var = tk.StringVar(value='Auto')

    ttk.Label(ai_panel, text='期望电路', style='Surface.TLabel').grid(row=0, column=0, sticky='w')
    self.expected_circuit_box = ttk.Combobox(
        ai_panel,
        textvariable=self.expected_circuit_var,
        values=EXPECTED_CIRCUIT_CHOICES,
        width=12,
        state='readonly',
    )
    self.expected_circuit_box.grid(row=0, column=1, columnspan=2, sticky='ew', padx=(8, 0))
    ttk.Button(ai_panel, text='智能诊断', command=self.run_ai_diagnosis_for_current, style='Primary.TButton').grid(
        row=1, column=0, columnspan=3, sticky='ew', pady=(10, 0), ipady=3
    )
    ttk.Button(ai_panel, text='智能补扫', command=self.command_smart_resweep, style='Secondary.TButton').grid(
        row=2, column=0, columnspan=3, sticky='ew', pady=(8, 0), ipady=3
    )
    utility = ttk.Frame(ai_panel, style='Surface.TFrame')
    utility.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(10, 0))
    utility.columnconfigure((0, 1, 2), weight=1)
    ttk.Button(utility, text='导入', command=self.import_text, style='Tool.TButton').grid(row=0, column=0, sticky='ew')
    ttk.Button(utility, text='CSV', command=self.save_csv, style='Tool.TButton').grid(row=0, column=1, sticky='ew', padx=4)
    ttk.Button(utility, text='图像', command=self.save_png, style='Tool.TButton').grid(row=0, column=2, sticky='ew')
    ai_panel.columnconfigure(1, weight=1)

    advanced = ttk.LabelFrame(main, text='高级分析设置', padding=(12, 8), style='Surface.TLabelframe')
    advanced.grid(row=2, column=0, sticky='ew', pady=(0, 10))
    for idx in range(9):
        advanced.columnconfigure(idx, weight=1 if idx in (1, 4, 7) else 0)

    ttk.Checkbutton(advanced, text='平滑', variable=self.smooth_var).grid(row=0, column=0, sticky='w')
    ttk.Checkbutton(advanced, text='首点修正', variable=self.fix_var).grid(row=0, column=1, sticky='w', padx=(10, 0))
    ttk.Checkbutton(advanced, text='交换方向', variable=self.swap_io_var).grid(row=0, column=2, sticky='w', padx=(10, 0))
    ttk.Label(advanced, text='窗口', style='Surface.TLabel').grid(row=0, column=3, sticky='e', padx=(14, 4))
    ttk.Entry(advanced, textvariable=self.window_var, width=5).grid(row=0, column=4, sticky='w')
    ttk.Label(advanced, text='阶数', style='Surface.TLabel').grid(row=0, column=5, sticky='e', padx=(10, 4))
    ttk.Entry(advanced, textvariable=self.poly_var, width=5).grid(row=0, column=6, sticky='w')
    ttk.Label(advanced, text='RHP P', style='Surface.TLabel').grid(row=0, column=7, sticky='e', padx=(10, 4))
    ttk.Entry(advanced, textvariable=self.open_loop_p_var, width=5).grid(row=0, column=8, sticky='w')

    body = ttk.PanedWindow(main, orient='horizontal')
    body.grid(row=3, column=0, sticky='nsew')

    left = ttk.Frame(body, padding=(0, 0, 8, 0), style='App.TFrame')
    right = ttk.Frame(body, padding=(8, 0, 0, 0), style='App.TFrame')
    body.add(left, weight=5)
    body.add(right, weight=2)

    self.fig = Figure(figsize=(11.4, 7.4), dpi=100, facecolor='#f8fafc', constrained_layout=True)
    grid = self.fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    self.ax_mag = self.fig.add_subplot(grid[0, 0])
    self.ax_phase = self.fig.add_subplot(grid[1, 0], sharex=self.ax_mag)
    self.ax_ny = self.fig.add_subplot(grid[:, 1])
    self.ax_stab = None

    self.canvas = FigureCanvasTkAgg(self.fig, master=left)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill='both', expand=True)

    tb = NavigationToolbar2Tk(self.canvas, left, pack_toolbar=False)
    tb.update()
    tb.pack(fill='x')

    logbox = ttk.LabelFrame(right, text='诊断报告', padding=8, style='Surface.TLabelframe')
    logbox.pack(fill='both', expand=True)

    self.text = tk.Text(
        logbox,
        wrap='word',
        font=('Consolas', 10),
        bg='#ffffff',
        fg=ink,
        insertbackground=ink,
        selectbackground='#e5e7eb',
        relief='flat',
        padx=12,
        pady=10,
        spacing1=2,
        spacing3=4,
    )
    self.text.pack(side='left', fill='both', expand=True)

    ys = ttk.Scrollbar(logbox, orient='vertical', command=self.text.yview)
    ys.pack(side='right', fill='y')
    self.text.configure(yscrollcommand=ys.set)

    self.text.insert('end', '智能频响分析仪已就绪\n主流程：选择串口 -> 开始扫频分析 -> 查看智能诊断 -> 按需智能补扫。\n旧的单帧/连续读取入口已从主界面移除，串口协议与底层调试函数保持兼容。\n\n')


def matlab_app_update_plots_v2(self, r: SystemAnalysisResult):
    for ax in (self.ax_mag, self.ax_phase, self.ax_ny):
        ax.clear()
        ax.set_facecolor('#ffffff')
        ax.grid(True, which='major', color='#cbd5e1', alpha=0.65, linewidth=0.8)
        ax.grid(True, which='minor', color='#e2e8f0', alpha=0.45, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')

    marker_omegas: list[tuple[float, str, str]] = []
    if r.filter_type in ('lowpass', 'highpass'):
        marker_omegas.append((float(r.magnitude_cutoff_omega), 'ωc', '#dc2626'))
    elif r.filter_type == 'bandpass':
        marker_omegas.append((float(r.magnitude_cutoff_omega), 'ωL', '#f97316'))
        if r.secondary_cutoff_omega is not None:
            marker_omegas.append((float(r.secondary_cutoff_omega), 'ωH', '#f97316'))
        marker_omegas.append((float(r.omega_c), 'ω0', '#dc2626'))
    elif r.filter_type == 'bandstop':
        marker_omegas.append((float(r.magnitude_cutoff_omega), 'ωL', '#f97316'))
        if r.secondary_cutoff_omega is not None:
            marker_omegas.append((float(r.secondary_cutoff_omega), 'ωH', '#f97316'))
        marker_omegas.append((float(r.omega_c), 'ω0', '#dc2626'))
    else:
        marker_omegas.append((float(r.omega_c), 'ω*', '#dc2626'))

    raw_mag_db = 20.0 * np.log10(np.clip(r.mag_raw, 1e-12, None))
    phase_display = phase_array_for_display(r.phase_deg, r.filter_type)
    cutoff_phase_display = normalize_phase_for_display(r.cutoff_phase_deg, r.filter_type)

    self.ax_mag.semilogx(r.omega, raw_mag_db, '.', color='#94a3b8', ms=3.5, alpha=0.55, label='原始点')
    self.ax_mag.semilogx(r.omega, r.mag_db, color='#2563eb', lw=2.1, label='幅值曲线')
    self.ax_mag.axhline(r.cutoff_mag_db, linestyle=':', color='#334155', lw=1.0)
    for omega_value, label, color in marker_omegas:
        self.ax_mag.axvline(omega_value, linestyle='--', color=color, lw=1.1, alpha=0.9)
        self.ax_mag.annotate(
            label,
            xy=(omega_value, 1.0),
            xycoords=('data', 'axes fraction'),
            xytext=(3, -16),
            textcoords='offset points',
            color=color,
            fontsize=9,
            ha='left',
            va='top',
        )
    self.ax_mag.set_title(f'幅频响应  {r.system_order}', loc='left', fontsize=11, color='#0f172a')
    self.ax_mag.set_ylabel('幅值 (dB)')
    mag_legend_loc = 'lower right' if r.filter_type == 'highpass' else 'lower left'
    self.ax_mag.legend(loc=mag_legend_loc, fontsize=8, frameon=False)

    self.ax_phase.semilogx(r.omega, phase_display, color='#e11d48', lw=2.0)
    self.ax_phase.axhline(cutoff_phase_display, linestyle=':', color='#334155', lw=1.0)
    for omega_value, _, color in marker_omegas:
        self.ax_phase.axvline(omega_value, linestyle='--', color=color, lw=1.1, alpha=0.9)
    self.ax_phase.set_title('相频响应', loc='left', fontsize=11, color='#0f172a')
    self.ax_phase.set_xlabel('角频率 ω (rad/s)')
    self.ax_phase.set_ylabel('相位 (°)')

    h_positive = r.real_part + 1j * r.imag_part
    h_full = build_closed_nyquist_curve(h_positive)
    self.ax_ny.plot(h_full.real, h_full.imag, color='#cbd5e1', linestyle='--', lw=1.0, label='镜像')
    self.ax_ny.plot(r.real_part, r.imag_part, color='#2563eb', lw=2.0, label='H(jω)')
    self.ax_ny.scatter(r.real_part, r.imag_part, s=14, color='#ef4444', alpha=0.62, zorder=3)
    self.ax_ny.plot(r.real_part[0], r.imag_part[0], 'o', color='#16a34a', ms=6, label='低频')
    self.ax_ny.plot(r.real_part[-1], r.imag_part[-1], 'x', color='#dc2626', ms=7, label='高频')
    self.ax_ny.plot(r.real_part[r.cutoff_index], r.imag_part[r.cutoff_index], '*', color='#111827', ms=11, label='特征点')
    self.ax_ny.plot(-1, 0, 's', color='#475569', ms=6)
    self.ax_ny.axhline(0, color='#94a3b8', linestyle=':', lw=0.8)
    self.ax_ny.axvline(-1, color='#94a3b8', linestyle=':', lw=0.8)

    ny_x = np.concatenate([np.asarray(h_full.real, dtype=float), np.asarray([-1.0])])
    ny_y = np.concatenate([np.asarray(h_full.imag, dtype=float), np.asarray([0.0])])
    x_mid = float((np.max(ny_x) + np.min(ny_x)) / 2.0)
    y_mid = float((np.max(ny_y) + np.min(ny_y)) / 2.0)
    span = max(float(np.max(ny_x) - np.min(ny_x)), float(np.max(ny_y) - np.min(ny_y)), 0.25) * 0.62
    self.ax_ny.set_xlim(x_mid - span, x_mid + span)
    self.ax_ny.set_ylim(y_mid - span, y_mid + span)
    self.ax_ny.set_aspect('equal', adjustable='box')
    self.ax_ny.set_title('Nyquist 图', loc='left', fontsize=11, color='#0f172a')
    self.ax_ny.set_xlabel('实部 Re{H(jω)}')
    self.ax_ny.set_ylabel('虚部 Im{H(jω)}')
    self.ax_ny.legend(loc='upper right', fontsize=8, frameon=False)

    self.canvas.draw_idle()


MatlabExactApp._build_ui = matlab_app_build_ui_v2
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
