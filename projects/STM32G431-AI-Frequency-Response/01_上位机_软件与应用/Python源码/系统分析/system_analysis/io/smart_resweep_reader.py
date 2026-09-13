from __future__ import annotations

import threading

try:
    import serial
except Exception:
    serial = None

TARGET_CONFIDENCE = 0.90
MIN_CONFIDENCE_IMPROVEMENT = 0.015

from system_analysis.analysis.ai_diagnosis import run_intelligent_diagnosis
from system_analysis.analysis.filter_analysis import analyze_system_v2, merge_measurement_frames
from system_analysis.io.serial_readers import read_measurement_frame_from_serial
from system_analysis.io.serial_transport import open_serial_transport, write_ascii_command


class SmartResweepReader(threading.Thread):
    def __init__(
        self,
        port,
        baudrate,
        timeout_sec,
        out_queue,
        stop_event,
        base_frame,
        sweep_steps,
        expected_circuit="Auto",
        mode_label="智能补扫",
        merged_source="智能补扫合并结果",
        analysis_settings=None,
        adaptive=True,
        max_steps=3,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout_sec = timeout_sec
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.base_frame = base_frame
        self.sweep_steps = list(sweep_steps or [])[:3]
        self.expected_circuit = expected_circuit or "Auto"
        self.mode_label = mode_label or "智能补扫"
        self.merged_source = merged_source or f"{self.mode_label}合并结果"
        self.analysis_settings = dict(analysis_settings or {})
        self.adaptive = bool(adaptive)
        self.max_steps = max(1, min(int(max_steps or 1), 8))
        self._ser = None
        self._ser_lock = threading.Lock()

    def send_stop_command(self):
        with self._ser_lock:
            if self._ser is None:
                return False
            write_ascii_command(self._ser, "STOP\n")
            return True

    def _line_callback(self, line: str):
        if line.startswith("PROGRESS") or line.startswith("STOPPED"):
            self.out_queue.put(("log", line))

    def _read_frame(self, ser):
        try:
            return read_measurement_frame_from_serial(
                ser,
                self.stop_event,
                self.timeout_sec,
                line_callback=self._line_callback,
            )
        except TypeError as exc:
            if "line_callback" not in str(exc):
                raise
            return read_measurement_frame_from_serial(ser, self.stop_event, self.timeout_sec)

    @staticmethod
    def _step_command_and_reason(step):
        if isinstance(step, str):
            return step.strip(), "诊断/手动补扫"
        command = getattr(step, "command", None)
        reason = getattr(step, "reason", None)
        if command is None and isinstance(step, dict):
            command = step.get("command")
            reason = step.get("reason")
        return str(command or "").strip(), str(reason or "诊断/手动补扫")

    @staticmethod
    def _command_frequency_span(command: str):
        parts = str(command or "").strip().split()
        if len(parts) < 4 or parts[0].upper() != "SWEEP":
            return None
        try:
            start = float(parts[1])
            stop = float(parts[2])
        except ValueError:
            return None
        if not (start > 0.0 and stop >= start):
            return None
        return start, stop

    @classmethod
    def _is_covered_by_sent(cls, command: str, sent_spans: list[tuple[float, float]]) -> bool:
        span = cls._command_frequency_span(command)
        if span is None:
            return False
        start, stop = span
        width = max(stop - start, 1.0e-9)
        for sent_start, sent_stop in sent_spans:
            overlap = max(0.0, min(stop, sent_stop) - max(start, sent_start))
            if overlap / width >= 0.80:
                return True
        return False

    @classmethod
    def _next_unsent_step(cls, steps, sent_commands: set[str], sent_spans: list[tuple[float, float]] | None = None):
        sent_spans = sent_spans or []
        for step in list(steps or []):
            command, reason = cls._step_command_and_reason(step)
            key = command.strip()
            if key and key.upper() == "DIAGNOSE":
                continue
            if key and key not in sent_commands and not cls._is_covered_by_sent(key, sent_spans):
                return step, command, reason
        return None, "", ""

    def _diagnose_merged_frames(self, frames):
        omega, mag, phase, diagnostics = merge_measurement_frames(frames)
        settings = dict(self.analysis_settings)
        control_settings = settings.pop("control_compensation_settings", None)
        result = analyze_system_v2(
            omega,
            mag,
            phase,
            diagnostics=diagnostics,
            **settings,
        )
        diagnosis = run_intelligent_diagnosis(
            omega,
            mag,
            phase,
            diagnostics=diagnostics,
            analysis_result=result,
            control_compensation_settings=control_settings,
        )
        refreshed_steps = list(getattr(diagnosis, "active_sweep_plan", None) or [])
        if not refreshed_steps:
            refreshed_steps = list(getattr(diagnosis, "adaptive_sweep_commands", None) or [])
        return (omega, mag, phase, diagnostics), refreshed_steps, diagnosis

    @staticmethod
    def _target_reached(diagnosis) -> bool:
        confidence = float(getattr(diagnosis, "confidence", 0.0) or 0.0)
        return confidence >= TARGET_CONFIDENCE

    def run(self):
        if serial is None:
            self.out_queue.put(("error", "未安装 pyserial，请先执行 pip install pyserial"))
            return
        if not self.sweep_steps:
            self.out_queue.put(("error", f"当前没有可执行的{self.mode_label}计划"))
            return

        ser = None
        try:
            ser = open_serial_transport(
                serial,
                self.port,
                self.baudrate,
                timeout=max(0.2, min(self.timeout_sec, 1.0)),
            )
            with self._ser_lock:
                self._ser = ser
            frames = [self.base_frame]
            pending_steps = list(self.sweep_steps)
            sent_commands: set[str] = set()
            sent_spans: list[tuple[float, float]] = []
            latest_merged = None
            last_confidence = None
            stagnant_rounds = 0
            for step_index in range(self.max_steps if self.adaptive else min(len(pending_steps), self.max_steps)):
                if self.stop_event.is_set():
                    self.out_queue.put(("log", f"{self.mode_label}已停止，剩余命令未发送"))
                    break
                if self.adaptive and step_index == 0:
                    try:
                        latest_merged, pending_steps, diagnosis = self._diagnose_merged_frames(frames)
                        confidence = float(getattr(diagnosis, "confidence", 0.0) or 0.0)
                        health = getattr(diagnosis, "measurement_health", "")
                        last_confidence = confidence
                        if self._target_reached(diagnosis):
                            self.out_queue.put((
                                "log",
                                f"{self.mode_label}: 当前合并数据已达到目标（置信度 {confidence:.2f}，{health}），无需继续自动补扫。",
                            ))
                            break
                    except Exception as exc:
                        self.out_queue.put(("log", f"{self.mode_label}: 初始目标诊断失败，使用已有补扫计划继续: {exc}"))
                _, command, reason = self._next_unsent_step(pending_steps, sent_commands, sent_spans)
                if not command:
                    self.out_queue.put(("log", f"{self.mode_label}: 当前合并数据没有新的可执行补扫命令"))
                    break
                command_to_send = command if command.endswith("\n") else command + "\n"
                command_key = command.strip()
                sent_commands.add(command_key)
                span = self._command_frequency_span(command_key)
                if span is not None:
                    sent_spans.append(span)
                self.out_queue.put(("log", f"{self.mode_label}: {command_key}，原因: {reason}"))
                write_ascii_command(ser, command_to_send)
                frames.append(self._read_frame(ser))

                if not self.adaptive:
                    continue
                try:
                    latest_merged, pending_steps, diagnosis = self._diagnose_merged_frames(frames)
                    health = getattr(diagnosis, "measurement_health", "")
                    confidence = float(getattr(diagnosis, "confidence", 0.0) or 0.0)
                    if last_confidence is not None and confidence - last_confidence < MIN_CONFIDENCE_IMPROVEMENT:
                        stagnant_rounds += 1
                    else:
                        stagnant_rounds = 0
                    last_confidence = confidence
                    next_count = len([
                        step for step in pending_steps
                        if (
                            self._step_command_and_reason(step)[0].strip() not in sent_commands
                            and not self._is_covered_by_sent(self._step_command_and_reason(step)[0], sent_spans)
                        )
                    ])
                    self.out_queue.put((
                        "log",
                        f"{self.mode_label}: 第 {step_index + 1} 段完成，已基于合并数据重新诊断"
                        f"（置信度 {confidence:.2f}，{health}，后续 {next_count} 条）。",
                    ))
                    if self._target_reached(diagnosis):
                        self.out_queue.put(("log", f"{self.mode_label}: 已达到目标置信度/健康度，自动收敛停止。"))
                        break
                    if stagnant_rounds >= 2:
                        self.out_queue.put((
                            "log",
                            f"{self.mode_label}: 连续补扫提升小于 {MIN_CONFIDENCE_IMPROVEMENT:.3f}，停止以避免无效重复扫频。",
                        ))
                        break
                    if next_count == 0:
                        self.out_queue.put(("log", f"{self.mode_label}: 未找到未覆盖的新频段，停止自动补扫。"))
                        break
                except Exception as exc:
                    self.out_queue.put(("log", f"{self.mode_label}: 动态诊断更新失败，保留已采集数据: {exc}"))
                    break

            if latest_merged is None:
                latest_merged = merge_measurement_frames(frames)
            omega, mag, phase, diagnostics = latest_merged
            self.out_queue.put(
                ("auto_frame", (omega, mag, phase, diagnostics, self.merged_source, self.expected_circuit))
            )
        except Exception as exc:
            self.out_queue.put(("error", f"{self.mode_label}失败: {exc}"))
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            self.out_queue.put(("log", f"{self.mode_label}线程已结束"))
