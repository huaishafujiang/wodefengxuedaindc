from __future__ import annotations

import threading

try:
    import serial
except Exception:
    serial = None

from filter_analysis import merge_measurement_frames
from serial_readers import read_measurement_frame_from_serial
from serial_transport import open_serial_transport, write_ascii_command


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

    @staticmethod
    def _step_command_and_reason(step):
        if isinstance(step, str):
            return step.strip(), "智能诊断建议补扫"
        command = getattr(step, "command", None)
        reason = getattr(step, "reason", None)
        if command is None and isinstance(step, dict):
            command = step.get("command")
            reason = step.get("reason")
        return str(command or "").strip(), str(reason or "智能诊断建议补扫")

    def run(self):
        if serial is None:
            self.out_queue.put(("error", "未安装 pyserial，请先执行 pip install pyserial"))
            return
        if not self.sweep_steps:
            self.out_queue.put(("error", "当前诊断没有可执行的智能补扫计划"))
            return

        ser = None
        try:
            ser = open_serial_transport(
                serial,
                self.port,
                self.baudrate,
                timeout=max(0.2, min(self.timeout_sec, 1.0)),
            )
            frames = [self.base_frame]
            for step in self.sweep_steps:
                if self.stop_event.is_set():
                    self.out_queue.put(("log", "智能补扫已停止，剩余命令未发送"))
                    break
                command, reason = self._step_command_and_reason(step)
                if not command:
                    continue
                command_to_send = command if command.endswith("\n") else command + "\n"
                self.out_queue.put(("log", f"智能补扫: {command.strip()}，原因: {reason}"))
                write_ascii_command(ser, command_to_send)
                frames.append(read_measurement_frame_from_serial(ser, self.stop_event, self.timeout_sec))

            omega, mag, phase, diagnostics = merge_measurement_frames(frames)
            self.out_queue.put(
                ("auto_frame", (omega, mag, phase, diagnostics, "智能补扫合并结果", self.expected_circuit))
            )
        except Exception as exc:
            self.out_queue.put(("error", f"智能补扫失败: {exc}"))
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            self.out_queue.put(("log", "智能补扫线程已结束"))
