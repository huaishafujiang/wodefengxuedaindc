from __future__ import annotations

import threading

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

from filter_analysis import (
    AUTO_COARSE_SWEEP,
    AUTO_SWEEP_SEGMENTS,
    EXPECTED_CIRCUIT_MAP,
    FILTER_TYPE_LABELS,
    analyze_system_v2,
    demo_segments_for_expected,
    merge_measurement_frames,
)
from serial_protocol import (
    build_sweep_command,
    read_measurement_frame_from_serial as read_protocol_measurement_frame_from_serial,
)
from serial_transport import open_serial_transport, write_ascii_command


def list_serial_ports():
    if serial is None:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]


def read_measurement_frame_from_serial(ser, stop_event, timeout_sec: float):
    frame = read_protocol_measurement_frame_from_serial(ser, stop_event, timeout_sec)
    return frame.as_legacy_tuple()


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

            fallback_sent = bool(self.command)
            if not self.command and self.fallback_command:
                self.out_queue.put(('log', '兼容模式：先等旧版自动三行输出，超时后自动补发 SWEEP。'))

            while not self.stop_event.is_set():
                read_timeout = self.timeout_sec if fallback_sent else min(max(0.8, self.timeout_sec * 0.1), 1.5)
                try:
                    frame = read_protocol_measurement_frame_from_serial(
                        ser,
                        self.stop_event,
                        read_timeout,
                    )
                except TimeoutError:
                    if not fallback_sent and self.fallback_command:
                        write_ascii_command(ser, self.fallback_command)
                        fallback_sent = True
                        self.out_queue.put(('log', f'Fallback command sent: {self.fallback_command.strip()}'))
                        continue
                    if not self.continuous:
                        raise
                    continue

                if frame.diagnostics:
                    self.out_queue.put(('log', f'已读取测量诊断数组: {", ".join(sorted(frame.diagnostics.keys()))}'))
                self.out_queue.put(('frame', frame.as_legacy_tuple()))
                if self.continuous and not self.command:
                    fallback_sent = False
                if not self.continuous:
                    break

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
            expected_target = EXPECTED_CIRCUIT_MAP.get(self.expected_circuit)
            demo_segments = demo_segments_for_expected(self.expected_circuit)
            if demo_segments is not None and expected_target is not None:
                candidate_type = expected_target[0]
                segments = demo_segments
                sent = set()
                self.out_queue.put((
                    'log',
                    f'按期望电路“{self.expected_circuit}”使用演示扫频预设，共 {len(segments)} 段。'
                ))
            else:
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
            if demo_segments is not None:
                source = f'演示预设({self.expected_circuit})'
            else:
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