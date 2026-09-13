from __future__ import annotations


def open_serial_transport(serial_module, port: str, baudrate: int, timeout: float):
    if serial_module is None:
        raise RuntimeError("未安装 pyserial，请先执行 pip install pyserial")

    ser = serial_module.Serial(port, baudrate, timeout=timeout)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


def write_ascii_command(ser, command: str) -> None:
    ser.write(command.encode("ascii"))
    ser.flush()
