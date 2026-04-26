import threading
import unittest

import numpy as np

from serial_protocol import (
    build_sweep_command,
    parse_measurement_frame_from_text,
    read_measurement_frame_from_serial,
)


class FakeSerial:
    def __init__(self, lines):
        self.lines = [line.encode("ascii") for line in lines]
        self.timeout = 1.0

    def readline(self):
        if not self.lines:
            return b""
        return self.lines.pop(0)


class SerialProtocolTests(unittest.TestCase):
    def test_parse_text_frame_with_diagnostics(self):
        frame = parse_measurement_frame_from_text(
            """
            READY STM32G431_SWEEP_V1
            omega=[1,2,3]
            Magnitude_data=[0.9,0.8,0.7]
            Phase_data_rad=[0,-0.1,-0.2]
            Input_rms_v=[0.2,0.2,0.2]
            DONE
            """
        )

        np.testing.assert_allclose(frame.omega, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(frame.magnitude, [0.9, 0.8, 0.7])
        self.assertIn("input_rms_v", frame.diagnostics)

    def test_stream_reader_skips_noise(self):
        ser = FakeSerial(
            [
                "OK SWEEP 100 300 100 1\n",
                "debug line\n",
                "omega=[1,2]\n",
                "Magnitude_data=[1,0.5]\n",
                "Phase_data_rad=[0,-1.57]\n",
                "Clip_flags=[0,1]\n",
                "DONE\n",
            ]
        )

        frame = read_measurement_frame_from_serial(ser, threading.Event(), 1.0)

        np.testing.assert_allclose(frame.phase, [0.0, -1.57])
        np.testing.assert_allclose(frame.diagnostics["clip_flags"], [0.0, 1.0])

    def test_build_sweep_command(self):
        self.assertEqual(build_sweep_command(100, 1000, 100, 0.6), "SWEEP 100 1000 100 0.6\n")


if __name__ == "__main__":
    unittest.main()
