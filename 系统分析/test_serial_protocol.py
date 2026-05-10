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
        self.lines = [line if isinstance(line, bytes) else line.encode("ascii") for line in lines]
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
                "Input_min_code=[1674,1674]\n",
                "Input_max_code=[2420,2420]\n",
                "Output_min_code=[1640,20]\n",
                "Output_max_code=[2390,4080]\n",
                "DONE\n",
            ]
        )

        frame = read_measurement_frame_from_serial(ser, threading.Event(), 1.0)

        np.testing.assert_allclose(frame.phase, [0.0, -1.57])
        np.testing.assert_allclose(frame.diagnostics["clip_flags"], [0.0, 1.0])
        np.testing.assert_allclose(frame.diagnostics["input_min_code"], [1674.0, 1674.0])
        np.testing.assert_allclose(frame.diagnostics["output_max_code"], [2390.0, 4080.0])

    def test_stream_reader_waits_for_non_adjacent_core_arrays(self):
        ser = FakeSerial(
            [
                "OK SWEEP 100 300 100 1\n",
                "omega=[1,2]\n",
                b"",
                "debug while sweep is still running\n",
                b"",
                "Magnitude_data=[1,0.5]\n",
                b"",
                "Phase_data_rad=[0,-1.57]\n",
            ]
        )

        frame = read_measurement_frame_from_serial(ser, threading.Event(), 1.0)

        np.testing.assert_allclose(frame.omega, [1.0, 2.0])
        np.testing.assert_allclose(frame.magnitude, [1.0, 0.5])
        np.testing.assert_allclose(frame.phase, [0.0, -1.57])

    def test_build_sweep_command(self):
        self.assertEqual(build_sweep_command(100, 1000, 100, 0.6), "SWEEP 100 1000 100 0.6\n")

    def test_parse_compact_diagnostics(self):
        frame = parse_measurement_frame_from_text(
            """
            omega=[1,2]
            Magnitude_data=[1,0.5]
            Phase_data_rad=[0,-0.1]
            Adc_code_range=[1674,2420,1640,4080]
            Clip_point_count=[0,1]
            """
        )

        np.testing.assert_allclose(frame.diagnostics["adc_code_range"], [1674, 2420, 1640, 4080])
        np.testing.assert_allclose(frame.diagnostics["clip_point_count"], [0, 1])

    def test_parse_nan_diagnostic_values(self):
        frame = parse_measurement_frame_from_text(
            """
            omega=[1,2]
            Magnitude_data=[1,0.5]
            Phase_data_rad=[0,-0.1]
            Magnitude_repeat_span_db=[nan,nan]
            Phase_repeat_span_rad=[nan,nan]
            """
        )

        self.assertTrue(np.isnan(frame.diagnostics["magnitude_repeat_span_db"]).all())
        self.assertTrue(np.isnan(frame.diagnostics["phase_repeat_span_rad"]).all())


if __name__ == "__main__":
    unittest.main()
