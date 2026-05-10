import unittest
from pathlib import Path

import numpy as np

from main import analyze_system, merge_measurement_frames
from serial_protocol import parse_measurement_frame_from_text


class LowpassNoiseFloorTests(unittest.TestCase):
    def test_noisy_lowpass_tail_does_not_promote_third_order_to_fourth(self):
        path = Path(__file__).resolve().parents[1] / "gui_default_100_20000_100_1p2.txt"
        if not path.exists():
            self.skipTest("live regression capture is not present")
        frame = parse_measurement_frame_from_text(path.read_text(encoding="utf-8"))

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 3)

    def test_fully_clipped_serial_frame_is_rejected_instead_of_classified(self):
        omega = np.linspace(2.0 * np.pi * 100.0, 2.0 * np.pi * 20000.0, 20)
        magnitude = np.zeros_like(omega)
        phase = np.linspace(0.5, 30.0, len(omega))

        with self.assertRaisesRegex(ValueError, "PA1|Magnitude_data|无效|削顶"):
            analyze_system(
                omega,
                magnitude,
                phase,
                diagnostics={
                    "input_rms_v": np.full_like(omega, 0.42),
                    "output_rms_v": np.zeros_like(omega),
                    "output_dc_v": np.full_like(omega, 3.3),
                    "clip_flags": np.full_like(omega, 2.0),
                    "valid_capture_count": np.full_like(omega, 3.0),
                    "adc_code_range": np.array([1287.0, 2816.0, 4095.0, 4095.0]),
                    "clip_point_count": np.array([0.0, float(len(omega))]),
                },
            )

    def test_lowpass_noise_floor_tail_is_ignored_for_order(self):
        freq_hz = np.geomspace(100.0, 60000.0, 160)
        omega = 2.0 * np.pi * freq_hz
        stage_wc = 10000.0
        input_rms = np.full_like(omega, 1.2 / (2.0 * np.sqrt(2.0)))
        ideal_mag = 1.0 / np.sqrt(1.0 + (omega / stage_wc) ** 2) ** 3
        ideal_phase = -3.0 * np.arctan(omega / stage_wc)

        output_rms = np.maximum(input_rms * ideal_mag, 0.002)
        measured_mag = output_rms / input_rms
        result = analyze_system(
            omega,
            measured_mag,
            ideal_phase,
            diagnostics={
                "input_rms_v": input_rms,
                "output_rms_v": output_rms,
                "clip_flags": np.zeros_like(omega),
                "valid_capture_count": np.full_like(omega, 3.0),
            },
        )

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 3)
        self.assertLess(len(result.omega), len(omega))

    def test_merge_measurement_frames_accepts_frame_objects_and_compact_diagnostics(self):
        frame1 = parse_measurement_frame_from_text(
            """
            omega=[1,2,3]
            Magnitude_data=[1,0.8,0.6]
            Phase_data_rad=[0,-0.1,-0.2]
            Input_rms_v=[0.2,0.2,0.2]
            Adc_code_range=[1700,2400,1650,2300]
            Clip_point_count=[0,1]
            """
        )
        frame2 = parse_measurement_frame_from_text(
            """
            omega=[4,5,6]
            Magnitude_data=[0.5,0.4,0.3]
            Phase_data_rad=[-0.3,-0.4,-0.5]
            Input_rms_v=[0.2,0.2,0.2]
            Adc_code_range=[1680,2420,1600,2350]
            Clip_point_count=[1,0]
            """
        )

        omega, _, _, diagnostics = merge_measurement_frames([frame1, frame2])

        np.testing.assert_allclose(omega, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(diagnostics["adc_code_range"], [1680.0, 2420.0, 1600.0, 2350.0])
        np.testing.assert_allclose(diagnostics["clip_point_count"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
