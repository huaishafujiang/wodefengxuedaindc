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

    def test_live_third_order_lowpass_default_sweep_stays_third_order(self):
        path = Path(__file__).resolve().parents[1] / "com6_live_lowpass3_single_20260512_210955_raw.txt"
        if not path.exists():
            self.skipTest("live third-order low-pass regression capture is not present")
        frame = parse_measurement_frame_from_text(path.read_text(encoding="utf-8"))

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 3)
        self.assertTrue(result.order_reliable)

    def test_live_third_order_lowpass_short_sweep_is_not_demoted_to_second_order(self):
        candidates = sorted(Path(__file__).resolve().parents[1].glob("com6_live_lowpass3_restore_*_raw.txt"))
        if not candidates:
            self.skipTest("live third-order low-pass short-sweep regression capture is not present")
        text = candidates[-1].read_text(encoding="utf-8")
        body = text.split(" ---", 1)[1].split("--- summary ---", 1)[0]
        frame = parse_measurement_frame_from_text(body)

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 3)

    def test_live_third_order_lowpass_gui_default_sweep_is_not_demoted_to_first_order(self):
        candidates = sorted(Path(__file__).resolve().parents[1].glob("com6_program_entry_recheck_*_raw.txt"))
        if not candidates:
            self.skipTest("live GUI-entry low-pass regression capture is not present")
        text = candidates[-1].read_text(encoding="utf-8")
        chunks = text.split("--- command: ")[1:]
        self.assertGreaterEqual(len(chunks), 2)
        _, rest = chunks[1].split(" ---", 1)
        body = rest.split("--- summary ---", 1)[0]
        frame = parse_measurement_frame_from_text(body)

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 3)

    def test_live_third_order_lowpass_demo_segments_merge_to_third_order(self):
        path = Path(__file__).resolve().parents[1] / "com6_live_lowpass3_20260512_210654_raw.txt"
        if not path.exists():
            self.skipTest("live third-order low-pass segmented capture is not present")
        text = path.read_text(encoding="utf-8")
        frames = []
        for chunk in text.split("--- command: ")[1:]:
            _, rest = chunk.split(" ---", 1)
            body = rest.split("--- summary ---", 1)[0]
            frames.append(parse_measurement_frame_from_text(body))

        omega, magnitude, phase, diagnostics = merge_measurement_frames(frames)
        result = analyze_system(omega, magnitude, phase, diagnostics=diagnostics)

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 3)
        self.assertTrue(result.order_reliable)

    def test_live_twin_t_bandstop_wide_and_merged_sweeps_stay_bandstop(self):
        path = Path(__file__).resolve().parents[1] / "com6_live_bandstop_plus222_both_20260512_204156_raw.txt"
        if not path.exists():
            self.skipTest("live Twin-T band-stop regression capture is not present")
        text = path.read_text(encoding="utf-8")
        frames = []
        for chunk in text.split("--- command: ")[1:]:
            _, rest = chunk.split(" ---", 1)
            body = rest.split("--- summary ---", 1)[0]
            frames.append(parse_measurement_frame_from_text(body))

        wide_result = analyze_system(
            frames[0].omega,
            frames[0].magnitude,
            frames[0].phase,
            diagnostics=frames[0].diagnostics,
        )
        self.assertEqual(wide_result.filter_type, "bandstop")
        self.assertEqual(wide_result.order_estimate, 2)

        omega, magnitude, phase, diagnostics = merge_measurement_frames(frames)
        merged_result = analyze_system(omega, magnitude, phase, diagnostics=diagnostics)

        self.assertEqual(merged_result.filter_type, "bandstop")
        self.assertEqual(merged_result.order_estimate, 2)
        self.assertTrue(merged_result.order_reliable)

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
        self.assertEqual(len(result.omega), len(omega))

    def test_highpass_noise_floor_head_is_ignored_for_order(self):
        freq_hz = np.geomspace(100.0, 60000.0, 160)
        omega = 2.0 * np.pi * freq_hz
        stage_wc = 10000.0
        input_rms = np.full_like(omega, 1.2 / (2.0 * np.sqrt(2.0)))
        ideal_mag = 1.0 / np.sqrt(1.0 + (stage_wc / omega) ** 2) ** 3
        ideal_phase = 3.0 * np.arctan(stage_wc / omega)

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

        self.assertEqual(result.filter_type, "highpass")
        self.assertEqual(result.order_estimate, 3)
        self.assertEqual(len(result.omega), len(omega))

    def test_finite_sweep_second_order_highpass_is_not_demoted_to_first(self):
        freq_hz = np.linspace(100.0, 20000.0, 200)
        omega = 2.0 * np.pi * freq_hz
        normalized = omega / (2.0 * np.pi * 1500.0)
        measured_mag = normalized**1.8 / (1.0 + normalized**1.8)
        phase_deg = 5.0 + 130.0 / (1.0 + (omega / (2.0 * np.pi * 1500.0)) ** 0.95)
        input_rms = np.full_like(omega, 1.2 / (2.0 * np.sqrt(2.0)))

        result = analyze_system(
            omega,
            measured_mag,
            np.radians(phase_deg),
            diagnostics={
                "input_rms_v": input_rms,
                "output_rms_v": input_rms * measured_mag,
                "input_dc_v": np.full_like(omega, 1.65),
                "output_dc_v": np.full_like(omega, 1.75),
                "clip_flags": np.zeros_like(omega),
                "valid_capture_count": np.full_like(omega, 3.0),
            },
        )

        self.assertEqual(result.filter_type, "highpass")
        self.assertEqual(result.order_estimate, 2)
        self.assertEqual(len(result.omega), len(omega))

    def test_clipped_bandpass_right_edge_is_not_promoted_to_third_order(self):
        path = Path(__file__).resolve().parents[1] / "com6_bandpass_default_100_20000_100_1p2_20260510_190645.txt"
        if not path.exists():
            self.skipTest("live band-pass regression capture is not present")
        frame = parse_measurement_frame_from_text(path.read_text(encoding="utf-8"))

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "bandpass")
        self.assertEqual(result.order_estimate, 2)
        self.assertEqual(len(result.omega), len(frame.omega))

    def test_live_first_order_bandpass_with_peak_near_left_side_is_detected(self):
        path = Path(__file__).resolve().parents[1] / "com6_current_102_default_20260512_165442_raw.txt"
        if not path.exists():
            self.skipTest("live first-order band-pass regression capture is not present")
        frame = parse_measurement_frame_from_text(path.read_text(encoding="utf-8"))

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "bandpass")
        self.assertEqual(result.order_estimate, 2)
        self.assertAlmostEqual(result.magnitude_cutoff_omega / (2.0 * np.pi), 493.0, delta=40.0)
        self.assertIsNotNone(result.secondary_cutoff_omega)
        self.assertAlmostEqual(result.secondary_cutoff_omega / (2.0 * np.pi), 17700.0, delta=900.0)
        self.assertTrue(any("two -3 dB crossings" in note for note in result.notes))

    def test_clipped_highpass_live_capture_still_returns_low_confidence_result(self):
        path = Path(__file__).resolve().parents[1] / "com6_live_highpass3_segments_0p2v.txt"
        if not path.exists():
            self.skipTest("live high-pass regression capture is not present")
        frame = parse_measurement_frame_from_text(path.read_text(encoding="utf-8"))

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertFalse(result.order_reliable)
        self.assertEqual(result.identification_summary, "需复测确认")
        self.assertTrue(any("PA1" in note and ("削顶" in note or "偏置" in note) for note in result.notes))

    def test_highpass_transition_band_recovers_third_order_live_capture(self):
        path = Path(__file__).resolve().parents[1] / "com6_live_judge_20260511_203326.txt"
        if not path.exists():
            self.skipTest("live high-pass transition-band regression capture is not present")
        frame = parse_measurement_frame_from_text(path.read_text(encoding="utf-8"))

        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)

        self.assertEqual(result.filter_type, "highpass")
        self.assertEqual(result.order_estimate, 3)
        self.assertTrue(any("transition-band fit prefers order 3" in note for note in result.notes))

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
