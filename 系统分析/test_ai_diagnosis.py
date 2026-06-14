from pathlib import Path
import queue
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

import smart_resweep_reader
from ai_diagnosis import (
    ActiveSweepStep,
    IntelligentDiagnosis,
    build_equivalent_transfer_function,
    build_active_sweep_plan,
    extract_diagnosis_features,
    fit_transfer_templates,
    format_ai_diagnosis_report_lines,
    load_fault_knowledge_base,
    rank_diagnosis_candidates,
    run_intelligent_diagnosis,
)
from filter_analysis import analyze_system_v2
from serial_protocol import parse_measurement_frame_from_text
from smart_resweep_reader import SmartResweepReader


DATA_DIR = Path(__file__).resolve().parents[1] / "测量数据"


STANDARD_CASES = [
    ("一阶低通", "lowpass", 1),
    ("二阶低通", "lowpass", 2),
    ("三阶低通", "lowpass", 3),
    ("一阶高通", "highpass", 1),
    ("二阶高通", "highpass", 2),
    ("三阶高通", "highpass", 3),
    ("带通", "bandpass", 2),
    ("带阻", "bandstop", 2),
]


def _lowpass_fixture(freq_hz=None):
    freq_hz = np.geomspace(100.0, 10000.0, 48) if freq_hz is None else np.asarray(freq_hz, dtype=float)
    omega = 2.0 * np.pi * freq_hz
    wc = 2.0 * np.pi * 1000.0
    magnitude = 1.0 / np.sqrt(1.0 + (omega / wc) ** 2)
    phase = -np.arctan(omega / wc)
    return omega, magnitude, phase


def _highpass_fixture():
    freq_hz = np.geomspace(100.0, 10000.0, 48)
    omega = 2.0 * np.pi * freq_hz
    wc = 2.0 * np.pi * 1000.0
    magnitude = (omega / wc) / np.sqrt(1.0 + (omega / wc) ** 2)
    phase = np.pi / 2.0 - np.arctan(omega / wc)
    return omega, magnitude, phase


def _bandpass_fixture():
    freq_hz = np.geomspace(80.0, 40000.0, 70)
    omega = 2.0 * np.pi * freq_hz
    f0 = 3500.0
    q = 1.4
    x = freq_hz / f0
    magnitude = 1.0 / np.sqrt(1.0 + (q * (x - 1.0 / x)) ** 2)
    phase = np.unwrap(np.angle(1.0 / (1.0 + 1j * q * (x - 1.0 / x))))
    return omega, magnitude, phase


def _bandstop_fixture():
    freq_hz = np.geomspace(80.0, 40000.0, 70)
    omega = 2.0 * np.pi * freq_hz
    f0 = 1600.0
    q = 3.0
    x = freq_hz / f0
    bp = 1.0 / np.sqrt(1.0 + (q * (x - 1.0 / x)) ** 2)
    magnitude = np.sqrt(np.clip(1.0 - bp**2, 1.0e-6, None))
    phase = np.unwrap(np.angle(1j * q * (x - 1.0 / x) / (1.0 + 1j * q * (x - 1.0 / x))))
    return omega, magnitude, phase


class IntelligentDiagnosisTests(unittest.TestCase):
    def _load_standard_case(self, name_part):
        matches = sorted(DATA_DIR.glob(f"{name_part}*_标准测试数据.txt"))
        self.assertEqual(len(matches), 1, f"Expected one data file for {name_part}, got {matches}")
        frame = parse_measurement_frame_from_text(matches[0].read_text(encoding="utf-8"))
        result = analyze_system_v2(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)
        diagnosis = run_intelligent_diagnosis(
            frame.omega,
            frame.magnitude,
            frame.phase,
            diagnostics=frame.diagnostics,
            analysis_result=result,
        )
        return frame, result, diagnosis

    def test_standard_eight_classes_are_identified_with_top1_candidate(self):
        for name_part, expected_type, expected_order in STANDARD_CASES:
            with self.subTest(name_part=name_part):
                _, result, diagnosis = self._load_standard_case(name_part)

                self.assertEqual(diagnosis.circuit_type, expected_type)
                self.assertEqual(diagnosis.order_estimate, expected_order)
                self.assertGreaterEqual(diagnosis.confidence, 0.80)
                self.assertEqual(diagnosis.candidates[0].circuit_type, expected_type)
                self.assertEqual(diagnosis.candidates[0].order_estimate, expected_order)
                self.assertTrue(diagnosis.adaptive_sweep_commands)
                self.assertTrue(diagnosis.adaptive_sweep_commands[0].startswith("SWEEP "))
                self.assertIn(result.filter_type, diagnosis.circuit_type)

    def test_transfer_template_fits_exist_for_standard_data(self):
        for name_part, expected_type, expected_order in STANDARD_CASES:
            with self.subTest(name_part=name_part):
                frame, _, diagnosis = self._load_standard_case(name_part)
                fits = fit_transfer_templates(frame.omega, frame.magnitude, frame.phase)
                expected_fit = next(
                    fit for fit in fits if fit.model_family == expected_type and fit.order == expected_order
                )

                self.assertTrue(diagnosis.model_fits)
                self.assertIsNotNone(diagnosis.best_fit)
                self.assertIsNotNone(diagnosis.best_fit.transfer_function)
                self.assertIsNotNone(diagnosis.equivalent_transfer_function)
                self.assertGreaterEqual(expected_fit.r_squared, 0.95)
                self.assertLess(expected_fit.rmse_db, 2.5)
                self.assertTrue(expected_fit.transfer_function.expression.startswith("H(s) = "))
                self.assertGreaterEqual(len(expected_fit.transfer_function.numerator), 1)
                self.assertGreaterEqual(len(expected_fit.transfer_function.denominator), 2)

    def test_equivalent_transfer_function_coefficients_are_generated(self):
        for fixture, expected_type in (
            (_lowpass_fixture, "lowpass"),
            (_highpass_fixture, "highpass"),
            (_bandpass_fixture, "bandpass"),
            (_bandstop_fixture, "bandstop"),
        ):
            with self.subTest(expected_type=expected_type):
                omega, magnitude, phase = fixture()
                fit = next(item for item in fit_transfer_templates(omega, magnitude, phase) if item.model_family == expected_type)
                tf = build_equivalent_transfer_function(fit)

                self.assertIsNotNone(tf)
                self.assertEqual(tf.model_family, expected_type)
                self.assertIn("H(s) =", tf.expression)
                self.assertTrue(all(np.isfinite(tf.numerator)))
                self.assertTrue(all(np.isfinite(tf.denominator)))

    def test_candidates_are_top3_and_sorted_descending(self):
        _, _, diagnosis = self._load_standard_case("二阶低通")
        confidences = [item.confidence for item in diagnosis.candidates]

        self.assertGreaterEqual(len(diagnosis.candidates), 3)
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertLessEqual(diagnosis.candidates[1].confidence, diagnosis.candidates[0].confidence)

    def test_public_candidate_ranking_helper(self):
        omega, mag, phase = _lowpass_fixture()
        features = extract_diagnosis_features(omega, mag, phase)
        fits = fit_transfer_templates(omega, mag, phase)
        candidates = rank_diagnosis_candidates(features, fits)

        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual(candidates[0].circuit_type, "lowpass")
        self.assertEqual(candidates[0].order_estimate, 1)

    def test_reliable_traditional_analysis_anchors_candidate_ranking(self):
        omega, mag, phase = _lowpass_fixture()
        features = extract_diagnosis_features(omega, mag, phase)
        fits = fit_transfer_templates(omega, mag, phase)
        reliable_result = SimpleNamespace(
            filter_type="highpass",
            order_estimate=2,
            order_reliable=True,
            identification_confidence=0.92,
        )

        candidates = rank_diagnosis_candidates(features, fits, analysis_result=reliable_result)

        self.assertEqual(candidates[0].circuit_type, "highpass")
        self.assertEqual(candidates[0].order_estimate, 2)
        self.assertIn("传统可靠锚定", candidates[0].evidence)
        self.assertGreaterEqual(candidates[0].confidence, 0.90)

    def test_unreliable_traditional_analysis_does_not_override_feature_and_model(self):
        omega, mag, phase = _lowpass_fixture()
        features = extract_diagnosis_features(omega, mag, phase)
        fits = fit_transfer_templates(omega, mag, phase)
        unreliable_result = SimpleNamespace(
            filter_type="highpass",
            order_estimate=2,
            order_reliable=False,
            identification_confidence=0.95,
        )

        candidates = rank_diagnosis_candidates(features, fits, analysis_result=unreliable_result)

        self.assertEqual(candidates[0].circuit_type, "lowpass")
        self.assertEqual(candidates[0].order_estimate, 1)

    def test_report_uses_engineering_instrument_v2_fields(self):
        _, _, diagnosis = self._load_standard_case("二阶低通")
        report = "\n".join(format_ai_diagnosis_report_lines(diagnosis))

        self.assertIn("智能识别: 二阶低通系统", report)
        self.assertIn("判定策略:", report)
        self.assertIn("置信度:", report)
        self.assertIn("关键频率:", report)
        self.assertIn("候选排名:", report)
        self.assertIn("模型拟合:", report)
        self.assertIn("等效传递函数:", report)
        self.assertIn("H(s) =", report)
        self.assertIn("测量质量:", report)
        self.assertIn("故障证据链:", report)
        self.assertIn("下一步测试建议:", report)
        self.assertIn("一键补扫计划:", report)
        self.assertIn("补扫SWEEP建议:", report)

    def test_feature_extraction_contains_required_diagnostic_features(self):
        frame, _, _ = self._load_standard_case("一阶低通")
        features = extract_diagnosis_features(
            frame.omega,
            frame.magnitude,
            frame.phase,
            diagnostics=frame.diagnostics,
        )

        self.assertGreater(features.low_gain_db, features.high_gain_db)
        self.assertLess(features.tail_slope_db_dec, -10.0)
        self.assertGreater(features.minus3_crossing_count, 0)
        self.assertIsNotNone(features.input_rms_median_v)
        self.assertEqual(features.clipped_points, 0)

    def test_fault_knowledge_base_loads_independent_rules(self):
        rules = load_fault_knowledge_base()
        rule_ids = {rule.rule_id for rule in rules}

        self.assertGreaterEqual(len(rules), 10)
        self.assertIn("pa0_clip", rule_ids)
        self.assertIn("pa1_clip", rule_ids)
        self.assertIn("output_noise_floor", rule_ids)
        self.assertIn("swap_io_suspected", rule_ids)
        self.assertIn("sweep_range_short", rule_ids)

    def test_clipping_and_dc_bias_faults_are_reported(self):
        omega, magnitude, phase = _lowpass_fixture()
        diagnostics = {
            "input_rms_v": np.full_like(omega, 0.20),
            "output_rms_v": np.maximum(0.20 * magnitude, 0.002),
            "input_dc_v": np.full_like(omega, 1.65),
            "output_dc_v": np.full_like(omega, 3.05),
            "clip_flags": np.r_[np.zeros(len(omega) - 4), np.full(4, 2.0)],
            "valid_capture_count": np.full_like(omega, 3.0),
        }

        diagnosis = run_intelligent_diagnosis(omega, magnitude, phase, diagnostics=diagnostics)
        text = "\n".join(format_ai_diagnosis_report_lines(diagnosis))
        finding_ids = {item.rule_id for item in diagnosis.fault_findings}

        self.assertEqual(diagnosis.circuit_type, "lowpass")
        self.assertLess(diagnosis.confidence, 0.90)
        self.assertIn("pa1_clip", finding_ids)
        self.assertIn("pa1_bias_rail", finding_ids)
        self.assertIn("PA1削顶", text)
        self.assertIn("PA1 DC", text)
        self.assertIn("降低激励幅值", text)

    def test_fault_knowledge_base_outputs_evidence_chain_for_common_faults(self):
        omega, magnitude, phase = _lowpass_fixture()
        diagnostics = {
            "input_rms_v": np.full_like(omega, 0.01),
            "output_rms_v": np.full_like(omega, 0.002),
            "input_dc_v": np.full_like(omega, 1.65),
            "output_dc_v": np.full_like(omega, 1.65),
            "clip_flags": np.r_[1.0, np.zeros(len(omega) - 1)],
            "valid_capture_count": np.full_like(omega, 1.0),
        }

        diagnosis = run_intelligent_diagnosis(omega, magnitude * 3.0, phase, diagnostics=diagnostics)
        finding_ids = {item.rule_id for item in diagnosis.fault_findings}

        self.assertIn("pa0_clip", finding_ids)
        self.assertIn("input_rms_low", finding_ids)
        self.assertIn("output_noise_floor", finding_ids)
        self.assertIn("swap_io_suspected", finding_ids)
        self.assertIn("valid_capture_low", finding_ids)
        self.assertTrue(all(item.evidence for item in diagnosis.fault_findings))
        self.assertTrue(all(item.suggestion for item in diagnosis.fault_findings))

    def test_sweep_range_insufficient_fault_is_reported(self):
        omega, magnitude, phase = _lowpass_fixture(np.linspace(900.0, 1400.0, 8))
        diagnosis = run_intelligent_diagnosis(omega, magnitude, phase)
        finding_ids = {item.rule_id for item in diagnosis.fault_findings}

        self.assertIn("sweep_range_short", finding_ids)
        self.assertLessEqual(diagnosis.confidence, 0.45)

    def test_unknown_response_expands_sweep_range(self):
        freq_hz = np.linspace(100.0, 2000.0, 20)
        omega = 2.0 * np.pi * freq_hz
        magnitude = np.full_like(omega, 0.8)
        phase = np.zeros_like(omega)

        diagnosis = run_intelligent_diagnosis(omega, magnitude, phase)

        self.assertEqual(diagnosis.circuit_type, "unknown")
        self.assertLess(diagnosis.confidence, 0.50)
        self.assertTrue(any("扩大扫频范围" in item for item in diagnosis.next_test_suggestions))
        self.assertTrue(diagnosis.adaptive_sweep_commands[0].startswith("SWEEP "))

    def test_active_sweep_plan_matches_diagnosis_family(self):
        for fixture, expected_type, command_count in (
            (_lowpass_fixture, "lowpass", 1),
            (_highpass_fixture, "highpass", 1),
            (_bandpass_fixture, "bandpass", 2),
            (_bandstop_fixture, "bandstop", 2),
        ):
            with self.subTest(expected_type=expected_type):
                omega, magnitude, phase = fixture()
                diagnosis = run_intelligent_diagnosis(omega, magnitude, phase)
                commands = build_active_sweep_plan(diagnosis)

                self.assertEqual(diagnosis.circuit_type, expected_type)
                self.assertEqual(len(commands), command_count)
                self.assertTrue(all(command.startswith("SWEEP ") for command in commands))

    def test_unknown_active_sweep_plan_expands_range(self):
        freq_hz = np.linspace(100.0, 2000.0, 20)
        omega = 2.0 * np.pi * freq_hz
        magnitude = np.full_like(omega, 0.8)
        phase = np.zeros_like(omega)
        diagnosis = run_intelligent_diagnosis(omega, magnitude, phase)
        command = build_active_sweep_plan(diagnosis)[0]
        parts = command.split()

        self.assertEqual(parts[0], "SWEEP")
        self.assertLessEqual(float(parts[1]), 25.0)
        self.assertGreaterEqual(float(parts[2]), 9000.0)


class SmartResweepReaderTests(unittest.TestCase):
    def test_smart_resweep_sends_commands_and_merges_frames(self):
        out_queue = queue.Queue()
        stop_event = threading.Event()
        base_frame = (np.array([1.0, 2.0, 3.0]), np.ones(3), np.zeros(3), {})
        sweep_steps = [
            ActiveSweepStep("SWEEP 100 1000 50 0.6", "cutoff"),
            ActiveSweepStep("SWEEP 900 1300 20 0.6", "right edge"),
        ]
        sent_commands = []
        fake_ser = unittest.mock.Mock()

        def fake_write(ser, command):
            sent_commands.append(command.strip())

        def fake_read(ser, stop, timeout):
            idx = len(sent_commands)
            return (np.array([10.0 + idx, 11.0 + idx, 12.0 + idx]), np.ones(3), np.zeros(3), {})

        def fake_merge(frames):
            self.assertEqual(len(frames), 3)
            return np.array([1.0, 2.0]), np.ones(2), np.zeros(2), {"input_rms_v": np.ones(2)}

        with patch.object(smart_resweep_reader, "serial", object()):
            with patch.object(smart_resweep_reader, "open_serial_transport", return_value=fake_ser):
                with patch.object(smart_resweep_reader, "write_ascii_command", side_effect=fake_write):
                    with patch.object(smart_resweep_reader, "read_measurement_frame_from_serial", side_effect=fake_read):
                        with patch.object(smart_resweep_reader, "merge_measurement_frames", side_effect=fake_merge):
                            reader = SmartResweepReader(
                                port="COM1",
                                baudrate=115200,
                                timeout_sec=2.0,
                                out_queue=out_queue,
                                stop_event=stop_event,
                                base_frame=base_frame,
                                sweep_steps=sweep_steps,
                                expected_circuit="Auto",
                            )
                            reader.run()

        self.assertEqual(sent_commands, ["SWEEP 100 1000 50 0.6", "SWEEP 900 1300 20 0.6"])
        events = list(out_queue.queue)
        self.assertTrue(any(kind == "auto_frame" for kind, _ in events))
        fake_ser.close.assert_called_once()

    def test_smart_resweep_stop_event_skips_remaining_commands(self):
        out_queue = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()
        base_frame = (np.array([1.0, 2.0, 3.0]), np.ones(3), np.zeros(3), {})
        sent_commands = []
        fake_ser = unittest.mock.Mock()

        def fake_merge(frames):
            self.assertEqual(len(frames), 1)
            return np.array([1.0, 2.0, 3.0]), np.ones(3), np.zeros(3), {}

        with patch.object(smart_resweep_reader, "serial", object()):
            with patch.object(smart_resweep_reader, "open_serial_transport", return_value=fake_ser):
                with patch.object(
                    smart_resweep_reader,
                    "write_ascii_command",
                    side_effect=lambda ser, command: sent_commands.append(command),
                ):
                    with patch.object(smart_resweep_reader, "merge_measurement_frames", side_effect=fake_merge):
                        reader = SmartResweepReader(
                            port="COM1",
                            baudrate=115200,
                            timeout_sec=2.0,
                            out_queue=out_queue,
                            stop_event=stop_event,
                            base_frame=base_frame,
                            sweep_steps=["SWEEP 100 1000 50 0.6"],
                        )
                        reader.run()

        self.assertEqual(sent_commands, [])
        events = list(out_queue.queue)
        self.assertTrue(any(kind == "log" and "已停止" in payload for kind, payload in events))
        self.assertTrue(any(kind == "auto_frame" for kind, _ in events))


if __name__ == "__main__":
    unittest.main()
