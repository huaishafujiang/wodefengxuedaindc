from pathlib import Path
import unittest

from main import (
    analyze_system,
    build_filter_report_lines,
    normalize_phase_for_display,
    phase_array_for_display,
)
from serial_protocol import parse_measurement_frame_from_text


DATA_DIR = Path(__file__).resolve().parents[1] / "测量数据"


STANDARD_CASES = [
    ("一阶低通", "lowpass", 1, 9988.31),
    ("二阶低通", "lowpass", 2, 6423.99),
    ("三阶低通", "lowpass", 3, 5091.52),
    ("一阶高通", "highpass", 1, 10019.52),
    ("二阶高通", "highpass", 2, 15587.88),
    ("三阶高通", "highpass", 3, 19685.92),
    ("带通", "bandpass", 2, 17407.59),
    ("带阻", "bandstop", 2, 10235.80),
]


class StandardFilterAnalysisTests(unittest.TestCase):
    def _load_case(self, name_part):
        matches = sorted(DATA_DIR.glob(f"{name_part}*_标准测试数据.txt"))
        self.assertEqual(len(matches), 1, f"Expected one data file for {name_part}, got {matches}")
        frame = parse_measurement_frame_from_text(matches[0].read_text(encoding="utf-8"))
        result = analyze_system(frame.omega, frame.magnitude, frame.phase, diagnostics=frame.diagnostics)
        report = "\n".join(build_filter_report_lines(result))
        return result, report

    def test_standard_data_identification(self):
        for name_part, expected_type, expected_order, expected_omega in STANDARD_CASES:
            with self.subTest(name_part=name_part):
                result, _ = self._load_case(name_part)

                self.assertEqual(result.filter_type, expected_type)
                self.assertEqual(result.order_estimate, expected_order)
                self.assertNotIn("疑似", result.system_order)
                self.assertAlmostEqual(result.omega_c, expected_omega, delta=expected_omega * 0.03)

    def test_default_report_is_filter_focused(self):
        forbidden = ("疑似", "奈奎斯特稳定性", "增益裕度", "相位裕度", "超调量")
        for name_part, expected_type, _, _ in STANDARD_CASES:
            with self.subTest(name_part=name_part):
                _, report = self._load_case(name_part)

                for word in forbidden:
                    self.assertNotIn(word, report)
                if expected_type in ("lowpass", "highpass"):
                    self.assertNotIn("谐振角频率", report)

    def test_phase_display_for_highpass_wraps_cleanly(self):
        result, report = self._load_case("三阶高通")

        displayed = normalize_phase_for_display(result.cutoff_phase_deg, result.filter_type)
        phase_curve = phase_array_for_display(result.phase_deg, result.filter_type)

        self.assertGreater(displayed, 0.0)
        self.assertLess(displayed, 180.0)
        self.assertGreater(float(phase_curve.min()), 0.0)
        self.assertLess(float(phase_curve.max()), 280.0)
        self.assertIn(f"特征频点相位 = {displayed:.2f} °", report)
        self.assertNotIn("-277.61", report)

    def test_first_order_phase_reference_matches_filter_family(self):
        _, lowpass_report = self._load_case("一阶低通")
        _, highpass_report = self._load_case("一阶高通")

        self.assertIn("一阶相位 -45° 参考角频率", lowpass_report)
        self.assertIn("一阶相位 +45° 参考角频率", highpass_report)
        self.assertNotIn("一阶相位 -45° 参考角频率", highpass_report)


if __name__ == "__main__":
    unittest.main()
