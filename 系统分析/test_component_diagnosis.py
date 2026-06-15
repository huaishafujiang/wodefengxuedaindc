import unittest

import numpy as np

from component_diagnosis import (
    diagnose_component_deviation,
    parse_component_value,
    profile_from_inputs,
    response_complex,
)


def _omega() -> np.ndarray:
    return 2.0 * np.pi * np.geomspace(40.0, 120000.0, 220)


class ComponentDiagnosisTests(unittest.TestCase):
    def test_parse_component_units(self):
        self.assertAlmostEqual(parse_component_value("10k"), 10000.0)
        self.assertAlmostEqual(parse_component_value("1k"), 1000.0)
        self.assertAlmostEqual(parse_component_value("10n"), 10e-9)
        self.assertAlmostEqual(parse_component_value("100n"), 100e-9)
        self.assertAlmostEqual(parse_component_value("4.7u"), 4.7e-6)
        self.assertAlmostEqual(parse_component_value("1M"), 1.0e6)

    def test_first_order_lowpass_detects_larger_capacitor(self):
        omega = _omega()
        nominal = profile_from_inputs("一阶低通", "10k", "10n", enabled=True, calibrated=True)
        actual = profile_from_inputs("一阶低通", "10k", "15n", enabled=True, calibrated=True)
        magnitude = np.abs(response_complex(omega / (2.0 * np.pi), actual))

        report = diagnose_component_deviation(omega, magnitude, np.angle(response_complex(omega / (2.0 * np.pi), actual)), nominal)

        self.assertTrue(report.candidates)
        self.assertEqual(report.candidates[0].component, "C1")
        self.assertAlmostEqual(report.candidates[0].fitted_value, 15e-9, delta=1.5e-9)

    def test_first_order_highpass_detects_smaller_resistor(self):
        omega = _omega()
        nominal = profile_from_inputs("一阶高通", "10k", "10n", enabled=True, calibrated=True)
        actual = profile_from_inputs("一阶高通", "6.8k", "10n", enabled=True, calibrated=True)
        h = response_complex(omega / (2.0 * np.pi), actual)

        report = diagnose_component_deviation(omega, np.abs(h), np.angle(h), nominal)

        self.assertTrue(report.candidates)
        self.assertEqual(report.candidates[0].component, "R1")
        self.assertAlmostEqual(report.candidates[0].fitted_value, 6800.0, delta=800.0)

    def test_second_order_lowpass_detects_second_stage_resistor(self):
        omega = _omega()
        nominal = profile_from_inputs("二阶低通", "10k,10k", "10n,10n", enabled=True, calibrated=True)
        actual = profile_from_inputs("二阶低通", "10k,15k", "10n,10n", enabled=True, calibrated=True)
        h = response_complex(omega / (2.0 * np.pi), actual)

        report = diagnose_component_deviation(omega, np.abs(h), np.angle(h), nominal)

        self.assertTrue(report.candidates)
        self.assertEqual(report.candidates[0].component, "R2")
        self.assertAlmostEqual(report.candidates[0].fitted_value, 15000.0, delta=1800.0)

    def test_nominal_data_does_not_report_significant_deviation(self):
        omega = _omega()
        nominal = profile_from_inputs("一阶低通", "10k", "10n", enabled=True, calibrated=True)
        h = response_complex(omega / (2.0 * np.pi), nominal)

        report = diagnose_component_deviation(omega, np.abs(h), np.angle(h), nominal)

        self.assertFalse(report.has_significant_deviation)
        self.assertFalse(report.candidates)


if __name__ == "__main__":
    unittest.main()
