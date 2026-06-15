import unittest

import numpy as np

from calibration import apply_calibration, build_calibration_profile, profile_summary
from filter_analysis import analyze_system_v2


class CalibrationTests(unittest.TestCase):
    def test_reference_calibration_removes_gain_and_phase_error(self):
        omega = 2.0 * np.pi * np.geomspace(100.0, 10000.0, 24)
        reference_mag = 0.82 + 0.06 * np.log10(omega / omega[0])
        reference_phase = 0.12 + 0.04 * np.sin(np.log(omega))
        dut_mag = 1.0 / np.sqrt(1.0 + (omega / (2.0 * np.pi * 1600.0)) ** 2)
        dut_phase = -np.arctan(omega / (2.0 * np.pi * 1600.0))

        profile = build_calibration_profile(omega, reference_mag, reference_phase, source="loopback")
        corrected = apply_calibration(omega, dut_mag * reference_mag, dut_phase + reference_phase, profile)

        np.testing.assert_allclose(corrected.magnitude, dut_mag, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(corrected.phase_rad, dut_phase, rtol=1e-10, atol=1e-12)
        self.assertIn("loopback", profile_summary(profile))

    def test_calibrated_lowpass_still_identifies_as_first_order_lowpass(self):
        omega = 2.0 * np.pi * np.geomspace(100.0, 60000.0, 160)
        cutoff = 2.0 * np.pi * 3000.0
        reference_mag = 0.78 * np.sqrt(1.0 + (omega / (2.0 * np.pi * 90000.0)) ** 2)
        reference_phase = 0.16 + 0.05 * np.log10(omega / omega[0])
        lowpass_mag = 1.0 / np.sqrt(1.0 + (omega / cutoff) ** 2)
        lowpass_phase = -np.arctan(omega / cutoff)

        profile = build_calibration_profile(omega, reference_mag, reference_phase, source="direct reference")
        corrected = apply_calibration(
            omega,
            lowpass_mag * reference_mag,
            lowpass_phase + reference_phase,
            profile,
        )
        result = analyze_system_v2(corrected.omega, corrected.magnitude, corrected.phase_rad)

        self.assertEqual(result.filter_type, "lowpass")
        self.assertEqual(result.order_estimate, 1)
        self.assertAlmostEqual(result.magnitude_cutoff_omega, cutoff, delta=cutoff * 0.08)


if __name__ == "__main__":
    unittest.main()
