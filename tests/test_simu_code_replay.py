"""Numerical and data-integrity checks for the no-OpenTrack replay plotter."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_new/07_visualization"))
import plot_simu_code_trip_compare as replay


class ReplayTests(unittest.TestCase):
    def test_full_physics_loop_parity(self):
        self.assertLess(max(replay.validate_fast_physics()), 1e-10)

    def test_resistance_at_map_boundaries(self):
        loads = np.array([38, 40, 42, 41, 39, 37], dtype=float)
        for reference, fast in ((replay.CurrentPhysics(), replay.FastCurrent()),
                                (replay.LegacyPhysics(), replay.FastLegacy())):
            positions = [-10, 0, 122, 122.001, 267, 1428, 29102, 32778, 36502]
            for position in positions:
                for method in ("_calc_gradient_distributed", "_calc_curve_distributed"):
                    a = getattr(reference, method)(position, loads)
                    b = getattr(fast, method)(position, loads)
                    self.assertAlmostEqual(a, b, delta=1e-8)

    def test_distance_display_only_and_reversal_guard(self):
        raw = np.array([0., -.0002, .001, 10.])
        before = raw.copy()
        shown = replay.line_distance(raw, 2)
        np.testing.assert_array_equal(raw, before)
        self.assertTrue(np.all(np.diff(shown) >= 0))
        self.assertEqual(shown[-1], replay.ENDS[2]/1000)
        with self.assertRaises(ValueError):
            replay.line_distance([0, -1, 10], 0)

    def test_first_energy_sample_is_preserved(self):
        result = dict(distance=np.array([0., 1.]), velocity=np.array([0., 2.]), simu_wh=np.array([1., 3.]))
        points = replay.points_frame(0, result, np.array([2., 4.]), 1, "history")
        self.assertEqual(len(points), 3)
        self.assertEqual(points.code_section_kwh.iloc[0], 0)
        self.assertEqual(points.code_section_kwh.iloc[-1], .006)
        self.assertEqual(points.simu_section_kwh.iloc[-1], .004)

    def test_percentage_sign_and_zero_denominator(self):
        self.assertEqual(replay.percentage(-5, 100), -5)
        self.assertTrue(np.isnan(replay.percentage(1, 0)))


if __name__ == "__main__":
    unittest.main()
