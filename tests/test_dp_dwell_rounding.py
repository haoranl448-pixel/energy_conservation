import ast
from pathlib import Path
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts_new/00_main_pipeline/08_ato_class_globall_v2.py"
)


class DwellRoundingTests(unittest.TestCase):
    def setUp(self):
        # Load only the allocator; importing the CLI configures shared output paths.
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8-sig"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "distribute_historical_dwell"
        )
        self.namespace = {"np": np, "TOTAL_TIME_TOLERANCE": 0.0}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT), "exec"),
            self.namespace,
        )

    def allocate(self, target, run, history, tolerance=0.0):
        self.namespace["T_TOTAL_TARGET"] = target
        return self.namespace["distribute_historical_dwell"](run, history, tolerance)

    def test_failed_trip_half_steps_keep_exact_historical_dwell(self):
        cases = [
            (35, 3814.25, 3023.2, [31.0] * 24 + [47.0]),
            (40, 3813.05, 3025.0, [31.0] * 24 + [44.0]),
            (57, 3735.95, 2968.0, [30.0] * 24 + [48.0]),
        ]
        for trip, target, run, history in cases:
            with self.subTest(trip=trip):
                ok, dwell = self.allocate(target, run, history)
                self.assertTrue(ok)
                self.assertEqual(dwell, history)
                self.assertAlmostEqual(abs(run + sum(dwell) - target), 0.05)

    def test_values_beyond_half_step_remain_infeasible(self):
        for delta in (-0.050001, 0.050001):
            with self.subTest(delta=delta):
                self.assertEqual(
                    self.allocate(3030.0 + delta, 3000.0, [30.0]), (False, None)
                )

    def test_station_bounds_survive_rounding_at_both_ends(self):
        history = [29.0, 30.0, 31.0]
        for tolerance in (0.02, 0.05, -3.0):
            lower = [max(0.0, t + tolerance) if tolerance < 0 else t * (1 - tolerance) for t in history]
            upper = history if tolerance < 0 else [t * (1 + tolerance) for t in history]
            for bounds, delta in ((lower, -0.05), (upper, 0.05)):
                with self.subTest(tolerance=tolerance, delta=delta):
                    ok, dwell = self.allocate(3000 + sum(bounds) + delta, 3000, history, tolerance)
                    self.assertTrue(ok)
                    np.testing.assert_allclose(dwell, bounds, rtol=0, atol=1e-9)
                    self.assertTrue(all(lo - 1e-9 <= d <= hi + 1e-9 for d, lo, hi in zip(dwell, lower, upper)))

    def test_zero_or_no_dwell_half_steps(self):
        for history in ([], [0.0]):
            with self.subTest(history=history):
                ok, dwell = self.allocate(3000.05, 3000, history)
                self.assertTrue(ok)
                self.assertEqual(dwell, history)
                self.assertFalse(self.allocate(3000.050001, 3000, history)[0])

    def test_explicit_total_time_tolerance_keeps_its_limit(self):
        self.namespace["TOTAL_TIME_TOLERANCE"] = 1.0
        self.assertTrue(self.allocate(3031.05, 3000, [30.0])[0])
        self.assertFalse(self.allocate(3031.050001, 3000, [30.0])[0])


if __name__ == "__main__":
    unittest.main()
