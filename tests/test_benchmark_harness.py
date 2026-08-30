# -*- coding: utf-8 -*-

import unittest

from src.inference.benchmark import _percentile, _summary


class BenchmarkHarnessTests(unittest.TestCase):
    def test_percentiles_and_summary_are_deterministic(self):
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(_percentile(values, 0.0), 10.0)
        self.assertEqual(_percentile(values, 1.0), 40.0)
        self.assertAlmostEqual(_percentile(values, 0.5), 25.0)
        summary = _summary(values)
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 25.0)
        self.assertAlmostEqual(summary["p50"], 25.0)
        self.assertGreaterEqual(summary["p95"], summary["p50"])
        self.assertGreaterEqual(summary["p99"], summary["p95"])

    def test_empty_summary_is_explicit(self):
        summary = _summary([])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["p50"])
        self.assertIsNone(summary["p95"])
        self.assertIsNone(summary["p99"])


if __name__ == "__main__":
    unittest.main()
