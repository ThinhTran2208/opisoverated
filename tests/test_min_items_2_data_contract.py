"""Pure-Python contract tests for min-items-2 data processing wrappers."""

import unittest
from pathlib import Path

from src.data.prepare_core7_dataset import DEFAULT_MIN_OUTFIT_ITEMS
from src.data.build_core7_scorer_dataset_min2 import (
    EXPERIMENT_ID,
    MIN_LOO_ORIGINAL_ITEMS,
    MIN_SCORER_ITEMS,
    _require_isolated_output_dir,
)


class MinItems2DataContractTests(unittest.TestCase):
    def test_core7_cleaning_default_is_two(self):
        self.assertEqual(DEFAULT_MIN_OUTFIT_ITEMS, 2)
        self.assertEqual(MIN_SCORER_ITEMS, 2)
        self.assertEqual(MIN_LOO_ORIGINAL_ITEMS, 3)
        self.assertEqual(EXPERIMENT_ID, "min-items-2-loo-min3-v1")

    def test_experiment_rejects_canonical_scorer_ready_directory(self):
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            _require_isolated_output_dir(Path("/tmp/scorer_ready_v2"))

    def test_experiment_accepts_isolated_output_directory(self):
        _require_isolated_output_dir(Path("/tmp/scorer_ready_v2_min2_exp"))


if __name__ == "__main__":
    unittest.main()
