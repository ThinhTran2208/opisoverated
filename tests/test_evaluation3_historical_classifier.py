"""Tests for source-aware NB11 historical evaluation."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation.evaluation3_historical_classifier import (
    CONFIRMED_DUPLICATE_SOURCE,
    CURRENT_DOMAIN_SOURCE,
    HARD_REVIEW_SOURCE,
    historical_training_weights,
    split_historical_domain_aware,
)


class Evaluation3HistoricalClassifierTests(unittest.TestCase):
    @staticmethod
    def _frame(group_count: int = 30) -> pd.DataFrame:
        rows = []
        for index in range(group_count):
            group = f"E3-{index:03d}"
            rows.append(
                {
                    "example_id": f"easy-{index}",
                    "group_id": group,
                    "source": CONFIRMED_DUPLICATE_SOURCE,
                    "human_label": "DUPLICATE",
                    "target": 1,
                }
            )
            rows.append(
                {
                    "example_id": f"hard-{index}",
                    "group_id": group,
                    "source": HARD_REVIEW_SOURCE,
                    "human_label": "DUPLICATE" if index % 4 == 0 else "NON_DUPLICATE",
                    "target": 1 if index % 4 == 0 else 0,
                }
            )
        return pd.DataFrame(rows)

    def test_domain_aware_holdouts_are_hard_only_and_group_disjoint(self):
        train, validation, test = split_historical_domain_aware(
            self._frame(), random_state=7
        )
        self.assertTrue(validation["source"].eq(HARD_REVIEW_SOURCE).all())
        self.assertTrue(test["source"].eq(HARD_REVIEW_SOURCE).all())

        train_groups = set(train["group_id"])
        validation_groups = set(validation["group_id"])
        test_groups = set(test["group_id"])
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(validation_groups.isdisjoint(test_groups))

        # Easy positive anchors for reserved E3 images must not leak into train.
        reserved = validation_groups | test_groups
        leaked_anchors = train[
            train["source"].eq(CONFIRMED_DUPLICATE_SOURCE)
            & train["group_id"].isin(reserved)
        ]
        self.assertTrue(leaked_anchors.empty)

    def test_source_weights_prioritize_current_labels(self):
        frame = pd.DataFrame(
            {
                "source": [
                    CONFIRMED_DUPLICATE_SOURCE,
                    HARD_REVIEW_SOURCE,
                    CURRENT_DOMAIN_SOURCE,
                    "unknown",
                ]
            }
        )
        weights = historical_training_weights(frame)
        np.testing.assert_allclose(weights, [0.25, 1.0, 4.0, 1.0])

    def test_domain_split_rejects_single_class_hard_queue(self):
        frame = self._frame()
        frame.loc[frame["source"].eq(HARD_REVIEW_SOURCE), "target"] = 0
        with self.assertRaisesRegex(ValueError, "both classes"):
            split_historical_domain_aware(frame)


if __name__ == "__main__":
    unittest.main()
