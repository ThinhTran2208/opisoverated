from src.scorer.paired_ranking_cv import (
    FamilySubset,
    build_grouped_family_folds,
    select_lambda_by_mean_auc,
    summarize_lambda_cv,
)


class DummyDataset:
    def __init__(self):
        # Six complete families, with two families sharing source kit A.
        self.pair_families = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
        ]
        sources = ["A", "A", "B", "C", "D", "E"]
        self.records = []
        for family_index, source in enumerate(sources):
            positive_id = f"p{family_index}"
            self.records.append(
                {
                    "sample_id": positive_id,
                    "source_kit_id": source,
                    "label": 1,
                    "paired_positive_sample_id": None,
                }
            )
            self.records.append(
                {
                    "sample_id": f"n{family_index}",
                    "source_kit_id": source,
                    "label": 0,
                    "paired_positive_sample_id": positive_id,
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def test_family_subset_preserves_complete_pairs():
    dataset = DummyDataset()
    subset = FamilySubset(dataset, [1, 4])
    assert len(subset) == 4
    assert subset.pair_families == [(0, 1), (2, 3)]
    assert subset[0]["sample_id"] == "p1"
    assert subset[1]["sample_id"] == "n1"
    assert subset[2]["sample_id"] == "p4"
    assert subset[3]["sample_id"] == "n4"


def test_grouped_folds_keep_source_kits_and_families_disjoint():
    dataset = DummyDataset()
    folds = build_grouped_family_folds(dataset, n_splits=3, split_seed=123)
    assert len(folds) == 3

    seen_valid = set()
    for fold in folds:
        train = set(fold["train_family_positions"])
        valid = set(fold["valid_family_positions"])
        assert train.isdisjoint(valid)
        assert not seen_valid.intersection(valid)
        seen_valid.update(valid)

        train_sources = {
            dataset.records[dataset.pair_families[pos][0]]["source_kit_id"]
            for pos in train
        }
        valid_sources = {
            dataset.records[dataset.pair_families[pos][0]]["source_kit_id"]
            for pos in valid
        }
        assert train_sources.isdisjoint(valid_sources)

    assert seen_valid == set(range(len(dataset.pair_families)))


def test_cv_summary_and_selection_use_mean_auc():
    rows = [
        {"ranking_weight": 0.1, "valid_roc_auc": 0.69, "valid_fitb_2way": 0.75, "mean_logit_margin": 0.8},
        {"ranking_weight": 0.1, "valid_roc_auc": 0.70, "valid_fitb_2way": 0.76, "mean_logit_margin": 0.9},
        {"ranking_weight": 0.5, "valid_roc_auc": 0.71, "valid_fitb_2way": 0.74, "mean_logit_margin": 1.0},
        {"ranking_weight": 0.5, "valid_roc_auc": 0.72, "valid_fitb_2way": 0.75, "mean_logit_margin": 1.1},
    ]
    summary = summarize_lambda_cv(rows)
    winner = select_lambda_by_mean_auc(summary)
    assert winner["ranking_weight"] == 0.5
    assert winner["mean_roc_auc"] == 0.715
