"""Tests for the locked Type-aware Pairwise Scorer V1 architecture."""

import unittest

from src.scorer.model import (
    SCORER_VERSION,
    TypeAwarePairwiseScorer,
    torch,
)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in portability CI")
class TypeAwarePairwiseScorerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = {
            "model": {
                "name": "type_aware_pairwise_v1",
                "embedding_dim": 512,
                "category_count": 7,
                "category_vocab_size": 8,
                "category_padding_idx": 0,
                "category_embedding_dim": 32,
                "item_projection_dim": 256,
                "item_hidden_dim": 128,
                "pair_hidden_dim": 128,
                "output_hidden_dim": 16,
                "activation": "relu",
                "dropout": 0.2,
                "aggregation": "mean",
                "pair_symmetry": "bidirectional_mean",
            },
            "data": {
                "min_items": 3,
                "max_items": 8,
            },
        }
        self.model = TypeAwarePairwiseScorer.from_config(self.config)

    def _batch(self):
        embeddings = torch.randn(2, 8, 512)
        categories = torch.tensor(
            [
                [1, 2, 5, 6, 0, 0, 0, 0],
                [1, 4, 2, 5, 7, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        item_mask = categories != 0
        embeddings[~item_mask] = 0.0
        return embeddings, categories, item_mask

    def test_config_builds_locked_dimensions(self):
        self.assertEqual(self.model.scorer_version, SCORER_VERSION)
        self.assertEqual(self.model.embedding_dim, 512)
        self.assertEqual(self.model.category_vocab_size, 8)
        self.assertEqual(self.model.item_hidden_dim, 128)
        self.assertEqual(self.model.pair_feature_dim, 576)

    def test_category_embedding_init_matches_fashionclip_scale(self):
        weights = self.model.category_embedding.weight.detach()
        padding_row = weights[self.model.category_padding_idx]
        self.assertTrue(torch.equal(padding_row, torch.zeros_like(padding_row)))

        expected_std = self.model.category_embedding_dim ** -0.5
        self.assertAlmostEqual(
            self.model.category_embedding_init_std,
            expected_std,
            places=12,
        )

        real_weights = weights[1:]
        empirical_rms = real_weights.square().mean().sqrt().item()
        self.assertLess(
            abs(empirical_rms - expected_std),
            expected_std * 0.25,
        )
        mean_norm = real_weights.norm(dim=1).mean().item()
        self.assertGreater(mean_norm, 0.7)
        self.assertLess(mean_norm, 1.3)

    def test_forward_output_shape_and_finite(self):
        self.model.eval()
        embeddings, categories, item_mask = self._batch()

        output = self.model(
            embeddings,
            categories,
            item_mask,
        )

        self.assertEqual(set(output), {"compatibility_logit"})
        logits = output["compatibility_logit"]
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_padding_embeddings_do_not_change_score(self):
        self.model.eval()
        embeddings, categories, item_mask = self._batch()

        clean_output = self.model(
            embeddings,
            categories,
            item_mask,
        )["compatibility_logit"]

        noisy_padding = embeddings.clone()
        noisy_padding[~item_mask] = torch.randn_like(noisy_padding[~item_mask])

        noisy_output = self.model(
            noisy_padding,
            categories,
            item_mask,
        )["compatibility_logit"]

        self.assertTrue(
            torch.allclose(clean_output, noisy_output, atol=1e-6, rtol=1e-6)
        )

    def test_permutation_invariance(self):
        self.model.eval()
        embeddings, categories, item_mask = self._batch()

        original = self.model(
            embeddings,
            categories,
            item_mask,
        )["compatibility_logit"]

        permutations = torch.tensor(
            [
                [3, 0, 2, 1, 7, 6, 5, 4],
                [4, 2, 0, 1, 3, 7, 6, 5],
            ],
            dtype=torch.long,
        )

        batch_indices = torch.arange(embeddings.shape[0]).unsqueeze(1)
        shuffled_embeddings = embeddings[batch_indices, permutations]
        shuffled_categories = categories[batch_indices, permutations]
        shuffled_mask = item_mask[batch_indices, permutations]

        shuffled = self.model(
            shuffled_embeddings,
            shuffled_categories,
            shuffled_mask,
        )["compatibility_logit"]

        self.assertTrue(
            torch.allclose(original, shuffled, atol=1e-6, rtol=1e-6)
        )

    def test_forward_accepts_canonical_pair_mask_and_rejects_wrong_mask(self):
        self.model.eval()
        embeddings, categories, item_mask = self._batch()
        expected_pair_mask = self.model._expected_pair_mask(item_mask)

        output = self.model(
            embeddings,
            categories,
            item_mask,
            pair_mask=expected_pair_mask,
        )
        self.assertEqual(tuple(output["compatibility_logit"].shape), (2,))

        wrong_pair_mask = expected_pair_mask.clone()
        wrong_pair_mask[0, 0, 1] = False
        with self.assertRaises(ValueError):
            self.model(
                embeddings,
                categories,
                item_mask,
                pair_mask=wrong_pair_mask,
            )

    def test_backward_produces_finite_gradients(self):
        self.model.train()
        embeddings, categories, item_mask = self._batch()

        logits = self.model(
            embeddings,
            categories,
            item_mask,
        )["compatibility_logit"]
        loss = logits.square().mean()
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
