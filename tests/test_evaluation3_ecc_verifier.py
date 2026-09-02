"""Synthetic tests for the experimental guarded-ECC EVALUATION3 verifier."""

import unittest
from io import BytesIO

from PIL import Image, ImageDraw, ImageEnhance

from src.evaluation.evaluation3_ecc_verifier import (
    DUPLICATE,
    MANUAL_REVIEW,
    aligned_pair_evidence,
    classify_aligned_pair,
)
from src.evaluation.evaluation3_rgb_verifier import normalize_rgb_for_verification


class Evaluation3EccVerifierTests(unittest.TestCase):
    @staticmethod
    def _catalog_image(
        *,
        canvas=(300, 300),
        box=(80, 50, 220, 250),
        fill=(40, 70, 120),
        local_detail=False,
    ):
        image = Image.new("RGB", canvas, "white")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(box, radius=10, fill=fill)
        draw.line(
            (box[0] + 10, box[1] + 30, box[2] - 10, box[1] + 30),
            fill=(185, 185, 185),
            width=2,
        )
        if local_detail:
            draw.rectangle(
                (box[0] + 20, box[1] + 45, box[0] + 55, box[1] + 75),
                outline=(185, 185, 185),
                width=4,
            )
        return image

    @staticmethod
    def _verify(left, right):
        left_array = normalize_rgb_for_verification(left, size=256)
        right_array = normalize_rgb_for_verification(right, size=256)
        evidence = aligned_pair_evidence(left_array, right_array)
        decision, method = classify_aligned_pair(evidence)
        return evidence, decision, method

    def test_jpeg_and_small_brightness_nuisance_can_auto_duplicate(self):
        original = self._catalog_image()
        buffer = BytesIO()
        original.save(buffer, format="JPEG", quality=55)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        recompressed = ImageEnhance.Brightness(recompressed).enhance(0.97)

        evidence, decision, method = self._verify(original, recompressed)

        self.assertTrue(evidence["alignment_success"])
        self.assertEqual(decision, DUPLICATE)
        self.assertEqual(method, "phash_ecc_consensus_auto")

    def test_canvas_scale_and_position_nuisance_can_auto_duplicate(self):
        left = self._catalog_image(
            canvas=(300, 300),
            box=(80, 50, 220, 250),
        )
        # Same aspect ratio/content, but a much larger source canvas/object.
        right = self._catalog_image(
            canvas=(420, 360),
            box=(115, 55, 304, 325),
        )

        evidence, decision, _ = self._verify(left, right)

        self.assertTrue(evidence["alignment_success"])
        self.assertGreater(evidence["rgb_ssim"], 0.90)
        self.assertEqual(decision, DUPLICATE)

    def test_same_shape_different_color_is_not_auto_duplicate(self):
        red = self._catalog_image(fill=(175, 35, 45))
        blue = self._catalog_image(fill=(35, 70, 175))

        evidence, decision, _ = self._verify(red, blue)

        self.assertGreater(evidence["mean_lab_delta"], 8.0)
        self.assertNotEqual(decision, DUPLICATE)
        self.assertEqual(decision, MANUAL_REVIEW)

    def test_small_local_detail_change_is_not_erased_by_alignment(self):
        plain = self._catalog_image(fill=(30, 30, 30))
        detailed = self._catalog_image(fill=(30, 30, 30), local_detail=True)

        evidence, decision, _ = self._verify(plain, detailed)

        # Global SSIM can remain deceptively high, but the worst interior patch
        # should expose the local change and veto automatic DUPLICATE.
        self.assertGreater(evidence["rgb_ssim"], 0.90)
        self.assertGreater(evidence["patch_mae_max"], 0.12)
        self.assertNotEqual(decision, DUPLICATE)
        self.assertEqual(decision, MANUAL_REVIEW)


if __name__ == "__main__":
    unittest.main()
