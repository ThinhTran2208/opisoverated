"""Synthetic tests for the experimental EVALUATION3 RGB verifier."""

import unittest

from PIL import Image, ImageDraw

from src.evaluation.evaluation3_rgb_verifier import (
    normalize_rgb_for_verification,
    pair_evidence,
    rgb_ssim_score,
)


class Evaluation3RgbVerifierTests(unittest.TestCase):
    @staticmethod
    def _catalog_image(
        *,
        canvas=(220, 180),
        box=(60, 35, 155, 150),
        fill=(35, 85, 170),
    ):
        image = Image.new("RGB", canvas, "white")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(box, radius=8, fill=fill)
        draw.line(
            (box[0] + 12, box[1] + 18, box[2] - 12, box[1] + 18),
            fill=(245, 245, 245),
            width=3,
        )
        return image

    def test_normalization_removes_canvas_scale_and_position_nuisance(self):
        left = self._catalog_image(
            canvas=(220, 180), box=(60, 35, 155, 150)
        )
        right = self._catalog_image(
            canvas=(360, 300), box=(110, 55, 275, 255)
        )

        left_array = normalize_rgb_for_verification(left, size=256)
        right_array = normalize_rgb_for_verification(right, size=256)

        self.assertEqual(left_array.shape, (256, 256, 3))
        self.assertEqual(right_array.shape, (256, 256, 3))
        self.assertGreater(rgb_ssim_score(left_array, right_array), 0.90)

    def test_color_change_is_visible_to_rgb_evidence(self):
        blue = self._catalog_image(fill=(30, 75, 170))
        red = self._catalog_image(fill=(175, 45, 50))

        blue_array = normalize_rgb_for_verification(blue, size=256)
        red_array = normalize_rgb_for_verification(red, size=256)
        evidence = pair_evidence(blue_array, red_array)

        self.assertGreater(
            evidence["grayscale_ssim"], evidence["rgb_ssim"]
        )
        self.assertGreater(evidence["mean_lab_delta"], 10.0)

    def test_identical_images_score_one(self):
        image = self._catalog_image()
        left = normalize_rgb_for_verification(image, size=256)
        right = normalize_rgb_for_verification(image.copy(), size=256)
        self.assertAlmostEqual(rgb_ssim_score(left, right), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
