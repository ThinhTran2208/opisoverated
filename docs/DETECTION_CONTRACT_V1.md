# DETECTION CONTRACT V1 — RF-DETR + FashionCLIP Core-7

**Version:** `rfdetr-fashionclip-core7-v1`  
**Status:** experimental feature branch  
**Goal:** convert one user outfit image into garment crops, frozen FashionCLIP embeddings, and canonical Core-7 category IDs that can be handed to the compatibility scorer.

## 1. Boundary of responsibility

Detection V1 does **not** infer the Polyvore `master_category` for a user image.

Training data and user-image inference use different sources of category information:

```text
TRAINING
Polyvore master_category
        ↓
category_mapping_core7_v2
        ↓
coarse_category

INFERENCE
user image
   ↓
RF-DETR garment boxes
   ↓
garment crops
   ↓
Frozen FashionCLIP image embedding
   ↓ cosine similarity against seven text prototypes
coarse_category
```

The two paths meet only at the canonical `coarse_category` taxonomy:

```text
TOP, BOTTOM, DRESS, OUTERWEAR, SHOES, BAG, HAT
```

A detector label such as `jacket` or `shoe` is **diagnostic metadata only**. It is not converted by a hard label map into the scorer category. This avoids coupling inference to a detector-specific taxonomy.

## 2. Detector

Detection V1 follows the RF-DETR setup explored in `fashion_detector_comparison.ipynb`:

```text
model repository: resoa/garment-detector-seg
architecture: RFDETRSegSmall
checkpoint selection: prefer a *.pth containing "best_ema"
default threshold: 0.35
```

The checkpoint is Fashionpedia-based and can emit garments, accessories, and garment parts. Only object types that can plausibly become Core-7 scorer items are allowed to proceed to crop classification:

```text
shirt, blouse
top, t-shirt, sweatshirt
sweater
cardigan
jacket
vest
pants
shorts
skirt
coat
dress
jumpsuit
cape
hat
shoe
bag, wallet
```

Other Fashionpedia outputs such as collar, sleeve, zipper, belt, scarf, watch, bead, or sequin are rejected before FashionCLIP classification.

This allow-list is a **candidate filter**, not a category mapping.

## 3. Crop contract

Each accepted RF-DETR box is expanded by a small configurable ratio and clamped to the source image. V1 defaults:

```text
crop_padding_ratio = 0.03
minimum crop side = 24 px
```

Crops smaller than the minimum are rejected and logged.

## 4. FashionCLIP embedding

Every accepted crop is encoded exactly once with the scorer-compatible frozen model:

```text
patrickjohncyh/fashion-clip
```

Required embedding contract:

```text
projected image embedding
shape = [512]
L2-normalized
finite values only
```

The same 512-d normalized image vector is reused for two purposes:

1. zero-shot Core-7 category classification;
2. compatibility scorer item embedding.

Detection must not run a second image encoder to produce scorer inputs.

## 5. Direct Core-7 classification

Detection V1 intentionally does not predict hundreds of source `master_category` values and then collapse them.

For each canonical category, FashionCLIP text embeddings are built from a small prompt ensemble. Each prompt vector is L2-normalized, vectors in the same category are averaged, and the resulting category prototype is L2-normalized.

For a garment image embedding `v` and category prototype `t_c`:

```text
score(c) = cosine(v, t_c)
predicted_coarse_category = argmax_c score(c)
```

The output records:

```text
coarse_category
coarse_category_id
category_similarity        # top-1 cosine
category_margin            # top-1 minus top-2
category_similarities      # all seven cosine values
coarse_category_source = fashionclip-zero-shot-core7-v1
```

V1 does not hard-code an unvalidated rejection threshold for FashionCLIP similarity. `min_similarity` is disabled (`null`) and `min_margin` is `0.0` by default. Both are versioned config fields and may be tuned after a labeled validation set exists.

## 6. Canonical category IDs

Detection must match the scorer vocabulary exactly:

| coarse_category | id |
|---|---:|
| `TOP` | 1 |
| `BOTTOM` | 2 |
| `DRESS` | 3 |
| `OUTERWEAR` | 4 |
| `SHOES` | 5 |
| `BAG` | 6 |
| `HAT` | 7 |
| padding | 0 |

The category mapping/taxonomy version is `core7-v2`.

## 7. Per-garment metadata

A garment detection result contains at least:

```json
{
  "detection_index": 0,
  "box_xyxy": [100.2, 40.1, 340.5, 300.8],
  "crop_box_xyxy": [93, 32, 348, 309],
  "detector_label": "shirt, blouse",
  "detector_confidence": 0.94,
  "coarse_category": "TOP",
  "coarse_category_id": 1,
  "coarse_category_source": "fashionclip-zero-shot-core7-v1",
  "category_similarity": 0.31,
  "category_margin": 0.05,
  "embedding_dimension": 512,
  "embedding_normalization": "l2"
}
```

`master_category` is deliberately absent from each inferred garment. At the result-taxonomy level it is explicitly documented as `null / not_inferred_for_user_images`.

## 8. Scorer handoff

For `N` accepted garments, the tensor handoff is:

```text
item_embeddings      FloatTensor [1, N, 512]
coarse_category_ids  LongTensor  [1, N]
item_mask            BoolTensor  [1, N]
```

Detection does not silently remove items just to satisfy scorer length. If `N` is outside the downstream scorer range, metadata/crops are still valid, but scorer handoff fails with an explicit message.

Current main-branch scorer contract used by this detection config:

```text
minimum = 3 items
maximum = 8 items
```

If the canonical scorer minimum changes, update the versioned detection config rather than hiding the mismatch in code.

## 9. Output files

The CLI writes under the selected output directory:

```text
detection_result.json
crops/
  garment_00_top.jpg
  ...
scorer_inputs.pt       # only when garment count satisfies scorer contract
```

Large runtime outputs and checkpoints remain outside Git.

## 10. Failure conditions

Detection/scorer handoff must surface rather than hide these cases:

- detector returns no supported garments;
- FashionCLIP embedding is not 512-d;
- FashionCLIP embedding contains NaN/Inf;
- crop is too small;
- category config does not contain exactly the seven canonical classes;
- accepted garment count violates scorer min/max;
- detector/runtime dependencies or checkpoint are unavailable.

## 11. Validation still required

Implementation completion is not the same as detector quality validation. Before treating Detection V1 as production-ready, create a manually reviewed image set and report at least:

- garment box recall/precision for scorer-relevant objects;
- Core-7 classification accuracy;
- per-class confusion matrix;
- distribution of top-1 cosine and top1-top2 margin;
- end-to-end percentage of images that produce a scorer-valid number of garments.

Those measurements should determine whether `0.35`, prompt wording, `min_similarity`, or `min_margin` need a V2 change.
