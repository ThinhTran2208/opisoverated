# Recommendation V1 — ZIP-direct Hybrid Retrieval

## Scope

Recommendation V1 is a standalone downstream pipeline:

```text
ML_Final ZIP artifacts
  -> problematic item / Evaluation3 swapped index
  -> Top-200 problematic-item cosine
  -> Top-200 context-centroid cosine
  -> union + de-duplicate
  -> exact master_category filter
  -> image + embedding availability filter
  -> frozen V5 scorer reranking
  -> public Top-3
```

There is no Qwen/VLM import or integration in this version. VLM Explanation V1
remains unchanged because no local Qwen checkpoint is available.

## Immutable artifacts

The implementation reads these archives without modifying them:

- `ML_Final-20260903T034319Z-1-001.zip`;
- `images-20260903T034922Z-1-001.zip`;
- `images-20260903T034922Z-1-002.zip`;
- `images-20260903T034922Z-1-003.zip`.

`MLFinalZipBundle` reads individual entries using `zipfile.ZipFile` and loads
PyTorch payloads through `BytesIO`. The relevant entries are:

- `ML_Final/fashionclip_item_embeddings.pt`;
- `ML_Final/embedding_manifest_v1.json`;
- `ML_Final/polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{split}.jsonl`;
- `ML_Final/polyvore_core7_v2/scorer_ready_v2/scorer_ready_v2_{split}.jsonl`;
- `ML_Final/scorer_runs/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt`.

The embedding SHA-256 and frozen V5 checkpoint SHA-256 are verified before use.
The loaded catalog contains 142,480 unique FashionCLIP 512-D, float16-on-disk,
L2-normalized embeddings. Search is performed in float32.

## ZIP image resolver

`ZipImageResolver` scans only ZIP central directories during initialization and
caches:

```text
item_id -> archive path + internal image path + byte size
```

Validation result:

- ZIP 001: 65,428 JPG files;
- ZIP 002: 64,955 JPG files;
- ZIP 003: 12,097 JPG files;
- total: 142,480;
- duplicate `item_id`: 0;
- image IDs equal embedding IDs exactly: yes.

The stable first mapping is:

```text
item_id:       100002074_1
archive:       images-20260903T034922Z-1-001.zip
internal path: images/100002074_1.jpg
```

Each request opens only the selected archive entry. No image archive is fully
extracted. `create_image_router()` exposes:

```text
GET /recommendation/images/{item_id}
```

The demo materializes exactly the three selected recommendation images.

## Retrieval and reranking

The context embedding is the L2-normalized centroid of all outfit items except
the problematic item. Recommendation V1 retrieves exact cosine Top-200 from
the problematic-item query and Top-200 from the context query. It then unions
and de-duplicates the two lists.

Filtering is strict and happens after retrieval:

1. candidate metadata must exist;
2. candidate `master_category` must equal the problematic item's exact
   `master_category`;
3. candidate embedding must exist;
4. candidate image must exist;
5. items already in the outfit remain excluded.

Every surviving candidate replaces the problematic position in the complete
outfit and is scored in batches by the frozen V5 scorer. Raw cosine values,
compatibility logits and logit improvements are internal metadata only.

## Public output

Successful output contains exactly three recommendations:

```json
{
  "status": "ok",
  "recommendation_version": "hybrid-retrieval-v1",
  "items": [
    {
      "item_id": "213383941_4",
      "rank": 1,
      "image_url": "/recommendation/images/213383941_4",
      "master_category": "Boots",
      "coarse_category": "SHOES"
    }
  ]
}
```

The public serializer never returns `compatibility_score`,
`compatibility_logit`, `score_uplift`, cosine similarity, or any equivalent
ranking score.

## Evaluation3

The ZIP entry `ML_Final/evaluation3/phash_ssim_threshold/` is empty. Therefore
the executable Evaluation3 protocol is defined from the packaged
`scorer_ready_v2_test.jsonl` one-item-swap pairs:

- query outfit: negative outfit;
- problematic position: known `negative_metadata.swapped_item_index`;
- relevant item: `negative_metadata.original_item_id`;
- purpose: isolate recommendation retrieval/reranking from LOO localization;
- split: full test split, 2,327 negatives;
- skipped rows: 0.

With one relevant item per query, Recall@K equals retrieval Hit@K. MRR is used
for final-rank quality because it remains informative beyond the public Top-3.

Full Evaluation3 result:

| Metric | Value |
|---|---:|
| Recall@50 | 0.039965620971207566 |
| Recall@100 | 0.05801461108723679 |
| Recall@200 | 0.08465835840137516 |
| Hit@1 | 0.012462397937258273 |
| Hit@3 | 0.024065320154705628 |
| MRR | 0.022436943771935322 |

The low retrieval recall is a measured limitation of applying exact
`master_category` filtering after global Top-200 retrieval. It is not hidden by
the reranker metric.

## Demo command

```powershell
.\.venv\Scripts\python.exe -m src.recommendation.demo `
  --ml-zip D:\BKU\VSC\ML_Final-20260903T034319Z-1-001.zip `
  --image-zip D:\BKU\VSC\images-20260903T034922Z-1-001.zip `
  --image-zip D:\BKU\VSC\images-20260903T034922Z-1-002.zip `
  --image-zip D:\BKU\VSC\images-20260903T034922Z-1-003.zip `
  --output-dir D:\BKU\VSC\opisoverated\outputs\recommendation_v1_demo `
  --evaluation-split test `
  --evaluation-max-samples 0
```

`--evaluation-max-samples 0` means the full split. Demo outputs:

- `outputs/recommendation_v1_demo/demo.json`;
- `outputs/recommendation_v1_demo/index.html`;
- exactly three `rank_{rank}_{item_id}.jpg` files.

## Limitations

- Evaluation3 does not test LOO localization; it deliberately uses the known
  swapped index to isolate recommendation quality.
- Global cosine Top-200 followed by exact master-category filtering has low
  ground-truth recall; future versions can compare category-aware retrieval,
  but that would be a separately versioned protocol.
- The image endpoint router must be attached to the product FastAPI app by the
  deployment composition layer.
- ZIP access avoids extraction but startup still loads the 142,480-row
  embedding tensor and indexes all image filenames in memory.
- Qwen/VLM Explanation is intentionally omitted.

