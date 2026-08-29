# Deploy Handoff V1 — Calibration + Production Inference

## Purpose

This document is the handoff contract between the ML pipeline and the web/backend deployment layer.

Production Inference V1 freezes the deploy-facing boundary around the components that are already stable:

```text
precomputed garments
(512-d L2 FashionCLIP embedding + Core-7 category)
        ↓
canonical frozen scorer V5
        ↓
compatibility_logit
        ├── Calibration V1 → compatibility_score [0,100]
        └── LOO Diagnosis V1 → problematic item evidence
        ↓
structured JSON response
        └── optional ExplanationProvider / VLM
```

Detection + crop + FashionCLIP image preprocessing is intentionally an adapter boundary until the detection runtime is finalized. VLM is also an adapter over structured evidence, not a replacement for the scorer.

## Versions frozen by this handoff

```text
pipeline_version          outfit-production-inference-v1
scorer_version            type_aware_pairwise_v1
calibration_version       platt-logistic-v1
category_mapping_version  core7-v2
embedding_version         fashionclip-512-l2-v1
LOO protocol              loo-diagnostic-v1
```

Production manifest:

`configs/production_inference_v1.json`

Canonical scorer checkpoint:

`artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt`

Calibration artifact:

`artifacts/calibration/type_aware_pairwise_v1/platt_logistic_v1.json`

The production loader verifies the checkpoint SHA-256 from the manifest before loading it.

## Stable Python API

```python
from pathlib import Path
from src.inference import ProductionInferencePipeline

repo_root = Path(".").resolve()
pipeline = ProductionInferencePipeline.load_from_manifest(
    repo_root / "configs" / "production_inference_v1.json",
    repo_root=repo_root,
    device="cpu",  # or "cuda" on a compatible runtime
)

result = pipeline.analyze_precomputed(items)
```

`items` must contain 3–8 garments. Every garment must contain:

```json
{
  "item_id": "garment-0",
  "embedding": "512 floating-point values, L2 norm ~= 1",
  "coarse_category_id": 1
}
```

Optional metadata preserved in the response:

```text
coarse_category
master_category
detection_label
detection_confidence
bbox
```

Raw embeddings are never echoed in the response.

## Runtime input contract

The production boundary validates these invariants before calling the scorer:

- outfit contains 3–8 garments;
- no silent truncation when more than 8 garments are supplied;
- every embedding has exactly 512 values;
- every embedding is finite;
- every embedding is L2-normalized within tolerance `1e-3`;
- every `coarse_category_id` is an integer in `1..7`;
- item IDs are unique within the outfit.

Current Core-7 IDs are the scorer contract IDs:

```text
1 TOP
2 BOTTOM
3 DRESS
4 OUTERWEAR
5 SHOES
6 BAG
7 HAT
0 PAD (never valid for a real garment)
```

The future detection/FashionCLIP adapter must satisfy this contract. It must not invent a separate category-ID convention.

## Product output contract

Successful response shape:

```json
{
  "status": "ok",
  "item_count": 3,
  "items": [
    {
      "item_index": 0,
      "item_id": "garment-0",
      "coarse_category_id": 1,
      "coarse_category": "TOP",
      "bbox": [100, 50, 400, 500],
      "detection_confidence": 0.94
    }
  ],
  "compatibility": {
    "compatibility_logit": 0.81,
    "compatibility_score": 55,
    "scorer_version": "type_aware_pairwise_v1",
    "calibration_version": "platt-logistic-v1"
  },
  "diagnosis": {
    "protocol_version": "loo-diagnostic-v1",
    "problematic_item_index": 2,
    "problematic_item_id": "garment-2",
    "ranked_item_indices": [2, 0, 1],
    "deltas_without_minus_full": [0.1, -0.2, 0.5],
    "uses_two_item_extrapolation": true
  },
  "versions": {
    "pipeline_version": "outfit-production-inference-v1",
    "scorer_version": "type_aware_pairwise_v1",
    "calibration_version": "platt-logistic-v1",
    "category_mapping_version": "core7-v2",
    "embedding_version": "fashionclip-512-l2-v1"
  }
}
```

The example numeric values above illustrate schema only; they are not a golden model output.

### Interpretation rule

`compatibility_score = 55` means the versioned calibration layer mapped the scorer output to a score of 55/100. It is not an objective “55% beautiful” probability and must not be presented as such in UI copy.

## Structured runtime errors

Expected user/input failures return a stable code instead of an internal model crash.

Examples:

```text
invalid_items
insufficient_garments
too_many_garments
missing_embedding
invalid_embedding_shape
non_finite_embedding
embedding_not_l2_normalized
missing_category
invalid_category
duplicate_item_id
image_preprocessor_unavailable
```

Example:

```json
{
  "status": "error",
  "error": {
    "code": "insufficient_garments",
    "message": "At least 3 garments are required",
    "details": {
      "detected_count": 2,
      "minimum_required": 3
    }
  },
  "versions": {}
}
```

Frontend/backend should branch on `error.code`, not parse the English message.

## HTTP service included in this branch

Runtime module:

`src.inference.http_api:app`

Endpoints:

```text
GET  /healthz
POST /v1/analyze-precomputed
```

`POST /v1/analyze-precomputed` accepts:

```json
{
  "items": [
    {
      "item_id": "garment-0",
      "embedding": ["512 numbers"],
      "coarse_category_id": 1
    }
  ]
}
```

Expected input errors return HTTP 422 with the structured error body above.

The final image endpoint should be added only after the detection/FashionCLIP adapter satisfies the same input contract.

## Docker handoff

Build:

```bash
docker build -t outfit-inference-v1 .
```

Run CPU service:

```bash
docker run --rm -p 8000:8000 outfit-inference-v1
```

Health check:

```bash
curl --fail http://127.0.0.1:8000/healthz
```

The default container uses:

```text
FASHION_INFERENCE_DEVICE=cpu
```

To use CUDA, the deploy environment must provide a compatible CUDA/PyTorch runtime and launch the container with the appropriate GPU access, then set:

```text
FASHION_INFERENCE_DEVICE=cuda
```

Detection/VLM dependencies are deliberately not included in this base image yet because those runtime implementations are still being finalized.

## Runtime environment variables

```text
FASHION_INFERENCE_MANIFEST
    optional path to a production inference manifest;
    defaults to configs/production_inference_v1.json

FASHION_INFERENCE_DEVICE
    scorer device;
    defaults to cpu
```

## Calibration V1

Calibration is post-hoc and does not alter scorer weights.

```text
p = sigmoid(scale * compatibility_logit + bias)
compatibility_score = round(100 * p)
```

Frozen parameters:

```text
scale = 0.47217959118640485
bias  = -0.17733224823027438
```

`scale > 0`, so calibration preserves scorer ranking.

Full evidence and calibration-only holdout sanity checks are documented in:

`docs/CALIBRATION_V1_RESULTS.md`

Reproduction CLI:

```bash
python -m src.calibration.fit \
  --checkpoint <best.pt> \
  --samples <scorer_ready_v2_valid.jsonl> \
  --metadata <core7_item_metadata_v1_valid.jsonl> \
  --embedding-cache <fashionclip_item_embeddings.pt> \
  --output <calibration.json>
```

The fitter loads validation only. Test must not be used to fit calibration.

## E2E and Docker CI gates

Unit/portability CI remains unchanged and covers Ubuntu/Windows × Python 3.10/3.11.

Production-specific workflow:

`.github/workflows/inference-integration.yml`

It performs on Ubuntu/Python 3.11:

```text
install runtime dependencies
        ↓
Calibration V1 tests
        ↓
real-checkpoint scorer → calibration → LOO E2E test
        ↓
Docker build
        ↓
container startup
        ↓
/healthz check
```

The real-checkpoint E2E test also verifies:

- product score stays in `[0,100]`;
- diagnosis returns a valid problematic index;
- original 3-item outfits explicitly mark two-item LOO extrapolation;
- `<3` is a structured error;
- `>8` is rejected rather than silently truncated;
- non-L2 embeddings are rejected;
- no embeddings leak into output JSON.

## Detection integration TODO

The detection/image team only needs to implement the `GarmentPreprocessor` boundary from `src/inference/adapters.py`:

```python
class GarmentPreprocessor:
    def prepare(self, image):
        ...
        return items
```

It owns:

```text
image
  ↓
detection
  ↓
crop garments
  ↓
FashionCLIP encode
  ↓
L2 normalize embedding
  ↓
runtime category resolution → Core-7 ID
  ↓
list of production garment records
```

Once available, inject it into `ProductionInferencePipeline(..., garment_preprocessor=...)`; scorer/calibration/diagnosis do not need to change.

## VLM integration TODO

The VLM team implements `ExplanationProvider`:

```python
class ExplanationProvider:
    def explain(self, evidence):
        ...
```

The evidence is the structured scorer/calibration/diagnosis response. VLM should explain these signals rather than independently redefine compatibility.

## Deployment ownership boundary

ML owns:

```text
model artifacts + versions
runtime input validation
scorer inference
calibration
LOO diagnosis
structured evidence schema
E2E model contract
```

Deploy/backend owns:

```text
public API/authentication
image/object storage
request queueing if needed
autoscaling/GPU provisioning
observability and service-level metrics
rate limiting
frontend integration
HTTPS/domain/release infrastructure
```

## Handoff status

Completed in Production Inference V1 branch:

- versioned Calibration V1 implementation and fitted artifact;
- reproducible calibration fitting CLI;
- stable scorer + calibration + diagnosis pipeline;
- immutable production manifest with checkpoint SHA verification;
- structured runtime errors;
- real-checkpoint E2E tests;
- deploy-facing FastAPI wrapper;
- Dockerfile and runtime dependency spec;
- production E2E + Docker CI workflow;
- this deployment handoff document.

Still pending before the full image-to-explanation web pipeline is feature-complete:

- concrete detection/crop/FashionCLIP/category resolver adapter;
- final VLM explanation provider;
- latency benchmark on the actual deployment hardware after those two components are integrated;
- final public image upload endpoint and frontend UX.
