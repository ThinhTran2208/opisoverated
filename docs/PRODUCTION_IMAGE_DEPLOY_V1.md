# Production Image Deployment V1

## Status

The production image path is implemented on `feat/production-inference-v1` after
Detection V1 and VLM Explanation V1 were merged into `main`.

Canonical request flow:

```text
POST /v1/analyze-outfit
        ↓
DetectionAdapter
        ↓
InferenceContext
├── garments
├── embeddings
├── categories
├── crop_image_refs
└── original_image
        ↓
Frozen scorer
        ↓
Calibration V1
        ↓
LOO diagnosis
        ↓
RemoteVLMAdapter
        ↓
VLM Explanation service
        ↓
final JSON
```

The scorer remains authoritative for compatibility and LOO remains authoritative
for the problematic item. The VLM consumes the existing structured evidence
contract and does not override the diagnosis.

## Why the runtime is split

The frozen Detection V1 runtime requires the RF-DETR-compatible Transformers 5.x
range while the canonical Qwen3-VL runtime is frozen on Transformers 4.x.
Production therefore uses two Python environments:

```text
inference-core
  RF-DETR
  FashionCLIP
  scorer
  calibration
  LOO
  FastAPI :8000

vlm
  Qwen3-VL-4B-Instruct
  VLM evidence validation
  explanation rendering
  FastAPI :8001
```

`RemoteVLMAdapter` preserves the same internal adapter call:

```python
explain(loo_result, garments, crop_image_refs, sample_id=...)
```

Crop images are base64-encoded before the `InferenceContext` temporary directory
is cleaned. The VLM service does not require shared filesystem access.

## HTTP endpoints

Inference core:

```text
GET  /healthz
POST /v1/analyze-precomputed
POST /v1/analyze-outfit
```

`/v1/analyze-outfit` accepts multipart form data with field `image`.

Example:

```bash
curl -X POST http://localhost:8000/v1/analyze-outfit \
  -F "image=@outfit.jpg"
```

Expected successful response includes:

```text
request_id
items
compatibility.compatibility_logit
compatibility.compatibility_score
diagnosis.problematic_item_index
diagnosis.deltas_without_minus_full
explanation               # when VLM service is configured
versions
preprocessing
```

The endpoint rejects empty uploads, non-image media types, invalid images, images
above the configured upload limit, fewer than 3 accepted garments, and more than
8 accepted garments.

VLM service:

```text
GET  /healthz
POST /v1/explain
```

The model is loaded lazily on the first `/v1/explain` request. This keeps health
checks and container startup independent of model download time.

## Environment variables

Inference core:

```text
FASHION_INFERENCE_MANIFEST
FASHION_INFERENCE_DEVICE
FASHION_DETECTION_CONFIG
FASHION_DETECTION_DEVICE
FASHION_VLM_SERVICE_URL
FASHION_REQUIRE_VLM
FASHION_VLM_TIMEOUT_SECONDS
FASHION_MAX_UPLOAD_BYTES
```

For the final product path set:

```text
FASHION_VLM_SERVICE_URL=http://vlm:8001
FASHION_REQUIRE_VLM=true
```

VLM service:

```text
FASHION_VLM_CONFIG
FASHION_VLM_MAX_CROP_BYTES
```

## Docker images

Build inference core:

```bash
docker build -f Dockerfile.inference-core -t outfit-inference-core-v1 .
```

Build VLM:

```bash
docker build -f Dockerfile.vlm -t outfit-vlm-v1 .
```

A GPU reference composition is provided in:

```text
docker-compose.gpu.yml
```

Run with a recent Docker Compose + NVIDIA Container Toolkit environment:

```bash
docker compose -f docker-compose.gpu.yml up --build
```

The reference compose file exposes core on port 8000 and VLM on port 8001. A
production platform may place the services on separate GPUs or hosts while
keeping the same HTTP contract.

## Integration tests

The contract-level E2E test exercises:

```text
image object
  ↓
real DetectionAdapter
  ↓
real InferenceContext lifecycle
  ↓
real frozen scorer checkpoint
  ↓
real Calibration V1 artifact
  ↓
real LOO implementation
  ↓
real VLMAdapter evidence builder
  ↓
fake VLM generation backend
```

The fake components are only the heavyweight model execution boundaries. This
lets normal CI verify the complete production wiring without downloading RF-DETR
or Qwen weights.

GitHub Actions additionally builds both split Docker images and health-checks
both services. The VLM health check intentionally does not load Qwen.

## GPU benchmark harness

CLI:

```bash
python -m src.inference.benchmark \
  --image path/to/outfit.jpg \
  --device cuda \
  --warmup-runs 2 \
  --runs 10 \
  --vlm-service-url http://localhost:8001 \
  --require-cuda \
  --require-vlm \
  --output benchmark-results/production_gpu_benchmark.json
```

The report records:

```text
hardware / GPU identity
CUDA + Torch versions
peak CUDA allocated/reserved memory in the inference-core process
image decode latency
detection adapter latency
RF-DETR model latency when exposed by Detection V1
post-detector detection/FashionCLIP overhead
scorer + calibration + LOO latency
VLM service latency
total ML latency
end-to-end local latency
mean / min / p50 / p95 / p99 / max
per-run records
```

A manual workflow is provided at:

```text
.github/workflows/gpu-benchmark.yml
```

It targets a deploy-owned self-hosted runner labeled:

```text
self-hosted, linux, x64, gpu
```

The workflow hard-fails when CUDA is unavailable and uploads the JSON report as
`production-gpu-benchmark`.

## Benchmark evidence policy

Do not report a GPU latency number from GitHub CPU runners, Colab assumptions, or
another machine as a deployment benchmark. A production benchmark is considered
complete only when `production_gpu_benchmark.json` was generated on the actual
deploy GPU (or a deploy-approved identical SKU) and the report records that GPU
identity.

No specific deployment GPU SKU or connected deploy GPU runner is frozen in the
project artifacts at the time this document was written. Repository work can
therefore complete the benchmark harness and GPU workflow, but the final measured
latency artifact must be generated by the deployment environment itself.
