# Deployment Handoff V1

## Purpose

This handoff freezes the deployment-facing ML contract for the current outfit
analysis system. It covers scorer/calibration/LOO plus the final image path that
uses Detection V1 and VLM Explanation V1.

The user-facing image flow is:

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
JSON response
```

Scoring and explanation remain separated. The scorer decides overall
compatibility; LOO decides the problematic item; VLM only explains the existing
structured evidence.

## Frozen scorer/calibration bundle

The production scorer is loaded from `configs/production_inference_v1.json`.
The manifest pins the scorer checkpoint, checkpoint hash, category mapping,
embedding version and Calibration V1 artifact.

Compatibility output always contains:

```text
compatibility_logit
compatibility_score [0,100]
scorer_version
calibration_version
```

Raw logit is internal model evidence, not a percentage. Product display should
use the calibrated score.

## Input invariants

The scorer boundary accepts 3 to 8 garments. Each accepted garment must have:

```text
512-d finite FashionCLIP embedding
L2 norm ~= 1
Core-7 category ID in 1..7
unique item ID
```

The image path gets those fields from Detection V1. The precomputed endpoint is
retained for integration/debug clients that already have embeddings.

No request is silently truncated when more than 8 garments are accepted.

## Image endpoint

Inference core exposes:

```text
GET  /healthz
POST /v1/analyze-precomputed
POST /v1/analyze-outfit
```

`/v1/analyze-outfit` accepts multipart form field `image`.

Example:

```bash
curl -X POST http://localhost:8000/v1/analyze-outfit \
  -F "image=@outfit.jpg"
```

Expected successful payload includes:

```text
status=ok
request_id
item_count
items[]
compatibility
diagnosis
explanation      # required in the final split deployment
versions
preprocessing
```

Stable request errors include invalid/empty/non-image upload, upload size limit,
insufficient garments, too many garments and invalid scorer inputs.

When `FASHION_REQUIRE_VLM=true`, the endpoint also fails if the VLM service is
unavailable or no explanation is returned.

## InferenceContext lifecycle

`DetectionAdapter` returns `InferenceContext`, not a loose item list. Temporary
crop files are request-scoped and remain available through VLM execution.
`ProductionInferencePipeline.analyze_image()` closes the context in `finally`,
so temporary crop storage is cleaned on both success and failure.

The raw `diagnose_outfit()` result is passed directly to VLMAdapter. It is not
reconstructed from the smaller public diagnosis response.

## Split runtime requirement

Do not force Detection V1 and Qwen VLM into one Python environment without a new
compatibility experiment. Their canonical Transformers major ranges are
incompatible.

Deployment is therefore split:

```text
inference-core :8000
  RF-DETR
  FashionCLIP
  scorer
  calibration
  LOO

vlm :8001
  Qwen3-VL-4B-Instruct
  VLM evidence validation
  explanation rendering
```

The core process uses `RemoteVLMAdapter`. It base64-encodes request-local crops
and sends them to `POST /v1/explain`. Shared filesystem access is not required.

## Docker

Build inference core:

```bash
docker build -f Dockerfile.inference-core -t outfit-inference-core-v1 .
```

Build VLM:

```bash
docker build -f Dockerfile.vlm -t outfit-vlm-v1 .
```

GPU reference composition:

```bash
docker compose -f docker-compose.gpu.yml up --build
```

A production platform may run both services on one sufficiently large GPU or
place them on separate GPU nodes. Keep the HTTP contract unchanged.

The original `Dockerfile` remains a lightweight scorer/calibration/LOO and API
smoke image for CI/backward compatibility; it is not the final two-service GPU
bundle.

## Runtime environment variables

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

Final product defaults should configure a VLM URL and set:

```text
FASHION_REQUIRE_VLM=true
```

VLM service:

```text
FASHION_VLM_CONFIG
FASHION_VLM_MAX_CROP_BYTES
```

## CI gates

`.github/workflows/inference-integration.yml` verifies:

```text
Calibration V1 tests
InferenceContext/adapter tests
real-checkpoint scorer + calibration + LOO E2E
image → DetectionAdapter → scorer → calibration → LOO → VLMAdapter contract E2E
final image HTTP endpoint contract
VLM service HTTP contract
benchmark aggregation tests
lightweight Docker build/start/health
inference-core Docker build/start/health
VLM Docker build/start/health without model download
```

The existing portability matrix remains a separate merge gate on Ubuntu/Windows
and Python 3.10/3.11.

Normal CI deliberately does not download RF-DETR or Qwen weights and does not
claim GPU latency results. Heavy model execution belongs on the deploy GPU.

## Benchmark harness

Use:

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

The report contains GPU identity, CUDA/Torch versions, core-process peak CUDA
memory and latency distributions for decode, detection, detector model time when
available, post-detector processing, scorer+calibration+LOO, VLM, total ML and
local end-to-end time.

`.github/workflows/gpu-benchmark.yml` runs this command on a deployment-owned
self-hosted runner labeled:

```text
self-hosted, linux, x64, gpu
```

It hard-fails if CUDA is unavailable and uploads the benchmark JSON artifact.

A deployment latency result is not considered valid until this workflow (or the
same harness) is run on the actual deploy GPU or an explicitly approved identical
GPU SKU. Do not substitute GitHub CPU CI or an assumed Colab GPU for this gate.

## ML / deploy ownership boundary

ML package owns:

```text
model/config/artifact versioning
DetectionAdapter and InferenceContext contract
scorer/calibration/LOO inference
VLM evidence contract
split service application code
model-level validation and benchmark harness
```

Deploy/backend owns:

```text
GPU provisioning
container registry/release promotion
service discovery and TLS
authentication/rate limits
request storage policy
observability and alerting
autoscaling/queues/timeouts
frontend integration
running and retaining the production GPU benchmark artifact
```

## Release gate

Before merge/release require:

```text
main is an ancestor of the branch
portability matrix PASS
production integration CI PASS
PR review complete
```

Before production traffic require additionally:

```text
actual deploy GPU benchmark report exists
acceptable latency/VRAM targets agreed with deploy team
both model services can download/access frozen model artifacts
/v1/analyze-outfit smoke test succeeds in the deploy environment
```

See also:

- `docs/INFERENCE_CONTEXT_ARCHITECTURE_V1.md`
- `docs/PRODUCTION_IMAGE_DEPLOY_V1.md`
- `docs/CALIBRATION_V1_RESULTS.md`
