# Inference Context Architecture V1

## Canonical internal flow

```text
DetectionAdapter
    ↓
InferenceContext
├── garments
├── embeddings
├── categories
├── crop_image_refs
└── original_image

        ↓
Scorer
Calibration
LOO
        ↓
VLMAdapter(
    loo_result,
    garments,
    crop_image_refs
)
```

## Why `InferenceContext` exists

The image path needs more information than the scorer itself consumes. The
scorer only needs FashionCLIP embeddings and Core-7 category IDs, while the VLM
needs stable garment metadata and one crop reference per item. Passing a single
managed context keeps those views aligned without putting image concerns into
the frozen scorer.

`InferenceContext` owns:

- ordered garment metadata;
- `[N, 512]` FashionCLIP embeddings;
- `[N]` Core-7 IDs;
- one crop reference per accepted garment when visual explanation is enabled;
- the decoded original image for request-local use;
- request ID and preprocessing metadata;
- cleanup callback for temporary crop storage.

The object is a context manager and `close()` is idempotent.

## Detection adapter

`DetectionAdapter` wraps merged Detection V1. It runs RF-DETR and FashionCLIP,
creates stable `garment-{index}` IDs, reuses the same L2-normalized FashionCLIP
vectors for the scorer, and writes temporary crop images for the VLM.

The temporary directory remains alive until scorer, calibration, LOO, and VLM
processing have completed. `ProductionInferencePipeline.analyze_image()` closes
the context in a `finally` block.

## Scorer / calibration / LOO

`ProductionInferencePipeline.analyze_context()` validates:

```text
3 <= item_count <= 8
embedding shape == [N, 512]
finite embeddings
L2 norm within tolerance
Core-7 IDs in 1..7
garment/category alignment
unique item IDs
```

It then runs the frozen scorer, converts the raw logit through Calibration V1,
and computes LOO. The complete raw `diagnose_outfit()` result is kept in memory
for the explanation adapter.

The public response exposes a smaller diagnosis object, but the VLM is not built
by reverse-engineering that public JSON.

## VLM adapter

The in-process `VLMAdapter` receives:

```text
raw LOO result
garment metadata
crop refs
sample/request ID
```

It builds the existing `vlm-evidence-v1` object and invokes VLM Explanation V1.
This preserves the existing grounding rule that the VLM cannot change the
problematic item chosen by LOO.

## Split runtime adapter

Detection V1 and the canonical Qwen VLM use incompatible Transformers major
ranges. Production therefore also provides `RemoteVLMAdapter`, which implements
the exact same adapter interface over HTTP.

Before context cleanup it reads each crop, base64-encodes it, and calls the
separate VLM service. The VLM service recreates request-local crop files and
runs the normal in-process `VLMAdapter` there.

This means deployment can be split as:

```text
inference-core                          vlm service
--------------                         -----------
Detection V1                           Qwen3-VL
FashionCLIP                            VLM evidence validator
Scorer                                 explanation renderer
Calibration V1
LOO

       RemoteVLMAdapter  ───────────→  POST /v1/explain
```

No shared filesystem is required.

## Precomputed compatibility path

`analyze_precomputed()` remains supported for backend/debug integration. It
constructs an `InferenceContext` without crop refs and deliberately skips VLM
visual explanation.

## HTTP boundary

The final image endpoint is:

```text
POST /v1/analyze-outfit
```

It decodes the uploaded image and calls `analyze_image()` through the same
context lifecycle. See `docs/PRODUCTION_IMAGE_DEPLOY_V1.md` for Docker and
benchmark instructions.
