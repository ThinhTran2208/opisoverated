# Inference Context Architecture V1

## Purpose

Production image inference now uses a single managed context object between image preprocessing and downstream reasoning. The scorer no longer reconstructs VLM inputs from the public JSON response.

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

## DetectionAdapter

`src.inference.adapters.DetectionAdapter` wraps the merged `src.detection.DetectionPipeline`.

It runs RF-DETR, crop extraction, FashionCLIP embedding and Core-7 classification once. The accepted FashionCLIP embedding is reused directly by the scorer.

The adapter returns `InferenceContext`, not a loose list of dictionaries.

## InferenceContext

`src.inference.context.InferenceContext` owns all state for one image request:

- `garments`: deployment-safe item metadata, without embedding duplication;
- `embeddings`: `[N, 512]` L2-normalized FashionCLIP vectors;
- `categories`: `[N]` Core-7 IDs;
- `crop_image_refs`: exactly one temporary crop reference per accepted garment when visual explanation is enabled;
- `original_image`: normalized original image object;
- `request_id`: stable ID shared by scorer/LOO/VLM outputs;
- `metadata`: preprocessing provenance, including Detection V1 metadata.

The context also owns crop cleanup. `ProductionInferencePipeline.analyze_image()` always closes the context in `finally`, so temporary crops live through VLM inference but are removed after the request completes.

## Scorer / Calibration / LOO

`ProductionInferencePipeline.analyze_context()` validates the context, then runs:

1. frozen compatibility scorer;
2. Calibration V1 to obtain the product-facing score;
3. LOO diagnosis.

The complete raw LOO object remains internal. The public diagnosis response may remain compact, but downstream VLM code receives the full object including `full_logit`, `without_item_logits`, `deltas_without_minus_full`, ranking and extrapolation flag.

## VLMAdapter

`src.inference.adapters.VLMAdapter` consumes exactly:

```python
VLMAdapter.explain(
    loo_result,
    garments,
    crop_image_refs,
    sample_id=...,
)
```

It builds the existing `vlm-evidence-v1` contract with `build_vlm_evidence()` and invokes `VLMExplanationPipeline` with one crop reference per item.

The VLM does not infer or override the problematic item. That decision remains authoritative from frozen scorer + LOO.

## Precomputed compatibility path

`analyze_precomputed()` is retained for backend testing and clients that already provide 512-d embeddings. It creates an `InferenceContext` internally but disables visual explanation because no crop references are available.

## Runtime dependency boundary

Adapters are lazy-loaded. Importing `src.inference` does not import RF-DETR, FashionCLIP, Transformers, or Qwen model weights.

This is important because the canonical Detection V1 and Qwen VLM environments currently have incompatible Transformers major-version constraints. The context contract remains valid whether the adapters eventually run in one compatible process or across separate services.
