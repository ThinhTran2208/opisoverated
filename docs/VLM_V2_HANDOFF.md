# VLM V2 Renderer and Handoff

## Scope

VLM V2 sits downstream of frozen diagnosis and recommendation logic.

```text
Frozen scorer + LOO
        +
Recommendation V2 Top-3
        ↓
full internal vlm-evidence-v2
        ↓
score-free vlm-prompt-context-v2
+ original garment crops
+ authoritative Top-3 candidate images
        ↓
Qwen3-VL closed-taxonomy visual analysis
        ↓
deterministic validator
        ↓
internal audit output
+ score-free handoff
+ concise user-facing renderer
```

The upstream scorer, LOO diagnosis, candidate identities, and Top-3 rank are not changed by VLM V2.

## Qwen responsibility

Qwen is a constrained visual-evidence extractor, not a second decision-maker.
It returns only structured taxonomy fields:

```text
color_harmony
pattern_coherence
silhouette_balance
formality_alignment
style_coherence
```

with the corresponding support / ambiguous / contradiction labels and visual confidence.

Qwen does **not** write natural-language user-facing explanations, does not rerank candidates, and does not invent replacement items.

Raw scorer / LOO / recommendation numerical values are projected out before prompt construction.

## Validator boundary

`validate_visual_analysis_v2(...)` keeps the following constraints deterministic:

- problematic item index / ID must remain unchanged;
- Top-3 candidate IDs and rank must remain unchanged;
- diagnosis observations must include the problematic item plus at least one other original item;
- recommendation observations may reference only the remaining original outfit context;
- taxonomy values must come from the closed enum sets;
- empty observations require an ambiguous overall label;
- exact cloned high-confidence Top-3 analyses are rejected;
- required limitation tokens must match exactly.

## User-facing renderer

`run["user_facing"]` is the payload intended for deploy/UI.

The renderer is intentionally concise. Rank is already represented by card order and the `rank` field, so prose does not repeat sentences such as:

```text
Mẫu 1 là lựa chọn ưu tiên nhất.
Mẫu 2 là lựa chọn thứ hai.
Mẫu 3 là lựa chọn thứ ba.
```

Instead, positive Qwen observations are translated deterministically into short, context-aware Vietnamese text.
For example, if a recommendation observation references the original top and shoes with `color_harmony`, the UI reason can be:

```text
Màu sắc phối hợp tốt với áo và giày.
```

If Qwen is ambiguous or internally contradicts the upstream decision, the renderer does **not** invent a fallback visual explanation. The corresponding `reason` is `null` and the UI may simply show the candidate image/card in authoritative rank order.

Example shape:

```json
{
  "schema_version": "vlm-user-facing-v2",
  "problematic_item": {
    "item_index": 3,
    "item_id": "...",
    "category": "túi",
    "headline": "Trong outfit này, chiếc túi hiện tại là món được ưu tiên thay.",
    "reason": null
  },
  "summary": "Ba mẫu túi bên dưới đều được đánh giá phù hợp hơn khi thay cho chiếc túi hiện tại.",
  "recommendations": [
    {
      "rank": 1,
      "item_id": "...",
      "display_name": "Mẫu túi 1",
      "reason": "Màu sắc phối hợp tốt với áo và giày."
    },
    {
      "rank": 2,
      "item_id": "...",
      "display_name": "Mẫu túi 2",
      "reason": null
    }
  ]
}
```

Frontend should bind images by authoritative `item_id` / `rank` and render the structured fields rather than relying on one long paragraph.

## Internal vs public data

Internal/debug only:

```text
run["evidence"]
run["visual_analysis"]
run["explanation"]
run["raw_response"]  # when enabled
```

Deploy-safe integration:

```text
run["handoff"]
run["user_facing"]
```

`run["handoff"]` remains score-free and keeps identity/rank/category plus validated visual evidence.
`run["user_facing"]` contains the final concise UI copy and no raw logits, LOO deltas, Qwen/validator jargon, or numerical score fields.

## Error analysis

Error analysis is **not part of the VLM runtime contract**.
Batch review, repair-rate measurement, hallucination review, and qualitative error tagging are offline evaluation tasks performed on saved outputs after inference. They must not affect deploy-time item identity, ranking, or rendering behavior.

## Runtime configuration

Canonical V2 still uses:

- `Qwen/Qwen3-VL-4B-Instruct`;
- FP16 / CUDA;
- greedy decoding;
- one deterministic schema-repair retry;
- canonical 262,144 pixels per image.

For the Colab/T4 functional test, a lower per-image visual budget may be used explicitly when VRAM is insufficient. This is only a runtime resource override and does not change the VLM contract.
