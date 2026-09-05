# VLM Protocol V2

**Protocol ID:** `vlm-explanation-v2`  
**Evidence schema:** `vlm-evidence-v2`  
**Prompt-context schema:** `vlm-prompt-context-v2`  
**Visual-analysis schema:** `vlm-visual-analysis-v2`  
**Handoff schema:** `vlm-handoff-v2`  
**User-facing schema:** `vlm-user-facing-v2`  
**Canonical model:** `Qwen/Qwen3-VL-4B-Instruct`  
**Baseline implementation:** branch `feat/vlm-v2-final`, based on commit `708ccb1b3b87fe8a6bf25a5d99e58299d92cb7bd` before this protocol document was added.

---

## 1. Purpose

VLM V2 is the explanation layer that sits **after** the frozen compatibility scorer, LOO diagnosis, and Recommendation V2.

Its job is to inspect the supplied garment images and extract constrained visual evidence that can help explain decisions that have already been made upstream.

The central contract is:

```text
ML decides.
VLM observes.
Code validates and renders.
```

VLM V2 is **not** a second recommendation system, not a reranker, and not an independent stylist.

---

## 2. Decision authority

Decision authority is intentionally asymmetric and must not be changed by the VLM layer.

```text
Frozen scorer
    ↓
LOO diagnosis
    ↓
problematic item is fixed
    ↓
Recommendation V2
    ↓
Top-3 candidate IDs + rank are fixed
    ↓
VLM V2
    ↓
visual evidence only
```

### 2.1 LOO authority

LOO is the authoritative source for:

- `problematic_item_index`;
- `problematic_item_id`;
- problematic coarse category.

Qwen must copy the problematic item identity exactly. It may not select another item.

### 2.2 Recommendation V2 authority

Recommendation V2 is the authoritative source for:

- the three candidate item IDs;
- their order/rank `1, 2, 3`;
- candidate category metadata.

Qwen must preserve all three candidate IDs and ranks exactly.

It may **not**:

- remove a candidate;
- add a candidate;
- replace a candidate;
- reorder or rerank candidates;
- decide that another candidate should be used instead.

### 2.3 Meaning of VLM disagreement

The internal VLM taxonomy allows:

- `supports_loo` / `supports_recommendation`;
- `ambiguous`;
- `contradicts_loo` / `contradicts_recommendation`.

These labels are **visual QA evidence only**.

A contradiction does not override upstream authority.

In particular:

```text
contradicts_loo
    ≠ choose another problematic item

contradicts_recommendation
    ≠ remove candidate
    ≠ rerank candidate
    ≠ replace candidate
```

The end-user renderer must not expose internal disagreement as a counter-argument against the authoritative Recommendation V2 result.

---

## 3. Runtime scope

VLM V2 runtime includes:

1. building and validating `vlm-evidence-v2`;
2. projecting the score-free Qwen context;
3. binding original garment crops and Top-3 candidate images;
4. generating constrained Qwen visual analysis;
5. deterministic validation;
6. one repair retry when validation fails;
7. internal/debug rendering;
8. score-free deploy handoff;
9. user-facing Vietnamese rendering.

### 3.1 Explicit non-scope

The following are **not part of the VLM runtime protocol**:

- scorer training or calibration;
- LOO algorithm design;
- Recommendation V2 retrieval or reranking;
- recommendation evaluation metrics;
- Hit@K / MRR / GT recovery;
- offline error analysis;
- offline hallucination-rate analysis;
- benchmark labels or synthetic negative metadata;
- frontend image retrieval infrastructure;
- HTTP/API server wiring.

Offline error analysis may consume saved VLM outputs, but it must remain outside the production VLM decision path and must never modify recommendation identity or rank.

---

## 4. Canonical pipeline

```text
Frozen scorer + LOO
        +
Recommendation V2 Top-3
        ↓
full internal vlm-evidence-v2
(raw numerical values retained internally)
        ↓
build_prompt_context_v2(...)
        ↓
score-free vlm-prompt-context-v2
        +
original garment crops
        +
exactly three authoritative candidate images
        ↓
Qwen3-VL constrained visual analysis
        ↓
validate_visual_analysis_v2(...)
        ↓
 ┌────────────────────┬────────────────────┬─────────────────────┐
 │ internal/debug     │ integration        │ end-user            │
 │ evidence/analysis  │ vlm-handoff-v2     │ vlm-user-facing-v2  │
 │ explanation/raw    │ score-free         │ UI-safe             │
 └────────────────────┴────────────────────┴─────────────────────┘
```

The full original outfit image is **not** part of the V2 baseline. The baseline uses one crop per original outfit item plus exactly three recommendation candidate images.

---

## 5. Evidence contract: `vlm-evidence-v2`

The full evidence object is an internal deterministic artifact joining:

- scorer/LOO evidence;
- authoritative Recommendation V2 output.

Recommendation evidence contains exactly three ranked candidates and validates that public Top-3 identity/order agrees with the same internal frozen scorer reranking call.

### 5.1 Recommendation invariants

The evidence builder must enforce:

- Recommendation status is canonical and available;
- Recommendation version is `category-aware-hybrid-v2`;
- exactly three candidates;
- ranks are exactly `1, 2, 3`;
- candidate IDs are unique;
- no candidate is already present in the original outfit;
- candidate coarse category matches the LOO problematic category;
- public Top-3 order matches internal reranked candidate order;
- all logits are finite;
- `improvement_logit = compatibility_logit - baseline_compatibility_logit` within tolerance;
- Top-3 ordering is consistent with descending frozen-scorer `compatibility_logit` with deterministic `item_id` tie-breaking.

### 5.2 Score semantics

Recommendation scores are stored as:

```text
ranking_semantics = frozen_scorer_descending_compatibility_logit
score_semantics   = uncalibrated_logits_not_probabilities
```

A `compatibility_logit` or `improvement_logit` must never be described as a probability or percentage.

---

## 6. Evaluation-leakage boundary

Synthetic benchmark or evaluation state must not become VLM explanation evidence.

The V2 leakage guard recursively scans the raw recommendation input, including nested mappings, **before** adapter projection can discard fields.

The following keys are forbidden in VLM Recommendation V2 evidence/input:

```text
label
negative_metadata
swapped_item_index
target_swapped_item_index
original_item_id
ground_truth
ground_truth_item_id
replacement_item_id
top1_correct
hit_at_1
hit_at_2
hit_at_3
mrr
source_split
ground_truth_rank
pre_rerank_ground_truth_rank
post_rerank_ground_truth_rank
```

If any forbidden key appears at any nested path, the builder must fail fast with `ValueError`.

This rule exists to prevent benchmark ground truth or synthetic-negative construction metadata from influencing explanation behavior.

---

## 7. Score-free Qwen boundary

Full `vlm-evidence-v2` retains raw numerical values because deterministic code needs them for validation, audit, and renderer logic.

Those raw values must **not** be serialized into the Qwen prompt context.

`build_prompt_context_v2(...)` exposes only:

### Original items

```text
item_index
item_id
coarse_category
```

### Diagnosis

```text
problematic_item_index
problematic_item_id
problematic_category
uses_two_item_extrapolation
```

### Recommendation Top-3

```text
rank
item_id
master_category
coarse_category
```

The following must remain outside the Qwen context:

```text
compatibility_logit
improvement_logit
full_logit
without_item_logits
without_item_logit
loo_delta
deltas_without_minus_full
top1_top2_delta_gap
checkpoint score state
scorer raw numerical state
```

The prompt explicitly tells Qwen not to reconstruct or guess omitted numerical values.

### 7.1 Why rank is still visible

Recommendation `rank` is allowed because it identifies authoritative presentation order. It is not an evaluation target and must not be interpreted as visual evidence.

Qwen may not derive a visual label from rank.

---

## 8. Visual input contract

For an outfit with `N` original items, Qwen receives:

- one full original outfit image when available;
- exactly `N` original garment crop images;
- exactly three Recommendation V2 candidate images.

Candidate images must be supplied as a mapping keyed by authoritative candidate `item_id`.

The supplied candidate-image keys must match the Top-3 candidate IDs exactly:

```text
missing candidate image → hard failure
extra candidate image   → hard failure
wrong candidate ID      → hard failure
```

Recommendation image binding is by ID, not by an assumed list position.

The full original image is context only: it helps Qwen inspect composition,
layering, color balance, silhouette, and overall style. It does not change the
authoritative problematic item or candidate ranking.

The original problematic item remains among the original outfit crops. Candidate visual observations must compare each candidate against the **remaining original context**, excluding the problematic original item.

---

## 9. Qwen role

Qwen is a constrained visual-evidence extractor.

It must not write user-facing prose.

Its task is to classify visible relations using a closed taxonomy.

### 9.1 Allowed visual dimensions

```text
color_harmony
pattern_coherence
silhouette_balance
formality_alignment
style_coherence
```

### 9.2 Confidence levels

```text
low
medium
high
```

Confidence means only how clearly the claimed visual relation is visible in the supplied images.

It is **not** confidence in:

- the frozen scorer;
- LOO correctness;
- Recommendation V2 correctness;
- candidate rank.

### 9.3 Diagnosis effects

```text
supports_loo
ambiguous
contradicts_loo
```

For diagnosis, Qwen first seeks visible evidence that helps explain why the already-fixed problematic item fits less well with the original outfit.

If no grounded supporting relation is visible, it should use `ambiguous` rather than inventing a reason.

`contradicts_loo` is reserved for clear internal-QA evidence and does not alter the problematic item.

### 9.4 Recommendation effects

```text
supports_recommendation
ambiguous
contradicts_recommendation
```

For each candidate independently, Qwen first seeks one concrete positive visible relation with the remaining outfit.

If no clear positive relation is visible, it should use `ambiguous` rather than create filler or weak negative commentary.

`contradicts_recommendation` is internal QA evidence only. The candidate remains part of the authoritative Top-3 with the same rank.

### 9.5 Prohibited Qwen behavior

Qwen must not:

- choose another problematic item;
- change or rerank Top-3 candidates;
- compare candidates against each other in order to recreate rank;
- infer scores hidden from the prompt;
- treat logits as probabilities;
- infer brand;
- infer material that is not directly visible;
- infer price;
- infer occasion;
- infer user intent;
- infer demographics;
- invent unsupported fashion facts;
- output arbitrary free-text fields;
- output Markdown or extra keys outside the required JSON contract.

---

## 10. Qwen output schema: `vlm-visual-analysis-v2`

Top-level keys must be exactly:

```json
{
  "schema_version": "vlm-visual-analysis-v2",
  "problematic_item_index": 0,
  "problematic_item_id": "...",
  "diagnosis": {},
  "recommendations": [],
  "limitations": []
}
```

The actual fixed index/ID must be copied from evidence.

### 10.1 Diagnosis object

Diagnosis keys must be exactly:

```text
overall_visual_support
visual_observations
```

Each diagnosis observation contains exactly:

```text
item_indices
dimension
effect
confidence
```

Every diagnosis observation must:

- reference only original outfit indices;
- include the problematic item index;
- include at least one other original outfit item;
- contain unique indices.

### 10.2 Recommendation objects

There must be exactly three recommendation rows, in authoritative rank order.

Each row contains exactly:

```text
rank
item_id
overall_visual_support
visual_observations
```

Each recommendation observation contains exactly:

```text
context_item_indices
dimension
effect
confidence
```

`context_item_indices` may reference only remaining original outfit items and must exclude the problematic original item.

---

## 11. Required limitation tokens

Every valid V2 visual analysis must carry these protocol tokens exactly:

```text
compatibility_logit_is_not_probability
recommendation_scores_are_not_probabilities
recommendation_identity_and_rank_are_authoritative
vlm_visual_observations_are_inferences
```

For three-item outfits where LOO relies on two-item subset scoring, also include:

```text
loo_uses_two_item_extrapolation
```

These are machine-readable protocol disclosures. They do not mean raw score values are exposed to Qwen.

---

## 12. Deterministic validator

`validate_visual_analysis_v2(...)` is the hard authority boundary after generation.

It must reject any output that:

- changes `problematic_item_index`;
- changes `problematic_item_id`;
- changes recommendation rank/order;
- changes candidate identity;
- changes schema keys;
- references invalid original/context indices;
- uses unknown taxonomy values;
- omits or changes required limitation tokens;
- uses non-ambiguous overall support without a matching observation;
- produces an exact cloned non-ambiguous high-confidence pattern across all three candidates.

The validator does not decide fashion quality. It enforces identity, schema, grounding shape, and the authority boundary.

---

## 13. Repair policy

Generation is deterministic/greedy.

If the first Qwen response fails parsing or V2 validation, the pipeline may issue **one** repair request using the same authoritative identities and contract.

Canonical configuration:

```text
max_validation_retries = 1
```

A repair attempt may fix structure or contract violations. It must not relax decision authority.

If validation still fails after the allowed retry, the VLM call is a hard failure and the invalid model output must not be exposed as a valid deploy payload.

---

## 14. Internal/debug output

The following are internal/audit artifacts:

```text
run["evidence"]
run["visual_analysis"]
run["explanation"]
run["raw_response"]   # when enabled
```

These may contain:

- scorer/LOO/recommendation numerical state;
- internal visual support/ambiguity/contradiction labels;
- technical protocol terminology;
- confidence taxonomy;
- debug summaries.

They are **not frontend payloads**.

Deploy code must not render internal/debug objects directly to users.

---

## 15. Score-free integration handoff: `vlm-handoff-v2`

`run["handoff"]` is the integration-safe structured payload.

It preserves:

- problematic item index/ID;
- Top-3 candidate rank/ID/category metadata;
- validated visual summaries and observations;
- protocol limitations.

It excludes raw numerical scorer state such as:

- compatibility logits;
- improvement logits;
- LOO deltas;
- raw score summaries;
- generation attempts;
- model internals;
- full evidence;
- raw Qwen response.

The handoff is useful when another application layer needs structured VLM metadata without raw model scores.

---

## 16. End-user payload: `vlm-user-facing-v2`

`run["user_facing"]` is the deploy-facing Vietnamese payload intended for UI rendering.

It contains:

- final `text`;
- authoritative item identity/category that should be replaced;
- exactly three authoritative candidates in frozen rank order;
- display names derived from coarse category;
- deterministic recommendation reasons/fallback text;
- candidate `item_id` and `rank` for frontend image binding.

### 16.1 User-facing authority rule

The user-facing layer must present the upstream decision as authoritative.

Internal Qwen `ambiguous` or `contradicts_*` labels must not become text such as:

```text
"VLM disagrees with this recommendation"
"candidate 2 should not be recommended"
"the model thinks candidate 3 is better than candidate 1"
```

Such statements would violate Recommendation V2 authority.

### 16.2 Raw-score privacy boundary

Raw scorer/LOO/recommendation numbers must not appear in the public UI payload.

Deterministic application code may use internal score information for fixed rendering logic, but raw values must not be passed through to Qwen or displayed to the user as probabilities.

---

## 17. Frontend/deploy contract

The deployment layer must treat `run["user_facing"]` as the primary end-user payload.

Recommendation images must be bound using the candidate `item_id` and `rank` from the authoritative Top-3.

Expected card structure:

```text
candidate image bound by item_id
candidate display_name
candidate reason
```

The frontend must preserve rank order `1 → 2 → 3`.

The frontend must not:

- rerank candidates based on VLM observations;
- hide a candidate because Qwen returned `ambiguous`;
- remove a candidate because Qwen returned `contradicts_recommendation`;
- substitute another image/candidate ID;
- render `run["explanation"]` or raw evidence as normal user copy;
- reinterpret compatibility logits as probabilities.

---

## 18. Canonical runtime configuration

Canonical config file:

```text
configs/vlm_qwen3_vl_4b_instruct_v2.json
```

Canonical settings:

```text
protocol_version:      vlm-explanation-v2
model:                 Qwen/Qwen3-VL-4B-Instruct
dtype:                 float16
device_map:            auto
require_cuda:          true
min_pixels/image:      262144
max_pixels/image:      262144
image_patch_size:      16
max_new_tokens:        1024
do_sample:             false
num_beams:             1
repetition_penalty:    1.05
max_validation_retries: 1
output language:       vi
include_raw_response:  true
```

A lower image pixel budget may be used as an explicit local/demo infrastructure override when the available GPU cannot fit the canonical visual budget.

Changing runtime pixel budget does not change:

- candidate identities;
- recommendation rank;
- evidence schema;
- prompt-context schema;
- validator contract;
- deploy handoff schema.

Canonical production behavior should use the canonical visual budget when hardware permits.

---

## 19. Failure behavior

The pipeline must fail closed for contract violations.

Examples of hard failures:

- recommendation evidence contains forbidden evaluation leakage;
- Recommendation Top-3 does not match frozen reranking order;
- candidate image map does not match authoritative Top-3 IDs;
- Qwen changes problematic item identity;
- Qwen changes candidate identity/rank;
- invalid taxonomy value;
- invalid context indices;
- invalid/missing limitation tokens;
- invalid JSON after repair budget is exhausted.

A malformed or authority-violating Qwen response must never be silently converted into a different recommendation decision.

---

## 20. Three-item outfit rule

When the original outfit has three items, LOO scoring necessarily evaluates two-item subsets.

The VLM must preserve the upstream LOO diagnosis rather than redefining it.

The machine-readable limitation:

```text
loo_uses_two_item_extrapolation
```

must be included when applicable.

This disclosure does not allow Qwen to change the problematic item.

---

## 21. Versioning and change control

The following are protocol-breaking changes and require an explicit new protocol/schema version or coordinated migration:

- allowing Qwen to choose the problematic item;
- allowing Qwen to rerank Recommendation V2;
- changing the number of authoritative candidates from three;
- exposing raw scorer/LOO/recommendation numerical values to Qwen;
- exposing benchmark ground truth to Qwen;
- adding unrestricted natural-language decision fields to Qwen output;
- changing candidate-image identity binding rules;
- changing required limitation semantics;
- changing deploy to consume internal/debug output instead of the public boundary.

Minor wording/documentation changes that do not alter schemas or authority do not require a protocol version change.

---

## 22. Required verification before merge/deploy

Before merging the VLM V2 branch into `main`, verify at minimum:

1. VLM V2 unit tests pass.
2. Leakage tests reject forbidden raw/nested Mapping fields.
3. Prompt-context tests confirm raw score values are not serialized to Qwen.
4. Validator tests confirm problematic item and Top-3 identity/rank cannot be changed.
5. Real Qwen NB10 smoke test completes with valid `vlm-visual-analysis-v2`.
6. `run["handoff"]` contains no raw score fields.
7. `run["user_facing"]` preserves the authoritative problematic item and all three Top-3 candidates in frozen rank order.
8. Frontend/deploy binds candidate images by `item_id` and rank.

Offline quality/error analysis can be performed separately, but it is not a blocker that changes VLM decision authority and it must not be inserted into the production decision path.

---

## 23. Canonical integration rule

For production integration, the intended ownership is:

```text
Scorer team
    owns compatibility scoring

LOO diagnosis
    owns problematic-item selection

Recommendation V2
    owns Top-3 candidate identities and rank

VLM V2
    owns constrained visual evidence only

Deterministic renderer
    owns public wording

Deploy/frontend
    owns API/UI presentation and candidate-image binding
```

No downstream component may silently take over an upstream authority role.

---

## 24. Final protocol statement

The final invariant of VLM V2 is:

> The problematic item and Recommendation V2 Top-3 are authoritative upstream outputs. Qwen may inspect and describe constrained visual relations, but it may never replace, remove, or rerank those decisions. Raw numerical scorer state and evaluation ground truth remain outside the Qwen prompt. Deterministic validation enforces the boundary, and deploy-facing code consumes only the intended score-free/public payloads.
