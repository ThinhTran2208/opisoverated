# SCORER CONTRACT V2 — Type-aware Pairwise, Frozen V5 Protocol

**Status:** `DEVELOPMENT_FROZEN_FINAL_TEST_DEFERRED`  
**Contract version:** `scorer-contract-v2`  
**Canonical scorer/model version:** `type_aware_pairwise_v1`  
**Canonical training profile:** `configs/scorer_type_aware_pairwise_v1_val_auc.yaml`  
**Canonical development checkpoint:** `artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt`

Tài liệu này **supersede `SCORER_CONTRACT_V1.md` cho protocol thực tế đang dùng** sau toàn bộ scorer diagnostics và V5 training. `SCORER_CONTRACT_V1.md` vẫn được giữ như baseline/history.

> Contract V2 **không đổi tên model** thành `type_aware_pairwise_v2`. Checkpoint đã freeze mang `scorer_version = type_aware_pairwise_v1`; vì vậy model/version identifier tiếp tục giữ nguyên để không phá checkpoint compatibility. “V2” ở đây là **contract/protocol version**, không phải architecture checkpoint version.

---

# 0. Những thay đổi chính so với Scorer Contract V1

| Thành phần | Contract V1 ban đầu | Contract V2 / V5 canonical |
|---|---|---|
| Category embedding init | PyTorch default | `N(0, 1/sqrt(32))`, PAD = 0 |
| Thứ tự category re-init | chưa khóa | **sau khi toàn bộ MLP đã construct** |
| Category init policy | chưa có | `post_mlp_scale_preserving` |
| Training precision | AMP được phép / baseline `true` | **FP32 canonical** |
| Validation precision | theo training path | **FP32 bắt buộc** |
| Loss | BCE baseline; ranking có thể thử sau | **BCE only; ranking experiments dừng** |
| Train batching | shuffled sample batches | **standard sample-level shuffled batches** |
| Paired-family batching | chưa canonical | **không dùng** |
| Learning-rate scheduler | none | **none**, giữ fixed `3e-4` |
| Max epochs | 30 | **60** |
| Early-stop patience | 5 | **10** |
| Minimum epochs | chưa có | **30** |
| Model selection | valid ROC-AUC | valid ROC-AUC, giữ nguyên |
| Canonical seed | 42 | 42, giữ nguyên |
| Frozen validation AUC | chưa có | **0.6905082489625538** |
| Frozen checkpoint | external/reference dự kiến | **`best.pt` đã track trong repo** |
| Final test | S7 sau S6 | **deferred; chưa chạy** |

Các thay đổi trên đến từ scorer diagnostics đã hoàn tất; không phải thay đổi ngầm trong notebook.

---

# 1. Mục tiêu scorer

Scorer nhận một outfit được biểu diễn bằng FashionCLIP embeddings và coarse category của từng item, rồi trả về một **raw compatibility logit** cho toàn outfit.

Quy ước:

```text
logit cao hơn -> outfit được model đánh giá compatible hơn
```

Canonical output:

```python
{
    "compatibility_logit": logits
}
```

với:

```text
logits.shape == [B]
```

Không sigmoid trong model output. Calibration sang score 0–100 vẫn deferred.

---

# 2. Frozen data dependency

Scorer V2 tiếp tục dùng Data Processing V2 đã freeze:

```text
dataset_version           = polyvore1000-core7-compat-v2
category_mapping_version  = core7-v2
negative_protocol_version = negative-v1
embedding_version         = fashionclip-512-l2-v1
```

Canonical scorer-ready data:

```text
scorer_ready_v2_train.jsonl
scorer_ready_v2_valid.jsonl
scorer_ready_v2_test.jsonl
```

Required conditions:

```text
READY_TO_TRAIN = true
embedding coverage = 100%
metadata coverage = 100%
negative sampling pass = true
cross-split leakage = 0
duplicate sample_id = 0
```

Artifact/version/hash mismatch phải hard-fail theo existing runtime/checkpoint provenance checks.

---

# 3. Input contract

## 3.1 Neural inputs

```text
item_embeddings      FloatTensor [B, MAX_ITEMS, 512]
coarse_category_ids  LongTensor  [B, MAX_ITEMS]
item_mask            BoolTensor  [B, MAX_ITEMS]
pair_mask            BoolTensor  [B, MAX_ITEMS, MAX_ITEMS]  # optional to model, canonical collator supplies it
```

Training target:

```text
labels               FloatTensor [B]
```

Current scorer lock:

```text
MIN_ITEMS = 3
MAX_ITEMS = 8
```

**Important downstream limitation:** canonical scorer vẫn reject outfit có ít hơn 3 real items. Vì vậy LOO trên một outfit gốc có đúng 3 items sẽ tạo outfit 2 items và **không hợp lệ với scorer contract hiện tại**. Việc hỗ trợ `MIN_ITEMS = 2` phải là một contract/model-data change riêng; chưa thuộc V2 này.

## 3.2 Không dùng trực tiếp làm neural feature

```text
master_category
product_name
price
dominant_color
kit description
negative_metadata
swapped_item_index
item_id
source_kit_id
source slot
```

Các field này chỉ dùng cho provenance, pairing, debugging, evaluation hoặc downstream diagnosis.

`swapped_item_index` tuyệt đối không được đưa vào scorer forward/inference.

---

# 4. Category vocabulary lock

Core-7:

```text
TOP
BOTTOM
DRESS
OUTERWEAR
SHOES
BAG
HAT
```

ID mapping:

```text
0 = PAD
1 = TOP
2 = BOTTOM
3 = DRESS
4 = OUTERWEAR
5 = SHOES
6 = BAG
7 = HAT
```

Config semantics:

```text
category_count       = 7
category_vocab_size  = 8
category_padding_idx = 0
```

Unknown category phải hard-fail.

---

# 5. FashionCLIP contract

FashionCLIP embedding được tạo trước ở data pipeline.

Scorer:

```text
không đọc ảnh trong training
không encode ảnh lại
không fine-tune FashionCLIP
không normalize lại embedding đã được manifest xác nhận L2-normalized
```

Mỗi item:

```text
x_i in R^512
||x_i||_2 ~= 1
```

---

# 6. Dataset / DataLoader contract

`src/scorer/dataset.py` giữ responsibility:

```text
JSONL
 -> item IDs
 -> metadata/category lookup
 -> FashionCLIP embedding lookup
 -> tensor sample
 -> collate + padding + masks
```

Dataset hard-fail nếu:

```text
missing embedding
missing metadata
unknown coarse_category
outfit length < 3
outfit length > 8
invalid label
invalid positive-negative pairing
```

Padding:

```text
MAX_ITEMS = 8
embedding PAD = zero vector
category PAD  = 0
item_mask PAD = false
```

Canonical loader behavior:

```text
train loader: shuffle = true
valid loader: shuffle = false
test loader : shuffle = false
```

Train loader phải dùng **standard sample-level batching**. Không dùng weighted sampler và không dùng paired-family batch sampler trong canonical V5.

Reproducible sample order:

```python
generator = torch.Generator()
generator.manual_seed(seed)
```

DataLoader-local generator phải được tạo fresh cho canonical run. Không reuse một loader đã bị iterate bởi experiment trước đó.

---

# 7. Pair generation, pair mask và permutation invariance

Với `n` real items:

```text
number_of_pairs = n(n - 1) / 2
```

Pair indices dùng `i < j`.

Canonical pair mask:

```python
pair_mask[b, i, j] = (
    item_mask[b, i]
    and item_mask[b, j]
    and i < j
)
```

Nếu `pair_mask` được truyền vào model thì phải đúng canonical upper-triangle mask; mismatch phải hard-fail.

Pair symmetry vẫn dùng **bidirectional scoring + mean**:

```text
f_ij = [h_i, h_j, |h_i-h_j|, h_i*h_j, c_i, c_j]
f_ji = [h_j, h_i, |h_i-h_j|, h_i*h_j, c_j, c_i]

pair_score = 0.5 * (PairMLP(f_ij) + PairMLP(f_ji))
```

Outfit score phải permutation-invariant theo item ordering.

---

# 8. Canonical architecture

Architecture dimensions không đổi so với V1.

## 8.1 Category embedding

```text
Embedding(
    num_embeddings = 8,
    embedding_dim = 32,
    padding_idx = 0
)
```

### V2 lock — category initialization

Đây là thay đổi architecture-initialization quan trọng nhất so với Contract V1.

FashionCLIP inputs có norm xấp xỉ 1, trong khi default `nn.Embedding(8, 32)` có category-vector norm ban đầu lớn hơn đáng kể. Diagnostics cho thấy category signal scale quá lớn làm model collapse/underfit.

Canonical initialization:

```python
category_embedding_init_std = category_embedding_dim ** -0.5

nn.init.normal_(
    category_embedding.weight,
    mean=0.0,
    std=category_embedding_init_std,
)

category_embedding.weight[0] = 0
```

Với dim 32:

```text
std = 1 / sqrt(32)
expected category-vector norm ~= 1
```

### V2 lock — initialization order

Category re-initialization phải xảy ra **sau khi** `item_mlp`, `pair_mlp` và `output_mlp` đã được construct.

Canonical sequence:

```text
construct nn.Embedding with PyTorch default init
construct Item MLP
construct Pair MLP
construct Output MLP
re-initialize only category_embedding with std = 1/sqrt(32)
zero PAD row
```

Reason: `nn.init.normal_()` consume RNG state. Nếu re-init category trước khi tạo downstream Linear layers, cùng seed vẫn tạo ra MLP weights khác. V2 khóa ordering này để category scale fix không perturb downstream MLP initialization trajectory.

Implementation marker:

```text
category_embedding_init_policy = post_mlp_scale_preserving
```

Required regression test: category scaling không được làm thay đổi downstream MLP initialization so với equivalent construction order.

## 8.2 Item MLP

Input:

```text
[x_i ; c_i] = 512 + 32 = 544
```

Canonical:

```text
Linear(544, 256)
ReLU
Dropout(0.2)
Linear(256, 128)
ReLU
```

Output:

```text
h_i in R^128
```

## 8.3 Pair feature

```text
[h_i, h_j, |h_i-h_j|, h_i*h_j, c_i, c_j]
```

Dimension:

```text
128 + 128 + 128 + 128 + 32 + 32 = 576
```

## 8.4 Pair MLP

```text
Linear(576, 128)
ReLU
Dropout(0.2)
Linear(128, 1)
```

## 8.5 Outfit aggregation

```text
mean over valid pair scores only
```

Không attention. Padding không được tham gia mean.

## 8.6 Output MLP

```text
Linear(1, 16)
ReLU
Linear(16, 1)
```

Output:

```text
compatibility_logit [B]
```

---

# 9. Model invariants và required tests

Required:

```text
output shape == [B]
output finite
pair count == n(n-1)/2
padding không tạo pair
padding không làm thay đổi score
pair swap symmetry
outfit permutation invariance
gradient flow tồn tại
backward không NaN/Inf
real item không dùng PAD category
padded position bắt buộc category ID = 0
category init mean norm xấp xỉ 1
PAD category embedding norm = 0
category init ordering không perturb downstream MLP RNG init
```

---

# 10. Loss và optimization lock

## 10.1 Loss

Canonical V2/V5:

```text
BCEWithLogitsLoss only
```

```python
loss = BCEWithLogitsLoss(
    compatibility_logit,
    labels,
)
```

Không sigmoid trước BCE.

### Ranking-loss decision

Các paired-ranking experiments đã được chạy trong development nhưng **không được chọn làm canonical path**.

V2 lock:

```text
ranking loss = OFF
paired-family BCE batching = OFF
standard sample-level BCE batching = ON
```

Không thêm ranking term vào frozen V5 checkpoint/protocol.

## 10.2 Optimizer

```text
AdamW
learning_rate = 3e-4
weight_decay  = 1e-4
```

Các AdamW parameters còn lại dùng PyTorch defaults.

## 10.3 Scheduler / clipping

Canonical:

```text
lr_scheduler = none
gradient_clipping = none
```

Một `ReduceLROnPlateau` ablation đã được thử sau V5 nhưng không thay thế V5. Vì vậy fixed `3e-4` vẫn là canonical scorer-development protocol.

---

# 11. Precision và reproducibility

## 11.1 Canonical precision

V2 thay đổi V1 ở điểm này.

Canonical V5:

```text
training precision   = FP32
validation precision = FP32
mixed_precision      = false
```

Validation **luôn FP32** trong canonical evaluator. Lý do: FP16 validation từng quantize các paired margins rất nhỏ thành exact ties, làm strict 2-way FITB bị sai lệch.

AMP vẫn có thể tồn tại trong source như một experimental capability, nhưng **không phải canonical V5 training protocol**.

## 11.2 Seeds

Mỗi run phải seed:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
DataLoader-local generator
```

Canonical seed:

```text
42
```

Predeclared confirmation seeds vẫn là:

```text
42, 43, 44
```

Tuy nhiên seed 43/44 confirmation chưa phải điều kiện blocking cho downstream work trong trạng thái hiện tại; xem Section 16.

---

# 12. Canonical V5 training protocol

Canonical config:

```text
optimizer                    = AdamW
learning_rate                = 0.0003
weight_decay                 = 0.0001
batch_size                   = 256
max_epochs                   = 60
early_stopping_patience      = 10
early_stopping_min_epochs    = 30
early_stopping_min_delta     = 0.0
lr_scheduler                 = none
gradient_clipping            = none
mixed_precision              = false
seed                         = 42
```

Early stopping không được kích hoạt trước epoch 30.

Improvement rule:

```text
valid_roc_auc > best_valid_roc_auc
```

Primary model-selection metric:

```text
validation ROC-AUC
```

Guardrail:

```text
validation fitb_2way
```

Diagnostics:

```text
validation loss
validation mean logit margin
validation median logit margin
```

FITB không được dùng làm hidden tie-breaker khi AUC không improve.

---

# 13. Evaluation contract

## 13.1 ROC-AUC

Dùng raw logits.

```text
primary development metric = roc_auc
```

Không sigmoid trước metric.

## 13.2 2-way FITB

Negative lookup paired positive qua:

```text
paired_positive_sample_id
```

Không dựa vào adjacency hoặc row ordering.

Correct iff:

```text
logit_positive > logit_negative
```

Tie = incorrect.

## 13.3 Paired margin

```text
margin = logit_positive - logit_negative
```

Report:

```text
mean_logit_margin
median_logit_margin
```

Evaluator output:

```python
{
    "loss": float,
    "roc_auc": float,
    "fitb_2way": float,
    "mean_logit_margin": float,
    "median_logit_margin": float,
    "sample_count": int,
    "paired_family_count": int,
}
```

Canonical validation forward không dùng autocast.

---

# 14. Frozen V5 development result

Canonical run:

```text
run name      = final_val_auc_v5_seed42
seed          = 42
precision     = FP32 train / FP32 valid
epochs ran    = 60
best epoch    = 52
stopped early = false
```

Frozen validation result at `best.pt`:

```text
validation loss            = 0.6972924366515071
validation ROC-AUC         = 0.6905082489625538
validation FITB 2-way      = 0.7626970227670753
mean logit margin          = 1.0499372052358245
median logit margin        = 0.7358774170279503
sample_count               = 2284
paired_family_count        = 1142
```

Model-selection interpretation:

- epoch 52 được chọn vì có validation ROC-AUC cao nhất;
- epoch 60 là `last.pt` trong training run nhưng không phải canonical best model;
- validation BCE tăng ở các epoch muộn không override AUC selection rule;
- V5 vẫn là canonical development checkpoint sau scheduler ablation.

---

# 15. Checkpoint contract và frozen checkpoint

Per run:

```text
best.pt = highest validation ROC-AUC checkpoint
last.pt = latest training checkpoint
```

Checkpoint schema:

```python
{
    "scorer_version": "type_aware_pairwise_v1",
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "global_step": ...,
    "config": ...,

    "dataset_version": "polyvore1000-core7-compat-v2",
    "category_mapping_version": "core7-v2",
    "negative_protocol_version": "negative-v1",
    "embedding_version": "fashionclip-512-l2-v1",

    "dataset_manifest_sha256": "...",
    "embedding_manifest_sha256": "...",
    "git_commit": "...",
    "git_tree_clean": true,

    "seed": ...,
    "best_valid_roc_auc": ...,
    "validation_metrics": {...},
}
```

Frozen V5 checkpoint metadata:

```text
path:
artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt

file size          = 2,978,247 bytes
sha256             = 7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c
checkpoint epoch   = 52
global_step        = 6292
seed               = 42
scorer_version     = type_aware_pairwise_v1
training git commit= 7cbbb19fd89352b7ef54038e57b4d8208b7ee1f6
git_tree_clean     = true
```

Lưu ý provenance:

> `git_commit` nằm **bên trong checkpoint** là commit của source/config tại thời điểm training. Việc sau đó copy checkpoint binary vào branch `main` tạo commit mới **không được phép rewrite provenance bên trong checkpoint**.

Repo có `.gitignore` chung cho `*.pt`; frozen canonical `best.pt` ở path trên là một deliberate tracked exception. Các checkpoint experiment khác vẫn nên external/ignored, trừ khi được freeze rõ ràng.

---

# 16. Development-stage status sau V5

## S0 — Interface / contract

```text
PASS (V1 historical, V2 now supersedes protocol)
```

## S1 — Dataset smoke test

```text
PASS
```

## S2 — Model implementation

```text
PASS
```

Architecture, symmetry, pair masking, permutation invariance và gradient tests đã được triển khai.

## S2.5 — Tiny-set overfit

```text
PASS
```

Tiny paired set đã có thể memorize gần hoàn toàn; implementation có capacity và gradient flow.

## S3 — Full training / scorer diagnosis

```text
PASS
```

Các vấn đề chính đã được diagnose:

```text
category-vs-FashionCLIP feature-scale imbalance
FP16 validation tie quantization
paired-family batching degradation
training trajectory / initialization-order sensitivity
```

## S4/S5 — Focused ablation / model selection

```text
PASS for V5 selection
```

Canonical decisions:

```text
scaled category embedding retained
category OFF not selected
ranking loss not selected
paired-family batching not selected
AMP training not selected for canonical V5
fixed LR retained
V5 selected
```

## S6 — Multi-seed confirmation

```text
DEFERRED
```

Seeds 43/44 có thể chạy sau để quantify stability. Không được dùng test set để chọn seed.

## S7 — Final test once

```text
DEFERRED / NOT RUN
```

Test split vẫn phải được coi là held-out. Khi quay lại final evaluation:

```text
load exactly frozen V5 best.pt
run test once
report test ROC-AUC / FITB / mean margin / median margin
không retrain hoặc tune dựa trên test
```

## S8 — Freeze status

Current status:

```text
SCORER_V1_DEVELOPMENT_FROZEN
FINAL_TEST_PENDING
DOWNSTREAM_USE_ALLOWED
```

Không được ghi `FINAL_CANONICAL_TESTED` trước khi S7 thực sự chạy.

---

# 17. Inference / downstream contract

Downstream stages phải load **frozen `best.pt`**, không dùng `last.pt`.

Inference inputs/output giống training model contract nhưng không cần label hoặc negative metadata.

Canonical downstream checkpoint:

```text
artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt
```

Use cases allowed trước final test:

```text
LOO diagnosis development
candidate reranking development
pipeline integration
validation-only downstream metrics
```

Không dùng deferred test split để tune downstream scorer behavior.

---

# 18. Source/config structure

Canonical scorer components:

```text
docs/
├── SCORER_CONTRACT_V1.md        # historical baseline
└── SCORER_CONTRACT_V2.md        # current protocol

configs/
├── scorer_type_aware_pairwise_v1.yaml
└── scorer_type_aware_pairwise_v1_val_auc.yaml

src/scorer/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── metrics.py
└── checkpoint.py

tests/
├── test_scorer_dataset.py
├── test_pair_generation.py
├── test_scorer_model.py
├── test_scorer_init_order.py
├── test_scorer_metrics.py
└── test_scorer_train.py

notebooks/experiments/
└── NB6_final_val_auc_v5.ipynb

artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/
└── best.pt
```

Core model/training logic phải nằm trong `src/scorer/`. Notebook là orchestration/verification surface, không phải source-of-truth architecture implementation.

---

# 19. Canonical V2/V5 config

```yaml
model:
  name: type_aware_pairwise_v1
  embedding_dim: 512
  category_count: 7
  category_vocab_size: 8
  category_padding_idx: 0
  category_embedding_dim: 32
  item_projection_dim: 256
  item_hidden_dim: 128
  pair_hidden_dim: 128
  output_hidden_dim: 16
  activation: relu
  dropout: 0.2
  aggregation: mean
  pair_symmetry: bidirectional_mean

data:
  min_items: 3
  max_items: 8

training:
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001
  batch_size: 256
  max_epochs: 60
  early_stopping_patience: 10
  early_stopping_min_epochs: 30
  early_stopping_min_delta: 0.0
  lr_scheduler: none
  gradient_clipping: none
  mixed_precision: false
  seed: 42

selection:
  primary_metric: roc_auc
  guardrail_metric: fitb_2way

confirmation:
  seeds:
    - 42
    - 43
    - 44
  canonical_seed: 42
```

---

# 20. Explicitly deferred beyond current scorer freeze

Không thuộc frozen V5 scorer contract:

```text
final test evaluation       # deferred, phải dùng exact frozen best.pt
multi-seed confirmation 43/44
calibration logit -> 0-100
threshold selection
attention aggregation
graph/hypergraph scorer
FashionCLIP fine-tuning
ranking-loss revival
broad hyperparameter search
MIN_ITEMS = 2 migration
```

LOO diagnosis và recommendation là **downstream stages**; chúng có thể sử dụng scorer V5 frozen nhưng không được âm thầm thay scorer contract.

---

# Final lock summary

```text
CONTRACT:
scorer-contract-v2

MODEL VERSION:
type_aware_pairwise_v1

ARCHITECTURE:
FashionCLIP 512-d
+ learned Core-7 category embedding 32-d
+ Item MLP 544 -> 256 -> 128
+ symmetric pair feature 576-d
+ Pair MLP 576 -> 128 -> 1
+ mean valid-pair aggregation
+ Output MLP 1 -> 16 -> 1

CATEGORY INIT:
post_mlp_scale_preserving
Normal(0, 1/sqrt(32))
PAD row = 0

TRAINING:
BCEWithLogitsLoss only
AdamW
lr = 3e-4 fixed
weight_decay = 1e-4
batch = 256
FP32 train
FP32 valid
max_epochs = 60
min_epochs = 30
early-stop patience = 10
no scheduler
no grad clipping
seed = 42
standard sample-level shuffle

SELECTION:
validation ROC-AUC only
FITB = guardrail
strict AUC improvement

FROZEN V5:
best epoch = 52
val ROC-AUC = 0.6905082489625538
val FITB = 0.7626970227670753
mean margin = 1.0499372052358245
median margin = 0.7358774170279503

CHECKPOINT:
artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt

STATUS:
SCORER_V1_DEVELOPMENT_FROZEN
FINAL_TEST_PENDING
DOWNSTREAM_USE_ALLOWED
```
