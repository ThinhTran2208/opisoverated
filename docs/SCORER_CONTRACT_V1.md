# SCORER CONTRACT V1 — Type-aware Pairwise

**Status:** `LOCKED_FOR_IMPLEMENTATION`  
**Canonical scorer name:** `type_aware_pairwise_v1`

Tài liệu này thay thế `SCORER_PLAN_V1_VI.md` và khóa interface, architecture details, training/evaluation protocol, checkpoint format và các implementation decisions cần thiết để triển khai Scorer V1.

---

## 1. Mục tiêu

Scorer V1 có một nhiệm vụ chính:

> Nhận một outfit đã được biểu diễn bằng FashionCLIP embeddings + coarse category của từng item, rồi trả về một **compatibility logit** cho toàn outfit.

Quy ước:

- logit càng cao → outfit càng compatible;
- scorer V1 chưa trả score 0–100 trực tiếp;
- calibration logit → 0–100 làm sau khi scorer ổn định;
- không dùng Low / Medium / High ở V1;
- ưu tiên simple / clean / debuggable trước khi tăng độ phức tạp.

---

## 2. Điều kiện để bắt đầu train

Scorer V1 chỉ train chính thức khi Data Processing V2 đã freeze và có:

```text
scorer_ready_v2_train.jsonl
scorer_ready_v2_valid.jsonl
scorer_ready_v2_test.jsonl

item metadata theo split
FashionCLIP embedding cache
dataset manifest
split manifest
final validation report
```

Frozen versions:

```text
dataset_version           = polyvore1000-core7-compat-v2
category_mapping_version  = core7-v2
negative_protocol_version = negative-v1
embedding_version         = fashionclip-512-l2-v1
```

Gate bắt buộc:

```text
READY_TO_TRAIN = true
embedding coverage = 100%
metadata coverage = 100%
negative sampling pass = true
cross-split leakage = 0
duplicate sample_id = 0
```

Nếu artifact/version/hash không khớp frozen reference thì scorer training phải hard-fail.

---

## 3. Input contract

### 3.1 Neural input

Mỗi item dùng:

```text
FashionCLIP embedding: 512-d
coarse_category
```

Batch-level input:

```text
item_embeddings      FloatTensor [B, MAX_ITEMS, 512]
coarse_category_ids  LongTensor  [B, MAX_ITEMS]
item_mask            BoolTensor  [B, MAX_ITEMS]
```

Training-only target:

```text
labels               FloatTensor [B]
```

V1 lock:

```text
MIN_ITEMS = 3
MAX_ITEMS = 8
```

### 3.2 Không dùng trực tiếp làm neural feature

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

Các field này vẫn được phép giữ cho provenance / evaluation / debugging.

Đặc biệt:

> `swapped_item_index` tuyệt đối không được đưa vào inference của scorer.

---

## 4. Category vocabulary lock

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

Category ID mapping được khóa:

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
category_count      = 7
category_vocab_size = 8
category_padding_idx = 0
```

Không tạo IDs theo thứ tự category xuất hiện trong dataset. Unknown coarse category phải hard-fail.

---

## 5. Output contract

API tối thiểu:

```python
output = {
    "compatibility_logit": logits
}
```

Trong batch:

```text
logits.shape == [B]
```

Output là raw logit. Không sigmoid trong model output contract.

Sau này calibration mới chuyển thành:

```text
compatibility_score ∈ [0, 100]
```

---

## 6. FashionCLIP contract

FashionCLIP embedding đã được tạo ở data pipeline.

Scorer:

- không encode ảnh lại;
- không fine-tune FashionCLIP;
- không normalize lại nếu embedding manifest đã xác nhận L2-normalized.

Input:

```text
x_i ∈ R^512
```

---

## 7. Dataset / DataLoader contract

### 7.1 Dataset responsibility

`src/scorer/dataset.py` chịu trách nhiệm:

```text
scorer-ready JSONL
    ↓
item IDs
    ↓
metadata lookup
    ↓
FashionCLIP embedding lookup
    ↓
coarse category IDs
    ↓
tensor sample
```

Không đọc ảnh trong training.

Embedding cache load một lần và dùng canonical lookup:

```text
item_id -> embedding row
```

Dataset phải hard-fail nếu:

```text
item thiếu embedding
item thiếu metadata
unknown coarse_category
outfit length < 3
outfit length > 8
label không thuộc {0, 1}
```

### 7.2 Padding

Mọi batch pad tới:

```text
MAX_ITEMS = 8
```

Padding values:

```text
embedding PAD = zero vector
category PAD  = 0
item_mask PAD = false
```

Ví dụ outfit 3 items:

```text
item_mask = [1, 1, 1, 0, 0, 0, 0, 0]
```

### 7.3 Loader behavior

```text
train loader: shuffle = true
valid loader: shuffle = false
test loader : shuffle = false
```

Không dùng weighted sampler ở V1.

---

## 8. Pair generation, pair mask và symmetry

Với `n` item thật:

```text
number_of_pairs = n(n-1)/2
```

Outfit 3–8 item → tối đa 28 valid pairs.

Pair generation dùng `i < j`. Pair mask chỉ true khi cả hai vị trí là item thật; padding không được tạo pair score.

Conceptually:

```python
pair_mask[b, i, j] = (
    item_mask[b, i]
    and item_mask[b, j]
    and i < j
)
```

### 8.1 Pair-order invariance lock

Outfit compatibility được xem là thuộc tính của một set garment, không phụ thuộc artificial array order.

Vì pair features có các thành phần order-sensitive (`h_i`, `h_j`, `c_i`, `c_j`), V1 khóa pair symmetry bằng **bidirectional scoring + mean**.

Với pair `(i, j)`:

```text
f_ij = [
    h_i,
    h_j,
    |h_i - h_j|,
    h_i * h_j,
    c_i,
    c_j
]

f_ji = [
    h_j,
    h_i,
    |h_i - h_j|,
    h_i * h_j,
    c_j,
    c_i
]
```

Dùng cùng một Pair MLP:

```text
s_forward = PairMLP(f_ij)
s_reverse = PairMLP(f_ji)
pair_score = 0.5 * (s_forward + s_reverse)
```

Do đó:

```text
score(pair i, j) = score(pair j, i)
```

Kết hợp với mean aggregation trên valid pairs, scorer phải permutation-invariant theo item ordering.

Bắt buộc có unit test:

```text
score(outfit) ≈ score(shuffled_outfit)
```

---

# 9. Type-aware Pairwise V1 architecture

## 9.1 Category embedding

Learned category embedding:

```text
Embedding(
    num_embeddings = 8,
    embedding_dim = 32,
    padding_idx = 0
)
```

Initialization được khóa để category vector không lấn át FashionCLIP vector
đã L2-normalized:

```text
category_embedding.weight ~ Normal(mean=0, std=0.02)
category_embedding.weight[PAD] = 0
```

Cho item `i`:

```text
c_i ∈ R^32
```

Không hard-code quan hệ thời trang; model học category representation trong training.

---

## 9.2 Item representation

Input item:

```text
[x_i ; c_i]
```

Baseline dimensions:

```text
512 + 32 = 544
```

Item MLP V1 được khóa:

```text
Linear(544, 256)
ReLU
Dropout(0.2)
Linear(256, 128)
ReLU
```

Output:

```text
h_i ∈ R^128
```

Các kích thước phải đọc từ config, không hard-code trong source.

---

## 9.3 Pair representation

Cho pair `(i, j)`:

```text
h_i
h_j
|h_i - h_j|
h_i * h_j
category_i embedding
category_j embedding
```

Baseline dimensions:

```text
h_i             128
h_j             128
|h_i - h_j|     128
h_i * h_j       128
c_i              32
c_j              32
--------------------
total            576
```

Mục tiêu:

> cùng một visual relation có thể mang ý nghĩa khác nhau theo loại item, ví dụ TOP ↔ BOTTOM, TOP ↔ SHOES, BAG ↔ SHOES.

---

## 9.4 Pair MLP

Pair MLP V1 được khóa:

```text
Linear(576, 128)
ReLU
Dropout(0.2)
Linear(128, 1)
```

Pair score cuối áp dụng bidirectional mean theo Section 8.

---

## 9.5 Outfit aggregation

Chỉ aggregate valid pair scores.

V1:

```text
mean pooling trên valid pairs
```

Không dùng attention ở V1.

Với tập valid pairs `P_b`:

```text
mean_pair_score_b
=
sum(pair_scores_b) / |P_b|
```

Không lấy mean trên padded pair positions.

---

## 9.6 Output MLP

Output head V1 được khóa:

```text
Linear(1, 16)
ReLU
Linear(16, 1)
```

Output cuối:

```text
compatibility_logit [B]
```

---

## 10. Model invariants và required tests

Phải test tối thiểu:

```text
output shape == [B]
output finite
pair count đúng n(n-1)/2
padding không tạo pair
padding không làm thay đổi score
pair swap symmetry
outfit permutation invariance
gradient flow tồn tại
backward không NaN/Inf
```

Model không được phụ thuộc source slot ordering.

---

# 11. Loss và optimization lock

## 11.1 Loss

Baseline V1:

```text
BCEWithLogitsLoss
```

Canonical usage:

```python
loss = criterion(
    compatibility_logit,
    labels,
)
```

Không sigmoid trước `BCEWithLogitsLoss`.

Không thêm ranking loss ở baseline V1. Chỉ cân nhắc `BCE + paired ranking loss` ở experiment sau nếu FITB / margin analysis cho thấy cần thiết.

## 11.2 Optimizer

Baseline:

```text
AdamW
learning_rate = 3e-4
weight_decay  = 1e-4
```

Các AdamW parameters không override dùng PyTorch defaults.

V1 khóa:

```text
lr_scheduler = none
gradient_clipping = none
```

Chỉ thêm gradient clipping ở experiment mới nếu có evidence của gradient instability.

---

## 12. Mixed precision và reproducibility

Nếu GPU hỗ trợ:

```text
AMP / autocast
```

được phép dùng để tăng tốc và giảm VRAM.

Mỗi run phải seed:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
DataLoader generator
```

Khi practical, bật deterministic settings phù hợp và tránh nondeterministic benchmark behavior.

Multi-seed values được khóa trước khi chạy:

```text
42
43
44
```

Canonical final-test seed:

```text
42
```

Không được chọn canonical seed sau khi nhìn test metrics.

---

# 13. Training protocol

Baseline:

```text
optimizer                 = AdamW
learning_rate             = 0.0003
weight_decay              = 0.0001
batch_size                = 256
max_epochs                = 30
early_stopping_patience   = 5
early_stopping_min_delta  = 0.0
mixed_precision           = true
seed                      = 42
```

Nếu GPU không đủ batch 256, giảm batch size phải được ghi thành config change / experiment mới; không silently thay đổi canonical run.

### Early stopping / checkpoint selection

Primary model-selection metric:

```text
validation ROC-AUC
```

Guardrail:

```text
validation 2-way FITB
```

Diagnostics:

```text
validation mean logit margin
validation median logit margin
```

Một epoch được coi là improvement khi:

```text
valid_roc_auc > best_valid_roc_auc
```

FITB không dùng làm hidden tie-breaker.

Test set không dùng để:

```text
chọn epoch
chọn learning rate
chọn architecture
chọn threshold
chọn dropout
chọn hidden size
chọn seed
```

---

# 14. Evaluation contract

## 14.1 ROC-AUC

ROC-AUC dùng raw compatibility logits. Không cần sigmoid trước metric.

## 14.2 2-way FITB

Mỗi negative lookup positive tương ứng bằng:

```text
paired_positive_sample_id
```

Không dựa vào row ordering.

Pair đúng khi:

```text
logit_positive > logit_negative
```

Tie được tính là incorrect.

## 14.3 Paired logit margin

```text
margin = logit_positive - logit_negative
```

Report:

```text
mean_logit_margin
median_logit_margin
```

`% margin > 0` tương đương FITB nhưng canonical metric name vẫn là:

```text
fitb_2way
```

Evaluator output tối thiểu:

```python
{
    "roc_auc": float,
    "fitb_2way": float,
    "mean_logit_margin": float,
    "median_logit_margin": float,
    "sample_count": int,
    "paired_family_count": int,
}
```

Evaluator phải hard-fail nếu paired positive/negative relationship không recover đầy đủ.

---

# 15. Development protocol S0–S8

## S0 — Interface lock

PASS khi:

```text
SCORER_CONTRACT_V1.md committed
YAML schema committed
input/output contract locked
category IDs locked
pair symmetry rule locked
checkpoint schema locked
metric names locked
seed protocol locked
```

## S1 — Dataset smoke test

Chạy khoảng 100–500 samples.

Check:

```text
embedding lookup đúng
category lookup đúng
padding đúng
item mask đúng
pair mask đúng
label đúng
positive-negative pairing đúng
embedding finite
category IDs hợp lệ
outfit length 3–8
metric synthetic tests đúng
```

Không full-train trước khi S1 PASS.

## S2 — Type-aware Pairwise implementation

PASS khi:

```text
category embedding PASS
item MLP PASS
pair feature generation PASS
pair MLP PASS
bidirectional pair symmetry PASS
mean valid-pair aggregation PASS
output MLP PASS
permutation invariance PASS
gradient flow PASS
```

## S2.5 — Tiny-set overfit

Tiny subset phải giữ complete positive-negative families.

Baseline sanity subset:

```text
32 positive + 32 paired negative = 64 samples
```

Có thể dùng 32–128 total samples nếu vẫn giữ complete paired families.

Expected sanity behavior:

```text
train loss giảm mạnh
train ROC-AUC tiến gần 1
train FITB tiến gần 1
paired margins chủ yếu dương
```

Nếu không overfit:

```text
debug trước
không tăng model size
không full-train
```

Debug order:

```text
labels
embedding lookup
category lookup
padding
item mask
pair mask
pair generation
logits
BCE usage
optimizer
gradient flow
```

## S3 — Full baseline training

Mỗi epoch:

```text
train
↓
validation inference
↓
valid loss
valid ROC-AUC
valid FITB
valid mean margin
valid median margin
↓
checkpoint decision bằng valid ROC-AUC
```

## S4 — Error analysis

Chỉ dùng validation set trong development.

Ưu tiên:

```text
high-confidence wrong positives
high-confidence wrong negatives
small paired margin
negative paired margin
```

Được phép join metadata để phân tích:

```text
coarse_category
master_category
outfit length
negative type
swap category
swapped_item_index
```

Canonical outputs:

```text
validation_predictions.jsonl
validation_metrics.json
```

Mỗi prediction row tối thiểu:

```text
sample_id
source_kit_id
paired_positive_sample_id
label
compatibility_logit
```

## S5 — Small tuning

Chỉ tune một số ít yếu tố:

```text
learning rate
item hidden size
pair hidden size
dropout
category embedding size
batch size
```

Không search rộng. Không tune trên test.

Sau S5:

```text
architecture LOCKED
hyperparameters LOCKED
selection rule LOCKED
```

## S6 — Multi-seed confirmation

Seeds:

```text
42
43
44
```

Giữ nguyên dataset / architecture / hyperparameters / protocol; chỉ thay seed.

S6 chỉ dùng train + validation.

Report:

```text
validation ROC-AUC mean ± std
validation FITB mean ± std
```

Nếu một seed collapse rõ rệt thì scorer chưa đủ stable để freeze.

## S7 — Final test once

Chỉ sau khi S5 config locked và S6 stability accepted.

Canonical protocol:

```text
canonical seed = 42
load best validation checkpoint của seed 42
run test once
```

Report:

```text
test ROC-AUC
test FITB
test mean logit margin
test median logit margin
```

Không chọn seed tốt nhất sau khi nhìn test và không tune lại sau test trong cùng scorer version.

## S8 — Freeze canonical scorer

Status:

```text
CANONICAL_SCORER_V1
```

Lưu:

```text
model checkpoint
model config
dataset version
dataset manifest hash
embedding version
embedding manifest hash
git commit
clean-tree status
metrics
random seed
training log
```

---

# 16. Checkpoint contract

Mỗi run lưu:

```text
best.pt
last.pt
```

`best.pt` = checkpoint có validation ROC-AUC tốt nhất.  
`last.pt` = checkpoint gần nhất, dùng để resume/debug.

Checkpoint schema tối thiểu:

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

Official freeze phải đến từ clean Git tree và resolvable commit.

Resume chỉ từ `last.pt` và phải restore model state, optimizer state, epoch/global step, config và provenance checks. Config/artifact hashes không khớp checkpoint phải hard-fail trừ khi tạo experiment version mới.

---

# 17. Inference contract

`src/scorer/inference.py` phải dùng cùng category mapping/model forward như training.

Inference không cần:

```text
label
negative metadata
swapped_item_index
```

Output tối thiểu:

```python
{
    "compatibility_logit": float
}
```

Batch inference:

```text
[B]
```

Inference test phải verify checkpoint load + forward + output finite.

---

# 18. Source structure

Canonical structure:

```text
docs/
└── SCORER_CONTRACT_V1.md

configs/
└── scorer_type_aware_pairwise_v1.yaml

src/scorer/
├── __init__.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── metrics.py
├── inference.py
└── checkpoint.py

tests/
├── test_scorer_dataset.py
├── test_pair_generation.py
├── test_scorer_model.py
├── test_scorer_metrics.py
└── test_scorer_inference.py

notebooks/experiments/
└── NB5_type_aware_pairwise_v1.ipynb
```

Core logic phải nằm trong `src/scorer/`.

Notebook không được chứa canonical duplicate implementation của Dataset, model classes, metrics, checkpoint logic hoặc training loop. Notebook chỉ là experiment wrapper / visualization surface.

Recommended notebook sections:

```text
0. Environment / runtime paths
1. Load config
2. Verify frozen artifacts
3. S1 Dataset/DataLoader smoke test
4. S2 model forward checks
5. S2.5 tiny-set overfit
6. S3 full training
7. Training curves
8. Validation metrics
9. S4 error analysis
10. S6 multi-seed run/results
11. S7 final test gate
```

Nếu notebook bị xóa, scorer training/evaluation vẫn phải chạy được từ source/CLI.

---

# 19. Dependency / portability lock

Scorer V1 core code được phép phụ thuộc vào:

```text
PyTorch
NumPy
scikit-learn
PyYAML
```

Không import:

```text
google.colab
```

Không hard-code:

```text
/content
Google Drive mount path
machine-specific absolute path
```

Exact package versions phải được record cho canonical run.

---

# 20. Canonical baseline config

```yaml
model:
  name: type_aware_pairwise_v1
  embedding_dim: 512
  category_count: 7
  category_vocab_size: 8
  category_padding_idx: 0
  category_embedding_dim: 32
  category_embedding_init_std: 0.02
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
  max_epochs: 30
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0
  lr_scheduler: none
  gradient_clipping: none
  mixed_precision: true
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

Các giá trị trên là starting baseline, không phải tuned optimum.

---

# 21. Canonical freeze reference

Checkpoint lớn lưu ở external artifact storage.

GitHub chỉ commit reference/provenance nhỏ, ví dụ:

```text
artifacts/scorer_v1_reference.json
```

Reference nên chứa tối thiểu:

```text
scorer_version
status
dataset_version
category_mapping_version
negative_protocol_version
embedding_version
dataset_manifest_sha256
embedding_manifest_sha256
git_commit
config_path
checkpoint_location
canonical_seed
validation_metrics
test_metrics
```

---

# 22. Explicitly deferred beyond V1

Không thuộc canonical baseline V1:

```text
Mean Pool ablation
ranking loss
attention aggregation
graph/hypergraph scorer
FashionCLIP fine-tuning
calibration 0–100
threshold selection
LOO diagnosis
recommendation reranking
large hyperparameter search
```

Các thay đổi trên phải là experiment/stage sau khi baseline `type_aware_pairwise_v1` đã ổn định.

---

# Final lock summary

```text
Architecture:
Type-aware Pairwise V1

Encoder:
Frozen FashionCLIP 512-d, precomputed, L2-normalized

Category:
Core-7 + PAD
fixed ID vocabulary

Item encoder:
[512 + 32] -> 256 -> 128
ReLU + Dropout

Pair feature:
h_i
h_j
|h_i - h_j|
h_i * h_j
c_i
c_j

Pair symmetry:
score both directions, then mean

Pair MLP:
576 -> 128 -> 1

Aggregation:
mean over valid pairs only

Output head:
1 -> 16 -> 1

Output:
compatibility_logit

Loss:
BCEWithLogitsLoss

Optimizer:
AdamW

Baseline LR:
3e-4

Batch:
256

Max epochs:
30

Early stopping:
validation ROC-AUC, patience 5

Guardrail:
2-way FITB

Diagnostics:
mean / median paired logit margin

Scheduler:
none

Gradient clipping:
none by default

Multi-seed:
42 / 43 / 44

Canonical final-test seed:
42

Test:
once after config lock

Canonical status:
CANONICAL_SCORER_V1
```
