# SCORER PLAN V1 — Type-aware Pairwise

## 1. Mục tiêu

Scorer V1 có một nhiệm vụ chính:

> Nhận một outfit đã được biểu diễn bằng FashionCLIP embeddings + category của từng item, rồi trả về một **compatibility logit** cho toàn outfit.

Quy ước:

- logit càng cao → outfit càng compatible;
- scorer V1 **chưa** cần trả score 0–100 trực tiếp;
- bước calibration logit → 0–100 sẽ làm sau khi scorer đã ổn định;
- không dùng Low / Medium / High ở V1.

Scorer được ưu tiên theo hướng:

> **đơn giản, sạch, dễ debug, dễ giải thích trước; mạnh hơn sẽ tối ưu sau.**

---

## 2. Điều kiện để bắt đầu train

Chỉ bắt đầu scorer chính thức khi Data Processing V2 đã freeze và có:

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

Gate bắt buộc:

```text
READY_TO_TRAIN = true
embedding coverage = 100%
metadata coverage = 100%
negative sampling pass = true
cross-split leakage = 0
duplicate sample_id = 0
```

Nếu một gate fail → chưa train scorer chính thức.

---

## 3. Input của model

Mỗi outfit gồm 3–8 item.

Mỗi item dùng:

```text
item_id
FashionCLIP embedding: 512-d
coarse_category
```

`coarse_category` thuộc:

```text
TOP
BOTTOM
DRESS
OUTERWEAR
SHOES
BAG
HAT
```

### Không dùng trực tiếp trong model V1

```text
master_category
product_name
price
dominant_color
kit description
negative metadata
swapped_item_index
```

`master_category` vẫn được giữ cho provenance / evaluation / debugging.

Đặc biệt:

> `swapped_item_index` tuyệt đối không được đưa vào inference của scorer.

---

## 4. Output của model

API tối thiểu:

```python
output = {
    "compatibility_logit": float
}
```

Trong batch:

```text
[B] logits
```

Sau này calibration mới chuyển thành:

```text
compatibility_score ∈ [0, 100]
```

---

# 5. Kiến trúc V1

## 5.1 FashionCLIP

FashionCLIP đã tạo embedding 512-d ở data pipeline.

Scorer:

- không encode ảnh lại;
- không fine-tune FashionCLIP;
- không normalize lại nếu embedding manifest đã xác nhận L2-normalized.

Input:

```text
x_i ∈ R^512
```

---

## 5.2 Category embedding

Mỗi coarse category được biến thành một embedding nhỏ học được.

Ví dụ:

```text
TOP -> vector
BOTTOM -> vector
SHOES -> vector
...
```

Không hard-code quan hệ thời trang.

Model tự học trong quá trình train.

---

## 5.3 Item representation

Cho item `i`:

```text
FashionCLIP embedding
        +
category embedding
        ↓
small projection MLP
        ↓
item representation h_i
```

V1 nên dùng MLP nhỏ, ví dụ:

```text
512 + category_dim
        ↓
256
        ↓
128
```

Kích thước chính xác để config, không hard-code trong source.

---

## 5.4 Pairwise interaction

Với outfit:

```text
TOP
BOTTOM
SHOES
BAG
```

model tự tạo tất cả cặp:

```text
TOP ↔ BOTTOM
TOP ↔ SHOES
TOP ↔ BAG
BOTTOM ↔ SHOES
BOTTOM ↔ BAG
SHOES ↔ BAG
```

Với `n` item:

```text
number_of_pairs = n(n-1)/2
```

Outfit 3–8 item → tối đa chỉ 28 pairs, rất nhẹ.

---

## 5.5 Type-aware pair representation

Cho cặp `(i, j)`:

```text
h_i
h_j
|h_i - h_j|
h_i * h_j
category_i embedding
category_j embedding
```

concatenate lại rồi cho qua Pair MLP:

```text
pair features
    ↓
Pair MLP
    ↓
pair compatibility score
```

Mục tiêu V1:

> để model học được cùng một visual relation nhưng ý nghĩa khác nhau theo loại item.

Ví dụ:

```text
TOP ↔ BOTTOM
TOP ↔ SHOES
BAG ↔ SHOES
```

không bị coi là cùng một loại interaction.

---

## 5.6 Outfit aggregation

Các pair scores được tổng hợp thành outfit representation / score.

V1 ưu tiên:

```text
mean pooling trên valid pairs
        ↓
small output MLP
        ↓
compatibility_logit
```

Lý do chọn mean:

- đơn giản;
- không phụ thuộc số item quá mạnh;
- dễ debug;
- phù hợp outfit 3–8 item.

Chưa dùng attention phức tạp ở V1.

---

# 6. Loss

Dataset là:

```text
positive = 1
negative = 0
```

Loss chính:

```text
BCEWithLogitsLoss
```

V1 giữ đơn giản.

Không thêm ranking loss ngay từ đầu.

Sau khi baseline ổn mới cân nhắc:

```text
BCE + paired ranking loss
```

nếu FITB / margin chưa tốt.

---

# 7. DataLoader

## 7.1 Dataset class

`dataset.py` chịu trách nhiệm:

```text
JSONL sample
    ↓
item IDs
    ↓
lookup metadata
    ↓
lookup FashionCLIP embeddings
    ↓
tensor sample
```

Không đọc ảnh trong training.

---

## 7.2 Padding

Outfit có 3–8 item.

Batch tensor:

```text
embeddings: [B, MAX_ITEMS, 512]
categories: [B, MAX_ITEMS]
mask:       [B, MAX_ITEMS]
labels:     [B]
```

`MAX_ITEMS` lấy từ config.

Nếu frozen dataset V2 vẫn max = 8:

```text
max_items = 8
```

Mask chỉ phục vụ batching.

---

## 7.3 Pair mask

Pairwise model chỉ tính các pair có hai item thật.

Padding item không được tạo pair score.

---

# 8. Training protocol

## 8.1 GPU

Không cố định T4.

Ưu tiên GPU mạnh nhất đang có:

```text
A100 > L4 > T4
```

Nếu Colab chỉ có T4 thì T4 vẫn đủ cho scorer V1 vì:

- embedding đã precompute;
- model nhỏ;
- outfit tối đa 8 item;
- không fine-tune FashionCLIP.

---

## 8.2 Mixed precision

Nếu GPU hỗ trợ:

```text
AMP / fp16
```

để tăng tốc và giảm VRAM.

Model weights có thể train fp32 + autocast fp16.

---

## 8.3 Optimizer

Baseline:

```text
AdamW
```

Các hyperparameter để trong YAML:

```text
learning_rate
weight_decay
batch_size
epochs
dropout
hidden_dim
category_embedding_dim
```

Không nhét trực tiếp vào source.

---

## 8.4 Early stopping

Theo:

```text
validation ROC-AUC
```

Test set không dùng để:

- chọn epoch;
- chọn learning rate;
- chọn architecture;
- chọn threshold.

---

# 9. Evaluation

## Metric chính

### 9.1 ROC-AUC

Metric chính để chọn scorer.

```text
higher = better
```

---

### 9.2 2-way FITB

Với mỗi positive và negative được ghép cặp:

```text
score(positive) > score(negative)
```

thì tính là đúng.

FITB cho biết scorer có thực sự nhận ra outfit gốc tốt hơn outfit bị swap không.

---

### 9.3 Paired logit margin

```text
margin = logit_positive - logit_negative
```

Theo dõi:

```text
mean margin
median margin
% margin > 0
```

Đây là diagnostic, không phải metric chính để chọn model.

---

## Metric chưa ưu tiên V1

```text
F1
accuracy thresholded
precision
recall
```

Có thể report nếu tiện, nhưng không dùng làm model-selection chính.

---

# 10. Quy trình phát triển

## Stage S0 — Interface lock

Chốt:

```text
input schema
output schema
config schema
checkpoint format
metric names
```

Output:

```text
SCORER_CONTRACT_V1.md
```

---

## Stage S1 — Dataset smoke test

Chạy khoảng:

```text
100–500 samples
```

Check:

```text
embedding lookup đúng
category lookup đúng
mask đúng
label đúng
positive-negative pairing đúng
```

Không train full ngay.

---

## Stage S2 — Overfit sanity check

Cố tình train trên rất ít sample:

```text
32–128 samples
```

Model phải overfit được.

Nếu không overfit:

> ưu tiên tìm bug trước, không tăng model size.

---

## Stage S3 — Baseline training

Train Type-aware Pairwise V1 trên full train.

Theo dõi:

```text
train loss
valid loss
valid ROC-AUC
valid FITB
valid margin
```

Save best checkpoint theo validation ROC-AUC.

---

## Stage S4 — Error analysis

Không nhìn hàng trăm ví dụ.

Chỉ lấy các nhóm:

```text
high-confidence wrong positives
high-confidence wrong negatives
small/negative paired margin
```

Check xem lỗi đến từ:

```text
category
data
embedding
model
negative sampling difficulty
```

---

## Stage S5 — Small tuning

Chỉ tune một số ít yếu tố:

```text
learning rate
hidden size
dropout
category embedding size
batch size
```

Không search quá rộng.

Mục tiêu là scorer ổn định, không phải thắng benchmark bằng tuning cực lớn.

---

## Stage S6 — Multi-seed confirmation

Sau khi có config tốt nhất:

```text
seed 1
seed 2
seed 3
```

Report:

```text
mean ± std
```

cho ROC-AUC và FITB.

---

## Stage S7 — Final test

Chỉ sau khi config đã khóa:

```text
run test once
```

Report:

```text
test ROC-AUC
test FITB
test paired margin
```

Không quay lại tune theo test.

---

## Stage S8 — Freeze canonical scorer

Lưu:

```text
model checkpoint
model config
dataset version
dataset manifest hash
embedding version
git commit
metrics
random seed
training log
```

Tên dự kiến:

```text
type_aware_pairwise_v1
```

Status:

```text
CANONICAL_SCORER_V1
```

---

# 11. Cấu trúc source code

```text
src/
└── scorer/
    ├── __init__.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── metrics.py
    ├── inference.py
    └── checkpoint.py

configs/
└── scorer_type_aware_pairwise_v1.yaml

tests/
├── test_scorer_dataset.py
├── test_pair_generation.py
├── test_scorer_model.py
├── test_scorer_metrics.py
└── test_scorer_inference.py
```

Core code:

```text
không import google.colab
không hard-code Google Drive
không hard-code /content
```

Notebook chỉ là wrapper chạy CLI.

---

# 12. Config dự kiến

```yaml
model:
  name: type_aware_pairwise_v1
  embedding_dim: 512
  category_count: 7
  category_embedding_dim: 32
  item_hidden_dim: 128
  pair_hidden_dim: 128
  dropout: 0.2
  aggregation: mean

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
  mixed_precision: true
  seed: 42

selection:
  primary_metric: roc_auc
  guardrail_metric: fitb_2way
```

Các giá trị trên là **starting baseline**, không phải kết quả tuning cuối.

---

# 13. Acceptance criteria cho Scorer V1

Scorer chỉ được freeze khi:

```text
data manifest đúng frozen version
train / valid / test loader PASS
overfit sanity check PASS
không dùng test để tune
best checkpoint reproducible
ROC-AUC > random rõ ràng
FITB > random rõ ràng
paired margin trung bình > 0
3-seed result ổn định
inference API PASS
checkpoint + config + manifest được lưu
```

Không đặt một ROC-AUC target tùy ý trước khi có baseline chính thức.

---

# 14. Sau Scorer V1

Sau khi scorer freeze:

```text
Type-aware Pairwise scorer
        ↓
Calibration
logit → score 0–100
        ↓
LOO diagnosis
        ↓
candidate retrieval
        ↓
reranking
        ↓
recommendation
```

LOO sẽ dùng chính canonical scorer:

```text
full outfit score
vs
score khi bỏ từng item
```

nhưng scorer không được đọc `swapped_item_index`.

---

# 15. Thứ tự thực thi thực tế

```text
Data V2 READY_TO_TRAIN
        ↓
S0 scorer contract
        ↓
S1 DataLoader smoke test
        ↓
S2 tiny-set overfit
        ↓
S3 full baseline training
        ↓
S4 error analysis
        ↓
S5 small tuning
        ↓
S6 3-seed confirmation
        ↓
S7 final test
        ↓
S8 freeze canonical scorer
        ↓
calibration + diagnosis
```

---

# 16. Quyết định đã chốt cho V1

```text
Architecture:
Type-aware Pairwise

Philosophy:
simple / clean / debuggable first

Image encoder:
FashionCLIP frozen

Embedding:
512-d precomputed + L2 normalized

Categories:
Core-7

Outfit length:
3–8

Primary metric:
ROC-AUC

Guardrail:
2-way FITB

Diagnostic:
paired logit margin

GPU:
dùng GPU mạnh nhất available;
A100 > L4 > T4;
T4 vẫn đủ cho V1

Test set:
chỉ dùng sau khi model/config đã khóa

Product 0–100:
làm sau bằng calibration
```
