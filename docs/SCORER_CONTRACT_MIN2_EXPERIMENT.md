# SCORER CONTRACT — MIN2 EXPERIMENT

**Branch:** `exp/min2-scorer-loo3`  
**Status:** `EXPERIMENTAL`  
**Base contract:** `docs/SCORER_CONTRACT_V2.md`  
**Experiment dataset version:** `polyvore1000-core7-compat-min2-exp-v1`  
**Scorer architecture:** `type_aware_pairwise_v1`

Tài liệu này là **experiment override** của `SCORER_CONTRACT_V2.md` trên branch này. Mọi điều khoản của Contract V2 vẫn giữ nguyên, trừ các thay đổi được nêu rõ bên dưới.

> Frozen Core-7 V2 benchmark trên `main` không được overwrite. Experiment MIN2 phải dùng artifact paths và dataset version riêng.

---

## 1. Mục tiêu experiment

Kiểm tra việc mở rộng pipeline từ data processing đến compatibility scorer để hỗ trợ outfit có **tối thiểu 2 items**, nhằm cho phép Leave-One-Out (LOO) trên outfit gốc 3 items tạo residual outfit 2 items mà scorer vẫn chấm được.

Hai boundary được tách riêng:

```text
Compatibility scorer:
    minimum = 2 items

LOO diagnosis:
    minimum original outfit = 3 items
```

Lý do: scorer Type-aware Pairwise vẫn có đúng một valid pair khi outfit có 2 items. Ngược lại, LOO trên outfit gốc 2 items sẽ tạo residual 1 item, không còn pair compatibility và không đủ context để localization một problematic item.

---

## 2. Data contract override

### 2.1 Outfit length

Experiment này khóa:

```text
MIN_SCORER_ITEMS = 2
MAX_SCORER_ITEMS = 8
LOO_MIN_ORIGINAL_ITEMS = 3
```

Các stage phải thống nhất cùng boundary `2–8`:

```text
raw Polyvore / Core-7 filtering
        ↓
category-clean positives: 2–8 items
        ↓
embedding coverage validation
        ↓
negative generation
        ↓
scorer-ready positives + negatives: 2–8 items
        ↓
ScorerDataset / collator
        ↓
Type-aware Pairwise scorer: 2–8 items
```

### 2.2 Version isolation

Không tái sử dụng tên dataset frozen V2 cho artifact mới.

Experiment version:

```text
polyvore1000-core7-compat-min2-exp-v1
```

Recommended runtime config:

```text
configs/data_paths.min2_experiment.json
```

Default isolated directories của experiment:

```text
data/core7_min2_exp_v1/
data/scorer_ready_min2_exp_v1/
```

Frozen artifacts sau đây trên main không được ghi đè:

```text
core7_drop_v2/
scorer_ready_v2/
polyvore1000-core7-compat-v2
```

---

## 3. Positive / negative semantics

Không thay negative protocol.

```text
negative_protocol_version = negative-v1
negative_type = same_category_different_kit
negatives_per_positive = 1
```

Negative vẫn thay đúng một item với replacement:

1. cùng `master_category`;
2. khác `item_id`;
3. đến từ kit khác;
4. không tồn tại sẵn trong positive outfit.

Positive và paired negative phải có cùng outfit length, bao gồm cả outfit length 2.

---

## 4. Scorer input contract

Neural inputs giữ nguyên:

```text
item_embeddings      FloatTensor [B, L, 512]
coarse_category_ids  LongTensor  [B, L]
item_mask            BoolTensor  [B, L]
pair_mask            BoolTensor  [B, L, L]
```

Experiment lock:

```text
2 <= real_item_count <= 8
```

Với `n` real items:

```text
number_of_pairs = n(n - 1) / 2
```

Do đó:

```text
n = 2 -> 1 pair -> valid scorer input
n = 1 -> 0 pair -> invalid scorer input
```

Model output vẫn là:

```python
{
    "compatibility_logit": logits
}
```

với `logits.shape == [B]`.

Architecture, category vocabulary, FashionCLIP embedding contract, Pair MLP, mean aggregation, Output MLP, loss và V5 optimization profile giữ nguyên Contract V2.

---

## 5. LOO diagnosis contract

LOO chỉ chạy khi original outfit có ít nhất 3 real items:

```text
LOO_MIN_ORIGINAL_ITEMS = 3
```

Với outfit `O = {x1, ..., xn}`:

```text
n >= 3
```

mỗi residual:

```text
O \ xi
```

có ít nhất 2 items và do đó vẫn nằm trong scorer input contract.

LOO delta giữ nguyên:

```text
Delta_i = C(O \ xi) - C(O)
```

Problematic item:

```text
argmax_i Delta_i
```

### Explicitly invalid

```text
original outfit size = 2
    ↓ LOO
residual size = 1
    ↓
INVALID for compatibility scorer
```

Do đó outfit 2 items có thể được dùng cho:

- compatibility scorer training;
- compatibility scorer evaluation;
- 2-way positive/negative ranking.

Nhưng **không** được đưa vào LOO Top-1 Localization Accuracy.

---

## 6. Scorer training profile

Experiment config:

```text
configs/scorer_type_aware_pairwise_min2_experiment.yaml
```

Chỉ thay outfit-length regime và regenerated data. Các V5 hyperparameters giữ nguyên để experiment có tính ablation:

```text
optimizer                 = AdamW
learning_rate             = 3e-4
weight_decay              = 1e-4
batch_size                = 256
max_epochs                = 60
early_stopping_patience   = 10
early_stopping_min_epochs = 30
mixed_precision           = false
seed                      = 42
```

Primary scorer metrics giữ nguyên:

```text
ROC-AUC
2-way FITB
```

Bắt buộc report thêm metric theo outfit length:

```text
n = 2
n = 3
n >= 4
overall
```

LOO evaluation chỉ report trên original outfits `n >= 3`, và nên tách:

```text
n = 3
n = 4
n >= 5
overall n >= 3
```

---

## 7. Acceptance criteria cho experiment

Không merge thay thế Contract V2 chỉ vì MIN2 pipeline chạy được.

Experiment chỉ được xem là có lợi nếu đồng thời:

1. data validation PASS với outfit length `2–8`;
2. scorer forward/backward PASS với 2-item outfits;
3. ROC-AUC / 2-way FITB trên shared subset `n >= 3` không regression đáng kể so với baseline;
4. LOO trên original 3-item outfits chạy hợp lệ vì residual 2-item được scorer hỗ trợ;
5. LOO Top-1 trên `n >= 3` đạt mức chấp nhận được;
6. frozen V2 artifacts trên main không bị thay đổi.

---

## 8. Required smoke tests

Data:

```text
2-item positive được giữ
1-item outfit bị drop
2-item negative có đúng một swap
positive/negative length bằng nhau
metadata + embedding coverage = 100%
```

Scorer:

```text
2 real items -> exactly 1 valid pair
1 real item -> hard fail
padding không tạo pair
2-item score finite
permutation invariance vẫn đúng
backward finite
```

Diagnosis:

```text
original 3 items -> tạo 3 residual outfits, mỗi residual có 2 items
original 2 items -> hard fail trước khi gọi scorer
```

---

## 9. Summary

```text
Compatibility scorer
    valid input: 2–8 items

LOO diagnosis
    valid original input: 3–8 items

Reason
    scorer cần >= 1 pair
    LOO residual cũng cần >= 1 pair
```

Đây là experiment branch contract; `docs/SCORER_CONTRACT_V2.md` trên `main` vẫn là frozen canonical reference cho benchmark 3–8 hiện tại.
