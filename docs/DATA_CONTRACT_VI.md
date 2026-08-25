# DATA CONTRACT — Hệ thống Fashion Outfit Compatibility

**Phiên bản:** `data-contract-v1.0`  
**Trạng thái:** Đã chốt cho giai đoạn phát triển scorer V1  
**Mục tiêu sản phẩm chính:** Chấm điểm compatibility tổng thể của outfit + hỗ trợ giải thích  
**Mục tiêu phụ:** Diagnosis và gợi ý thay item  
**Định dạng lưu trữ chuẩn:** JSONL + FashionCLIP embedding cache bên ngoài

---

## 1. Mục đích

Tài liệu này là **source of truth** cho cách dữ liệu được biểu diễn và trao đổi giữa các thành phần:

- data processing;
- embedding;
- scorer;
- diagnosis;
- recommendation;
- product/backend.

Data Contract được thiết kế theo hướng **architecture-neutral**, tức là không khóa toàn bộ pipeline vào một kiến trúc scorer cụ thể.

Contract phải hỗ trợ được:

- Mean Pooling;
- Type-aware Pairwise;
- hoặc các scorer khác trong tương lai.

Yêu cầu sản phẩm cốt lõi:

> Với một outfit, hệ thống phải đưa ra được một đánh giá compatibility tổng thể đáng tin cậy.  
> Phần giải thích là quan trọng, nhưng evidence để giải thích có thể đến từ nhiều nguồn khác nhau và không bắt buộc scorer phải tự sinh toàn bộ evidence.

---

## 2. Các quyết định thiết kế chính

### D1 — Trách nhiệm chính của scorer là đánh giá compatibility tổng thể

**Quyết định**

Scorer bắt buộc phải trả về một output tổng thể cho toàn bộ outfit.

**Lý do**

Mục tiêu sản phẩm hiện tại được ưu tiên theo thứ tự:

1. chấm compatibility tổng thể của outfit;
2. giải thích vì sao outfit được chấm như vậy;
3. diagnosis và suggestion là chức năng phụ.

Vì vậy core scorer contract không nên phụ thuộc vào một architecture phục vụ riêng cho recommendation hoặc pairwise analysis.

---

### D2 — Pairwise evidence là optional, không bắt buộc

**Quyết định**

Scorer **có thể** trả thêm:

- item-pair evidence;
- attention weights;
- item contributions;
- hoặc các diagnostic outputs khác.

Nhưng các output này là **optional**.

**Lý do**

Phần explanation có thể sử dụng evidence từ nhiều nguồn:

- pairwise interaction score;
- Leave-One-Out (LOO);
- category/type information;
- candidate replacement experiments;
- hoặc các diagnostic module khác.

Nếu bắt buộc mọi scorer phải trả pairwise evidence thì project vô tình bị khóa vào kiến trúc pairwise.

---

### D3 — Output chuẩn của model là raw compatibility logit

**Quyết định**

Mọi scorer bắt buộc phải trả:

```text
compatibility_logit
```

Quy ước:

```text
compatibility_logit càng cao
→ model đánh giá outfit càng compatible
```

**`compatibility_logit` được dùng cho:**

- training;
- model comparison;
- ROC-AUC / ranking metrics;
- FITB;
- Leave-One-Out;
- candidate reranking;
- debugging;
- calibration.

Raw logit **không được hiển thị trực tiếp cho user như phần trăm hoặc beauty score**.

---

### D4 — Product cuối phải có compatibility score dạng số từ 0–100

**Quyết định**

Pipeline production cuối cùng phải trả:

```text
compatibility_score ∈ [0, 100]
```

Luồng xử lý:

```text
Scorer
  ↓
compatibility_logit
  ↓
Calibration
  ↓
calibrated compatibility
  ↓
compatibility_score 0–100
```

Ví dụ:

```text
compatibility_logit = 1.73
        ↓
calibration
        ↓
calibrated value = 0.782
        ↓
compatibility_score = 78
```

**Lý do**

Product cần một con số cụ thể để người dùng dễ hiểu.

Tuy nhiên không được lấy raw logit rồi tự ý map trực tiếp thành phần trăm, vì điều đó có thể tạo ra ý nghĩa sai.

Phương pháp calibration cụ thể **chưa được chọn trong Data Contract**. Quyết định đó sẽ nằm trong `EVALUATION_PROTOCOL.md`.

### Quy tắc về ý nghĩa

Nếu:

```text
compatibility_score = 78
```

thì có nghĩa:

> Hệ thống đã calibration và gán cho outfit này compatibility score là 78/100.

Không được diễn giải thành:

- “outfit đẹp 78%”;
- “outfit objectively fashionable 78%”;
- “78% stylist sẽ thích outfit này”.

Training labels hiện tại không hỗ trợ các kết luận đó.

Các nhãn chữ như:

```text
Low / Medium / High
```

tạm thời **ngoài phạm vi V1**.

---

## 3. Ý nghĩa của dataset

### Positive sample

```text
label = 1
```

Positive là outfit gốc từ Polyvore.

Ý nghĩa:

> Sample này thuộc positive class của benchmark compatibility hiện tại.

Không có nghĩa:

> Outfit này đẹp một cách khách quan hoặc tất cả mọi người đều thấy đẹp.

---

### Negative V1

```text
label = 0
negative_type = "same_category_different_kit"
```

Negative V1 được tạo bằng cách thay **đúng một item** trong outfit gốc.

Replacement item phải thỏa các điều kiện:

1. cùng `master_category` với item bị thay;
2. có `item_id` khác;
3. đến từ kit khác;
4. không tồn tại sẵn trong outfit hiện tại.

Benchmark V1 được freeze ở:

```text
1 negative / 1 positive outfit
```

---

### Quy tắc freeze V1

Sau khi train/validation/test artifacts của V1 được tạo và kiểm tra, chúng được xem là **immutable benchmark artifacts**.

Nếu sau này muốn thử negative strategy mới thì phải tạo version khác.

Ví dụ:

```text
negative-v1 = same_category_different_kit
negative-v2 = semi_hard
negative-v3 = style_or_attribute_aware
```

Không được âm thầm ghi đè V1.

**Lý do**

Tất cả scorer phải được so sánh trên cùng một bộ dữ liệu.

Nếu mỗi scorer dùng negative khác nhau thì không thể biết improvement đến từ:

```text
model tốt hơn
```

hay chỉ vì:

```text
dataset dễ hơn
```

---

## 4. Sample Identity và Provenance

Mỗi sample phải truy ngược được về outfit gốc mà nó được tạo từ đó.

### Các field bắt buộc

| Field | Kiểu | Bắt buộc | Ý nghĩa |
|---|---|---:|---|
| `sample_id` | string | Có | ID duy nhất của sample |
| `source_kit_id` | string | Có | ID outfit Polyvore gốc |
| `label` | int | Có | `1` positive, `0` synthetic negative |
| `items` | list[string] | Có | Danh sách item ID của outfit |

Ví dụ một sample family:

```text
source kit: 100

100_pos
100_neg_1
100_hard_neg_1
```

Tất cả đều phải truy ra:

```text
source_kit_id = "100"
```

### Vì sao `source_kit_id` bắt buộc?

Dùng cho:

- chống data leakage khi split;
- debugging;
- reproduce negative;
- hard-negative generation;
- error analysis.

Nguyên tắc:

> Một original outfit và toàn bộ synthetic variants của nó phải được xem là cùng một outfit family.

Tỷ lệ split và cách tạo split cụ thể sẽ được định nghĩa trong `EVALUATION_PROTOCOL.md`.

---

## 5. Canonical Scorer-ready Sample Schema

### Positive

```json
{
  "sample_id": "214181831_pos",
  "source_kit_id": "214181831",
  "items": [
    "214181831_1",
    "214181831_2",
    "214181831_3"
  ],
  "label": 1,
  "negative_metadata": null
}
```

### Negative

```json
{
  "sample_id": "214181831_neg_1",
  "source_kit_id": "214181831",
  "items": [
    "214181831_1",
    "987654321_2",
    "214181831_3"
  ],
  "label": 0,
  "negative_metadata": {
    "negative_type": "same_category_different_kit",
    "swapped_item_index": 1,
    "original_item_id": "214181831_2",
    "replacement_item_id": "987654321_2",
    "swap_category": "example_master_category",
    "replacement_kit_id": "987654321"
  }
}
```

---

### Tương thích với artifacts hiện tại

Các JSONL cũ có thể đang dùng:

- `kit_id`;
- `original_item_id`;
- `replacement_item_id`;
- các metadata negative ở top-level.

Đối với canonical artifact mới:

- dùng `sample_id` để định danh sample;
- dùng `source_kit_id` để định danh outfit gốc;
- gom metadata chỉ dành cho negative vào `negative_metadata`.

Loader có thể hỗ trợ backward compatibility với artifact cũ.

---

## 6. Item Metadata Contract

Mọi `item_id` trong scorer-ready dataset phải truy được đến:

```text
master_category
coarse_category
```

### `master_category`

Taxonomy gốc / fine-grained category lấy từ dataset.

### `coarse_category`

Taxonomy do project định nghĩa để phục vụ modeling.

Ví dụ conceptual:

```text
fine category
    ↓
coarse garment type
```

Exact mapping phải được lưu ở artifact riêng có version.

Ví dụ:

```text
category_mapping_v1.json
```

### Invariant

```text
coarse_category
```

không bao giờ được overwrite:

```text
master_category
```

**Lý do**

`master_category` cần cho:

- reproducibility;
- diagnosis;
- negative provenance.

`coarse_category` có thể thay đổi qua experiment mà không làm mất source taxonomy.

---

## 7. FashionCLIP Embedding Contract

Embedding được lưu **bên ngoài JSONL**.

Canonical lookup:

```text
item_id
  ↓
FashionCLIP embedding
```

### Yêu cầu embedding V1

| Thuộc tính | Contract |
|---|---|
| Encoder | Frozen FashionCLIP image encoder |
| Representation | Projected image embedding |
| Dimension | 512 |
| Normalization | L2-normalized |
| Cache precision | FP16 được phép / ưu tiên |
| Key | `item_id` |
| Missing embedding | Không được phép với scorer-ready data |

### Embedding manifest bắt buộc

Artifact embedding phải có manifest versioned chứa ít nhất:

```text
embedding_version
model_name_or_version
preprocessing_version
embedding_dimension
normalization
dtype
item_count
```

**Lý do**

Việc chỉ biết:

```text
shape = [512]
```

là chưa đủ.

Teammate cần biết:

- model nào tạo embedding;
- preprocessing nào;
- output ở stage nào;
- có normalize hay chưa.

---

## 8. Batch Representation và Mask

Các outfit có số lượng item khác nhau.

Ví dụ:

```text
Outfit A:
shirt
pants
shoes
```

```text
Outfit B:
shirt
pants
shoes
bag
hat
```

Để GPU xử lý cùng batch, cần padding:

```text
A: shirt pants shoes PAD PAD
B: shirt pants shoes bag hat
```

Mask dùng để cho model biết vị trí nào là item thật:

```text
A mask: 1 1 1 0 0
B mask: 1 1 1 1 1
```

Trong đó:

```text
1 = item thật
0 = padding
```

### Canonical batch-level features

```text
item_embeddings      FloatTensor [B, L, 512]
item_mask            BoolTensor  [B, L]
master_category_ids  LongTensor  [B, L]
coarse_category_ids  LongTensor  [B, L]
```

Trong đó:

```text
B = số outfit trong batch
L = độ dài outfit lớn nhất trong batch
```

### Model không bắt buộc dùng mọi feature

Ví dụ:

```text
Mean Pool:
    item_embeddings
    item_mask
```

Type-aware scorer có thể dùng:

```text
item_embeddings
category/type IDs
item_mask
```

**Lý do**

Data interface phải đủ rộng để hỗ trợ nhiều architecture, nhưng không được ép mọi model dùng toàn bộ metadata.

---

## 9. Outfit Ordering

Danh sách item có thể giữ source slot order để:

- reproducibility;
- debugging;
- trace item swap.

Tuy nhiên compatibility được xem là thuộc tính của **một set các garment**, không phải phụ thuộc vào vị trí item trong array.

Do đó:

- source slot/order có thể lưu làm metadata;
- padding phải bị ignore qua `item_mask`;
- scorer không được tận dụng artificial position artifacts trừ khi một future model có định nghĩa rõ ràng và rationale hợp lý.

---

## 10. Core Scorer I/O Contract

### Input bắt buộc tối thiểu

```text
item_embeddings
item_mask
```

Các architecture cần category/type có thể dùng thêm:

```text
master_category_ids
coarse_category_ids
```

### Output bắt buộc

```json
{
  "compatibility_logit": 1.73
}
```

Quy ước:

```text
logit cao hơn
=
model đánh giá compatibility cao hơn
```

### Optional diagnostic output

Scorer có thể trả thêm:

```text
pairwise_evidence
item_contributions
attention_weights
other architecture-specific diagnostics
```

Nhưng các field trên **không phải core requirement**.

---

## 11. Product Scoring Contract

Pipeline scoring cuối:

```text
outfit
  ↓
scorer
  ↓
compatibility_logit
  ↓
calibration
  ↓
compatibility_score [0,100]
```

### Production output bắt buộc

```json
{
  "compatibility_logit": 1.73,
  "compatibility_score": 78,
  "scorer_version": "example_scorer_v1",
  "calibration_version": "example_calibration_v1"
}
```

### Ý nghĩa

`compatibility_logit`

- raw model output;
- dùng nội bộ ML;
- không giới hạn range.

`compatibility_score`

- score đã qua calibration;
- range `[0,100]`;
- dùng cho product/user-facing output.

`scorer_version`

- version của scorer/checkpoint đang sử dụng.

`calibration_version`

- version của calibration artifact.

Phương pháp calibration cụ thể và tiêu chí accept/reject sẽ được định nghĩa trong `EVALUATION_PROTOCOL.md`.

---

## 12. Boundary giữa Scorer và Explanation / Diagnosis

Scorer chịu trách nhiệm chính cho:

```text
overall outfit compatibility
```

Diagnosis/evidence layer có thể thu thập:

```text
Leave-One-Out score changes
pairwise evidence nếu scorer có
category/type information
candidate replacement score changes
```

Sau đó một structured evidence schema trong tương lai sẽ tổng hợp các signal này trước khi đưa vào VLM.

Thiết kế này giúp explanation không bị khóa vào một architecture cụ thể.

---

## 13. Validation bắt buộc cho scorer-ready dataset

Artifact phải fail validation nếu có bất kỳ lỗi nào sau:

- duplicate `sample_id`;
- thiếu `source_kit_id`;
- thiếu item metadata;
- item không có embedding;
- embedding V1 không phải 512-d;
- embedding có NaN / Inf;
- `label` không thuộc `{0,1}`;
- negative thiếu provenance;
- replacement item đã tồn tại sẵn trong original outfit;
- replacement category vi phạm negative rule V1;
- V1 artifact bị thay đổi nhưng dataset version không đổi.

---

## 14. Versioning

Ít nhất các component sau phải có version riêng:

```text
dataset_version
negative_protocol_version
category_mapping_version
embedding_version
scorer_version
calibration_version
```

Ví dụ:

```json
{
  "dataset_version": "polyvore-compat-v1",
  "negative_protocol_version": "same-category-different-kit-v1",
  "category_mapping_version": "coarse-category-v1",
  "embedding_version": "fashionclip-512-l2-v1",
  "scorer_version": "type-aware-v1",
  "calibration_version": "calibration-v1"
}
```

Thay đổi một component không được âm thầm làm thay đổi ý nghĩa của component khác.

---

## 15. Các quyết định cố ý để sang Evaluation Protocol

Data Contract **không tự chọn ngẫu nhiên** các quyết định sau:

- tỷ lệ train / validation / test;
- split manifest và split seed;
- primary metric;
- secondary metrics;
- model-selection rule;
- checkpoint-selection rule;
- multi-seed policy;
- mức regression ROC-AUC / FITB được phép;
- calibration algorithm;
- calibration metrics;
- calibration acceptance criteria;
- threshold selection;
- end-to-end acceptance criteria.

Các nội dung trên sẽ được chốt trong:

```text
EVALUATION_PROTOCOL.md
```

dựa trên mục tiêu thực tế của project và benchmark hiện có.

---

## 16. Teammate Implementation Checklist

- [ ] Mỗi sample có `sample_id` duy nhất.
- [ ] Mỗi sample có `source_kit_id` hợp lệ.
- [ ] Negative V1 tuân theo `same_category_different_kit`.
- [ ] Negative V1 đã freeze không được regenerate riêng cho từng scorer experiment.
- [ ] Mỗi item truy được `master_category`.
- [ ] Mỗi item truy được `coarse_category` từ mapping có version.
- [ ] Mỗi item scorer-ready có FashionCLIP embedding 512-d, L2-normalized.
- [ ] Padding position bị loại khỏi computation qua `item_mask`.
- [ ] Core scorer luôn trả `compatibility_logit`.
- [ ] Logit cao hơn luôn mang nghĩa compatibility cao hơn.
- [ ] Pairwise evidence chỉ là optional.
- [ ] Product score 0–100 chỉ được tạo qua calibration có version.
- [ ] Không gọi compatibility score là phần trăm “đẹp khách quan”.
- [ ] Evaluation result phải log dataset, embedding, scorer và calibration version.

---

## 17. Tóm tắt Contract

```text
Polyvore outfit / synthetic negative
        ↓
canonical JSONL + provenance
        ↓
item metadata
(master_category + coarse_category)
        ↓
item_id → frozen FashionCLIP 512-d embedding
        ↓
batch + padding mask
        ↓
architecture-neutral scorer
        ↓
compatibility_logit
        ├── diagnosis / evidence
        └── calibration
                ↓
        compatibility_score 0–100
```

### Nguyên tắc V1

```text
Core scorer contract
→ đơn giản và ổn định

Explanation evidence
→ modular, không khóa architecture

Dataset / embedding / scorer / calibration
→ versioned + reproducible
```
