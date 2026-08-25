# Team Workflow — Hướng dẫn cho người mới

## 1. Ownership tối giản

### Leader

Leader chịu trách nhiệm:
- giữ roadmap chung;
- chốt version dataset / embedding / scorer đang dùng;
- chốt experiment nào được chạy;
- kiểm tra Data Contract và Metrics không bị thay đổi tùy tiện;
- xử lý dependency giữa các assistants;
- quyết định cuối: `KEEP`, `REJECT`, hoặc `RETRY`.

Leader không cần tự code tất cả các module.

### Assistant 1 — Data + Evaluation

Owner:

```text
src/data/
src/evaluation/
```

Làm:
- canonical JSONL;
- `sample_id`, `source_kit_id`, provenance;
- `master_category`, `coarse_category`;
- embedding manifest;
- validation;
- evaluation chuẩn.

Output:
- dataset/version;
- validation report;
- metric report.

### Assistant 2 — Scorer + Diagnosis + Recommendation

Owner:

```text
src/scorer/
src/diagnosis/
src/recommendation/
```

Làm:
- train scorer;
- output `compatibility_logit`;
- LOO diagnosis;
- retrieve candidate;
- scorer rerank.

Metrics:
- Scorer: ROC-AUC, 2-way FITB.
- Diagnosis: LOO Top-1 Localization Accuracy, LOO Hit@2.
- Retrieval: Recall@200 là chính.
- Rerank: Recall@5 là chính.

### Assistant 3 — Detection + VLM

Owner:

```text
src/detection/
src/vlm/
```

Làm:
- detect/crop garment;
- map category;
- dùng đúng FashionCLIP preprocessing;
- kiểm tra scorer trên ảnh/crop thật;
- tạo prompt VLM;
- VLM chỉ giải thích structured evidence, không tự tạo compatibility score mới.

---

## 2. Luồng của Leader

### Bước 1 — Freeze input

Xác nhận các version đang dùng:

```text
dataset_version
negative_protocol_version
category_mapping_version
embedding_version
```

### Bước 2 — Chốt experiment

Mỗi experiment phải có:
- câu hỏi;
- owner;
- dataset version;
- config;
- primary metric;
- success condition.

Ví dụ:

```text
Type-aware + ranking loss có tăng 2-way FITB
mà không làm ROC-AUC giảm đáng kể không?
```

### Bước 3 — Cho assistants làm song song

Có thể chạy song song:

```text
Data/Eval
Scorer/Recommendation
Detection/VLM
```

nhưng mọi phần phải tuân theo cùng Data Contract.

### Bước 4 — Integration check

Kiểm tra pipeline:

```text
Data
→ Embedding
→ Scorer
→ LOO
→ Recommendation
→ Detection input
→ VLM explanation
```

### Bước 5 — Chốt kết quả

Experiment chỉ được xem là hoàn thành khi có:
- config;
- metrics;
- checkpoint/artifact;
- failure cases;
- decision.

Decision:

```text
KEEP
REJECT
RETRY
```

---

## 3. Luồng Assistant 1 — Data + Evaluation

1. Đọc `DATA_CONTRACT_VI.md`.
2. Tạo/kiểm tra canonical JSONL.
3. Validate:
   - `sample_id` unique;
   - có `source_kit_id`;
   - negative provenance đầy đủ;
   - replacement đúng rule V1;
   - mọi item có metadata;
   - mọi item có FashionCLIP embedding 512-d, L2-normalized.
4. Freeze version.
5. Không ghi đè V1 nếu muốn thử negative strategy mới.
6. Chạy evaluation bằng metric definitions cố định.

---

## 4. Luồng Assistant 2 — Scorer + Diagnosis + Recommendation

### Scorer

```text
batch scorer-ready
→ model
→ compatibility_logit
```

Đánh giá:
- ROC-AUC;
- 2-way FITB;
- mean/median logit margin;
- F1.

### Diagnosis

```text
score outfit
→ remove từng item
→ score lại
→ tính LOO delta
→ rank item
```

Đánh giá:
- LOO Top-1 Localization Accuracy;
- LOO Hit@2.

### Recommendation

```text
problem item
→ filter candidate
→ retrieve Top-200
→ scorer rerank
→ Top-5
```

Đánh giá retrieval:
- Recall@200;
- Recall@100;
- Recall@50.

Đánh giá rerank:
- Recall@5;
- Recall@1;
- Recall@3;
- Recall@10;
- Replacement Success Rate.

---

## 5. Luồng Assistant 3 — Detection + VLM

### Detection

```text
Ảnh thật
→ detect garment
→ crop
→ category
→ FashionCLIP embedding
→ scorer
```

Không chỉ đo detector riêng; phải kiểm tra ảnh/crop thực tế có làm downstream scorer giảm chất lượng hay không.

### VLM

Input nên là structured evidence:
- compatibility score/logit;
- problematic item;
- LOO delta;
- recommendation;
- score gain.

VLM chỉ làm:

```text
evidence
→ explanation
```

Không làm:

```text
image
→ tự chấm compatibility riêng
```

---

## 6. Luồng một task mới

Ví dụ:

```text
Thử ranking loss cho Type-aware scorer
```

Thực hiện:

```text
Leader
→ chốt dataset/version/metric

Data assistant
→ xác nhận benchmark không đổi

Scorer assistant
→ train + evaluate

Detection/VLM assistant
→ smoke test ảnh thật nếu cần

Leader
→ đọc metrics + failure cases
→ KEEP / REJECT / RETRY
```

---

## 7. Quy tắc GitHub

Branch:

```text
feat/data-...
feat/scorer-...
feat/diagnosis-...
feat/recommendation-...
feat/detection-...
feat/vlm-...
```

PR tối thiểu phải ghi:
- mục tiêu;
- dataset/version;
- config;
- metrics trước/sau;
- artifact/checkpoint;
- failure cases.

Không push trực tiếp vào `main`.

---

## 8. Notebook

Notebook chỉ để experiment:

```text
notebooks/experiments/
```

Nếu experiment được chọn:

```text
notebook
→ chuyển logic chính vào src/
→ thêm config
→ thêm test tối thiểu
```

---

## 9. Source of truth

| Nội dung | Nơi lưu |
|---|---|
| Data rules | `docs/DATA_CONTRACT_VI.md` |
| Metrics | `docs/PROJECT_METRICS_VI.md` |
| Team workflow | `docs/TEAM_WORKFLOW_VI.md` |
| Code | `src/` |
| Experiment | `notebooks/experiments/` |
| Config | `configs/` |
| Tests | `tests/` |
| Artifact lớn | Drive / external storage |
| Artifact metadata | `artifacts/README.md` |
