# ML Final — Fashion Outfit Compatibility

Repo này chứa phần ML của project.

Pipeline chính:

```text
Data
  ↓
FashionCLIP Embedding
  ↓
Compatibility Scorer
  ↓
Diagnosis (LOO)
  ↓
Recommendation
  ↓
Detection / Real-image Input
  ↓
VLM Explanation
```

Đọc trước khi code:

1. `docs/DATA_CONTRACT_VI.md`
2. `docs/PROJECT_METRICS_VI.md`
3. `docs/TEAM_WORKFLOW_VI.md`

Nguyên tắc:
- `DATA_CONTRACT_VI.md` là source of truth cho dữ liệu và scorer I/O.
- `PROJECT_METRICS_VI.md` là source of truth cho evaluation.
- Dataset V1 đã freeze thì không được âm thầm regenerate cho từng scorer.
- Experiment có thể nằm trong notebook, nhưng logic được chọn phải chuyển vào `src/`.
- Không commit checkpoint, embedding cache, dataset lớn lên GitHub.
