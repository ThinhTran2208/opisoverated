# MIN2 Experiment Runbook

Branch:

```text
exp/min2-scorer-loo3
```

Contract:

```text
docs/SCORER_CONTRACT_MIN2_EXPERIMENT.md
```

## 1. Runtime paths

Dùng config riêng để không overwrite frozen Core-7 V2 artifacts:

```text
configs/data_paths.min2_experiment.json
```

Mặc định experiment ghi vào:

```text
data/core7_min2_exp_v1/
data/scorer_ready_min2_exp_v1/
```

FashionCLIP cache/manifest vẫn reuse artifact hiện tại.

## 2. Regenerate data với minimum = 2

Chạy toàn bộ data stages:

```bash
python -m src.data.min2_experiment all \
  --paths-config configs/data_paths.min2_experiment.json
```

Hoặc chạy từng stage:

```bash
python -m src.data.min2_experiment prepare \
  --paths-config configs/data_paths.min2_experiment.json

python -m src.data.min2_experiment validate \
  --paths-config configs/data_paths.min2_experiment.json

python -m src.data.min2_experiment build \
  --paths-config configs/data_paths.min2_experiment.json
```

Expected contract:

```text
category-clean positives: 2–8 items
scorer-ready positives:   2–8 items
scorer-ready negatives:   2–8 items
negative protocol:        same_category_different_kit
positive : negative:      1 : 1
```

Build chỉ được dùng cho training khi final validation trả:

```text
READY_TO_TRAIN
```

## 3. Train scorer với minimum = 2

Config:

```text
configs/scorer_type_aware_pairwise_min2_experiment.yaml
```

Run:

```bash
python -m src.scorer.min2_experiment \
  --paths-config configs/data_paths.min2_experiment.json \
  --config configs/scorer_type_aware_pairwise_min2_experiment.yaml \
  --checkpoint-dir artifacts/checkpoints/type_aware_pairwise_v1/min2_exp_v1_seed42
```

Scorer contract:

```text
2 <= real_item_count <= 8
```

2-item outfit có đúng 1 valid pair và phải trả finite `compatibility_logit`.

1-item outfit phải hard-fail.

## 4. LOO diagnosis

LOO helper:

```python
from src.diagnosis.loo import build_leave_one_out_outfits
```

Valid:

```python
build_leave_one_out_outfits(["top", "bottom", "shoes"])
```

Kết quả có 3 residual outfits, mỗi residual còn 2 items.

Invalid:

```python
build_leave_one_out_outfits(["top", "shoes"])
```

Case này hard-fail trước scorer vì residual chỉ còn 1 item.

Boundary:

```text
Compatibility scorer minimum = 2
LOO original outfit minimum  = 3
```

## 5. Evaluation

Scorer report:

```text
ROC-AUC
2-way FITB
mean / median logit margin
```

Ngoài overall metrics, tách theo outfit length:

```text
n = 2
n = 3
n >= 4
```

LOO localization chỉ evaluate original outfits `n >= 3`, nên tách:

```text
n = 3
n = 4
n >= 5
overall n >= 3
```

Để so với baseline 3–8 công bằng, model MIN2 phải được so sánh thêm trên **shared evaluation subset `n >= 3`**. Không kết luận improvement chỉ từ overall metric vì MIN2 benchmark có thêm distribution `n = 2`.
