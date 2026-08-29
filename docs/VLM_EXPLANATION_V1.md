# VLM Explanation V1 — Qwen3-VL grounded explanation

## 1. Trạng thái và phạm vi

V1 là inference-only explanation layer cho pipeline hiện tại:

```text
Frozen V5 scorer
→ LOO diagnosis
→ structured evidence
→ Qwen3-VL explanation
```

V1 không:

- train hoặc fine-tune VLM;
- thay đổi V5 scorer/checkpoint;
- thay đổi item mà LOO đã chọn;
- đọc synthetic ground-truth để tạo lời giải thích;
- tạo recommendation. Recommendation hiện ghi rõ `not_implemented`;
- load hoặc đánh giá test split.

## 2. Model được chọn

Canonical model:

```text
Qwen/Qwen3-VL-4B-Instruct
```

Lý do:

- là bản Qwen3-VL Instruct chính thức, phù hợp nhiệm vụ image + structured text;
- 4B là điểm cân bằng thực dụng giữa 2B và 8B cho Colab T4 16 GB;
- Apache-2.0;
- Hugging Face Transformers hỗ trợ trực tiếp;
- dùng greedy decoding để giảm biến động và dễ validate JSON;
- không cần Thinking model vì VLM chỉ diễn đạt evidence, không làm scorer mới.

Runtime canonical dùng FP16, `device_map=auto`, không cài FlashAttention và
không dùng quantization. Mỗi item image được giới hạn đúng 262,144 pixels,
tương đương ngân sách khoảng 256 visual tokens theo Qwen3-VL pixel control.
Điều này giữ input 3–8 ảnh trong giới hạn thực dụng của T4.

Nguồn chính thức:

- Model card: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Qwen3-VL repository: https://github.com/QwenLM/Qwen3-VL

## 3. Evidence contract

`build_vlm_evidence(...)` nhận output inference của `diagnose_outfit(...)`, item
IDs và Core-7 categories. Nó tạo `vlm-evidence-v1` gồm:

```json
{
  "schema_version": "vlm-evidence-v1",
  "sample_id": "...",
  "output_language": "vi",
  "scorer": {
    "version": "type_aware_pairwise_v1",
    "checkpoint": "final_val_auc_v5_seed42/best.pt",
    "compatibility_logit": -0.31,
    "semantics": "uncalibrated_logit_not_probability"
  },
  "items": [],
  "diagnosis": {
    "problematic_item_index": 2,
    "problematic_item_id": "...",
    "problematic_category": "SHOES",
    "ranked_items": [],
    "top1_top2_delta_gap": 0.08,
    "certainty": "not_calibrated",
    "uses_two_item_extrapolation": false
  },
  "recommendation": {
    "status": "not_implemented",
    "items": []
  },
  "grounding_rules": []
}
```

Các field ground-truth/evaluation sau bị cấm tuyệt đối trong evidence:

```text
label
negative_metadata
swapped_item_index
target_swapped_item_index
top1_correct
hit_at_2
```

Vì vậy VLM không thể “giải thích đúng” bằng cách nhìn đáp án synthetic.

## 4. Image binding

Runtime bắt buộc đúng một image cho mỗi item. Message ghi rõ:

```text
item_index=...
item_id=...
coarse_category=...
→ image
```

Image order phải trùng canonical item order. VLM được phép nhận xét chi tiết
trực tiếp nhìn thấy như màu, pattern, silhouette và formality. Nó không được
bịa brand, chất liệu, giá, hoàn cảnh sử dụng hoặc ý định người dùng.

## 5. Output contract

Qwen phải trả đúng một JSON object `vlm-explanation-v1`:

```json
{
  "schema_version": "vlm-explanation-v1",
  "problematic_item_index": 2,
  "problematic_item_id": "...",
  "headline": "...",
  "evidence_summary": ["..."],
  "visual_observations": [
    {
      "item_indices": [0, 2],
      "observation": "..."
    }
  ],
  "explanation": "...",
  "uncertainty_note": "...",
  "limitations": [
    "recommendation_not_implemented",
    "compatibility_logit_is_not_probability",
    "vlm_visual_observations_are_inferences"
  ]
}
```

Validator hard-fail nếu:

- JSON/schema sai;
- problematic ID/index khác LOO evidence;
- visual observation reference item ngoài outfit;
- thiếu limitation bắt buộc;
- xuất hiện recommendation field;
- output quá dài hoặc sai kiểu dữ liệu.

Pipeline cho phép đúng một schema-repair retry. Nếu retry vẫn sai, case đó fail
thay vì trả một explanation không kiểm soát.

## 6. Chạy trên Colab

Mở:

```text
notebooks/experiments/NB8_vlm_explanation_v1.ipynb
```

Notebook thực hiện:

1. checkout `feat/vlm-explanation-v1`;
2. mount `ML_Final`;
3. cài `requirements-vlm.txt`;
4. chạy VLM + LOO/scorer regression tests;
5. load frozen V5 `best.pt`;
6. chọn deterministic validation negative đầu tiên có ít nhất 4 items;
7. chạy LOO mà không đưa ground truth vào evidence;
8. lấy đúng item images từ split `valid` của Polyvore1000;
9. chạy Qwen3-VL-4B-Instruct;
10. validate và lưu `evidence.json` + `vlm_run.json` vào Drive.

Không load test split. Demo mặc định tránh original-size-3 để không dùng two-item
extrapolation, nhưng contract vẫn bắt buộc disclosure nếu caller dùng case đó.

## 7. API

```python
from src.vlm import build_vlm_evidence, load_vlm_config, VLMExplanationPipeline
from src.vlm.qwen_backend import Qwen3VLBackend

evidence = build_vlm_evidence(
    loo_result,
    sample_id=sample_id,
    item_ids=item_ids,
    coarse_categories=coarse_categories,
)
config = load_vlm_config("configs/vlm_qwen3_vl_4b_instruct_v1.json")
backend = Qwen3VLBackend.from_config(config)
pipeline = VLMExplanationPipeline(backend, config)
run = pipeline.explain(evidence, item_image_paths)
```

CLI tương đương:

```bash
python -m src.vlm.cli \
  --evidence evidence.json \
  --images item0.jpg item1.jpg item2.jpg \
  --output vlm_run.json
```

## 8. Handoff contract

Khi detection hoàn thành, nó chỉ cần cung cấp một image/crop theo đúng item order
và giữ mapping `item_index → item_id → category`. VLM module không phụ thuộc
vào detector cụ thể.

Recommendation có thể được thêm bằng evidence schema version mới. Không được
âm thầm chèn recommendation vào V1 hiện tại.
