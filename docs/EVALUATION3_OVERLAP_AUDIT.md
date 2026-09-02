# EVALUATION3 — kiểm tra overlap với Polyvore

## Mục đích

Script này trả lời một câu hỏi duy nhất:

> Outfit hoặc item của EVALUATION3 có từng xuất hiện trong dữ liệu Polyvore mà scorer đã nhìn thấy hay không?

Nó kiểm tra ba loại tín hiệu:

1. `e3_outfit_id` trùng `source_kit_id` — chỉ là **candidate**, vì EVALUATION3 gộp nhiều nguồn và có thể va ID.
2. Ảnh item trùng hoàn toàn sau khi decode về RGB — bằng chứng exact-image overlap.
3. Ảnh gần trùng theo dHash — candidate cho trường hợp resize hoặc nén lại.

Script đọc mọi item thực sự có trong scorer-ready JSONL, bao gồm item gốc và item được đưa vào synthetic negative. Như vậy một replacement item từng xuất hiện khi train cũng được tính là model đã thấy.

## Dữ liệu EVALUATION3 đã kiểm tra

Hai workbook hiện có nối được 1-1 theo `ITEM`:

- `Cmt_ALL_20190325`: 34.479 `ITEM`, chứa `Cmt` và `Reason`;
- `Attribute_ALL_UBSGsimple.xlsx`, sheet `Num`: 34.479 `Item#`, chứa thuộc tính và một số giá trị `Group`.

Lưu ý: `Group` **không phải cột split đầy đủ** trong bản workbook hiện tại. Có 34.355/34.479 dòng để trống; chỉ 30 dòng mang tag `A-Test2000`. Vì vậy không được coi `A-Test2000` là toàn bộ official test gồm 2.000 outfit. Chạy overlap trên toàn bộ 34.479 outfit trước; chỉ lọc theo `Group` nếu nhóm xác minh được ý nghĩa chính thức của tag.

Mapping `Cmt = 1/2/3` chưa cần cho bước overlap và script cố ý giữ nguyên `cmt_raw`.

## Chạy trên Colab

Chạy từ repository đã checkout nhánh chứa module này:

```python
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path("/content/opisoverated")
E3_ROOT = Path("/content/drive/MyDrive/EVALUATION3/outfit")
CMT_FILE = Path("/content/drive/MyDrive/EVALUATION3/Cmt_ALL_20190325.xlsx")
ATTRIBUTE_FILE = Path("/content/drive/MyDrive/EVALUATION3/Attribute_ALL_UBSGsimple.xlsx")
SCORER_DIR = Path("/content/drive/MyDrive/opisoverated/scorer_ready_v2")
OUTPUT_DIR = Path("/content/drive/MyDrive/opisoverated/evaluation3_overlap_audit")

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REPO_ROOT / "requirements-evaluation.txt"),
    ],
    check=True,
)

command = [
    sys.executable,
    "-m",
    "src.evaluation.evaluation3_overlap",
    "--evaluation3-root", str(E3_ROOT),
    "--evaluation3-annotations", str(CMT_FILE),
    "--annotation-sheet", "CMT",
    "--evaluation3-metadata", str(ATTRIBUTE_FILE),
    "--metadata-sheet", "Num",
    "--development-jsonl", f"train={SCORER_DIR / 'scorer_ready_v2_train.jsonl'}",
    "--development-jsonl", f"valid={SCORER_DIR / 'scorer_ready_v2_valid.jsonl'}",
    "--development-jsonl", f"test={SCORER_DIR / 'scorer_ready_v2_test.jsonl'}",
    "--polyvore-hf-dataset", "codewaly/polyvore1000",
    "--model-development-splits", "train,valid",
    "--near-hamming-threshold", "4",
    "--output-dir", str(OUTPUT_DIR),
]
subprocess.run(command, cwd=REPO_ROOT, check=True)
```

Nếu nhóm đã lưu ảnh Polyvore ra thư mục theo tên `item_id`, thay:

```text
--polyvore-hf-dataset codewaly/polyvore1000
```

bằng:

```text
--polyvore-image-root /path/to/polyvore/images
```

Script mặc định dừng nếu không tìm được ảnh của bất kỳ item nào từng được đưa vào scorer. Không thêm `--allow-incomplete-image-index` cho kết quả chính thức; tùy chọn đó chỉ dùng để debug.

## Output

| File | Ý nghĩa |
|---|---|
| `evaluation3_overlap_summary.json` | Tổng số overlap, độ đầy đủ của image index và số lượng từng manifest |
| `evaluation3_overlap_audit.jsonl` | Kết quả chi tiết theo outfit và từng slot ảnh |
| `evaluation3_overlap_evidence.csv` | Các cặp ảnh trùng/gần trùng để kiểm tra thủ công |
| `evaluation3_full.jsonl` | Toàn bộ EVALUATION3; dùng cho diagnostic |
| `evaluation3_model_clean.jsonl` | Không overlap với các split dùng train/model selection (`train,valid`) |
| `evaluation3_strict_clean.jsonl` | Không overlap với bất kỳ split Polyvore nào đã cung cấp, kể cả `test` |

Chỉ coi hai manifest clean là sẵn sàng khi summary có:

```json
{
  "status": "PASS",
  "official_clean_manifests_ready": true
}
```

## Cách dùng kết quả

- Báo cáo chính chặt nhất: `strict_clean`.
- Báo cáo bổ sung hợp lý: `model_clean`, nếu Polyvore test thật sự chưa từng dùng để train, chọn checkpoint, chỉnh threshold hoặc calibration.
- `full` chỉ là diagnostic nếu có overlap; không gọi nó là independent external test.
- Các hàng `near_dhash` trong CSV là candidate. Script loại chúng khỏi clean set theo hướng bảo thủ, nhưng nhóm nên xem một mẫu để biết threshold 4 có tạo nhiều false positive không.

Không được xem kết quả EVALUATION3 rồi đổi threshold overlap để giữ lại các case thuận lợi cho model. Threshold và quy tắc loại overlap phải được freeze trước khi chạy scorer trên EVALUATION3.
