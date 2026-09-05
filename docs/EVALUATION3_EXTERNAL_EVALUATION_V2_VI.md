# Giao thức Đánh giá Ngoài EVALUATION3 v2 — EVAL3-Test2000-Full

## 1. Trạng thái

Trạng thái giao thức:

```text
FROZEN
```

Đánh giá xác nhận trên EVALUATION3 chỉ được giới hạn trong khối test 2.000 outfit đã được tái dựng:

```text
EVAL3-Test2000-Full
```

Kế hoạch dự phòng trước đây là sử dụng toàn bộ 34.479 outfit làm tập đánh giá ngoài chính không còn là giao thức mặc định.

Không được tính bất kỳ metric model-vs-human nào trước khi hoàn tất và đóng băng audit overlap cuối cùng, bước tổng hợp theo outfit, evaluation manifest và protocol artifact.

---

## 2. Mục tiêu

Đánh giá xem compatibility scorer đã đóng băng được huấn luyện trên Polyvore1000 có chuyển giao được sang các đánh giá outfit do con người gán nhãn trong EVALUATION3 hay không.

Đánh giá xác nhận phải:

- chỉ sử dụng Scorer V1 đã đóng băng;
- sử dụng khối test EVALUATION3 đã tái dựng là EVAL3-Test2000-Full;
- kiểm soát rõ ràng overlap với Polyvore1000 TRAIN, VALID và TEST;
- chỉ sử dụng các metric tương quan thứ bậc;
- không thực hiện bất kỳ điều chỉnh mô hình nào dựa trên EVALUATION3.

---

## 3. Tái dựng Split của EVALUATION3

### 3.1 Kích thước split mở rộng đã công bố

Bản EVALUATION3 mở rộng được mô tả là có:

```text
Train      = 29.479 outfits
Validation =  3.000 outfits
Test       =  2.000 outfits
Total      = 34.479 outfits
```

### 3.2 Tái dựng theo thứ tự dòng trong workbook

Dự án tái dựng ba khối theo thứ tự các dòng dữ liệu trong workbook như sau:

```text
data rows 0–29.478
→ Train
→ 29.479 outfits

data rows 29.479–32.478
→ Validation
→ 3.000 outfits

data rows 32.479–34.478
→ Test
→ 2.000 outfits
```

Python slicing tương đương:

```python
train = rows[0:29479]
valid = rows[29479:32479]
test  = rows[32479:34479]
```

Các chỉ số dòng ở trên được tính trên 34.479 dòng dữ liệu, không bao gồm header của workbook.

### 3.3 Định nghĩa EVAL3-Test2000-Full

Khối test đã tái dựng được đóng băng với tên:

```text
EVAL3-Test2000-Full
```

Định nghĩa:

```text
2.000 dòng dữ liệu cuối
data-row indices 32.479–34.478 inclusive
Python slice rows[32479:34479]
```

Biên outfit-ID quan sát được:

```text
row 32.478 → 200519201000
row 32.479 → 200519201001  ← EVAL3-Test2000-Full bắt đầu

...

row 34.478 → 200519203000  ← EVAL3-Test2000-Full kết thúc
```

Do đó EVAL3-Test2000-Full tương ứng với dải outfit-ID liên tục:

```text
200519201001
...
200519203000
```

### 3.4 Bằng chứng hỗ trợ việc tái dựng

Việc tái dựng split được chấp nhận trong dự án này vì các bằng chứng sau đồng thời nhất quán:

1. kích thước các split mở rộng đã công bố chính xác là `29.479 / 3.000 / 2.000`;
2. `Attribute_ALL_UBSGsimple.xlsx` chứa chính xác 34.479 dòng dữ liệu;
3. cắt theo các biên đã công bố tạo ra chính xác ba kích thước split trên;
4. audit độ phủ ảnh cho kết quả:

```text
Candidate Train:
29.479 total
23.318 with complete image availability
6.161 missing
79,10% coverage

Candidate Validation:
3.000 total
3.000 with complete image availability
0 missing
100% coverage

Candidate Test:
2.000 total
2.000 with complete image availability
0 missing
100% coverage
```

5. toàn bộ 6.161 outfit thiếu ảnh đều nằm trong 29.479 dòng đầu;
6. không có dòng nào trong 5.000 dòng cuối bị thiếu trong dữ liệu ảnh;
7. 3.000 dòng cuối tạo thành một khối ID liên tục:

```text
200519200001
...
200519203000
```

8. 2.000 dòng cuối tạo thành khối EVAL3-Test2000-Full liên tục:

```text
200519201001
...
200519203000
```

9. `Attribute_ALL_UBSGsimple.xlsx` và `Cmt_ALL_20190325.xlsx` khớp với nhau trên 3.000 outfit ID cuối theo cùng thứ tự;
10. cả ba khối được tái dựng đều có đầy đủ nhãn `Cmt`.

### 3.5 Thuật ngữ khi báo cáo

Dự án xem EVAL3-Test2000-Full là candidate cho official test đã được tái dựng.

Cách diễn đạt ưu tiên:

```text
khối test 2.000 outfit EVALUATION3 được tái dựng (EVAL3-Test2000-Full)
```

hoặc, khi bằng chứng tái dựng được mô tả rõ:

```text
EVALUATION3 EVAL3-Test2000-Full
```

Không được khẳng định rằng mapping theo thứ tự dòng đã được công bố trực tiếp trong workbook phát hành, trừ khi có một nguồn độc lập xác nhận trực tiếp mapping đó.

---

## 4. Phạm vi Đánh giá

Tất cả metric xác nhận trên EVALUATION3 chỉ được tính trong:

```text
EVAL3-Test2000-Full
```

29.479 dòng TRAIN và 3.000 dòng VALIDATION được tái dựng không được sử dụng cho:

- huấn luyện scorer;
- fine-tune scorer;
- lựa chọn checkpoint;
- tinh chỉnh metric;
- tinh chỉnh threshold;
- calibration;
- báo cáo hiệu năng xác nhận trên EVALUATION3.

Mọi phân tích trên 32.479 outfit EVALUATION3 còn lại đều là exploratory và không được thay thế kết quả xác nhận EVAL3-Test2000-Full.

---

## 5. Scorer Đã Đóng băng

### 5.1 Định danh mô hình

```text
scorer_version = type_aware_pairwise_v1
checkpoint = V5 seed42 epoch52
```

Checkpoint SHA-256:

```text
7b3d0b6e0d44e3de517565f5725ded198bbc762b02a4dece26a58ee145cfed9c
```

Commit code train/evaluation:

```text
7cbbb19fd89352b7ef54038e57b4d8208b7ee1f6
```

### 5.2 Output của mô hình

Chỉ sử dụng:

```text
compatibility_logit
```

Logit càng cao nghĩa là outfit càng tương thích theo scorer đã đóng băng.

Không sử dụng:

- calibrated score 0–100;
- classification threshold;
- score bands;
- calibration riêng cho EVALUATION3.

---

## 6. Input Contract của EVALUATION3

EVAL3-Test2000-Full sử dụng bốn ảnh item cho mỗi outfit:

```text
U → TOP
B → BOTTOM
S → SHOES
G → BAG
```

Không sử dụng RF-DETR.

Luồng đánh giá:

```text
U/B/S/G item images
→ FashionCLIP
→ Core-7 category IDs
→ frozen Scorer V1
→ compatibility_logit
```

Luồng này đánh giá:

```text
FashionCLIP representation + frozen compatibility scorer
```

Đây không phải là đánh giá detector end-to-end.

---

## 7. FashionCLIP Contract

```text
model_name = patrickjohncyh/fashion-clip
embedding_version = fashionclip-512-l2-v1
preprocessing_version = fashionclip-hf-clipprocessor-rgb-v1
embedding_dimension = 512
normalization = L2
```

Không được fine-tune FashionCLIP trên EVALUATION3.

---

## 8. Nhãn Con người

### 8.1 Nhãn human gốc (`Cmt`)

Giữ nguyên chính xác giá trị từ workbook gốc:

```text
Raw human label:
Cmt:
    Good   = 1
    Normal = 2
    Bad    = 3
```

Giá trị `Cmt` gốc phải được giữ nguyên trong raw manifest và frozen manifest.

### 8.2 Chất lượng thứ bậc dẫn xuất cho metric

Scorer contract sử dụng:

```text
higher compatibility_logit = more compatible / better
```

Do `Cmt` gốc chạy theo chiều ngược lại, ta dẫn xuất:

```text
Derived ordinal quality:
    human_ordinal_quality = 4 - Cmt

Therefore:
    Bad    = 1
    Normal = 2
    Good   = 3
```

Python tương đương:

```python
human_ordinal_quality = 4 - cmt
```

Tất cả metric tương quan thứ bậc phải sử dụng `human_ordinal_quality`, không sử dụng trực tiếp `Cmt` gốc.

Phép biến đổi này chỉ thay đổi chiều của metric. Nó không sửa đổi hoặc thay thế trường `Cmt` gốc.

### 8.3 Reason

Các nhóm reason được phép:

```text
Color
Print
Material
Silhouette
Design
```

Chính sách:

```text
provided reason → preserve
blank reason    → remain blank
```

Không suy luận hoặc điền bù các reason bị thiếu.

---

## 9. Metrics

### 9.1 Metric chính

```text
Kendall tau-b
```

Được tính giữa:

```text
compatibility_logit
and
human_ordinal_quality
```

trong đó:

```text
Raw human label:
Cmt:
    Good   = 1
    Normal = 2
    Bad    = 3

Derived ordinal quality:
    human_ordinal_quality = 4 - Cmt

Therefore:
    Bad    = 1
    Normal = 2
    Good   = 3
```

Kendall tau-b dương nghĩa là logit scorer cao hơn có xu hướng đồng thuận với chất lượng human tốt hơn.

### 9.2 Metric phụ

```text
Spearman rho
```

Được tính trên cùng hai input:

```text
compatibility_logit
human_ordinal_quality
```

Spearman rho dương nghĩa là thứ hạng scorer cao hơn có xu hướng tương ứng với chất lượng human tốt hơn.

### 9.3 Khoảng tin cậy

Đối với cả Kendall tau-b và Spearman rho, sử dụng cấu hình bootstrap đã đóng băng:

```text
seed       = 42
resamples  = 10,000
CI         = 95%
CI method  = percentile
unit       = outfit
```

Bootstrap được thực hiện ở cấp outfit: mỗi lần resample lấy lại `N` outfit có hoàn lại từ chính evaluation subset đang được đánh giá.

Cùng một bootstrap seed, số lần resample, CI level, CI method và bootstrap unit phải được sử dụng cho cả Kendall tau-b và Spearman rho trên cả ba evaluation subset.

### 9.4 Tóm tắt logit theo từng class bắt buộc

Đối với mỗi evaluation subset được báo cáo, cần báo cáo:

```text
N_total
N_Bad
N_Normal
N_Good
```

Đối với từng human class riêng:

```text
Bad
Normal
Good
```

báo cáo phân phối `compatibility_logit` tối thiểu gồm:

```text
N
median
Q1
Q3
IQR
```

Có thể báo cáo thêm mean và standard deviation như thống kê mô tả bổ sung, nhưng chúng không phải confirmatory metrics.

Các classwise summary chỉ là descriptive diagnostics và không được sử dụng để chọn threshold hoặc sửa đổi mô hình.

### 9.5 Pairwise ordering diagnostics

Đối với mỗi evaluation subset được báo cáo, cần báo cáo:

```text
P(logit_Good   > logit_Bad)
P(logit_Good   > logit_Normal)
P(logit_Normal > logit_Bad)
```

Mỗi diagnostic được tính trên toàn bộ các cặp outfit khác class trong subset đó.

Nếu hai outfit có scorer-logit bằng nhau chính xác:

```text
tie contribution = 0.5
```

Do đó mỗi pairwise ordering diagnostic được tính bằng:

```text
(wins + 0.5 * ties) / number_of_cross_class_pairs
```

Đây là các descriptive ordering diagnostics, không thay thế Kendall tau-b hoặc Spearman rho.

### 9.6 Giao thức metric giống nhau trên các overlap subset

Phải áp dụng độc lập cùng một reporting protocol cho cả ba frozen overlap subset:

```text
EVAL3-Test2000-No-Full-Image-Overlap-Candidate
EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate
EVAL3-Test2000-Full
```

Với mỗi subset, báo cáo:

```text
N_total
N_Bad
N_Normal
N_Good

Kendall tau-b
95% bootstrap CI

Spearman rho
95% bootstrap CI

classwise compatibility_logit summary

P(logit_Good > logit_Bad)
P(logit_Good > logit_Normal)
P(logit_Normal > logit_Bad)
```

Sử dụng cùng một phép biến đổi label, implementation metric, bootstrap policy, seed, số lần resample và pairwise tie rule cho cả ba subset.

Không được thay đổi định nghĩa metric hoặc các trường báo cáo sau khi đã quan sát hiệu năng theo từng subset.

### 9.7 Các metric bị loại khỏi đánh giá xác nhận EVALUATION3

Không sử dụng:

```text
ROC-AUC
accuracy
F1
3-class accuracy
threshold-based metrics
calibrated 0–100 score metrics
```

---

## 10. Phạm vi Overlap Audit

Overlap audit so sánh các item trong EVAL3-Test2000-Full với toàn bộ các split Polyvore1000 đã đóng băng:

```text
Polyvore TRAIN
Polyvore VALID
Polyvore TEST
```

Mục đích:

```text
TRAIN overlap
→ direct training exposure

VALID overlap
→ model-selection exposure

TEST-only overlap
→ no model-development exposure,
  but Polyvore source/item overlap remains
```

Overlap chỉ với Polyvore TEST không được gọi là training leakage.

Có thể sử dụng một combined Polyvore visual index nếu mọi candidate đều giữ lại source split của nó.

---

## 11. Quy tắc Visual Duplicate Đã Đóng băng

### 11.1 Ngưỡng và phạm vi pHash

Cấu hình audit đã đóng băng:

```text
T_pHash     = 14
T_SSIM_auto = 0.92
```

Các exact pHash band được audit:

```text
0
2
4
6
8
10
12
14
```

Candidate có:

```text
pHash > 14
```

không nằm trong manual-search scope của protocol overlap hiện tại.

### 11.2 Quy tắc auto duplicate

Một visual pair được tự động gán:

```text
DUPLICATE
```

khi:

```text
pHash_distance <= 14
AND
SSIM >= 0.92
```

`T_SSIM_auto = 0.92` là threshold thực nghiệm của dự án, không phải universal threshold.

### 11.3 Quy tắc manual review theo exact pHash band

Đối với một EVALUATION3 item và một reference scope, manual review không duyệt toàn bộ raw pHash candidates.

Trong mỗi exact pHash band:

```text
0, 2, 4, 6, 8, 10, 12, 14
```

chọn:

```text
candidate có SSIM lớn nhất trong band đó
```

làm representative pair cho manual review.

Do đó manual rule được đóng băng là:

```text
one maximum-SSIM candidate
per exact pHash band
within each reference scope
```

Reference scopes:

```text
TRAIN_VALID = Polyvore TRAIN + VALID
FULL        = Polyvore TRAIN + VALID + TEST
```

Nếu cùng một visual pair phục vụ nhiều scope/band requirement, pair chỉ cần được manual-review một lần và label phải được tái sử dụng nhất quán.

### 11.4 Thứ tự review và early stopping

Trong từng outfit và từng phase:

```text
primary sort   = SSIM descending
exact tie only = pHash_distance ascending
```

Không làm tròn SSIM và không dùng pHash thấp hơn để vượt một candidate có SSIM cao hơn.

Review theo hai phase:

```text
Phase 1: TRAIN + VALID
Phase 2: TEST
```

Nếu phát hiện `DUPLICATE` trong TRAIN/VALID:

```text
→ outfit có TRAIN_VALID image overlap
→ dừng review toàn bộ outfit
```

Nếu TRAIN/VALID đã clean, mới xét TEST.

Nếu phát hiện `DUPLICATE` trong TEST:

```text
→ outfit có FULL image overlap
→ dừng phần TEST còn lại
```

Auto-duplicate phải được áp dụng trước manual review và cũng kích hoạt cùng early-stop logic.

### 11.5 Phạm vi diễn giải

Các quy tắc trên nhằm phát hiện:

```text
same-image / visual duplicate overlap
```

và không được mô tả như một bảo đảm tuyệt đối rằng không tồn tại mọi dạng product overlap.

---

## 12. Nhãn Manual Overlap

Các nhãn review cuối cùng được phép:

```text
DUPLICATE
SAME_PRODUCT_DIFFERENT_IMAGE
NON_DUPLICATE
UNCERTAIN
SKIP
```

Cách xử lý cho **image-overlap protocol**:

```text
DUPLICATE
→ image overlap
→ kích hoạt early stop

SAME_PRODUCT_DIFFERENT_IMAGE
→ không phải image overlap
→ không kích hoạt early stop
→ ghi riêng product-overlap diagnostic

NON_DUPLICATE
→ không phải image overlap

UNCERTAIN
→ audit unresolved

SKIP
→ audit unresolved
```

`SAME_PRODUCT_DIFFERENT_IMAGE` nghĩa là cùng sản phẩm nhưng khác ảnh chụp. Nhãn này không được gộp vào `DUPLICATE` khi xây hai image-clean subset.

Nếu sau này cần đánh giá product-level overlap, phải tạo một analysis/subset riêng; không thay đổi định nghĩa của các image-overlap subset đã đóng băng.

`UNCERTAIN` và `SKIP` không được âm thầm chuyển thành clean.

Các quyết định manual overlap phải được đưa ra mà không sử dụng scorer logits hoặc các metric hiệu năng EVALUATION3.

---

## 13. Output Audit ở Cấp Item

Các trường item-level tối thiểu:

```text
eval3_outfit_id
eval3_item_id
role

best_polyvore_item_id
best_polyvore_split

best_phash_distance
best_ssim

auto_duplicate
borderline
manual_label

overlap_type
is_image_overlap
is_same_product_different_image
audit_unresolved
```

Các trường theo từng split được khuyến nghị:

```text
train_best_item_id
train_best_phash
train_best_ssim
train_overlap

valid_best_item_id
valid_best_phash
valid_best_ssim
valid_overlap

test_best_item_id
test_best_phash
test_best_ssim
test_overlap
```

Quy ước:

```text
is_image_overlap = true
↔ final audit status == DUPLICATE

is_same_product_different_image = true
↔ final audit status == SAME_PRODUCT_DIFFERENT_IMAGE
```

Hai field này phải được giữ riêng.

---

## 14. Tổng hợp ở Cấp Outfit

Mỗi outfit trong EVAL3-Test2000-Full chứa:

```text
TOP
BOTTOM
SHOES
BAG
```

Với mỗi outfit, tính:

```text
train_overlap_count = 0..4
valid_overlap_count = 0..4
test_overlap_count  = 0..4
```

Đồng thời tính:

```text
any_train_overlap
any_valid_overlap
any_test_overlap
audit_unresolved
```

Một item được tính vào `train_overlap_count`, `valid_overlap_count` hoặc `test_overlap_count` chỉ khi trạng thái audit cuối cùng của visual pair tương ứng là:

```text
DUPLICATE
```

`SAME_PRODUCT_DIFFERENT_IMAGE` không làm tăng image-overlap count.

Khuyến nghị ghi thêm product-level diagnostic riêng:

```text
same_product_different_image_count
any_same_product_different_image
```

Nếu bất kỳ item nào còn `UNCERTAIN`, `SKIP` hoặc chưa được resolve:

```text
audit_unresolved = true
```

---

## 15. TH1 / TH2 / TH3 trong EVAL3-Test2000-Full

### 15.1 TH1 — Overlap với dữ liệu phát triển mô hình

```text
train_overlap_count > 0
OR
valid_overlap_count > 0
```

Diễn giải:

```text
known overlap with data involved in model development
```

### 15.2 TH2 — Overlap chỉ với Polyvore TEST

```text
train_overlap_count == 0
AND
valid_overlap_count == 0
AND
test_overlap_count > 0
AND
audit_unresolved == false
```

Diễn giải:

```text
no known model-development overlap
but known overlap with untouched Polyvore TEST
```

### 15.3 TH3 — Strict clean

```text
train_overlap_count == 0
AND
valid_overlap_count == 0
AND
test_overlap_count == 0
AND
audit_unresolved == false
```

Diễn giải:

```text
no known overlap with any Polyvore split under the frozen audit protocol
```

### 15.4 Chưa resolve

```text
audit_unresolved == true
```

Các outfit chưa resolve vẫn được giữ trong thống kê đầy đủ của EVAL3-Test2000-Full nhưng không được đi vào các clean subset.

---

## 16. Các Evaluation Subset Cuối cùng của EVAL3-Test2000-Full

Ba tập báo cáo đều thuộc cùng test universe:

```text
EVAL3-Test2000-Full
```

Hai tập còn lại là subset được tạo từ image-overlap audit, không phải test split mới.

### 16.1 EVAL3-Test2000-No-Full-Image-Overlap-Candidate

Định nghĩa:

```text
TH3 only
```

Tức là:

```text
không có known image overlap với Polyvore TRAIN
không có known image overlap với Polyvore VALID
không có known image overlap với Polyvore TEST
audit_unresolved == false
```

Vai trò:

```text
PRIMARY evaluation subset
```

### 16.2 EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate

Định nghĩa:

```text
TH2 + TH3
```

Tức là không có known image overlap với dữ liệu tham gia model development:

```text
Polyvore TRAIN
Polyvore VALID
```

Outfit vẫn có thể có TEST-only image overlap.

Vai trò:

```text
SECONDARY evaluation subset
```

### 16.3 EVAL3-Test2000-Full

Định nghĩa:

Toàn bộ outfit EVAL3-Test2000-Full vượt qua các yêu cầu integrity cơ bản.

Có thể bao gồm:

```text
TH1
TH2
TH3
UNRESOLVED
```

Vai trò:

```text
DIAGNOSTIC full-test result
```

### 16.4 Thứ tự báo cáo

```text
1. EVAL3-Test2000-No-Full-Image-Overlap-Candidate
2. EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate
3. EVAL3-Test2000-Full
```

Primary subset không được thay đổi sau khi đã quan sát hiệu năng mô hình.

---

## 17. Chính sách Integrity

EVAL3-Test2000-Full hiện có:

```text
2.000 / 2.000 outfits with image coverage
```

Trước scorer inference vẫn phải kiểm tra:

- tồn tại đủ cả bốn reference U/B/S/G;
- cả bốn ảnh đều decode thành công;
- role mapping hợp lệ;
- output FashionCLIP là finite;
- dimension FashionCLIP bằng 512;
- thỏa mãn L2 normalization contract.

Không được thay thế item bị thiếu bằng collage image `0`.

Bất kỳ outfit nào không vượt qua integrity check bắt buộc đều bị loại khỏi scorer evaluation và phải được ghi lại với exclusion reason rõ ràng.

---

## 18. Raw Manifest

Tạo:

```text
a_test2000_raw_manifest.csv
```

Các trường tối thiểu:

```text
row_index
outfit_id

top_item_id
bottom_item_id
shoes_item_id
bag_item_id

top_image_path
bottom_image_path
shoes_image_path
bag_image_path

cmt_original
human_label
reason

split_reconstruction = test
split_protocol = EVAL3-Test2000-Full
integrity_pass
```

Số dòng kỳ vọng trước exclusions:

```text
2.000
```

---

## 19. Borderline Audit Artifact

Tạo:

```text
a_test2000_borderline_candidates.csv
```

Các trường tối thiểu:

```text
eval3_outfit_id
eval3_item_id
role

polyvore_item_id
polyvore_split

phash_distance
ssim

manual_label
manual_notes
```

Manual review phải hoàn tất trước khi đóng băng final evaluation manifest.

---

## 20. Frozen Evaluation Manifest

Tạo:

```text
a_test2000_eval_manifest_frozen.csv
```

Các trường tối thiểu:

```text
row_index
outfit_id

top_item_id
bottom_item_id
shoes_item_id
bag_item_id

top_image_path
bottom_image_path
shoes_image_path
bag_image_path

cmt_original
human_label
reason

train_overlap_count
valid_overlap_count
test_overlap_count

any_train_overlap
any_valid_overlap
any_test_overlap
audit_unresolved

overlap_group

subset_full
subset_no_trainvalid_image_overlap
subset_no_full_image_overlap

integrity_pass
```

Các giá trị `overlap_group` được phép:

```text
TH1
TH2
TH3
UNRESOLVED
```

---

## 21. Protocol Artifact

Tạo:

```text
a_test2000_protocol_v2.json
```

Tối thiểu phải ghi lại:

```text
protocol_version
protocol_status

evaluation_dataset
evaluation_split_name
split_reconstruction_rule
test_row_start
test_row_end
test_outfit_id_start
test_outfit_id_end
expected_test_count

scorer_version
checkpoint_sha256
checkpoint_epoch
seed
code_commit

fashionclip_model
embedding_version
embedding_dimension
embedding_normalization
preprocessing_version

role_mapping

original_cmt_mapping
human_ordinal_derivation
metric_ordinal_mapping

primary_metric
secondary_metric
bootstrap_policy
classwise_logit_summary_policy
pairwise_ordering_policy
reporting_subsets

phash_threshold
exact_phash_bands
ssim_auto_threshold
manual_band_selection_rule
manual_review_order_rule
early_stop_rule

manual_overlap_labels
image_overlap_positive_labels
same_product_different_image_policy
uncertain_policy
skip_policy

TH1_definition
TH2_definition
TH3_definition

primary_subset
secondary_subset
diagnostic_subset

reason_missing_policy
calibration_policy
detection_policy
```

Bản ghi split bắt buộc:

```text
evaluation_dataset = EVALUATION3
evaluation_split_name = EVAL3-Test2000-Full
expected_test_count = 2000
test_data_row_start = 32479
test_data_row_end_inclusive = 34478
test_outfit_id_start = 200519201001
test_outfit_id_end = 200519203000
```

---

## 22. Ranh giới Đóng băng

Thứ tự bắt buộc:

```text
1. Tái dựng EVAL3-Test2000-Full từ thứ tự dòng workbook

2. Xác minh 2.000 outfit ID kỳ vọng

3. Tạo a_test2000_raw_manifest.csv

4. Chạy image/integrity audit

5. Chạy pHash search với Polyvore TRAIN + VALID + TEST
   với T_pHash = 14

6. Tính SSIM cho candidate pairs cần thiết

7. Áp dụng frozen auto-duplicate rule:
   pHash <= 14 AND SSIM >= 0.92 → DUPLICATE

8. Với manual candidates, chọn một maximum-SSIM candidate
   trên mỗi exact pHash band:
   0,2,4,6,8,10,12,14

9. Chạy manual review theo thứ tự:
   TRAIN_VALID trước, TEST sau
   SSIM descending, pHash ascending chỉ khi SSIM tie chính xác

10. Áp dụng early stopping ở cấp outfit

11. Hoàn tất manual review cho toàn bộ actionable pairs
    hoặc resolve rõ mọi UNCERTAIN/SKIP

12. Hoàn thiện item-level image-overlap audit

13. Tổng hợp lên cấp outfit

14. Gán TH1 / TH2 / TH3 / UNRESOLVED

15. Tạo frozen evaluation manifest

16. Tạo protocol artifact JSON

17. Hash/freeze overlap audit outputs, evaluation manifest và protocol artifact

========== CONFIRMATORY EVALUATION FROZEN ==========

18. Sinh FashionCLIP embeddings

19. Chạy frozen Scorer V1

20. Join frozen scorer logits với `Cmt` gốc

21. Dẫn xuất `human_ordinal_quality = 4 - Cmt`

22. Với từng tập:
    - EVAL3-Test2000-No-Full-Image-Overlap-Candidate
    - EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate
    - EVAL3-Test2000-Full

    báo cáo:
    - N_total, N_Bad, N_Normal, N_Good
    - Kendall tau-b + 95% bootstrap percentile CI
    - Spearman rho + 95% bootstrap percentile CI
    - classwise compatibility_logit summaries
    - pairwise ordering diagnostics

    Bootstrap cố định:
    seed=42
    resamples=10,000
    CI=95%
    method=percentile
    unit=outfit

23. Run reason-based error analysis

24. Freeze final EVALUATION3 result
```

---

## 23. Phân tích Lỗi Theo Reason

Chỉ chạy sau khi các metric xác nhận chính đã hoàn tất.

Các nhóm:

```text
Color
Print
Material
Silhouette
Design
```

Với mỗi reason, tối thiểu báo cáo:

```text
N
human-label distribution
compatibility-logit distribution
representative large model-human disagreement cases
```

Không tính hoặc báo cáo Good-vs-Bad ROC-AUC như một phần của frozen EVALUATION3 metric protocol.

Reason trống:

```text
remain missing
not imputed
not assigned to a reason category
```

Reason analysis không được sử dụng để sửa đổi rồi chạy lại confirmatory protocol.

---

## 24. Các Bảng Báo cáo

Cùng một tập metric và diagnostic field là bắt buộc cho cả ba evaluation subset.

### 24.1 Bảng metric thứ bậc chính

| EVAL3-Test2000 subset | N total | N Bad | N Normal | N Good | Kendall tau-b | 95% CI | Spearman rho | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate |  |  |  |  |  |  |  |  |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate |  |  |  |  |  |  |  |  |
| EVAL3-Test2000-Full |  |  |  |  |  |  |  |  |

### 24.2 Tóm tắt compatibility-logit theo từng class

Báo cáo một dòng cho mỗi subset × human class:

| Subset | Human class | N | Median logit | Q1 | Q3 | IQR |
|---|---|---:|---:|---:|---:|---:|
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate | Bad |  |  |  |  |  |
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate | Normal |  |  |  |  |  |
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate | Good |  |  |  |  |  |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate | Bad |  |  |  |  |  |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate | Normal |  |  |  |  |  |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate | Good |  |  |  |  |  |
| EVAL3-Test2000-Full | Bad |  |  |  |  |  |
| EVAL3-Test2000-Full | Normal |  |  |  |  |  |
| EVAL3-Test2000-Full | Good |  |  |  |  |  |

### 24.3 Pairwise ordering diagnostics

| Subset | P(Good > Bad) | P(Good > Normal) | P(Normal > Bad) |
|---|---:|---:|---:|
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate |  |  |  |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate |  |  |  |
| EVAL3-Test2000-Full |  |  |  |

Ví dụ, `Good > Bad` có nghĩa:

```text
compatibility_logit(Good outfit) > compatibility_logit(Bad outfit)
```

Các scorer-logit tie chính xác đóng góp `0.5`.

Ngoài ra báo cáo overlap accounting:

```text
N_EVAL3_Test2000_Full = 2000
N_TH1
N_TH2
N_TH3
N_UNRESOLVED
```

Và số lượng image-overlap ở cấp item theo từng Polyvore split.

`SAME_PRODUCT_DIFFERENT_IMAGE` được báo cáo riêng như product-overlap diagnostic và không được cộng vào image-overlap count.

---

## 25. Quy tắc Diễn giải

Được phép:

```text
Scorer đã đóng băng được huấn luyện trên Polyvore cho thấy mức liên hệ thứ bậc
[dương/yếu/không có] với đánh giá của con người trên khối test EVALUATION3
EVAL3-Test2000-Full đã được tái dựng.
```

Được phép:

```text
Kết quả EVAL3-Test2000-No-Full-Image-Overlap-Candidate kiểm soát known same-image/visual duplicate overlap
được phát hiện theo frozen audit protocol.
```

Được phép:

```text
Khác biệt giữa EVAL3-Test2000-Full và hai image-clean subset cho thấy ảnh hưởng có thể có
của mức độ quen thuộc với source/item.
```

Không được khẳng định:

```text
EVALUATION3 is fully independent of Polyvore1000.
```

Không được khẳng định:

```text
TH3 guarantees absence of all possible product overlap.
```

Cách diễn đạt ưu tiên:

```text
no known overlap under the frozen audit protocol
```

Không được khẳng định:

```text
the scorer objectively understands fashion quality.
```

---

## 26. Tóm tắt Các Quyết định Đã Đóng băng

```text
Confirmatory evaluation base:
EVAL3-Test2000-Full

EVAL3-Test2000-Full definition:
last 2.000 EVALUATION3 workbook data rows

Data-row indices:
32.479–34.478 inclusive

Python slice:
rows[32479:34479]

Outfit-ID range:
200519201001–200519203000

Expected N:
2.000

Primary metric:
Kendall tau-b

Secondary metric:
Spearman rho

Bootstrap:
seed       = 42
resamples  = 10,000
CI         = 95%
CI method  = percentile
unit       = outfit

Required descriptive reporting:
N_total
N_Bad
N_Normal
N_Good
classwise compatibility_logit median/Q1/Q3/IQR

Pairwise ordering diagnostics:
P(logit_Good > logit_Bad)
P(logit_Good > logit_Normal)
P(logit_Normal > logit_Bad)

Same metric protocol on:
EVAL3-Test2000-No-Full-Image-Overlap-Candidate
EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate
EVAL3-Test2000-Full

Raw human label:
Cmt:
Good = 1
Normal = 2
Bad = 3

Derived ordinal quality:
human_ordinal_quality = 4 - Cmt

Therefore:
Bad = 1
Normal = 2
Good = 3

Scorer signal:
raw compatibility_logit

Overlap search:
Polyvore TRAIN + VALID + TEST

T_pHash:
14

T_SSIM_auto:
0.92

Exact pHash bands:
0,2,4,6,8,10,12,14

Manual selection:
one maximum-SSIM candidate
per exact pHash band
within each reference scope

Manual ordering:
TRAIN_VALID first
then TEST
SSIM descending
pHash ascending only for exact SSIM tie

Image-overlap positive label:
DUPLICATE

SAME_PRODUCT_DIFFERENT_IMAGE:
not image overlap
does not trigger early stop
reported separately as product-overlap diagnostic

UNCERTAIN / SKIP:
unresolved
excluded from image-clean subsets

TH1:
TRAIN or VALID image overlap

TH2:
TEST-only image overlap

TH3:
no known TRAIN/VALID/TEST image overlap

Primary subset:
EVAL3-Test2000-No-Full-Image-Overlap-Candidate

Secondary subset:
EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate

Diagnostic subset:
EVAL3-Test2000-Full

RF-DETR:
not used

ROC-AUC:
not used

Calibration:
not used

EVALUATION3 fine-tuning:
not allowed
```
