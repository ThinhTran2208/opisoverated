# Recommendation V2 — Process Report (04/09/2026)

## Mục tiêu trong ngày

Hoàn thiện Recommendation V2 để khắc phục bottleneck retrieval của V1, chạy VALID/TEST, chọn bộ metric dễ giải thích và freeze kết quả cho final report.

## 1. Vấn đề của Recommendation V1

V1 làm retrieval theo thứ tự:

```text
global cosine Top-200
→ filter exact master_category
→ scorer rerank
```

Top-200 có thể bị chiếm bởi nhiều item sai category, nên sau filter số candidate hữu ích bị giảm mạnh. Trên TEST V1, Hybrid Recall@200 chỉ đạt **8.47%** và có **59** query không đủ 3 candidate cuối.

## 2. Lựa chọn cho Recommendation V2

V2 đổi thứ tự retrieval thành:

```text
xác định category của problematic item
→ lọc candidate pool theo exact master_category trước
→ item cosine Top-200
→ context cosine Top-200
→ union + dedup
→ frozen scorer rerank toàn bộ union
→ Top-3
```

Lý do: dành toàn bộ retrieval budget cho candidate có khả năng thay thế đúng loại item. Nếu runtime thiếu exact `master_category`, hệ thống fallback sang cùng Core-7 category; benchmark offline vẫn dùng exact master category khi metadata có sẵn.

Không retrain FashionCLIP, scorer hoặc LOO.

## 3. Điều chỉnh artifact loading

Drive thực tế không có ZIP artifact như code V1 giả định, nên Recommendation V2 được bổ sung directory mode portable:

```text
--artifact-root /path/to/ML_Final
--image-root /path/to/images
```

Cách này không phụ thuộc Colab; cùng interface có thể dùng trên local Linux, server hoặc Docker.

Folder ảnh trên Google Drive rất lớn và bị timeout khi scan toàn bộ, nên image resolver được đổi sang lazy lookup theo `item_id`, tránh enumerate 142k file mỗi lần chạy.

## 4. VALID và lựa chọn metric

VALID chạy sạch:

- **N = 1,142**
- excluded = 0
- scorer error = 0
- image read error = 0

Kết quả retrieval VALID:

| Metric | VALID |
| --- | ---: |
| Item Recall@200 | 14.80% |
| Context Recall@200 | 36.87% |
| Hybrid Recall@200 | 31.79% |
| Full-union GT coverage | 40.89% |

Hit@1/Hit@3 ban đầu được xem xét nhưng không dùng làm metric chính vì benchmark chỉ có một exact reference: `original_item_id`. Một replacement khác GT vẫn có thể hợp lý nhưng sẽ bị tính miss.

Conditional Hit@3 cũng được thử để tách retrieval failure khỏi reranking, nhưng vẫn phụ thuộc mạnh vào single-GT assumption. Cuối cùng reranking được báo bằng **rank diagnostics** trên các query mà GT đã có trong full union.

VALID reranking:

| Metric | VALID |
| --- | ---: |
| GT rank improved | 60.39% |
| GT rank worsened | 38.12% |
| Median rank change | +16 positions |

MRR vẫn được giữ trong raw evaluation để audit, nhưng không đưa vào main report để tránh làm bảng rối và vì giá trị diễn giải không mạnh bằng rank movement.

## 5. Frozen TEST result

TEST full chạy với **N = 2,327**, excluded = 0, không có scorer/image runtime error.

### Retrieval

| Metric | V1 TEST | V2 TEST |
| --- | ---: | ---: |
| Hybrid Recall@200 | 8.47% | **33.95%** |

Supporting V2 retrieval metrics:

| Metric | V2 TEST |
| --- | ---: |
| Item Recall@200 | 17.58% |
| Context Recall@200 | 39.62% |
| Hybrid Recall@200 | **33.95%** |
| Full-union GT coverage | **44.13%** |

V2 cũng loại bỏ failure `fewer_than_three_final_candidates`: **59 → 0**.

### Frozen scorer reranking

Trong **1,027** query mà exact GT có mặt trong full union:

| Metric | V2 TEST |
| --- | ---: |
| GT rank improved | **58.71%** |
| GT rank unchanged | 1.85% |
| GT rank worsened | 39.44% |
| Median rank change | **+13 positions** |

Replacement Success Rate = **99.28%**, nhưng chỉ giữ như scorer self-consistency diagnostic, không coi là human recommendation quality.

## 6. Reporting policy cuối cùng

Main report sẽ ưu tiên:

- **Hybrid Recall@200** — retrieval metric chính để so V1/V2;
- **Full-union GT coverage** — GT có thực sự được đưa tới scorer hay không;
- **GT rank improved / worsened** và **Median rank change** — reranking diagnostics.

Không dùng Hit@1/Hit@3, Conditional Hit@3 hoặc MRR làm headline metric. Chúng vẫn có thể giữ trong raw artifact để audit.

Tất cả metric là **exact-reference diagnostics trên synthetic one-item-swap benchmark**, không phải recommendation accuracy hoặc human preference accuracy.

## 7. Freeze

Canonical branch:

```text
feat/recommendation-rank-diagnostics-v2
```

Previous experimental branch name:

```text
feat/recommendation-conditional-hit-v2
```

được xem là deprecated vì Conditional Hit@3 không còn là protocol chính.

Frozen evaluation protocol:

```text
polyvore-one-item-swap-recovery-v2-rank-diagnostics
```

Frozen result artifact:

```text
artifacts/recommendation_v2_rankdiag_freeze.json
```

Evaluation code commit dùng cho VALID/TEST đã ghi nhận:

```text
3472abf7a39ae9fc51683bcb19b5dcd3ac3e8ec4
```
