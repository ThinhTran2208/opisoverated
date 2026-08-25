# Bộ Metrics của Project

Tài liệu này định nghĩa các metrics dùng để đánh giá ba thành phần ML chính của project: **Compatibility Scorer**, **Recommendation**, và **Diagnosis**.

Mục tiêu là giữ protocol đánh giá đơn giản, nhất quán và dễ so sánh giữa các experiment. **Primary metrics** dùng để so sánh và chọn model. **Secondary metrics** chủ yếu dùng để phân tích hành vi model và tìm failure mode.

---

## 1. Compatibility Scorer

### Primary metrics

#### ROC-AUC

ROC-AUC là metric chính để đánh giá khả năng scorer phân biệt outfit compatible và incompatible.

Metric này được ưu tiên hơn các metric phụ thuộc threshold vì nó đánh giá chất lượng ranking trên toàn bộ các ngưỡng quyết định có thể có.

Trong project này, ROC-AUC trả lời câu hỏi:

> Scorer có thường gán điểm cho positive outfit cao hơn negative outfit hay không?

ROC-AUC càng cao thì khả năng phân tách hai nhóm càng tốt.

---

#### 2-way FITB

Dataset tạo negative outfit bằng cách thay đúng một item trong positive outfit bằng một item khác cùng category.

Vì vậy mỗi positive outfit có một negative outfit tương ứng.

Với mỗi cặp:

\[
(O_i^+, O_i^-)
\]

2-way FITB kiểm tra:

\[
s(O_i^+) > s(O_i^-)
\]

Metric được tính:

\[
\text{2-way FITB}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[
s(O_i^+) > s(O_i^-)
]
\]

Metric này cần thiết vì ROC-AUC đo khả năng **phân tách positive và negative trên toàn dataset**, trong khi 2-way FITB đo một câu hỏi cục bộ hơn:

> Scorer có xếp original outfit cao hơn đúng negative outfit được tạo bằng cách swap một item từ outfit đó hay không?

Điều này liên quan trực tiếp tới các bước diagnosis và replacement sau này, vì hệ thống sẽ thường xuyên sửa một item rồi chạy scorer lại.

Lưu ý: đây là **2-way FITB riêng của project**, không phải FITB truyền thống với 4 lựa chọn thường dùng trong các paper về outfit compatibility.

---

### Secondary metrics

#### Mean logit margin

Với mỗi cặp positive/negative:

\[
m_i = z_i^+ - z_i^-
\]

trong đó \(z\) là raw logit của scorer trước sigmoid.

Mean logit margin:

\[
\text{Mean Margin}
=
\frac{1}{N}
\sum_{i=1}^{N}
m_i
\]

Metric này cho biết scorer phân tách positive và negative mạnh đến mức nào trên trung bình.

2-way FITB chỉ quan tâm margin dương hay âm. Mean margin cho biết thêm khoảng cách giữa hai score lớn hay nhỏ.

---

#### Median logit margin

Median logit margin là trung vị của tất cả paired margins:

\[
\text{Median Margin}
=
\operatorname{median}(m_1,\dots,m_N)
\]

Metric này được báo cáo cùng mean vì mean có thể bị ảnh hưởng bởi một số ít pair có margin rất lớn hoặc rất nhỏ.

Median cho cái nhìn ổn định hơn về mức separation điển hình của model.

---

#### F1-score

F1-score kết hợp precision và recall:

\[
F1
=
2
\frac{
\text{Precision}\times\text{Recall}
}{
\text{Precision}+\text{Recall}
}
\]

Metric này dùng để kiểm tra scorer có bị lệch quá nhiều về dự đoán compatible hoặc incompatible hay không.

Tuy nhiên, F1 phụ thuộc vào threshold được chọn, nên không dùng làm metric chính để chọn model.

---

## 2. Recommendation

Recommendation được đánh giá theo hai giai đoạn:

1. **Candidate Retrieval**
2. **Scorer Reranking**

Việc tách hai stage giúp xác định lỗi recommendation đến từ retrieval hay từ reranker.

---

## 2.1 Candidate Retrieval

Retrieval có nhiệm vụ giảm số lượng candidate trước khi chạy compatibility scorer, vì scorer tốn chi phí tính toán hơn cosine retrieval.

### Primary metric: Recall@200

Recall@200 đo xem ground-truth replacement item có xuất hiện trong 200 candidate được retrieve hay không.

Với query \(q\):

\[
Recall@K(q)
=
\frac{
|\text{GT}_q \cap TopK_q|
}{
|\text{GT}_q|
}
\]

Sau đó lấy trung bình trên toàn bộ queries.

Recall@200 được chọn làm metric chính vì pipeline hiện tại dự kiến lấy khoảng 200 item sau retrieval để đưa vào reranker.

Nếu ground-truth item không nằm trong Top-200, reranker sẽ không còn cơ hội đưa item đó lên Top-K cuối.

---

### Secondary metrics: Recall@100 và Recall@50

Recall@100 và Recall@50 dùng để phân tích ground-truth item xuất hiện sớm đến mức nào trong danh sách retrieval.

Hai metric này giúp trả lời:

> Có thể giảm candidate pool từ 200 xuống 100 hoặc 50 mà không làm mất quá nhiều ground-truth hay không?

Nếu Recall@50 gần Recall@200 thì có thể giảm số candidate và tiết kiệm computation cho reranking.

---

### Recall@K và Hit@K

Với một query \(q\), Recall@K được định nghĩa:

\[
Recall@K(q)
=
\frac{
|\text{GT}_q \cap TopK_q|
}{
|\text{GT}_q|
}
\]

Hit@K là metric nhị phân:

\[
Hit@K(q)
=
\mathbf{1}
[
|\text{GT}_q \cap TopK_q| > 0
]
\]

Ý nghĩa:

- **Recall@K**: trong tổng số ground-truth relevant items, tìm được bao nhiêu item trong Top-K.
- **Hit@K**: chỉ kiểm tra có ít nhất một ground-truth item xuất hiện trong Top-K hay không.

Trong dataset hiện tại, mỗi outfit chỉ có **một ground-truth item**.

Do đó trong project này:

\[
Recall@K = Hit@K
\]

về mặt giá trị số.

Project chọn dùng **Recall@K** vì các paper retrieval thường sử dụng Recall@K, và metric này cũng mở rộng tự nhiên nếu sau này mỗi query có nhiều ground-truth items.

---

## 2.2 Reranking

Sau retrieval, compatibility scorer đánh giá từng candidate replacement trong context của toàn bộ outfit và rerank lại danh sách.

### Primary metric: Recall@5

Recall@5 đo xem ground-truth item có nằm trong 5 recommendation cuối hay không.

Đây là metric chính của reranking vì hệ thống cuối cùng dự kiến chỉ hiển thị một số lượng nhỏ recommendation cho người dùng.

Do đó Recall@5 trực tiếp trả lời:

> Ground-truth replacement có sống sót qua cả retrieval lẫn scorer reranking và xuất hiện trong shortlist cuối hay không?

---

### Secondary metrics: Recall@1, Recall@3, Recall@10

Các metric này dùng để phân tích chi tiết hơn chất lượng ranking:

- **Recall@1**: ground-truth item có đứng đầu danh sách hay không.
- **Recall@3**: ground-truth item có nằm trong Top-3 hay không.
- **Recall@10**: ground-truth item có được scorer đưa lên vùng đầu danh sách hay không, ngay cả khi chưa vào Top-5.

Nhìn đồng thời các giá trị này giúp đánh giá quality của ranking thay vì chỉ nhìn một cutoff duy nhất.

---

### Secondary metric: Replacement Success Rate

Recommendation không chỉ cần recover ground-truth item. Một candidate khác ground truth vẫn có thể là recommendation tốt nếu nó làm compatibility score tăng đủ nhiều.

Gọi:

\[
C(O)
\]

là compatibility score hiện tại, và:

\[
C(O^{(c)})
\]

là score sau khi thay problematic item bằng candidate \(c\).

Một replacement được xem là thành công nếu:

\[
C(O^{(c)}) - C(O) > \epsilon
\]

trong đó \(\epsilon\) là mức cải thiện tối thiểu cần đạt.

Replacement Success Rate:

\[
\text{Replacement Success Rate}
=
\frac{
\#\text{successful replacements}
}{
\#\text{evaluated recommendations}
}
\]

Metric này trả lời câu hỏi thực tế:

> Recommendation có làm compatibility score tăng một mức đủ đáng kể hay không?

Metric được xếp secondary vì kết quả phụ thuộc vào hyperparameter \(\epsilon\).

---

## 3. Diagnosis

Diagnosis dùng Leave-One-Out (LOO) để xác định item có ảnh hưởng tiêu cực nhất đến compatibility.

Với outfit \(O\), trước tiên tính:

\[
C(O)
\]

Sau đó lần lượt bỏ từng item \(i\):

\[
\Delta_i
=
C(O \setminus x_i) - C(O)
\]

Item có \(\Delta_i\) lớn nhất được xem là problematic item:

\[
\hat y
=
\arg\max_i \Delta_i
\]

Do synthetic negative được tạo bằng cách swap đúng một item, dataset có ground-truth `swapped_index` là \(y\). Vì vậy diagnosis có thể được đánh giá trực tiếp.

---

### Primary metric: Localization Accuracy

Localization Accuracy kiểm tra item do LOO chọn có đúng là item đã bị swap hay không.

Với mỗi outfit:

\[
\mathbf{1}[\hat y = y]
\]

Sau đó lấy trung bình:

\[
\text{Localization Accuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
[
\hat y_i = y_i
]
\]

Metric này trực tiếp đo mục tiêu chính của diagnosis:

> Model có xác định đúng problematic item hay không?

---

### LOO Top-1

Trong project này, **LOO Top-1 và Localization Accuracy là cùng một metric**.

LOO Top-1 chọn item có LOO delta lớn nhất:

\[
\hat y
=
\arg\max_i \Delta_i
\]

sau đó kiểm tra:

\[
\hat y = y
\]

Do đó không cần báo cáo LOO Top-1 và Localization Accuracy như hai metric riêng biệt.

Tên đề xuất dùng trong report:

**LOO Top-1 Localization Accuracy**

---

### Secondary metric: LOO Hit@2

LOO Hit@2 kiểm tra ground-truth swapped item có nằm trong hai item có LOO delta cao nhất hay không.

Với mỗi outfit:

\[
Hit@2
=
\mathbf{1}
[
y \in Top2(\Delta)
]
\]

LOO Hit@2 luôn lớn hơn hoặc bằng LOO Top-1, vì mọi trường hợp đúng ở Top-1 cũng chắc chắn nằm trong Top-2.

Metric này được dùng để phân tích độ chắc chắn của diagnosis.

Ví dụ:

```text
LOO Top-1 = 65%
LOO Hit@2 = 70%
```

Khoảng cách nhỏ cho thấy nếu model nhận ra swapped item thì thường đã xếp nó lên vị trí đầu.

Ngược lại:

```text
LOO Top-1 = 45%
LOO Hit@2 = 80%
```

Khoảng cách lớn cho thấy model thường nhận ra swapped item là một trong những item đáng nghi, nhưng không đủ khả năng xếp đúng nó là problematic item số một.

Vì vậy, nếu **LOO Top-1 và LOO Hit@2 chênh lệch quá lớn**, diagnosis ranking cần được kiểm tra thêm.

---

## Tổng hợp Metrics

| Thành phần | Mức ưu tiên | Metrics |
|---|---|---|
| **Scorer** | Primary | ROC-AUC, 2-way FITB |
| | Secondary | Mean logit margin, Median logit margin, F1-score |
| **Recommendation — Retrieval** | Primary | Recall@200 |
| | Secondary | Recall@100, Recall@50 |
| **Recommendation — Reranking** | Primary | Recall@5 |
| | Secondary | Recall@1, Recall@3, Recall@10, Replacement Success Rate |
| **Diagnosis** | Primary | LOO Top-1 Localization Accuracy |
| | Secondary | LOO Hit@2 |

Primary metrics được dùng để so sánh và chọn model.

Secondary metrics được dùng để phân tích sâu hơn hành vi của model, phát hiện failure mode và hỗ trợ quyết định trong quá trình experiment.
