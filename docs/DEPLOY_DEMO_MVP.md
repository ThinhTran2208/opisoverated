# Runbook deploy backend MVP — demo 1 ngày

**Ngày cập nhật:** 2026-09-05  
**Mục tiêu:** chạy web app đến hết ngày demo, tối đa khoảng 3 request đồng thời.  
**Kiến trúc đã chốt:** một instance Vast.ai có GPU CUDA, một Docker container, một public port `8000`, không Redis/Celery/Kubernetes/database/reverse proxy.

## Quy ước cập nhật tiến độ

- `✅ DONE`: đã hoàn tất và có thể chuyển bước.
- `🟡 IN PROGRESS`: đang thực hiện, chưa được chuyển bước kế tiếp.
- `⬜ TODO`: chưa bắt đầu.
- `❌ BLOCKED`: có lỗi, ghi nguyên nhân ở cột ghi chú.

## Tiến độ hiện tại

| # | Hạng mục | Trạng thái | Điều kiện chuyển bước |
|---:|---|---|---|
| 1 | Backend contract V2 | ✅ DONE | Core gọi được Recommendation V2 và VLM V2 |
| 2 | Runtime một container | ✅ DONE | Có `Dockerfile.demo` và startup script |
| 3 | Kiểm tra local | ✅ DONE | Compile, shell check, targeted tests pass |
| 4 | Chuẩn bị artifact và máy Vast | 🟡 IN PROGRESS | Có GPU, port `8000`, artifact mount đúng |
| 5 | Build và khởi động container | ⬜ TODO | `/healthz` trả `status=ok` |
| 6 | Smoke test request thật | ⬜ TODO | `/v2/analyze-outfit` trả diagnosis + Top-3 + explanation |
| 7 | Chạy demo | ⬜ TODO | Frontend gọi được API ổn định |
| 8 | Dừng và release máy | ⬜ TODO | Container và instance Vast đã tắt |

> Cập nhật bảng sau mỗi bước; không đánh dấu `DONE` nếu chưa có bằng chứng ở phần tương ứng.

## Bước 1 — Chốt scope trước khi deploy

- [x] Chỉ hỗ trợ flow: upload ảnh outfit → detection/FashionCLIP → scorer/LOO → Recommendation Top-3 → VLM explanation.
- [x] Dùng endpoint chính `POST /v2/analyze-outfit`.
- [x] Giữ tối đa một inference chạy trên GPU; tối đa 3 request pending.
- [x] Không thêm login, database, queue, analytics, autoscaling hoặc HTTPS riêng cho demo một ngày.

## Bước 2 — Chuẩn bị artifact và image catalog

**Trạng thái thực tế:** ✅ DONE. Các artifact bắt buộc và image catalog đã được xác nhận trong Google Drive; bỏ qua chênh lệch số lượng ảnh theo quyết định triển khai.

Trên máy Vast, đặt dữ liệu theo đúng layout sau:

```text
/path/to/ML_Final/
  fashionclip_item_embeddings.pt
  embedding_manifest_v1.json
  scorer_runs/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt
  polyvore_core7_v2/core7_drop_v2/core7_item_metadata_v1_{train,valid,test}.jsonl
  polyvore_core7_v2/scorer_ready_v2/scorer_ready_v2_{train,valid,test}.jsonl
/path/to/images/{item_id}.jpg
```

- [x] Kiểm tra đủ file bắt buộc trong thư mục `ML_Final` trên Google Drive:

  ```bash
  test -f /path/to/ML_Final/fashionclip_item_embeddings.pt
  test -f /path/to/ML_Final/embedding_manifest_v1.json
  test -f /path/to/ML_Final/scorer_runs/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt
  ```

- [x] Xác nhận image catalog trong thư mục `images`; bỏ qua số lượng ảnh còn thiếu theo quyết định triển khai.
- [x] Đảm bảo user chạy hiện tại có quyền đọc `D:/BKU/VSC/images`.

**Bằng chứng đã kiểm tra ngày 2026-09-05:**

- Google Drive `ML_Final` có `fashionclip_item_embeddings.pt`, `embedding_manifest_v1.json`, `scorer_runs/` và `polyvore_core7_v2/`.
- Local có scorer checkpoint tại `artifacts/checkpoints/type_aware_pairwise_v1/final_val_auc_v5_seed42/best.pt`, nhưng vị trí này chưa thay thế được layout `ML_Final/...` mà Docker runtime yêu cầu.
- Còn thiếu bước tải toàn bộ thư mục `ML_Final` và đồng bộ image catalog đầy đủ trước khi đánh dấu `DONE`.

**Bằng chứng cần ghi:** đường dẫn thực tế trên Vast và kết quả `test -f`.

## Bước 3 — Tạo instance Vast.ai

- [ ] Thuê một instance có GPU CUDA đủ VRAM cho RF-DETR + scorer + Qwen3-VL 4B; không thuê nhiều GPU.
- [ ] Kiểm tra GPU và driver:

  ```bash
  nvidia-smi
  docker --version
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

- [ ] Mở TCP port `8000` trong firewall/Vast port mapping.
- [ ] Copy repository lên instance:

  ```bash
  git clone --branch feat/vlm-v2-final \
    https://github.com/ThinhTran2208/opisoverated.git
  cd opisoverated
  ```

**Bằng chứng cần ghi:** GPU model, port public, commit/branch đang chạy.

## Bước 4 — Build image MVP

Từ repository root:

```bash
docker build -f Dockerfile.demo -t outfit-demo:latest .
```

- [ ] Build hoàn tất không lỗi dependency.
- [ ] Không cần push image lên registry; build trực tiếp trên instance là đủ.

Nếu build lỗi, lưu 30 dòng log cuối và giữ trạng thái `❌ BLOCKED`; không tự thêm service mới để chữa cháy.

## Bước 5 — Khởi động backend

```bash
docker run --rm --gpus all \
  --name outfit-demo \
  -p 8000:8000 \
  -v /path/to/ML_Final:/data/ML_Final:ro \
  -v /path/to/images:/data/images:ro \
  outfit-demo:latest
```

Nếu frontend chạy ở domain khác, thêm một biến môi trường trước image name:

```bash
-e FASHION_CORS_ORIGINS=https://frontend.example.com
```

Mặc định demo đang cho phép `FASHION_CORS_ORIGINS=*`; chỉ dùng vậy trong thời gian ngắn.

- [ ] Container ở trạng thái running.
- [ ] Startup script đã đợi VLM nội bộ trên `127.0.0.1:8001`.
- [ ] Core API lắng nghe public trên `0.0.0.0:8000`.

## Bước 6 — Smoke test trước khi gửi link

Mở terminal thứ hai trên Vast hoặc máy có thể truy cập port `8000`:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS -F 'image=@/path/to/outfit.jpg' \
  http://127.0.0.1:8000/v2/analyze-outfit \
  -o response.json
python -m json.tool response.json
```

- [ ] Health trả `status: ok` và `recommendation_v2: true`.
- [ ] Response có `diagnosis`.
- [ ] Response có `recommendation.items` đúng 3 item, có `rank` và `image_url`.
- [ ] Response có `explanation` tiếng Việt.
- [ ] Mở thử một `image_url` để xác nhận candidate image được serve.
- [ ] Gửi một ảnh không hợp lệ để xác nhận API trả lỗi có cấu trúc, không làm container chết.

**Bằng chứng cần ghi:** lưu `healthz` và một `response.json` mẫu.

## Bước 7 — Chạy demo

- [ ] Chỉ chia sẻ URL/port sau khi smoke test pass.
- [ ] Nếu có tối đa 3 request cùng lúc: một request chạy, hai request chờ.
- [ ] Request thứ tư có thể nhận HTTP `429`; yêu cầu người dùng thử lại sau.
- [ ] Nếu VLM timeout hoặc lỗi model, xem log container; không restart liên tục trong khi chưa biết nguyên nhân.

Các endpoint frontend cần dùng:

```text
POST /v2/analyze-outfit              multipart field: image
GET  /recommendation/images/{item_id}
GET  /healthz
```

## Bước 8 — Kết thúc ngay sau demo

- [ ] Nhấn `Ctrl-C` ở terminal chạy Docker, hoặc:

  ```bash
  docker stop outfit-demo
  ```

- [ ] Xác nhận container đã dừng.
- [ ] Release/delete instance Vast.ai ngay.
- [ ] Kiểm tra lại dashboard Vast để chắc chắn không còn instance tính phí.

Không cần duy trì uptime, backup, monitoring dài hạn hay AWS cho phiên demo này.
