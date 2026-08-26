# Artifacts

GitHub chứa code, config, tests và **reference nhỏ**. Dữ liệu sinh ra được lưu
bên ngoài GitHub để repository không phình to và binary vẫn có thể được tái sử
dụng giữa các experiment.

## Không commit

- scorer-ready JSONL lớn;
- FashionCLIP embedding cache (`.pt`);
- model checkpoint;
- ảnh;
- large evaluation outputs.

## Được commit trong folder này

- link tới external storage;
- artifact/dataset version;
- commit và config đã dùng để tạo artifact;
- expected filenames;
- validation status, record counts và SHA256 khi đã freeze.

Reference hiện tại nằm trong `data_v1_reference.json`.

## Cấu trúc external artifact chuẩn

```text
artifact_root/
├── cache/
│   └── fashionclip_item_embeddings.pt
├── core7_drop_v1/
│   ├── category_clean_{train,valid,test}.jsonl
│   ├── core7_item_metadata_v1_{train,valid,test}.jsonl
│   └── core7_embedding_validation_report.json
└── scorer_ready_v1/
    ├── scorer_ready_v1_{train,valid,test}.jsonl
    ├── negative_v1_{train,valid,test}.jsonl
    ├── dataset_manifest_v1.json
    └── final_validation_v1.json
```

Người chạy có thể đặt `artifact_root` ở bất kỳ đâu và khai báo qua
`FASHION_ARTIFACT_ROOT` hoặc `configs/data_paths.local.json`.
