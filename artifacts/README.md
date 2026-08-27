# Artifacts

GitHub chứa code, config, tests và **reference nhỏ**. Dữ liệu sinh ra được lưu
bên ngoài GitHub để repository không phình to và binary có thể được tái sử dụng
giữa các experiment.

## Không commit

- scorer-ready JSONL lớn;
- FashionCLIP embedding cache (`.pt`);
- embedding manifest thực tế của cache external;
- model checkpoint;
- ảnh;
- large evaluation outputs.

## Được commit trong folder này

- link tới external storage;
- artifact/dataset version;
- commit và config đã dùng để tạo artifact;
- expected filenames;
- validation status, record counts và SHA256 khi đã freeze;
- template của embedding manifest.

`category_mapping_core7_v1.json` là mapping frozen cũ và không được mutate.
Các category-policy thay đổi trong PR #3 được version thành `core7-v2` và output
được ghi sang folder V2 riêng.

## Version canonical

Mỗi loại artifact có namespace riêng; không dùng mapping version thay cho
dataset version:

```text
dataset_version          = polyvore1000-core7-compat-v2
category_mapping_version = core7-v2
negative_protocol_version = negative-v1
embedding_version        = fashionclip-512-l2-v1
```

`dataset_version` mô tả toàn bộ benchmark đã build. Ba version còn lại mô tả
đúng component tương ứng và được ghi riêng trong manifest.

## Embedding manifest bắt buộc

FashionCLIP cache không đủ để xác định provenance chỉ bằng shape `[N, 512]`.
Cạnh cache phải có `embedding_manifest_v1.json` chứa ít nhất:

```text
embedding_version
model_name_or_version
preprocessing_version
embedding_dimension
normalization
dtype
item_count
cache_sha256
```

Template: `artifacts/embedding_manifest_v1.example.json`.

## Cấu trúc external artifact chuẩn cho V2

```text
artifact_root/
├── cache/
│   ├── fashionclip_item_embeddings.pt
│   └── embedding_manifest_v1.json
├── core7_drop_v2/
│   ├── category_clean_{train,valid,test}.jsonl
│   ├── core7_item_metadata_v1_{train,valid,test}.jsonl
│   └── core7_embedding_validation_report.json
└── scorer_ready_v2/
    ├── scorer_ready_v2_{train,valid,test}.jsonl
    ├── negative_v1_{train,valid,test}.jsonl
    ├── dataset_manifest_v2.json
    └── final_validation_v2.json
```

Lưu ý: `core7_item_metadata_v1` là **schema version** của item metadata; mỗi row
trong V2 phải ghi `category_mapping_version = core7-v2`. Tương tự,
`negative_v1` vẫn là protocol `same_category_different_kit`; dataset được bump
sang V2 vì category mapping semantics đã thay đổi.

NB3 ghi SHA-256 của exact inputs vào embedding-validation report. Vì Core-7 V2
là override trên V1, report bind cả file V2, frozen V1 base và resolved mapping.
NB4 phải hash lại mapping, cache, manifest, positives và metadata; bất kỳ
mismatch nào đều hard-fail trước negative sampling.

Official freeze còn yêu cầu `git rev-parse --verify HEAD` thành công và
`git status --porcelain` không có entry. Nếu thiếu commit hoặc working tree
dirty, NB4 hard-fail trước khi tạo output và không được ghi
`READY_TO_TRAIN`. `dataset_manifest_v2.json` phải lưu exact `git_commit` và
`git_tree_clean = true`.

Người chạy có thể đặt `artifact_root` ở bất kỳ đâu và khai báo qua
`FASHION_ARTIFACT_ROOT` hoặc `configs/data_paths.local.json`.
