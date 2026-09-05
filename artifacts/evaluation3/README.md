# EVALUATION3 human-alignment artifacts

This folder contains only the small frozen references needed to document and
verify the EVALUATION3 Test2000 human-alignment evaluation.

The evaluation asks whether the frozen compatibility scorer's raw
`compatibility_logit` follows the same direction as EVALUATION3 human ordinal
quality (`Bad=1`, `Normal=2`, `Good=3`).

## External artifact storage

Canonical/staging Drive root:

https://drive.google.com/drive/folders/13nbrOlCeyFdNkBBJYzeJdN1twDGvMk6i

Large/generated artifacts remain on Drive and are intentionally not committed
to GitHub, including images, FashionCLIP embedding caches, scorer-ready JSONL,
overlap-audit dumps/manual-review material, and joined/intermediate metric
outputs.

## Frozen protocol

- protocol version: `EVALUATION3_EXTERNAL_EVALUATION_V2`
- protocol status: `FROZEN`
- evaluation split: `EVAL3-Test2000-Full`
- full set: `N=2000`
- no-train/valid-image-overlap candidate subset: `N=609`
- no-full-image-overlap candidate subset: `N=422`
- primary metric: Kendall tau-b
- secondary metric: Spearman rho
- bootstrap: seed=42, 10,000 outfit-level resamples, 95% percentile CI

### Frozen hashes

- `EVALUATION3_EXTERNAL_EVALUATION_V2_VI.md`
  - SHA256 `cd9f44a109201f010ecd0bcd33a7a5655301346e3c427ff7b3804b5dd6d757bf`
- `EVAL3-Test2000-protocol-v2-final.json`
  - SHA256 `74ab73bd935b628a1923372d2bd27cd82aebcce4c97f52b594e756191142470f`
- `EVAL3-Test2000-Full-run-manifest.json`
  - SHA256 `fb244ffa93b21fdf8709a45a3fb571228150d9ada6ff13dccbfb22f8aa23da07`
- `metrics/EVAL3-Test2000-metrics-main.csv`
  - SHA256 `aad1ddfa25cfbb183c7c253e8b648370d9d24b036c3e3a98dfb379f57770feb3`
- `metrics/EVAL3-Test2000-classwise-logit-summary.csv`
  - SHA256 `2f82f08a2f5378c74e2fac3015b9149be975f11225cf3cb5c3242cb47835f99e`
- `metrics/EVAL3-Test2000-pairwise-ordering.csv`
  - SHA256 `25c74a81c00ab20006d42253bb43803bb469ae7240a25e00a397da50f792cfda`
- `metrics/EVAL3-Test2000-metrics-result.json`
  - SHA256 `cf72d7e58bd41cebc14a3a5da6b6dfe1ad80a258f45747534d781b30eb056670`

The evaluation manifest, human labels, predictions, embeddings and other
generated inputs stay in external storage. Their exact provenance/hashes are
recorded by the frozen protocol/result artifacts and run manifest.

## GitHub contents

GitHub keeps:
- stdlib-only evaluator metric code;
- unit tests;
- frozen protocol JSON + SHA sidecar;
- scorer run manifest;
- small metric result tables;
- the human-readable protocol and result summary.

GitHub does not keep:
- `EVAL3-Test2000-Full-scorer-ready.jsonl`;
- `EVAL3-Test2000-metric-input-joined.csv`;
- EVALUATION3 images;
- FashionCLIP `.pt` embeddings;
- overlap/manual-review audit dumps.
