# EVALUATION3 NB11B domain-aware fix

## Why NB11 historical looked perfect but did not transfer

The historical feature cache contains 622 rows:

- 322 confirmed-duplicate positives;
- 28 duplicate rows from the hard-review source;
- 272 non-duplicate rows from the hard-review source.

The confirmed queue is an easy, positive-only source. On this cache, the rule
`confirmed_duplicate source -> positive; otherwise -> negative` already reaches
about 95.5% row accuracy. A random group-disjoint split of the union therefore
still permits source/domain provenance to make the held-out task much easier
than the current near-duplicate queue.

The 30 rows in `round_01_query.xlsx` are also not an IID test set. They were
chosen by the previous model for being closest to `p=0.5`; having all 30 land in
`MANUAL_REVIEW` is expected for an uncertainty query.

## What changes

`NB11B_historical_domain_aware_classifier.ipynb` uses:

1. source-aware, E3-group-disjoint validation and test drawn only from
   `hard_negative_review`;
2. confirmed duplicates as low-weight (`0.25`) training anchors;
3. current active-learning labels as high-weight (`4.0`) training rows instead
   of misusing them as a test;
4. a separate resolution queue that ranks high-probability duplicates and
   emits at most one candidate per unresolved E3 group; and
5. the explicit label `SAME_PRODUCT_DIFFERENT_IMAGE`, mapped to the binary
   duplicate target while retaining the subtype for audit.

On the cached data with seed 42, the honest hard-only test has 60 rows but only
6 positives. The selected random forest obtains ROC-AUC about 0.93, while
recall at the arbitrary 0.5 threshold is only 0.33. This is much more useful
than the old perfect score: ranking is promising, but automatic binary decisions
are not yet well supported for the difficult positive subtype.

## Why the resolution queue matters

The project goal is to decide whether an E3 outfit/image overlaps Polyvore, not
to classify every candidate pair independently. Active learning should still
sample uncertain rows to improve the model. Production review should instead
inspect the most likely duplicate first for each unresolved group. Once a
relevant duplicate is confirmed, redundant candidates in that group need not
be reviewed for the strict-clean decision.

## Remaining policy decision

The team must explicitly decide whether a same-design item in a different color
counts as contamination. It should not be silently mixed into `DUPLICATE` and
`NON_DUPLICATE` labels. If color/viewpoint variants remain the dominant error
after two or three current-domain rounds, the next experiment should add a
learned instance-similarity feature such as DINO, with FashionCLIP only as an
auxiliary semantic feature. Further blind SSIM-threshold tuning is not justified.
