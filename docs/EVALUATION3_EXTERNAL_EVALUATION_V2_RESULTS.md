# EVALUATION3 Test2000 — Kendall / Spearman / Diagnostics

Bootstrap: 10,000 outfit-level resamples, seed=42, 95% percentile CI.

Protocol artifact SHA256: `74ab73bd935b628a1923372d2bd27cd82aebcce4c97f52b594e756191142470f`

Evaluation manifest SHA256: `1a45f6dc77f11eb57cb9d4e0ff2f10a26686f0f6d131bbc134dda40de9e7a76d`

## Main metrics

| subset                                               |   N_total |   N_Bad |   N_Normal |   N_Good |   kendall_tau_b |   kendall_ci_low |   kendall_ci_high |   spearman_rho |   spearman_ci_low |   spearman_ci_high |
|:-----------------------------------------------------|----------:|--------:|-----------:|---------:|----------------:|-----------------:|------------------:|---------------:|------------------:|-------------------:|
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate       |       422 |      61 |        255 |      106 |        0.023315 |        -0.051823 |          0.098222 |       0.029103 |         -0.066610 |           0.123796 |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate |       609 |      91 |        365 |      153 |        0.049037 |        -0.014089 |          0.112124 |       0.062987 |         -0.017340 |           0.142963 |
| EVAL3-Test2000-Full                                  |      2000 |     296 |       1230 |      474 |        0.014264 |        -0.020781 |          0.049445 |       0.018752 |         -0.025930 |           0.063730 |

## Classwise compatibility_logit

| subset                                               | human_class   |    N |   median_logit |        Q1 |       Q3 |      IQR |   mean_logit |   std_logit |
|:-----------------------------------------------------|:--------------|-----:|---------------:|----------:|---------:|---------:|-------------:|------------:|
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate       | Bad           |   61 |      -0.584926 | -1.616529 | 0.335999 | 1.952528 |    -0.598904 |    1.484823 |
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate       | Normal        |  255 |      -0.517462 | -1.430061 | 0.112373 | 1.542434 |    -0.674034 |    1.119285 |
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate       | Good          |  106 |      -0.510009 | -1.026250 | 0.187013 | 1.213263 |    -0.481414 |    1.010522 |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate | Bad           |   91 |      -0.830940 | -1.712150 | 0.002569 | 1.714719 |    -0.830657 |    1.362415 |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate | Normal        |  365 |      -0.517462 | -1.379663 | 0.077064 | 1.456728 |    -0.659293 |    1.101994 |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate | Good          |  153 |      -0.547468 | -1.196305 | 0.069485 | 1.265789 |    -0.563189 |    1.036421 |
| EVAL3-Test2000-Full                                  | Bad           |  296 |      -0.813865 | -1.756569 | 0.008051 | 1.764620 |    -0.792609 |    1.292198 |
| EVAL3-Test2000-Full                                  | Normal        | 1230 |      -0.591430 | -1.502122 | 0.109686 | 1.611808 |    -0.705951 |    1.176547 |
| EVAL3-Test2000-Full                                  | Good          |  474 |      -0.619970 | -1.522441 | 0.028823 | 1.551264 |    -0.715799 |    1.113038 |

## Pairwise ordering diagnostics

| subset                                               |   P_Good_gt_Bad |   P_Good_gt_Normal |   P_Normal_gt_Bad |
|:-----------------------------------------------------|----------------:|-------------------:|------------------:|
| EVAL3-Test2000-Full                                  |        0.526970 |           0.492017 |          0.531271 |
| EVAL3-Test2000-No-Full-Image-Overlap-Candidate       |        0.511135 |           0.540067 |          0.475217 |
| EVAL3-Test2000-No-TrainValid-Image-Overlap-Candidate |        0.564893 |           0.518131 |          0.544242 |

Exact compatibility_logit ties contribute 0.5.
