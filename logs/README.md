# Completed ReLiSS runs

Only the released BraTS2021 five-fold, ISLES2022 five-fold and WMH2017 official
train/test runs are included here. Each `fold_*.log` covers epochs 0–299 and ends
with the recorded completion marker. Startup configuration dumps and machine
paths are omitted; model identifiers are normalized to ReLiSS. Numerical log
records are preserved.

ISLES folds 1–4 resumed at epoch 200. Their logs join the retained epochs 0–199
with the completed resumed epochs 200–299. Superseded records after epoch 199
from the interrupted attempts and startup-only logs are excluded.

| Dataset | Case-condition rows | Subjects | Nonempty modality subsets |
|---|---:|---:|---:|
| BraTS2021 | 18,765 | 1,251 | 15 |
| ISLES2022 | 1,750 | 250 | 7 |
| WMH2017 | 330 | 110 | 3 |

`per_case_metrics.csv` contains segmentation metrics and case/condition identity.
`condition_summary.csv` contains recorded fold and pooled summaries. Metric
values are retained from completed evaluations; machine paths and auxiliary
uncertainty columns are omitted. Training-time validation metrics in the logs
are distinct from the native-grid held-out evaluations in these CSV tables.

The BraTS result-table mask order is **FLAIR, T1ce, T1, T2**. The physical input
channel order used by `predict.py --keep` is **T1, T1ce, T2, FLAIR**. Use
`Available_Modalities` to interpret table masks. ISLES uses FLAIR/ADC/DWI and
WMH uses FLAIR/T1 for both. Cross-validation rows match each subject's released
validation fold; WMH rows match the official test list.

`manifest.json` records file checksums and completeness counts. Training source
and experiment launch scripts are not part of this release.
