# Inputs and fixed splits

Supply channel-aligned NIfTI images with a shared shape and affine, named
`CASE_0000.nii.gz`, `CASE_0001.nii.gz`, and so on. Use identifiers from the
released split files when comparing with the included results.

| Dataset | Physical input channel order | Reference labels |
|---|---|---|
| BraTS2021 | T1, T1ce, T2, FLAIR | Convert original 0→0, 2→1, 1→2, 4→3 |
| ISLES2022 | FLAIR, ADC, DWI | Nonzero → 1 |
| WMH2017 | FLAIR, T1 | 0 = background; 1 = WMH; 2 = other pathology, excluded from evaluation |

For BraTS, converted regions are WT=[1,2,3], TC=[2,3], ET=[3]. Do not apply
the label conversion twice. For ISLES, align/resample FLAIR linearly into the
ADC/DWI reference grid before supplying the images. ISLES identifiers follow
sorted source subject order (`ISLES2022_0000`, etc.). WMH uses the official
pre-registered FLAIR/T1 images. Reference labels must be aligned with the images.

`configs/*/splits_final.json` contains the fixed five training/validation splits.
BraTS validation sizes are 251/250/250/250/250; ISLES has five sets of 50.
WMH weights use all 60 training cases and the separate 110-case official test list
in `configs/Dataset752_WMH2017/official_split.json`. Its training split file does
not imply five released WMH models.

Plans specify the inference crop/normalization/resampling parameters. Dataset
images and masks are obtained separately under the corresponding dataset terms.
