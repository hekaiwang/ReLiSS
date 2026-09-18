# ReLiSS

ReLiSS for missing-modality MRI segmentation on BraTS2021, ISLES2022 and WMH2017.
This release provides pretrained weights, completed-run logs, per-case metrics,
inference, evaluation and fixed dataset splits. Training code and the full
experiment pipeline are not included in this release.

| Dataset | Released weights | Evaluation split |
|---|---|---|
| BraTS2021 | Five best checkpoints | Five folds, 1,251 subjects |
| ISLES2022 | Five final checkpoints | Five folds, 250 subjects |
| WMH2017 | One final checkpoint, fold all | Official 60 train / 110 test subjects |

## Installation

Use Linux, Python 3.10 and a CUDA-capable GPU. CUDA extensions require a matching
CUDA compiler and PyTorch build.

```bash
conda create -n reliss python=3.10 -y
conda activate reliss
python -m pip install 'numpy<2' torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
python -m pip install packaging ninja
python -m pip install causal-conv1d==1.1.1 mamba-ssm==1.2.2 --no-build-isolation
python -m pip install -e '.[hub]'
export RELISS_DATA_ROOT=/absolute/path/to/reliss_data
source scripts/env.sh
```

Run commands from the repository root. Source `scripts/env.sh` in each shell.
Download the eleven weights using [docs/WEIGHTS.md](docs/WEIGHTS.md).

## Inference

Supply aligned NIfTI volumes named `CASE_0000.nii.gz`, `CASE_0001.nii.gz`, etc.
Channel order, label mapping and fixed splits are described in
[docs/INPUTS.md](docs/INPUTS.md). Inference performs the normalization and
resampling specified in the supplied plans; no training/preprocessing runner is needed.

Example for BraTS fold 0, using that fold's held-out subjects:

```bash
export MODEL_DIR="$nnUNet_results/Dataset137_BraTS2021/nnUNetTrainerReLiSS_300epochs__nnUNetPlans__3d_fullres"
python - <<'PYCODE'
import json
from pathlib import Path
splits = json.loads(Path('configs/Dataset137_BraTS2021/splits_final.json').read_text())
Path('/tmp/reliss_cases.json').write_text(json.dumps(splits[0]['val']))
PYCODE
python scripts/predict.py --model-dir "$MODEL_DIR" --fold 0 \
  --checkpoint checkpoint_best.pth --input /path/to/aligned_images \
  --cases /tmp/reliss_cases.json --output "$RELISS_DATA_ROOT/predictions/brats_fold0"
```

For ISLES use `checkpoint_final.pth` and its held-out fold. For WMH use
`--fold all`, `checkpoint_final.pth` and the `test` list from
`configs/Dataset752_WMH2017/official_split.json`.

`--keep` selects the available input channels, for example `--keep 0 3` for
BraTS T1 and FLAIR. The released missing-modality protocol preprocesses the
complete input using a shared crop and then zeros unavailable channels in
normalized space. All channel files are required for this protocol. Omit
`--keep` for all channels. Use a fresh output directory for each prediction.

Inference uses sliding-window step 0.5, Gaussian blending and mirroring unless
`--no-mirroring` is specified. Network parameters are loaded strictly.

## Evaluation and logs

```bash
python scripts/evaluate.py --dataset brats \
  --predictions "$RELISS_DATA_ROOT/predictions/brats_fold0" \
  --references /path/to/converted_reference_masks \
  --output "$RELISS_DATA_ROOT/evaluation/brats_fold0.csv"
```

Use `--dataset isles` or `--dataset wmh` for the corresponding lesion masks.
Reference and prediction geometry must agree. The script reports Dice, IoU,
HD95 in mm and ASSD in mm. Both-empty regions have NaN metrics; one-empty
regions have Dice 0 and infinite surface distance. Match these conventions
when aggregating results.

[logs/README.md](logs/README.md) describes the completed-run records and
case-wise result tables. Only the three datasets and checkpoint protocols above
are included. Use each subject's held-out fold for cross-validation evaluation.

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Dataset and dependency terms remain separate.
