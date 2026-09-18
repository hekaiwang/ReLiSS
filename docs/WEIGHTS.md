# Pretrained weights

Weights: [wanghekai/ReLiSS](https://huggingface.co/wanghekai/ReLiSS).
Pinned revision: `1a2ec73921668b3a28eccf87bbd5b6513815b826`.

The release contains five BraTS best checkpoints, five ISLES final checkpoints,
and one WMH final checkpoint trained with fold all (official train/test protocol).
File sizes and SHA-256 are listed in [weights_manifest.json](weights_manifest.json).

```bash
source scripts/env.sh
python scripts/download_weights.py \
  --repo wanghekai/ReLiSS \
  --revision 1a2ec73921668b3a28eccf87bbd5b6513815b826 \
  --output "$nnUNet_results"
```

The downloader validates every checkpoint before copying it and installs the
matching dataset metadata and plans beside the fold directories. Use the
inference entry point in the README to load the released checkpoint files.
The code repository and source archive contain no weights.
