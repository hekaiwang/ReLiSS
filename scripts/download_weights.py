"""Fetch only the eleven approved weights from an explicitly supplied HF repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,help='Published Hugging Face model repo: owner/name')
    p.add_argument('--revision',required=True,help='Published commit hash or release tag')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from huggingface_hub import hf_hub_download
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'docs/weights_manifest.json').read_text())
    for row in manifest['files']:
        cached=Path(hf_hub_download(repo_id=a.repo,filename=row['file'],revision=a.revision))
        h=hashlib.sha256()
        with cached.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if cached.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError(f'Weight identity mismatch: {row["file"]}')
        dst=a.output/row['file']
        if dst.exists():raise FileExistsError(dst)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(cached,dst)
        trainer=dst.parent.parent
        for source,target in [('nnUNetPlans.json','plans.json'),('dataset.json','dataset.json')]:
            shutil.copy2(root/'configs'/row['dataset']/source,trainer/target)
        print(dst)

if __name__=='__main__':main()
