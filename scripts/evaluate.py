"""Compute native-grid Dice/IoU/HD95/ASSD from predicted and reference NIfTI."""
import argparse
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def metrics(pred, ref, spacing):
    n_pred, n_ref = int(pred.sum()), int(ref.sum())
    if n_pred+n_ref == 0:
        return dict(Dice=float('nan'),IoU=float('nan'),HD95_mm=float('nan'),ASSD_mm=float('nan'))
    intersection = int((pred & ref).sum())
    result = dict(Dice=2*intersection/(n_pred+n_ref), IoU=intersection/(n_pred+n_ref-intersection))
    if not n_pred or not n_ref:
        result.update(HD95_mm=float('inf'),ASSD_mm=float('inf'))
    else:
        conn=generate_binary_structure(3,1)
        p_edge=pred ^ binary_erosion(pred,structure=conn,border_value=0)
        r_edge=ref ^ binary_erosion(ref,structure=conn,border_value=0)
        pr=distance_transform_edt(~r_edge,sampling=spacing)[p_edge]
        rp=distance_transform_edt(~p_edge,sampling=spacing)[r_edge]
        result.update(HD95_mm=float(np.percentile(np.concatenate((pr,rp)),95)),ASSD_mm=float((pr.mean()+rp.mean())/2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--dataset',choices=['brats','isles','wmh'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for path in sorted(a.predictions.glob('*.nii.gz')):
        pred=nib.load(path); ref=nib.load(a.references/path.name)
        if pred.shape != ref.shape or not np.allclose(pred.affine,ref.affine,atol=1e-4):raise ValueError(f'Geometry mismatch: {path.name}')
        pv=np.asanyarray(pred.dataobj);rv=np.asanyarray(ref.dataobj)
        regions={'WT':[1,2,3],'TC':[2,3],'ET':[3]} if a.dataset=='brats' else {'lesion':[1]}
        for region,labels in regions.items():
            rows.append({'Case':path.name.removesuffix('.nii.gz'),'Region':region,**metrics(np.isin(pv,labels),np.isin(rv,labels),ref.header.get_zooms()[:3])})
    if not rows:raise ValueError('No predictions found')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'case_count':len({r['Case'] for r in rows}),'mean_Dice':float(np.nanmean([r['Dice'] for r in rows])),'empty_empty':'NaN excluded','empty_one_surface_distance':'Infinity; not silently excluded'}))

if __name__=='__main__':main()
