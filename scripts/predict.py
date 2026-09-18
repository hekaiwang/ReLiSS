"""Native-geometry prediction; missing channels are zeroed after preprocessing."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True,help='Trainer directory containing plans.json and dataset.json')
    p.add_argument('--fold',default='0')
    p.add_argument('--checkpoint',default='checkpoint_final.pth')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path,help='JSON list of case IDs; use only validation IDs for OOF evaluation')
    p.add_argument('--keep',type=int,nargs='+',help='Present channel indices; default all channels')
    p.add_argument('--no-mirroring',action='store_true')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from reliss.inference import MissingModalityWrapper, build_inference_network
    plans=PlansManager(json.loads((a.model_dir/'plans.json').read_text()))
    dataset=json.loads((a.model_dir/'dataset.json').read_text())
    config=plans.get_configuration('3d_fullres')
    channels=len(dataset['channel_names'])
    keep=sorted(set(a.keep if a.keep is not None else range(channels)))
    if not keep or any(i<0 or i>=channels for i in keep):p.error('--keep must contain valid channel indices')
    ckpt=torch.load(a.model_dir/f'fold_{a.fold}'/a.checkpoint,map_location='cpu',weights_only=False)
    network=build_inference_network(plans,dataset,config,ckpt['network_weights'])
    network=MissingModalityWrapper(network,channels,keep)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=not a.no_mirroring,perform_everything_on_gpu=True,device=torch.device('cuda',0))
    predictor.manual_initialization(network,plans,config,None,dataset,"ReLiSS",ckpt.get('inference_allowed_mirroring_axes',(0,1,2)))
    cases=json.loads(a.cases.read_text()) if a.cases else sorted(x.name.removesuffix('_0000.nii.gz') for x in a.input.glob('*_0000.nii.gz'))
    if not cases or len(cases)!=len(set(cases)):raise ValueError('Expected a nonempty unique case list')
    inputs=[[str(a.input/f'{case}_{i:04d}.nii.gz') for i in range(channels)] for case in cases]
    for files in inputs:
        for path in files:
            if not Path(path).is_file():raise FileNotFoundError(path)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use a fresh prediction output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    outputs=[str(a.output/case) for case in cases]
    predictor.predict_from_files(inputs,outputs,save_probabilities=False,num_processes_preprocessing=a.workers,num_processes_segmentation_export=a.workers)
    (a.output/'prediction_protocol.json').write_text(json.dumps({'model':'reliss','checkpoint':str(a.model_dir/f'fold_{a.fold}'/a.checkpoint),'cases':cases,'keep_channels':keep,'channel_names':dataset['channel_names'],'missingness':'exact zero after full-input preprocessing; shared crop','step':.5,'gaussian':True,'mirroring':not a.no_mirroring},indent=2))

if __name__=='__main__':main()
