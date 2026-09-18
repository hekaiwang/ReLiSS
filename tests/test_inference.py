import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from reliss.inference import MissingModalityWrapper, normalize_network_state
from reliss.network import ReLiSSUNet

ROOT=Path(__file__).resolve().parents[1]


def test_missing_channels_are_zero():
    model=MissingModalityWrapper(torch.nn.Identity(),4,[0,3])
    x=torch.randn(1,4,3,3,3);y=model(x)
    assert torch.equal(x[:,[0,3]],y[:,[0,3]])
    assert torch.count_nonzero(y[:,[1,2]])==0


def test_wrapped_weights_have_unambiguous_names():
    x=torch.ones(1)
    assert normalize_network_state({'module._orig_mod.a':x})['a'] is x
    with pytest.raises(ValueError):normalize_network_state({'a':x,'module.a':x})


def test_inference_checkpoint_roundtrip():
    config=dict(in_channels=4,num_classes=3,depths=[0]*4,feat_size=[8,16,24,32],patch_size=(16,16,16))
    network=ReLiSSUNet(**config).eval();restored=ReLiSSUNet(**config).eval()
    state=network.state_dict();restored.load_state_dict(state,strict=True)
    x=torch.randn(1,4,16,16,16);x[:,1]=0
    with torch.inference_mode():torch.testing.assert_close(network(x),restored(x),rtol=0,atol=0)
    assert (restored.last_fusion_weights[:,1]==0).all()
    bad=dict(state);bad.pop(next(iter(bad)))
    with pytest.raises(RuntimeError):restored.load_state_dict(bad,strict=True)


def test_evaluation_empty_and_spacing_rules():
    spec=importlib.util.spec_from_file_location('evaluate',ROOT/'scripts/evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    empty=np.zeros((5,5,5),dtype=bool);a=empty.copy();a[1,1,1]=True;b=empty.copy();b[2,1,1]=True
    assert np.isnan(module.metrics(empty,empty,(1,1,1))['Dice'])
    assert module.metrics(a,empty,(1,1,1))['Dice']==0
    assert np.isinf(module.metrics(a,empty,(1,1,1))['HD95_mm'])
    assert module.metrics(a,b,(2,1,1))['HD95_mm']==2


def test_fixed_folds_and_official_test_are_disjoint():
    for folder in (ROOT/'configs').iterdir():
        splits=json.loads((folder/'splits_final.json').read_text());assert len(splits)==5
        cases=set(splits[0]['train'])|set(splits[0]['val']);seen=set()
        for fold in splits:
            tr,va=set(fold['train']),set(fold['val'])
            assert not tr&va and tr|va==cases and not seen&va
            seen|=va
        assert seen==cases
        if (folder/'official_split.json').exists():
            official=json.loads((folder/'official_split.json').read_text())
            assert set(official['train'])==cases and not cases&set(official['test'])
            assert len(official['test'])==110
