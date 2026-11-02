"""r3 original encoder, SHA-bound r2 meta projections and heads, no test selection."""
from stage2_common import ROOT,read,sha
from pathlib import Path

def initialize(policy):
    import torch
    plan=read(ROOT/'STAGE2_PLAN.json');parent=read(ROOT/'PARENT_R2.json');source_path=ROOT/'pretrained/meta_complete.pt'
    assert sha(source_path)==plan['original_meta_sha256']
    original=torch.load(source_path,map_location=policy.device,weights_only=False)['model']
    warm_path=Path(parent['archive'])/'stage2_training/meta_retrained.pt'
    assert sha(warm_path)==parent['r2_meta_sha256']
    warm=torch.load(warm_path,map_location=policy.device,weights_only=False)['model']
    target=policy.state_dict();copied=[]
    for name,value in target.items():
        key='heads.0.'+name.split('.',2)[2] if name.startswith('heads.') else name
        source=original if name.startswith('encoder.') else warm
        if key not in source or source[key].shape!=value.shape:raise RuntimeError('Warm source shape mismatch '+name)
        target[name]=source[key].detach().clone();copied.append(name)
    policy.load_state_dict(target,strict=True)
    return policy,dict(source_sha256=sha(source_path),warm_meta_sha256=sha(warm_path),warm_meta_path=str(warm_path),copied_keys=copied,encoder_from_original=True,projections_and_heads_from_r2=True,clean_fallback_used=False)
