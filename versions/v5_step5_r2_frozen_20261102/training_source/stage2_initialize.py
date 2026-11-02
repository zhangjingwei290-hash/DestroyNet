"""Exact allowed warm transfer and neutral unlocked q initialization."""
from stage2_common import ROOT,read,sha
import math

def initialize(policy):
    import torch
    source_path=ROOT/'pretrained/meta_complete.pt'
    assert sha(source_path)==read(ROOT/'STAGE2_PLAN.json')['original_meta_sha256']
    payload=torch.load(source_path,map_location=policy.device,weights_only=False)
    source=payload['model'];target=policy.state_dict();copied=[];new=[]
    for key,value in target.items():
        allowed=key.startswith(('encoder.','context.','film.','q_history_norm.')) or (key.startswith('heads.') and ('.node.' in key or key.endswith('.physical_bilinear')))
        origin=key
        if key.startswith('heads.'):
            origin='heads.0.'+key.split('.',2)[2]
        if allowed:
            if origin not in source or source[origin].shape!=value.shape:raise RuntimeError('Warm-transfer key mismatch: '+key)
            target[key]=source[origin].detach().clone();copied.append(key)
        else:new.append(key)
    policy.load_state_dict(target,strict=True)
    with torch.no_grad():
        for head in policy.heads:
            head.q_mu.weight.zero_();head.q_mu.bias.copy_(torch.tensor([math.log(((mu-.02)/.96)/(1-(mu-.02)/.96)) for mu in (.35,.40,.30)],device=policy.device))
            head.mode_logits.weight.zero_();head.mode_logits.bias.copy_(torch.tensor([.2,.2,-.2],device=policy.device))
            head.q_kappa.weight.zero_();head.q_kappa.bias.fill_(math.log(((4.-2.)/18.)/(1-(4.-2.)/18.)))
    receipt=dict(copied_keys=copied,new_neutral_keys=new,source_sha256=sha(source_path),
        initial_mu=[.35,.40,.30],initial_kappa=4.,removed_parameter='q_state_mu',phys_idx=[18,17,19,12,13])
    return policy,receipt
