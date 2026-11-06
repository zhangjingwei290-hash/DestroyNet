"""New trainable search-state adapter and joint-action PPO utilities.
Call only from a separately authorized training runner. No implicit execution.
Frozen graph representations may be cached; adapter inputs must refresh per action.
"""
import torch
from torch import nn
from session_contract import STATE_FIELDS

class SearchStateAdapter(nn.Module):
    def __init__(self, hidden_dim, set_dim):
        super().__init__()
        self.set_dim=set_dim
        self.fusion=nn.Sequential(nn.Linear(hidden_dim+len(STATE_FIELDS)+set_dim,hidden_dim),nn.Tanh())
        self.mode_delta=nn.Linear(hidden_dim,3)
        self.beta_delta=nn.Linear(hidden_dim,6)
        self.value=nn.Linear(hidden_dim,1)
        # Start actor as the transferred policy; value and fusion can learn immediately.
        for layer in (self.mode_delta,self.beta_delta):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self, hidden, search_state, set_summary, mode_logits, alpha, beta):
        if search_state.shape[-1]!=len(STATE_FIELDS) or set_summary.shape[-1]!=self.set_dim:
            raise ValueError('State schema mismatch')
        z=self.fusion(torch.cat((hidden,search_state,set_summary),dim=-1))
        da,db=self.beta_delta(z).chunk(2,-1)
        return dict(mode_logits=mode_logits+self.mode_delta(z),
                    alpha=alpha*torch.exp(da.clamp(-5,5)),beta=beta*torch.exp(db.clamp(-5,5)),
                    value=self.value(z).squeeze(-1))

def joint_logprob(terms):
    # No root/length-normalized probability ratios: this is the sampled augmented action.
    return terms['logprob_macro']+terms['logprob_micro']

def returns_and_advantages(rewards, values, terminated, truncated, bootstrap_value, gamma=1.0, lam=.95):
    """One contiguous sequence; true K terminal zeros bootstrap; time cut uses supplied value."""
    if not (len(rewards)==len(values)==len(terminated)==len(truncated)) or not len(rewards):
        raise ValueError('Bad trajectory shape')
    if any(t and c for t,c in zip(terminated,truncated)):raise ValueError('Conflicting ending flags')
    if any(terminated[:-1]) or any(truncated[:-1]):raise ValueError('Do not concatenate sessions')
    adv=torch.zeros_like(values);carry=torch.zeros_like(bootstrap_value)
    for t in range(len(rewards)-1,-1,-1):
        next_value=bootstrap_value if t==len(rewards)-1 else values[t+1]
        nonterminal=0. if terminated[t] else 1.
        delta=rewards[t]+gamma*nonterminal*next_value-values[t]
        carry=delta+gamma*lam*nonterminal*carry
        adv[t]=carry
    return (adv+values).detach(),adv.detach()

def ppo_loss(new_logprob, old_logprob, advantages, values, targets, entropy, clip=.2, value_weight=.5, entropy_weight=.01):
    tensors=(new_logprob,old_logprob,advantages,values,targets,entropy)
    if any(t.shape!=values.shape for t in tensors):raise ValueError('PPO shape mismatch')
    if any(not torch.isfinite(t).all() for t in tensors):raise ValueError('Nonfinite PPO batch')
    ratio=(new_logprob-old_logprob.detach()).exp()
    surrogate=torch.minimum(ratio*advantages.detach(),ratio.clamp(1-clip,1+clip)*advantages.detach())
    return -surrogate.mean()+value_weight*(values-targets.detach()).square().mean()-entropy_weight*entropy.mean()

@torch.no_grad()
def reptile_update(meta, adapted, step_size):
    if not adapted or not 0<step_size<=1:raise ValueError('Invalid meta update')
    states=[m.state_dict() for m in adapted]
    for key,tensor in meta.state_dict().items():
        if any(s[key].shape!=tensor.shape for s in states):raise ValueError('Task parameter mismatch')
        if tensor.is_floating_point():tensor.lerp_(torch.stack([s[key] for s in states]).mean(0),step_size)
        elif any(not torch.equal(tensor,s[key]) for s in states):raise ValueError('Nonfloating buffers differ')
