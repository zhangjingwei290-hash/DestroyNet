"""User-specified child .7parent+.3meta, with original encoder copied bit-exact."""
def initialize_direction(actor,parent,meta):
    import torch
    if parent is None:actor.start_inner(meta);return
    with torch.no_grad():
        source=parent.state_dict();base=meta.state_dict()
        state={name:(base[name].detach().clone() if name.startswith('encoder.') else .7*value+.3*base[name]) for name,value in source.items()}
        actor.load_state_dict(state,strict=True)
