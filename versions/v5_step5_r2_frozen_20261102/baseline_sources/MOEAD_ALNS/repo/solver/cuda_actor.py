"""Shared frozen-address CUDA hosts; input values and source weights stay dynamic."""
from contextlib import nullcontext
import copy,time
import torch
from .model import Policy,pack

_CUDA_GRAPH_RUNNERS={}


def runner_for(policy,n,batch,greedy):
    key=(str(policy.device),n,batch,bool(greedy))
    if key not in _CUDA_GRAPH_RUNNERS:
        _CUDA_GRAPH_RUNNERS[key]=CapturedActor(policy,n,batch,greedy)
    return _CUDA_GRAPH_RUNNERS[key]


class CapturedActor:
    def __init__(self,policy,n,batch,greedy):
        self.n=n;self.batch=batch;self.greedy=greedy;self.signature=None
        cpu_rng=torch.get_rng_state()
        self.host=Policy(copy.deepcopy(policy.cfg),tasks=batch).to(policy.device).eval()
        torch.set_rng_state(cpu_rng)
        # One fixed contiguous GPU parameter buffer; graph-visible parameter
        # views keep their addresses while task/model weights are refreshed.
        self.parameter_buffer=torch.cat([p.detach().reshape(-1) for p in self.host.parameters()])
        torch.nn.utils.vector_to_parameters(self.parameter_buffer,list(self.host.parameters()))
        self.graph=None;self.buffers=None;self.capture_count=0
        self.setup_seconds=0.;self.weight_copy_seconds=0.

    def _bind(self,policy,ids):
        signature=(id(policy),tuple(ids),tuple(p._version for p in policy.parameters()))
        if signature==self.signature:return
        started=time.perf_counter()
        source=dict(policy.named_parameters())
        values=[]
        for name,param in self.host.named_parameters():
            if name.startswith('heads.'):
                _,slot,suffix=name.split('.',2)
                name=f'heads.{ids[int(slot)]}.{suffix}'
            values.append(source[name].detach().reshape(-1))
        with torch.no_grad():self.parameter_buffer.copy_(torch.cat(values))
        self.signature=signature
        self.source_policy=policy  # Keep current owner alive; guard Python id reuse.
        self.weight_copy_seconds+=time.perf_counter()-started

    def run(self,policy,observations,ids):
        self._bind(policy,ids)
        fields,nodes=pack(observations)
        if nodes!=self.n+1 or len(observations)!=self.batch:raise ValueError('Wrong graph shape key')
        if self.buffers is None:
            began=time.perf_counter();self.buffers={k:v.to(policy.device) for k,v in fields.items()}
            # Warm allocations before capturing Beta/Gamma RNG and log probability.
            stream=torch.cuda.Stream(device=policy.device);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(4):self.host._tensor_sample(self.buffers,nodes,list(range(self.batch)),self.greedy)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize(policy.device)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph),torch.no_grad():
                self.outputs=self.host._tensor_sample(self.buffers,nodes,list(range(self.batch)),self.greedy)
            self.capture_count+=1;torch.cuda.synchronize(policy.device)
            self.setup_seconds+=time.perf_counter()-began
        else:
            if any(v.shape!=self.buffers[k].shape for k,v in fields.items()):raise ValueError('Edge shape invariant violated; no silent recapture')
            for k,v in fields.items():self.buffers[k].copy_(v)
        self.graph.replay()
        return self.outputs
