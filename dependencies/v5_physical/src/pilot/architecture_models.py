"""Three architectures sharing the pilot insertion action space."""
import torch
from torch import nn
from src.pilot.model import RepairNet


class LightweightRepair(nn.Module):
    def __init__(self, once=False):
        super().__init__()
        self.conditioned = True
        self.once = once
        self.chunk_size = 128
        self.nodes = nn.Sequential(nn.Linear(16, 128), nn.ReLU())
        self.preference = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 128))
        if once:
            self.adjacency = nn.Linear(256, 128)
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.dynamic = nn.Sequential(nn.Linear(16, 128), nn.ReLU())
        self.route = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 * 6 + 4, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self._context = None

    def reset_context(self):
        self._context = None

    def forward(self, features, weights):
        x, neighbors, membership, indices, extras, mask = features
        p = self.preference(weights)
        if self.once:
            if self._context is None:
                h = self.nodes(x)
                h = h + self.adjacency(torch.cat((h[neighbors[:, 0]], h[neighbors[:, 1]]), -1)) + p
                self._context = self.encoder(h[None]).squeeze(0)
            h = self._context
        else:
            h = self.nodes(x) + p
        dynamic = self.dynamic(x)
        current = h + dynamic
        routes = self.route(torch.cat((membership @ h, membership @ dynamic), -1))
        global_h = current.mean(0)
        chunks = []
        for start in range(0, len(indices), self.chunk_size):
            ids = indices[start:start + self.chunk_size]
            features_a = torch.cat((current[ids[:, 0]], current[ids[:, 1]], current[ids[:, 2]],
                                    routes[ids[:, 3]], global_h.expand(len(ids), -1),
                                    p.expand(len(ids), -1), extras[start:start + len(ids)]), -1)
            chunks.append(self.head(features_a).squeeze(-1))
        return torch.cat(chunks).masked_fill(~mask, -torch.inf)


def build_model(name):
    if name == 'transformer':
        return RepairNet(True)
    if name in ('mlp', 'encode_once'):
        return LightweightRepair(once=name == 'encode_once')
    raise ValueError(name)
