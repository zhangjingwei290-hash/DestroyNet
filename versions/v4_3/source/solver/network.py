import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,softmax

class GATConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(GATConv, self).__init__(aggr='add')
        self.negative_slope = negative_slope
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, index, size_i):
        # x_j -> x_i
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, num_nodes=size_i)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out

class Encoder(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.node_dim=width;self.embedding_dim=20
        self.W_in=nn.Linear(20,width,bias=False)
        self.W_z=nn.Linear(2*width,width);self.W_r=nn.Linear(2*width,width);self.W_h=nn.Linear(2*width,width)
        self.gat_0=GATConv(20,width);self.gat_n=GATConv(width,width)
        self.gat_1=GATConv(width,width);self.gat_2=GATConv(width,width)
    def encode(self,data,n_nodes):
        state=data.state_.reshape(-1,self.node_dim)
        inp=self.W_in(data.input_.reshape(state.shape[0],-1,self.embedding_dim)).mean(1)
        z=torch.sigmoid(self.W_z(torch.cat([state,inp],-1)))
        r=torch.sigmoid(self.W_r(torch.cat([state,inp],-1)))
        h=torch.tanh((1-z)*state+z*torch.tanh(self.W_h(torch.cat([r*state,inp],-1))))
        a=self.gat_0(data.x,data.edge_index_n);b=self.gat_1(a,data.edge_index_r1);c=self.gat_1(b,data.edge_index_r1)
        d=self.gat_n(c,data.edge_index_n)+a;e=self.gat_2(d,data.edge_index_r0);f=self.gat_2(e,data.edge_index_r0)
        g=(self.gat_n(f,data.edge_index_n)+d).reshape(-1,n_nodes,self.node_dim)
        return h,g*(1+h[:,None])

