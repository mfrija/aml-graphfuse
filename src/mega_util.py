import torch
import torch.nn as nn
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_scatter import scatter
from torch_geometric.utils import degree

def find_parallel_edges(edge_index):
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

    return simplified_edge_batch


class PnaAgg(nn.Module):
    def __init__(self , n_hidden, deg):
        super().__init__()
        
        aggregators = ['mean', 'min', 'max', 'std']
        self.num_aggregators = len(aggregators)
        scalers = ['identity', 'amplification', 'attenuation']

        self.agg = DegreeScalerAggregation(aggregators, scalers, deg)
        self.lin = nn.Linear(len(scalers)*len(aggregators)*n_hidden, n_hidden)

    def forward(self, x, index):
        out = self.agg(x, index)
        return self.lin(out)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = nn.init.kaiming_normal_(param.detach())
            elif 'bias' in name:
                param = nn.init.constant_(param.detach(), 0)


class GinAgg(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.nn = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(), 
                nn.Linear(n_hidden, n_hidden)
                )
    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.nn(out)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class IdentityAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return x


class MultiEdgeAggModule(nn.Module):
    def __init__(self, n_hidden=None, agg_type=None, index=None):
        super().__init__()
        self.agg_type = agg_type

        if agg_type == 'gin':
            self.agg = GinAgg(n_hidden=n_hidden)
        elif agg_type == 'pna':
            uniq_index, inverse_indices = torch.unique(index, return_inverse=True)
            d = degree(inverse_indices, num_nodes=uniq_index.numel(), dtype=torch.long)
            deg = torch.bincount(d, minlength=1)
            self.agg = PnaAgg(n_hidden=n_hidden, deg=deg)
        else:
            self.agg = IdentityAgg()
        
    def forward(self, edge_index, edge_attr, simp_edge_batch):
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        # inverse_indices: assigns the unique ID of the simple edge that parallel edges correspond to # [E]
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index
        # new_edge_index: tensor of shape [2,N] where N is the number of unique simple edges, where each column represents the (src, dst) node indices
        new_edge_attr = self.agg(x=edge_attr, index=inverse_indices)
        # new_edge_attr: embeddings of artificial nodes
        return new_edge_index, new_edge_attr, inverse_indices
    
    def reset_parameters(self):
        self.agg.reset_parameters()