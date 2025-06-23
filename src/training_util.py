import torch
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.data import Data, HeteroData
from typing import Union


class AddEgoIds(BaseTransform):
    r"""Add EgoIDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    # Apply transformation to each batch
    def __call__(self, data: Union[Data, HeteroData]):
        # Node attributes
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        # IDs tensor
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            # Indices of nodes corresponding to seed edges
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        # Assign Ego IDs to nodes of seed edges
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data