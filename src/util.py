import numpy as np
import torch
import random
import logging
import math
import os
import json


def extract_param(parameter_name: str, config) -> float:
    """
    Extract the value of the specified parameter for the given model.
    
    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - config: Arguments given to this specific run.
    
    Returns:
    - float: Value of the specified parameter.
    """

    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(config.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:    
        data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)


def set_seed(seed: int = 0) -> None:
    """
    Ensuring deterministic results and reproducability.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def save_model(model, optimizer, epoch, config):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 
                os.path.join(config.checkpoint_dir, f'epoch_{epoch+1}.tar')
            )
    
def compute_lsh_buckets(edge_attr, config):
    # According to the Reformerpaper
    num_bits = 16
    edge_attr = torch.nn.functional.normalize(edge_attr, p=2, dim=-1)
    d = edge_attr.size(-1)
    R = torch.randn(d, num_bits, device=edge_attr.device)
    hash_bits = (edge_attr @ R) > 0  # [M, num_bits] boolean tensor
    hash_bits = hash_bits.int() # convert to int tensor
    powers = 2 ** torch.arange(num_bits, device=edge_attr.device)
    bucket_ids = (hash_bits * powers).sum(dim=-1)
    
    num_buckets = max(1, math.ceil(edge_attr.size(0)/config.cluster_size))
    bucket_ids = bucket_ids % num_buckets
    return bucket_ids  # [M]

def compute_random_buckets(num_edges, device, config):
    num_buckets = max(1, math.ceil(num_edges / config.cluster_size))
    return torch.randint(low=0, high=num_buckets, size=(num_edges,), device=device)
