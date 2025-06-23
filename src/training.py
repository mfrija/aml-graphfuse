import re
import math
import torch
from typing import Any, Tuple

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import summary
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import logging
import numpy as np
import sklearn.metrics

import copy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
import wandb

from .util import add_arange_ids, save_model
from .models import GraphFuse
from .training_util import AddEgoIds


def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, config):

    tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=config.num_neighs, batch_size=config.batch_size, shuffle=True, transform=transform)
    val_loader = LinkNeighborLoader(val_data,num_neighbors=config.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                    edge_label=val_data.y[val_inds], batch_size=config.batch_size, shuffle=False, transform=transform)
    te_loader =  LinkNeighborLoader(te_data,num_neighbors=config.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                            edge_label=te_data.y[te_inds], batch_size=config.batch_size, shuffle=False, transform=transform)
        
    return tr_loader, val_loader, te_loader


def compute_binary_metrics(preds: np.array, labels: np.array):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """
    probs = preds[:,1]
    preds = preds.argmax(axis=-1)

    precisions, recalls, _ = sklearn.metrics.precision_recall_curve(labels, probs) # probs: probabilities for the positive class
    f1 = sklearn.metrics.f1_score(labels, preds, zero_division=0)
    auc = sklearn.metrics.auc(recalls, precisions)

    precision = sklearn.metrics.precision_score(labels, preds, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, preds, zero_division=0)

    return f1, auc, precision, recall


def train_epoch(
    loader: Any, 
    model: Module, 
    optimizer: Optimizer,
    scheduler: LambdaLR,
    accum_steps: int, 
    loss_fn: Module, 
    tr_inds: torch.Tensor, 
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trains the model for one epoch.
    
    Args:
        loader: Data loader for training
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        tr_inds: Training indices
        device: Device to train on
        
    Returns:
        Tuple containing:
            - Average loss for the epoch
            - Model predictions
            - Ground truth labels
    """
    # Set model in training mode
    model.train()
    total_loss = total_examples = 0
    preds = []
    ground_truths = []
    optimizer.zero_grad()
    for it, batch in enumerate(loader, start=1):        
        # Select the seed edges from which the batch was created
        inds = tr_inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        # Remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        batch.to(device)

        out = model(batch)

        pred = out[mask]
        ground_truth = batch.y[mask]
        loss = loss_fn(pred, ground_truth) / accum_steps

        loss.backward()
        
        if (it % accum_steps == 0) or (it == len(loader)):
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({"lr": current_lr})

        total_loss += float(loss) * pred.numel() * accum_steps
        total_examples += pred.numel()

        preds.append(pred.detach().cpu())
        ground_truths.append(ground_truth.detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()
    
    return total_loss / total_examples, pred, ground_truth

@torch.no_grad()
def eval_epoch(
    loader: Any, 
    inds: torch.Tensor, 
    model: Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Evaluates the model on the given loader.
    
    Args:
        loader: Data loader for evaluation
        inds: Evaluation indices
        model: Model to evaluate
        device: Device to evaluate on
        
    Returns:
        Tuple containing evaluation metrics:
            - F1 score
            - AUC score
            - Precision
            - Recall
    """
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    preds = []
    ground_truths = []
    for batch in loader:
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)

            out = model(batch)
            pred = out[mask]
            preds.append(pred.detach().cpu())
            ground_truths.append(batch.y[mask].detach().cpu())
            
    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)
    return f1, auc, precision, recall


def train(
    tr_loader: Any, 
    val_loader: Any, 
    te_loader: Any, 
    tr_inds: torch.Tensor, 
    val_inds: torch.Tensor, 
    te_inds: torch.Tensor, 
    model: Module, 
    optimizer: Optimizer,
    scheduler: LambdaLR, 
    loss_fn: Module, 
    config: Any
) -> Module:
    """Main training loop with validation and model selection.
    
    Args:
        tr_loader: Training data loader
        val_loader: Validation data loader
        te_loader: Test data loader
        tr_inds: Training indices
        val_inds: Validation indices
        te_inds: Test indices
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        config: Training configuration
        
    Returns:
        Module: Best model
    """
    best_val_f1 = 0
    best_test_f1 = 0
    best_state_dict = None
    
    for epoch in range(config.epochs):
        logging.info(f'****** EPOCH {epoch} ******')
        
        # Training phase
        total_loss, pred, ground_truth = train_epoch(tr_loader, model, optimizer, scheduler, config.accum_steps, loss_fn, tr_inds, config.device)
        
        # Compute training metrics
        f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)
        
        # Log training metrics
        logging.info({"Train": {"F1": f"{f1:.4f}", "Precision": f"{precision:.4f}", "Recall": f"{recall:.4f}", "PR-AUC": f"{auc:.4f}"}})

        # Evaluation phase
        val_f1, val_auc, val_precision, val_recall = eval_epoch(val_loader, val_inds, model, config.device)
        te_f1, te_auc, te_precision, te_recall = eval_epoch(te_loader, te_inds, model, config.device)

        # Log validation metrics
        logging.info({"Val": {"F1": f"{val_f1:.4f}", "Precision": f"{val_precision:.4f}", "Recall": f"{val_recall:.4f}", "PR-AUC": f"{val_auc:.4f}"}})

        # Log test metrics
        logging.info({"Test": {"F1": f"{te_f1:.4f}", "Precision": f"{te_precision:.4f}", "Recall": f"{te_recall:.4f}", "PR-AUC": f"{te_auc:.4f}"}})

        # Log loss
        logging.info(f"Loss: {total_loss}")

        # Model selection based on validation F1 score
        if epoch == 0:
            logging.info({"best_test_f1": f"{te_f1:.4f}"})
            best_state_dict = copy.deepcopy(model.state_dict())
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_f1 = te_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            logging.info({"best_test_f1": f"{best_test_f1:.4f}"})
            
            # Save best model
            if config.save_model:
                save_model(model, optimizer, epoch, config)

        wandb.log({
            "epoch": epoch,
            "train/f1": f1,
            "train/precision": precision,
            "train/recall": recall,
            "train/pr_auc": auc,
            "train/loss": total_loss,
            "val/f1": val_f1,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "val/pr_auc": val_auc,
            "test/f1": te_f1,
            "test/precision": te_precision,
            "test/recall": te_recall,
            "test/pr_auc": te_auc,
            "test/best_test_f1": best_test_f1
        })
        if model.use_ec:
            wandb.log({"train/ec_alpha_attn": model.ec_alpha_attn.item()})
        
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return model


def get_model(sample_batch, config):
    """Creates a model instance based on the provided configuration.
    
    Args:
        sample_batch: Sample batch for model initialization
        config: Model configuration
        
    Returns:
        Module: Initialized model
    """
    n_feats = sample_batch.x.shape[1] 
    e_dim = (sample_batch.edge_attr.shape[1])
    
    if config.use_mega:
        index_ = sample_batch.simp_edge_batch
    else:
        index_=None

    if config.use_mega:
        # Simple edges
        s_edges = torch.unique(sample_batch.edge_index, dim=1)
        in_deg = degree(s_edges[1], num_nodes=sample_batch.num_nodes, dtype=torch.long)
    else:
        # Compute the in-degree of each node
        in_deg = degree(sample_batch.edge_index[1], num_nodes=sample_batch.num_nodes, dtype=torch.long)
    
    # Histogram based on the in-degree values of the nodes
    deg = torch.bincount(in_deg, minlength=1)

    # Modify the model that is being trained
    model = GraphFuse(num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2, 
                      gnn_hidden=round(config.gnn_hidden), attn_hidden=round(config.attn_hidden), edge_updates=config.emlps, edge_dim=e_dim, 
                      dropout=config.dropout, attn_dropout=config.attn_dropout, final_dropout=config.final_dropout, index_=index_, deg=deg, config=config)

    return model


def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, config):
    """Main entry point for training the GNN model.
    
    Args:
        tr_data: Training data
        val_data: Validation data
        te_data: Test data
        tr_inds: Training indices
        val_inds: Validation indices
        te_inds: Test indices
        
    Returns:
        Module: Trained model
    """
    device = config.device
    # Add unique IDs to later find the seed edges (target transactions)
    add_arange_ids([tr_data, val_data, te_data])

    # Get data loaders
    # if config.use_pe:
    #     transform = T.AddLaplacianEigenvectorPE(k=20, attr_name="node_pe")
    # else:
    #     transform = None
    if config.use_ego:
        transform = AddEgoIds()
    else:
        transform = None

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform=transform, config=config)
    
    # Get a sample batch and initialize the model   
    sample_batch = next(iter(tr_loader))
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    model = get_model(sample_batch, config)
    model.to(device)
    # Move sample batch to device and log model summary
    sample_batch.to(device)
    
    if config.wandb:
        # Initialize Weights and Biases
        wandb.init(
            project="graphfuse-aml",
            name=f"run_{config.run_name}",
            group=re.sub(r'-s\d+$', "", config.run_name),
            config=vars(config),
        )

    logging.info(summary(model, sample_batch))

    # Define loss function and Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    num_epochs = config.n_epochs
    batches_per_epoch = len(tr_loader)
    logging.info(f"Number of batches per epoch: {batches_per_epoch}")
    accum_steps = config.accum_steps
    
    if config.use_scheduler:
        num_warmup_epochs = 5
        
        num_training_steps = math.ceil(batches_per_epoch
                                    / accum_steps
                                    * num_epochs)
        
        num_warmup_steps = math.ceil(batches_per_epoch
                                / accum_steps
                                * num_warmup_epochs)

        scheduler = get_cosine_schedule_with_warmup(
        optimizer             = optimizer,
        num_warmup_steps      = num_warmup_steps,
        num_training_steps    = num_training_steps,
        num_cycles            = 0.5,
        last_epoch            = -1,
        )
    else:
        scheduler = None

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    # Train the model
    model = train(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, scheduler, loss_fn, config)
    
    if config.wandb:
        wandb.finish()
    
    return model


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

