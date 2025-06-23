import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import degree
from .util import compute_lsh_buckets, compute_random_buckets
from .mega_util import MultiEdgeAggModule

class GraphFuse(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                 gnn_hidden=100, attn_hidden=32,
                 edge_updates=True, edge_dim=None, 
                 dropout=0.1, attn_dropout=0.2, final_dropout=0.5, 
                 index_=None, deg=None, config=None):
        super().__init__()
        self.config = config
        self.gnn_hidden = gnn_hidden
        self.use_pe = config.use_pe
        self.use_ec = config.use_ec
        self.use_aug = config.use_aug
        self.use_attn = config.use_attn
        self.attn_hidden = attn_hidden
        self.dim_fusion = gnn_hidden + attn_hidden if self.use_attn else 3 * gnn_hidden
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.final_dropout = final_dropout

        # Node and Edge features are mapped to the latents space of dimensionality n_hidden
        self.node_encoder = nn.Linear(num_features, self.gnn_hidden)
        # Configure Laplacian Eigenvector PEs
        if self.use_pe:
            self.pe_encoder = nn.Linear(config.pe_dim, self.attn_hidden, bias=True)
            self.pe_dropout = nn.Dropout(self.dropout)
            self.pe_bn = nn.BatchNorm1d(self.attn_hidden)
            self.pe_alpha = nn.Parameter(torch.tensor(1.0))
        if self.use_ec:
            self.ec_encoder = nn.Linear(2, self.attn_hidden)
            self.ec_dropout = nn.Dropout(self.dropout)
            self.ec_alpha_attn = nn.Parameter(torch.tensor(1.0))
        if self.use_attn:
                if self.use_aug:
                    self.edge_encoder_attn = nn.Linear(edge_dim+16, self.attn_hidden)
                else:
                    self.edge_encoder_attn = nn.Linear(edge_dim, self.attn_hidden)
        
        self.edge_encoder_gnn = nn.Linear(edge_dim, self.gnn_hidden)

        # Define the GNN-based Module
        self.gnn_module = GNNModule(num_gnn_layers=num_gnn_layers, 
                                    dim_node=self.gnn_hidden, 
                                    dim_edge=self.gnn_hidden, 
                                    edge_updates=edge_updates,
                                    index_=index_,
                                    deg=deg, 
                                    config=config)
        
        if self.use_attn:
            # Define the Edge Global Attention Module
            self.edge_transformer = EdgeTransformer(n_hidden=self.attn_hidden,
                                                    dropout=self.attn_dropout,
                                                    config=config)

            # Define the Late Fusion Layer
            if self.config.gated_fusion:
                self.late_fusion_layer = LateGatedFusionLayer(gnn_hidden=self.gnn_hidden,
                                                              attn_hidden=self.attn_hidden,
                                                              dim_fusion=self.dim_fusion,
                                                              dropout=self.dropout,
                                                              final_dropout=self.final_dropout,
                                                              config=config)
            else:
                self.late_fusion_layer = LateFusionLayer(gnn_hidden=self.gnn_hidden,
                                                         attn_hidden = self.attn_hidden,
                                                         dim_fusion=self.dim_fusion,
                                                         dropout=self.dropout,
                                                         config=config)

        # Define the Prediction Head
        self.mlp = nn.Sequential(Linear(self.dim_fusion, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                            Linear(25, n_classes))
    
    def reset_parameters(self):
        self.gnn_module.reset_parameters()
        if self.use_attn:
            self.edge_transformer.reset_parameters()
            self.late_fusion_layer.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):        
        # Initial Embedding Layer: Local Message Passing Module
        h_n = self.node_encoder(data.x)
        h_e_gnn = self.edge_encoder_gnn(data.edge_attr)
        simp_edge_batch = data.simp_edge_batch if self.config.use_mega else None

        # Local Message Passing Layers
        h_n, h_e_gnn = self.gnn_module(h_n, data.edge_index, h_e_gnn, simp_edge_batch) # [V, gnn_hidden], [E, gnn_hidden]

        if self.use_attn:
            if self.config.attn_mechanism == "sparse":
                if self.config.sparse_clustering == "lsh":
                    edge_bucket_ids = compute_lsh_buckets(edge_attr=data.edge_attr, config=self.config)
                elif self.config.sparse_clustering == "rnd":
                    edge_bucket_ids = compute_random_buckets(num_edges=data.edge_attr.shape[0], device=data.edge_attr.device, config=self.config)
                else:
                    raise ValueError("Wrong sparse clustering method provided, either lsh or rnd.")
            
            # Initial Embedding Layer: Global Attention Module
            if self.use_aug:
                node_aug = data.x_aug
                src, dest = data.edge_index
                h_e_attn = self.edge_encoder_attn(torch.cat((data.edge_attr, node_aug[src], node_aug[dest]),1))
            else:
                h_e_attn = self.edge_encoder_attn(data.edge_attr)
            
            # Edge Centrality Layer
            if self.use_ec:
                # Compute the in-degree of each node
                in_deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long) # [V,]
                # Compute the out-degree of each node
                out_deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long) # [V,]
                # Log scaling
                in_deg = torch.log(in_deg+1)
                out_deg = torch.log(out_deg+1)
                # 
                in_deg = (in_deg - in_deg.mean()) / (in_deg.std() + 1e-5)
                out_deg = (out_deg - out_deg.mean()) / (out_deg.std() + 1e-5)

                src, dst = data.edge_index
                src_centrality = (in_deg[src] + out_deg[src]) / 2
                dst_centrality = (in_deg[dst] + out_deg[dst]) / 2
                edge_centrality = torch.cat((src_centrality.unsqueeze(1), dst_centrality.unsqueeze(1)),1) # [E,2]

                h_e_ec = self.ec_dropout(self.ec_encoder(edge_centrality)) # [E, attn_hidden]
                h_e_attn += self.ec_alpha_attn * h_e_ec
            
            # Global Attention Layer
            if self.config.attn_mechanism == "sparse":
                h_e_ga = self.edge_transformer(h_e_attn, edge_bucket_ids)
            else:
                h_e_ga = self.edge_transformer(h_e_attn)
            
            # Local MP representation
            h = h_n[data.edge_index.T].reshape(-1, 2*self.gnn_hidden).relu()
            h_e_gnn = torch.cat((h, h_e_gnn.view(-1, h_e_gnn.shape[1])), 1) # [E, 3*gnn_hidden]
            
            # Late Fusion Layer
            h_e_fused = self.late_fusion_layer(h_e_gnn, h_e_ga)

            # Prediction Head
            out = self.mlp(h_e_fused)
        else:
            # Local MP representation
            h = h_n[data.edge_index.T].reshape(-1, 2*self.gnn_hidden).relu()
            h = torch.cat((h, h_e_gnn.view(-1, h_e_gnn.shape[1])), 1)
            
            # Prediction Head
            out = self.mlp(h)
        
        return out

class EdgeTransformer(torch.nn.Module):
    """
    Stack of encoder blocks consisting of attention-based layers operating on the edges of the sampled neighbourhood.
    """
    def __init__(self, n_hidden, dropout, config=None):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_layers = config.attn_num_layers
        self.dropout = dropout
        self.config = config

        self.encoder_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            encoder_layer = EdgeEncoder(n_hidden, dropout, config=config)
            self.encoder_layers.append(encoder_layer)
    
    def reset_parameters(self):
        for layer in self.encoder_layers:
            layer.reset_parameters()
        self.ln_layer.reset_parameters()

    def forward(self, x, edge_bucket_ids=None):        
        for k in range(len(self.encoder_layers)):
            if self.config.attn_mechanism == "sparse":
                x = self.encoder_layers[k](x, x, edge_bucket_ids)
            else:
                x = self.encoder_layers[k](x, x)
        return x

class EdgeEncoder(torch.nn.Module):
    """
    Attention-based encoder which computes contextualized representations of the edge in the sampled sub-graph.
    """
    def __init__(self, n_hidden, dropout, config=None):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_heads = config.attn_num_heads
        self.head_dim = n_hidden // self.num_heads
        self.dropout = dropout
        self.attention_mechanism = config.attn_mechanism
        self.config = config
        if self.attention_mechanism == "linear":
            # Define the matrices associated with the Linear Attention mechanism (as defined in the SGFormer model)
            self.Wk_lin = nn.Linear(n_hidden, self.head_dim * self.num_heads)
            self.Wq_lin = nn.Linear(n_hidden, self.head_dim * self.num_heads)
            self.Wv_lin = nn.Linear(n_hidden, self.head_dim * self.num_heads)
            self.Wh_lin = nn.Linear(self.head_dim * self.num_heads, n_hidden)
            self.norm1_kv_lin = nn.LayerNorm(n_hidden)
            self.norm1_q_lin = nn.LayerNorm(n_hidden)
            self.ffn_lin = nn.Sequential(nn.Linear(n_hidden, 4 * n_hidden, bias=True), nn.GELU(), nn.Linear(4 * n_hidden, n_hidden, bias=True))
            self.norm2_lin = nn.LayerNorm(n_hidden)
            self.dropout_lin = nn.Dropout(dropout)
        elif self.attention_mechanism == "full":
            # Define the Transformer Encoder Layer which will full-attentive represenations of the edges
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=self.num_heads, dim_feedforward=4*n_hidden,
                                                            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True)

        else:
            # Define the matrices associated with the Sparse Attention mechanism
            self.Wk_sparse = nn.Linear(n_hidden, self.head_dim * self.num_heads)
            self.Wq_sparse = nn.Linear(n_hidden, self.head_dim * self.num_heads)
            self.Wv_sparse = nn.Linear(n_hidden, self.head_dim * self.num_heads)
            self.Wh_sparse = nn.Linear(self.head_dim * self.num_heads, n_hidden)
            self.norm1_kv_sparse = nn.LayerNorm(n_hidden)
            self.norm1_q_sparse = nn.LayerNorm(n_hidden)
            self.ffn_sparse = nn.Sequential(nn.Linear(n_hidden, 4 * n_hidden, bias=True), nn.GELU(), nn.Linear(4 * n_hidden, n_hidden, bias=True))
            self.norm2_sparse = nn.LayerNorm(n_hidden)
            self.dropout_sparse = nn.Dropout(dropout)
    
    def reset_parameters(self):
        if self.attention_mechanism == "linear":
            self.Wk_lin.reset_parameters()
            self.Wq_lin.reset_parameters()
            self.Wv_lin.reset_parameters()
            self.Wh_lin.reset_parameters()
            self.norm1_kv_lin.reset_parameters()
            self.norm1_q_lin.reset_parameters()
            for layer in self.ffn_lin:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            self.norm2_lin.reset_parameters()
        elif self.attention_mechanism == "sparse":
            self.Wk_sparse.reset_parameters()
            self.Wq_sparse.reset_parameters()
            self.Wv_sparse.reset_parameters()
            self.Wh_sparse.reset_parameters()
            self.norm1_kv_sparse.reset_parameters()
            self.norm1_q_sparse.reset_parameters()
            for layer in self.ffn_sparse:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            self.norm2_sparse.reset_parameters()
        else:
            return

    def forward(self, query_input, source_input, edge_bucket_ids=None):
        if self.attention_mechanism == "linear":
            # Peform the forward pass considering the Linear Attention Mechanism (as defined in the SGFormer model)
            # Construct the Q and K matrices
            # Pre-LayerNorm
            ln_source_input = self.norm1_kv_lin(source_input)
            ln_query_input = self.norm1_q_lin(query_input)

            qs = self.Wq_lin(ln_query_input).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
            ks = self.Wk_lin(ln_source_input).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
            vs = self.Wv_lin(ln_source_input).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
            
            # Normalize the Q and K matrices
            qs = qs / torch.norm(qs, p=2) # [N, H, d_h]
            ks = ks / torch.norm(ks, p=2) # [N, H, d_h]
            N = qs.shape[0]

            # Compute the all-pair attentive representations
            # Numerator
            kvs = torch.einsum("nhd,nhk->hdk", ks, vs) # [H, d_h, d_h]
            attention_num = torch.einsum("nhd,hdd->nhd", qs, kvs) # [N, H, d_h]
            attention_num += vs * N # [N, H, d]

            # Denominator
            all_ones = torch.ones([ks.shape[0]], device=ks.device)
            ks_sum = torch.einsum("nhd,n->hd", ks, all_ones) # [H, d_h]
            attention_normalizer = torch.einsum("nhd,hd->nh", qs, ks_sum) # [N, H]

            # Attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape)) # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer, device=attention_normalizer.device) * N
            attn_output = attention_num / attention_normalizer # [N, H, d_h]

            attn_output = attn_output.reshape(attn_output.size(0), -1) # [N, H*d_h]

            h_post_attention = self.Wh_lin(attn_output) # [N, d]
            
            h_pre_ffn = source_input + self.dropout_lin(h_post_attention)
            
            h_post_ffn = self.ffn_lin(self.norm2_lin(h_pre_ffn))

            final_output = h_pre_ffn + self.dropout_lin(h_post_ffn) # [N, d]
    
            return final_output

        elif self.attention_mechanism == "full":
            # Perform the forward pass and compute the all-pair attentive representations of the input
            source_input_ = source_input.unsqueeze(0) # [1, N, d]
            out = self.encoder_layer(source_input_)
            out = out.squeeze(0)
            return out
        
        elif self.attention_mechanism == "sparse":
            ln_source_input = self.norm1_kv_sparse(source_input)
            ln_query_input = self.norm1_q_sparse(query_input)

            sorted_ids = torch.argsort(edge_bucket_ids)
            ln_source_input_sorted = ln_source_input[sorted_ids]
            ln_query_input_sorted = ln_query_input[sorted_ids]
            # clusters sorted in ascending order
            edge_bucket_ids_sorted = edge_bucket_ids[sorted_ids]

            # Project
            qs = self.Wq_sparse(ln_query_input_sorted).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
            ks = self.Wk_sparse(ln_source_input_sorted).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]
            vs = self.Wv_sparse(ln_source_input_sorted).reshape(-1, self.num_heads, self.head_dim) # [N, H, d_h]

            # Group by cluster
            unique, counts = edge_bucket_ids_sorted.unique_consecutive(return_counts=True)
            # Find the ending index of each cluster by cumulatively summing the counts of each cluster
            cluster_ptr = counts.cumsum(0)
            # Find the starting index of each cluster
            cluster_start = torch.cat([torch.tensor([0], device=source_input.device), cluster_ptr[:-1]])

            output = torch.zeros_like(qs)
            for i in range(len(unique)):
                s, e = cluster_start[i].item(), cluster_ptr[i].item()
                q = qs[s:e].transpose(0, 1)  # [H, N, d_h]
                k = ks[s:e].transpose(0, 1)
                v = vs[s:e].transpose(0, 1)

                # Intra-cluster scaled-dot product attention
                attn_out = F.scaled_dot_product_attention(q, k, v)  # [H, N, d_h]
                output[s:e] = attn_out.transpose(0, 1) # [N, H, d_h]

            # Merge heads and unsort
            output = output.reshape(output.size(0), -1) # [N, H*d_h]
            
            output = self.Wh_sparse(output) # [N, d]

            h_post_attention = torch.zeros_like(output, device=query_input.device)
            
            h_post_attention[sorted_ids] = output

            h_pre_ffn = source_input + self.dropout_sparse(h_post_attention) 

            h_post_ffn = self.ffn_sparse(self.norm2_sparse(h_pre_ffn))
            
            final_output = h_pre_ffn + self.dropout_sparse(h_post_ffn)

            return final_output
        else:
            raise ValueError("Invalid attention mechanism indicated.")

class LateFusionLayer(torch.nn.Module):
    def __init__(self, gnn_hidden, attn_hidden, dim_fusion=None, dropout=0.1,config=None):
        super().__init__()
        self.gnn_hidden = gnn_hidden
        self.attn_hidden = attn_hidden 
        self.dim_fusion = dim_fusion if dim_fusion else gnn_hidden
        self.dropout = dropout
        self.config = config
        # Two-layered MLP
        self.mlp = nn.Sequential(Linear(3 * self.gnn_hidden + self.attn_hidden, self.dim_fusion), nn.GELU(), nn.Dropout(self.dropout),
                                 Linear(self.dim_fusion, self.dim_fusion))

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, edge_attr_gnn, edge_attr_et):
        out = self.mlp(torch.cat((edge_attr_gnn, edge_attr_et), -1)) # [N, dim_fusion]
        return out

class LateGatedFusionLayer(torch.nn.Module):
    def __init__(self, gnn_hidden, attn_hidden, dim_fusion=None, dropout=0.1, final_dropout=0.5,config=None):
        super().__init__()
        self.gnn_hidden = gnn_hidden
        self.attn_hidden = attn_hidden 
        self.dim_fusion = dim_fusion if dim_fusion else gnn_hidden
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.config = config
        # Global MP Gate
        self.gmp_gate = Linear(self.attn_hidden, self.attn_hidden, bias=True)
        # Local MP Gate
        self.lmp_gate = Linear(3 * self.gnn_hidden, 3 * self.gnn_hidden, bias=True)
        # Better initial bias so sigmoid approx eq. 0.5 instead of saturating at 0
        nn.init.constant_(self.gmp_gate.bias, -1.0)
        nn.init.constant_(self.lmp_gate.bias, -1.0)
        # Fusion Layer
        self.mlp = nn.Sequential(Linear(3* self.gnn_hidden + self.attn_hidden, self.dim_fusion), nn.GELU(), nn.Dropout(self.dropout))

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, edge_attr_lmp, edge_attr_gmp):
        filtered_edge_atrr_lmp = F.sigmoid(self.lmp_gate(edge_attr_lmp)) * edge_attr_lmp
        filtered_edge_attr_gmp = F.sigmoid(self.gmp_gate(edge_attr_gmp)) * edge_attr_gmp
        out = self.mlp(torch.cat([filtered_edge_atrr_lmp, filtered_edge_attr_gmp], -1)) # [N, dim_fusion]
        return out

class GNNModule(torch.nn.Module):
    def __init__(self, num_gnn_layers, dim_node=100, dim_edge=100, edge_updates=False, 
                 index_ = None, deg = None, config=None):
        super().__init__()

        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.num_gnn_layers = num_gnn_layers
        self.config = config    
        self.edge_updates = edge_updates
        self.use_mega = config.use_mega

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.edge_aggrs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            if config.model == 'gin':
                conv = GINEConv(nn.Sequential(
                    nn.Linear(self.dim_node, self.dim_node), 
                    nn.ReLU(), 
                    nn.Linear(self.dim_node, self.dim_node)
                    ), 
                    edge_dim=self.dim_edge,
                    train_eps=True
                    )
            elif config.model == 'pna':
                # Consider other aggregators (nth moments)
                aggregators = ['mean', 'min', 'max', 'std']
                # Consider different scalers
                scalers = ['identity', 'amplification', 'attenuation']
                conv = PNAConv(in_channels=self.dim_node, out_channels=self.dim_node,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=self.dim_edge, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
                
            # Append the Edge-Updating MLP
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(2 * self.dim_node + self.dim_edge, self.dim_edge),
                nn.ReLU(),
                nn.Linear(self.dim_edge, self.dim_edge),
            ))
            
            # Append the Multi-Edge Aggregators
            if self.use_mega:
                edge_agg = MultiEdgeAggModule(n_hidden=self.dim_edge, agg_type=config.agg_type, index=index_)
                self.edge_aggrs.append(edge_agg)
                
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(dim_node))
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for emlp in self.emlps:
            emlp.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        if self.use_gru:
            for gru in self.grus:
                gru.reset_parameters()
            
    def forward(self, x, edge_index, edge_attr, simp_edge_batch=None):
        src, dst = edge_index
        for i in range(self.num_gnn_layers):
            # Multi-Edge Aggregation
            if self.use_mega:
                n_edge_index, n_edge_attr, inverse_indices  = self.edge_aggrs[i](edge_index, edge_attr, simp_edge_batch)
                # n_edge_attr: artificial node attributes
                # n_edge_index: index involving nodes and artificial nodes # [2,N_simple]
                
                # Standard MP using the artificial nodes: Node-Level Aggregation
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, n_edge_index, n_edge_attr)))) / 2
                if self.edge_updates: 
                    remapped_edge_attr = torch.index_select(n_edge_attr, 0, inverse_indices) # artificial node attributes mapped to #[E, d_e] 
                    edge_attr_new = edge_attr + self.emlps[i](torch.cat([x[src], remapped_edge_attr, edge_attr], dim=-1)) / 2
            # Standard MP
            else:
                n_new = self.convs[i](x, edge_index, edge_attr) 
                x = (x + F.relu(self.batch_norms[i](n_new))) / 2 # [V, dim_node]
                if self.edge_updates: 
                    edge_attr_new = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
        return x, edge_attr_new