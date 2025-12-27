"""
Model definitions and utilities for PPO GNN-based RL agent.
Contains model architecture, helper functions, and utility functions.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    SCHEDULE_TOKEN_DIM,
    NODE_MAX_LENGTH,
    SCHEDULE_TOKENS_MAX_LENGTH,
    ENTROPY_COEF as ENTROPY_COEFFICIENT,
    RANDOM_WARMUP_STEPS,
    EPSILON_GREEDY,
)


# -------------------------------------------------
# Helper: masked categorical sampling & log prob
# -------------------------------------------------
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps=1e-8):
    """
    logits: (..., A)
    mask: same shape as logits, boolean (True=allowed)
    returns probs with zeros on masked entries and normalized over allowed actions.
    """
    neg_inf = -1e9
    masked_logits = logits.masked_fill(~mask, neg_inf)
    probs = F.softmax(masked_logits, dim=dim)
    # ensure masked positions are zero (numerical safety)
    probs = probs * mask.to(probs.dtype)
    # renormalize in case of numerical issues
    denom = probs.sum(dim=dim, keepdim=True)
    denom = torch.clamp(denom, min=eps)
    probs = probs / denom
    return probs


def masked_categorical_sample(probs: torch.Tensor):
    """
    probs: (..., A) with rows that sum to 1
    returns actions: long tensor shape (...)
    """
    # flatten leading dims, sample, then reshape
    shape = probs.shape
    flat = probs.view(-1, shape[-1])
    # sample via multinomial
    acts = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return acts.view(shape[:-1])


def masked_log_prob_from_logits(logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
    """
    Compute log prob of chosen actions given logits and mask
    logits: (B, A)
    actions: (B,) long
    mask: (B, A) bool
    """
    probs = masked_softmax(logits, mask, dim=-1)
    # gather prob
    action_prob = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    return torch.log(action_prob + 1e-12)


# ------------------------------------------
# Model components
# ------------------------------------------
class NodeEncoder(nn.Module):
    """Simple MLP encoder for node features -> initial node embeddings"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, N, in_dim) or (N, in_dim)
        return self.net(x)


class ScheduleAttentionEncoder(nn.Module):
    """
    Per-node schedule encoder:
    - Accepts a batch of schedule token lists (variable length).
    - We'll represent tokens per node as a padded tensor with lengths.
    - Use a CLS token per node (learnable) and a small multi-head attention.
    For simplicity: implement single-head scaled dot-product attention.
    """

    def __init__(self, token_dim=3, d_model=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_proj = nn.Linear(token_dim, d_model)    # project tokens -> d_model
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        # QKV projections
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        # small output projector
        self.out = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_pad: torch.Tensor, lengths: torch.Tensor):
        """
        token_pad: (B, N, K, token_dim)  padded schedule tokens per node
        lengths: (B, N) number of tokens per node (0..K)
        Returns schedule_embedding: (B, N, d_model)
        Implementation: for each node, build token sequence [CLS, tok1..tokL, PAD...],
        run single-head attention where query=CLS and keys/values=all tokens,
        return CLS output (projected).
        """
        B, N, K, token_dim = token_pad.shape
        device = token_pad.device
        # project tokens
        tok = self.token_proj(token_pad)  # (B,N,K,d)
        # create CLS tokens per (B,N)
        cls = self.cls_token.expand(B, N, -1)  # (B, N, d)
        cls = cls.unsqueeze(2)  # (B,N,1,d)
        seq = torch.cat([cls, tok], dim=2)  # (B,N,1+K,d)

        # mask for valid positions
        # valid_mask shape (B,N,1+K) bool, True where token exists (CLS always True)
        valid_mask = torch.zeros((B, N, 1 + K), dtype=torch.bool, device=device)
        valid_mask[:, :, 0] = True
        # lengths: (B,N), mark first lengths entries (1..length) as True in positions 1..length
        for b in range(B):
            for n in range(N):
                L = int(lengths[b, n].item())
                if L > 0:
                    valid_mask[b, n, 1:1+L] = True

        # reshape to (B*N, 1+K, d)
        seq2 = seq.view(B * N, 1 + K, self.d_model)
        mask2 = valid_mask.view(B * N, 1 + K)

        # Q from CLS only (index 0)
        q = self.q(seq2[:, 0:1, :])          # (B*N,1,d)
        k = self.k(seq2)                     # (B*N,1+K,d)
        v = self.v(seq2)

        # scaled dot product
        scale = math.sqrt(self.d_model)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / scale  # (B*N,1,1+K)
        # mask: set -inf where invalid
        neg_inf = -1e9
        attn_scores = attn_scores.masked_fill(~mask2.unsqueeze(1), neg_inf)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B*N,1,1+K)

        attn_output = torch.bmm(attn_weights, v)  # (B*N,1,d)
        attn_output = attn_output.squeeze(1)  # (B*N,d)
        out = self.out(attn_output)
        out = self.dropout(out)  # Apply dropout before layernorm
        out = out.view(B, N, self.d_model)
        out = self.layernorm(out)  # (B,N,d)
        return out


class FusionMLP(nn.Module):
    def __init__(self, node_dim, sched_dim, fused_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + sched_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

    def forward(self, node_emb, sched_emb):
        # both are (B,N,d)
        x = torch.cat([node_emb, sched_emb], dim=-1)
        return self.net(x)


class SimpleGNNLayer(nn.Module):
    """A tiny message passing layer using adjacency (dense) and an MLP for messages."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, h, adj, node_mask=None):
        """
        h: (B, N, D)
        adj: (B, N, N) edge weights (we will use 1/(1+travel_time) as affinity or just binary)
        node_mask: (B, N) bool, True for valid nodes, False for padded nodes (optional)
        """
        # compute messages: aggregate neighbor features weighted by normalized adjacency
        # adj may contain inf distances; user env provides shortest_paths, we convert externally to affinity
        # Here we expect adj to be affinity matrix already (B,N,N)
        m = self.msg(h)  # (B,N,D)
        
        # Mask padded nodes: set messages to zero for padded nodes
        if node_mask is not None:
            m = m * node_mask.unsqueeze(-1).to(m.dtype)  # (B,N,D)
            # Mask adjacency: set connections to/from padded nodes to zero
            adj_masked = adj * node_mask.unsqueeze(1).to(adj.dtype) * node_mask.unsqueeze(2).to(adj.dtype)  # (B,N,N)
        else:
            adj_masked = adj
        
        # aggregate: neighbors -> sum_j adj_ij * m_j
        m_agg = torch.bmm(adj_masked, m)  # (B,N,D)
        # concat and update
        out = self.update(torch.cat([h, m_agg], dim=-1))
        
        # Mask output for padded nodes
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        
        return out


class ActorCritic(nn.Module):
    """
    Complete model:
      - NodeEncoder: node features -> node_emb
      - ScheduleEncoder: schedule tokens -> sched_emb
      - FusionMLP -> fused node features
      - 2 x SimpleGNNLayer (message passing)
      - Actor head: produce per-node logits + special action logits (RETURN_TO_BASE, REST, WAIT)
      - Critic head: global value from pooling
    """
    def __init__(self, node_feat_dim=10, token_dim=3, sched_d_model=16,
                 node_emb_dim=32, fused_dim=64, gnn_dim=64, hidden_v=128, dropout=0):
        super().__init__()

        self.dropout_rate = dropout  # Store for reference

        # encoders - pass dropout to all
        self.node_encoder = NodeEncoder(node_feat_dim, node_emb_dim, dropout=dropout)
        self.sched_encoder = ScheduleAttentionEncoder(token_dim=token_dim, d_model=sched_d_model, dropout=dropout)

        # fusion - pass dropout    [it combines the node and schedule embeddings]
        self.fusion = FusionMLP(node_emb_dim, sched_d_model, fused_dim, dropout=dropout)

        # make fused -> gnn_dim    [it projects the fused embeddings to the gnn dimension]
        # [here both the fusion dim and the gnn dim are the same]
        self.fused_proj = nn.Linear(fused_dim, gnn_dim)

        # GNN layers - pass dropout
        self.gnn1 = SimpleGNNLayer(gnn_dim, dropout=dropout)
        self.gnn2 = SimpleGNNLayer(gnn_dim, dropout=dropout)
        
        # Shared dropout layer for between GNN layers
        self.gnn_dropout = nn.Dropout(dropout)

        # actor parts: unified action embedding architecture
        # Action embeddings (all 64-dim)
        self.go_to_node_action_embedding = nn.Parameter(torch.randn(1, gnn_dim) * 0.01)  # Shared embedding for all GO_TO_NODE actions
        self.return_to_base_action_embedding = nn.Parameter(torch.randn(1, gnn_dim) * 0.01)
        self.rest_action_embedding = nn.Parameter(torch.randn(1, gnn_dim) * 0.01)
        self.wait_action_embedding = nn.Parameter(torch.randn(1, gnn_dim) * 0.01)
        
        # Action embedding fusion: combines node embedding with action embedding -> 64-dim
        # Input: node_emb (64) + action_emb (64) = 128, output: 64
        self.action_fusion = nn.Sequential(
            nn.Linear(gnn_dim * 2, gnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Common MLP for all actions: takes 64-dim action embedding -> logit
        self.action_logit_mlp = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_dim, 1)
        )

        # actor scaling
        self.logits_scale = 1.0

        # critic - use dropout
        self.critic = nn.Sequential(
            nn.Linear(gnn_dim + 2, hidden_v),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_v, 1)
        )

    def forward(self, node_feats: torch.Tensor, schedule_tokens_pad: torch.Tensor,
                schedule_lengths: torch.Tensor, adj_affinity: torch.Tensor,
                worker_pos: torch.LongTensor, base_idx: int, mask: torch.Tensor,
                node_lengths: torch.Tensor = None):
        """
        Inputs:
         - node_feats: (B, max_N, node_feat_dim) - padded node features
         - schedule_tokens_pad: (B, max_N, K, token_dim) - padded schedule tokens
         - schedule_lengths: (B, max_N) ints - number of tokens per node
         - adj_affinity: (B, max_N, max_N) affinity matrix (higher = stronger) - padded
         - worker_pos: (B,) long - worker position (within valid nodes)
         - base_idx: int - base node index (within valid nodes)
         - mask: (B, max_N+3) bool  (action mask provided by env)
         - node_lengths: (B,) long - actual number of nodes per graph (for masking padded nodes)
        Returns:
         - logits: (B, max_N+3)
         - value: (B,)
         - extra dict (node_logits etc)
        """

        B, max_N, _ = node_feats.shape
        device = node_feats.device

        # Create node mask for padded nodes (True for valid nodes, False for padded)
        if node_lengths is not None:
            node_mask = torch.arange(max_N, device=device).unsqueeze(0) < node_lengths.unsqueeze(1)  # (B, max_N)
        else:
            node_mask = None

        # 1) node encoder
        node_emb = self.node_encoder(node_feats)            # (B,max_N,node_emb_dim)

        # 2) schedule encoder -> schedule embeddings
        sched_emb = self.sched_encoder(schedule_tokens_pad, schedule_lengths)  # (B,max_N,sched_d_model)

        # 3) fusion & project
        fused = self.fusion(node_emb, sched_emb)            # (B,max_N,fused_dim)
        h = self.fused_proj(fused)                          # (B,max_N,gnn_dim)

        # 4) message passing (two layers) - with node masking
        h = self.gnn1(h, adj_affinity, node_mask=node_mask)  # (B,max_N,gnn_dim)
        h = self.gnn_dropout(h)  # Dropout between GNN layers
        h = self.gnn2(h, adj_affinity, node_mask=node_mask)  # (B,max_N,gnn_dim)

        # 5) Compute action embeddings and logits using unified architecture
        base_emb = h[:, base_idx, :]                        # (B, gnn_dim)
        cur_emb = h[torch.arange(B, device=device), worker_pos, :]  # (B, gnn_dim)
        
        # Expand action embeddings to batch size
        go_to_node_emb = self.go_to_node_action_embedding.expand(B, -1)  # (B, gnn_dim)
        return_to_base_emb = self.return_to_base_action_embedding.expand(B, -1)  # (B, gnn_dim)
        rest_emb = self.rest_action_embedding.expand(B, -1)  # (B, gnn_dim)
        wait_emb = self.wait_action_embedding.expand(B, -1)  # (B, gnn_dim)
        
        # GO_TO_NODE actions: for each node, combine node embedding with action embedding
        # h: (B, max_N, gnn_dim), go_to_node_emb: (B, gnn_dim)
        # We need to combine each node embedding with the shared go_to_node_action_embedding
        go_to_node_emb_expanded = go_to_node_emb.unsqueeze(1).expand(-1, max_N, -1)  # (B, max_N, gnn_dim)
        go_to_node_input = torch.cat([h, go_to_node_emb_expanded], dim=-1)  # (B, max_N, gnn_dim*2)
        go_to_node_action_embeddings = self.action_fusion(go_to_node_input)  # (B, max_N, gnn_dim)
        # Reshape for MLP: (B*max_N, gnn_dim) -> (B*max_N, 1) -> (B, max_N)
        go_to_node_logits = self.action_logit_mlp(go_to_node_action_embeddings.view(B * max_N, -1)).view(B, max_N)  # (B, max_N)
        
        # RETURN_TO_BASE: combine cur_emb with return_to_base_action_embedding
        return_to_base_input = torch.cat([cur_emb, return_to_base_emb], dim=-1)  # (B, gnn_dim*2)
        return_to_base_action_emb = self.action_fusion(return_to_base_input)  # (B, gnn_dim)
        return_to_base_logit = self.action_logit_mlp(return_to_base_action_emb).squeeze(-1)  # (B,)
        
        # REST: combine base_emb with rest_action_embedding
        rest_input = torch.cat([base_emb, rest_emb], dim=-1)  # (B, gnn_dim*2)
        rest_action_emb = self.action_fusion(rest_input)  # (B, gnn_dim)
        rest_logit = self.action_logit_mlp(rest_action_emb).squeeze(-1)  # (B,)
        
        # WAIT: combine cur_emb with wait_action_embedding
        wait_input = torch.cat([cur_emb, wait_emb], dim=-1)  # (B, gnn_dim*2)
        wait_action_emb = self.action_fusion(wait_input)  # (B, gnn_dim)
        wait_logit = self.action_logit_mlp(wait_action_emb).squeeze(-1)  # (B,)
        
        # Stack logits: (B, max_N+3)
        logits = torch.cat([go_to_node_logits, return_to_base_logit.unsqueeze(-1), rest_logit.unsqueeze(-1), wait_logit.unsqueeze(-1)], dim=-1)
        logits = logits * self.logits_scale

        # apply mask outside usually, but we'll return raw logits + mask
        # value: global pooling (mean over valid nodes only) plus worker_pos info
        if node_mask is not None:
            # Masked mean: sum over valid nodes, divide by count of valid nodes
            h_masked = h * node_mask.unsqueeze(-1).to(h.dtype)  # (B, max_N, gnn_dim)
            pooled = h_masked.sum(dim=1) / node_lengths.unsqueeze(-1).to(h.dtype)  # (B, gnn_dim)
        else:
            pooled = h.mean(dim=1)   # (B, gnn_dim)
        
        # optionally include worker_pos distance to base or current_time as extra features: here we just add dummy zeros
        extra = torch.zeros(B, 2, device=device)
        value_in = torch.cat([pooled, extra], dim=-1)
        value = self.critic(value_in).squeeze(-1)  # (B,)

        return logits, value, {
            "node_logits": go_to_node_logits,
            "rtn_logit": return_to_base_logit,
            "rest_logit": rest_logit,
            "wait_logit": wait_logit,
            "node_embeddings": h
        }

    def act(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ------------------------------------------------------------
# Utilities: Device detection
# ------------------------------------------------------------
def get_device():
    """
    Automatically detect and return the best available device.
    Returns 'cuda' if GPU is available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Utilities: Graph registry for efficient storage (GPU-optimized)
# ------------------------------------------------------------
def build_graph_registry(graph_dataset: List, device=None):
    """
    Build a device-resident registry of graph-level static data (adj_affinity, base_idx, num_nodes) 
    keyed by dataset index. All tensors are pre-allocated on the specified device to avoid 
    CPU->GPU transfers during training. Adjacency matrices are padded to NODE_MAX_LENGTH.
    
    Args:
        graph_dataset: List of GraphData objects
        device: Target device (torch.device or str). If None, auto-detects GPU/CPU.
    
    Returns:
        graph_registry: dict {graph_id: {"adj_affinity": tensor (NODE_MAX_LENGTH, NODE_MAX_LENGTH), "base_idx": int, "num_nodes": int}}
        max_nodes: int - always returns NODE_MAX_LENGTH (for compatibility)
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    graph_registry = {}
    max_nodes = NODE_MAX_LENGTH  # Use constant from config
    
    # Pad all adjacency matrices to NODE_MAX_LENGTH
    for graph_id, graph in enumerate(graph_dataset):
        num_nodes = graph.num_nodes
        
        # Convert shortest_paths to affinity matrix
        sp = graph.shortest_paths
        aff = 1.0 / (1.0 + sp)
        aff[np.isinf(sp)] = 0.0
        
        # Pad adjacency matrix to NODE_MAX_LENGTH x NODE_MAX_LENGTH
        if num_nodes < NODE_MAX_LENGTH:
            aff_padded = np.zeros((NODE_MAX_LENGTH, NODE_MAX_LENGTH), dtype=np.float32)
            aff_padded[:num_nodes, :num_nodes] = aff
            aff = aff_padded
        elif num_nodes > NODE_MAX_LENGTH:
            # Truncate if graph is larger than max (shouldn't happen, but handle gracefully)
            aff = aff[:NODE_MAX_LENGTH, :NODE_MAX_LENGTH]
        
        # Pre-allocate on GPU (one-time cost, avoids CPU->GPU transfers during training)
        graph_registry[graph_id] = {
            "adj_affinity": torch.from_numpy(aff).float().to(device),  # (NODE_MAX_LENGTH, NODE_MAX_LENGTH) - already on GPU!
            "base_idx": int(graph.base),
            "num_nodes": int(num_nodes)  # Store actual number of nodes for this graph
        }
    return graph_registry, max_nodes

