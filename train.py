"""
PPO training loops and agent implementation.
Contains PPOAgent class and training functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import (
    ActorCritic,
    masked_softmax,
    get_device,
    build_graph_registry,
    ENTROPY_COEFFICIENT,
    RANDOM_WARMUP_STEPS,
    EPSILON_GREEDY,
)
from config import (
    LEARNING_RATE,
    CLIP_EPS,
    VALUE_COEF,
    ENTROPY_COEF,
    GAMMA,
    LAM,
    NUM_UPDATES,
    NUM_STEPS_PER_UPDATE,
    EPOCHS,
    MINIBATCH_SIZE,
    MAX_GRAD_NORM,
    NODE_EMB_DIM,
    SCHED_D_MODEL,
    FUSED_DIM,
    GNN_DIM,
    HIDDEN_V,
    DROPOUT,
    NODE_FEATURE_DIM,
    SCHEDULE_TOKEN_DIM,
)


# ---------------------------------------------------
# PPO training machinery
# ---------------------------------------------------
class PPOAgent:
    def __init__(self, model: ActorCritic, lr=None, clip_eps=None, value_coef=None, ent_coef=None, gamma=None, lam=None):
        self.model = model
        # Use config defaults if not provided
        lr = lr if lr is not None else LEARNING_RATE
        clip_eps = clip_eps if clip_eps is not None else CLIP_EPS
        value_coef = value_coef if value_coef is not None else VALUE_COEF
        ent_coef = ent_coef if ent_coef is not None else ENTROPY_COEF
        gamma = gamma if gamma is not None else GAMMA
        lam = lam if lam is not None else LAM
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.ent_coef = ent_coef  # entropy regularization term
        self.gamma = gamma
        self.lam = lam

    def compute_gae(self, rewards, values, dones):
        """
        rewards: (T,) numpy
        values: (T+1,) numpy
        dones: (T,) numpy (True if terminal)
        returns:
            advantages (T,)
            returns (T,)
        """
        # CPU: numpy operations
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):  # Accumulate advantages backward from terminal state
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]  # TD error δₜ
            adv[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam

        returns = adv + values[:-1]  # Return_t = A_t + V(s_t)
        return adv, returns

    def ppo_update(self, batch, clip_eps=None, epochs=4, batch_size=64):
        """
        PPO training update: full batched GPU training.
        
        Args:
            batch: dict with all rollout data (all tensors on CPU)
        """
        # Get device from model (model is already on target device)
        device = next(self.model.parameters()).device
        
        # ========== CPU OPERATIONS: Batch Preparation ==========
        M = batch["actions"].shape[0]
        inds = np.arange(M)  # CPU: numpy array for shuffling
        graph_registry = batch["graph_registry"]  # GPU registry (passed through)
        if clip_eps is None:
            clip_eps = self.clip_eps

        for _ in range(epochs):
            np.random.shuffle(inds)  # CPU: numpy shuffle
            for start in range(0, M, batch_size):
                mb_inds = inds[start:start + batch_size]
                
                # ========== GPU OPERATIONS: Transfer Minibatch to GPU ==========
                node_feats = batch["obs_node_feats"][mb_inds].to(device, non_blocking=True)
                token_pad = batch["obs_token_pad"][mb_inds].to(device, non_blocking=True)
                token_len = batch["obs_token_lengths"][mb_inds].to(device, non_blocking=True)
                graph_ids = batch["obs_graph_id"][mb_inds]  # Keep on CPU for indexing
                worker_pos = batch["obs_worker_pos"][mb_inds].to(device, non_blocking=True)
                mask = batch["obs_mask"][mb_inds].to(device, non_blocking=True)
                actions = batch["actions"][mb_inds].to(device, non_blocking=True)
                old_logp = batch["old_log_probs"][mb_inds].to(device, non_blocking=True)
                returns = batch["returns"][mb_inds].to(device, non_blocking=True)
                advantages = batch["advantages"][mb_inds].to(device, non_blocking=True)
                node_lengths = batch["obs_node_lengths"][mb_inds].to(device, non_blocking=True)

                # ========== GPU OPERATIONS: Build Adjacency Matrix for Minibatch ==========
                unique_graph_ids = torch.unique(graph_ids.cpu()).numpy()  # CPU: get unique graph IDs
                mb_size = len(mb_inds)
                
                # Build adj_affinity tensor for minibatch (all operations on GPU)
                if len(unique_graph_ids) == 1:
                    # All samples from same graph - efficient broadcast
                    graph_id = int(unique_graph_ids[0])
                    adj_aff_single = graph_registry[graph_id]["adj_affinity"]  # Already on GPU
                    adj_aff = adj_aff_single.unsqueeze(0).expand(mb_size, -1, -1)  # (mb_size, N, N) - GPU
                else:
                    # Multiple graphs in minibatch - stack per sample (all on GPU)
                    adj_aff_list = []
                    for gid in unique_graph_ids:
                        adj_aff_list.append(graph_registry[int(gid)]["adj_affinity"])  # Already on GPU
                    # Create mapping: graph_id -> index in unique list
                    gid_to_idx = {int(gid): i for i, gid in enumerate(unique_graph_ids)}
                    # Stack only unique graphs, then index (all on GPU)
                    unique_adj = torch.stack(adj_aff_list, dim=0)  # (num_unique, N, N) - GPU
                    # Map each sample's graph_id to its unique index
                    gid_indices = torch.tensor([gid_to_idx[int(gid.item())] for gid in graph_ids], device=device)
                    adj_aff = unique_adj[gid_indices]  # (mb_size, N, N) - GPU
                
                # Get base_idx from stored rollout data (CPU operation for indexing)
                base_idx_stored = batch["obs_base_idx"][mb_inds]  # CPU tensor
                if len(torch.unique(base_idx_stored)) == 1:
                    base_idx = int(base_idx_stored[0].item())
                else:
                    base_idx = int(base_idx_stored[0].item())  # Use first one when they differ

                # ========== GPU OPERATIONS: Model Forward Pass ==========
                logits, values, _ = self.model(node_feats, token_pad, token_len, adj_aff, worker_pos, 
                                              base_idx=base_idx, mask=mask, node_lengths=node_lengths)
                
                # ========== GPU OPERATIONS: Loss Computation ==========
                # Mask logits and compute logp (GPU)
                probs = masked_softmax(logits, mask, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Critic loss (MSE) - GPU
                value_loss = F.mse_loss(values, returns)

                # Policy loss - GPU
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Total loss - GPU
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                # Backward pass and optimizer step - GPU
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, device='cpu'):
        self.model.load_state_dict(torch.load(path, map_location=device))


# -------------------------
# Training loop (collect episodes -> update)
# -------------------------

def collect_rollout_batch(env, agent: PPOAgent, graph_registry: dict,
                         num_steps_per_update: int, total_steps_collected: int = 0):
    """
    Collect a single batch of rollouts from the environment.
    Assumes env.train=True, so observations are already padded.
    
    Args:
        env: The environment to collect from (must have train=True)
        agent: PPOAgent with the model (model should already be on target device)
        graph_registry: Pre-built graph registry with adj_affinity and base_idx (on GPU)
        num_steps_per_update: Number of steps to collect in this batch
        total_steps_collected: Total number of steps collected so far (for warmup tracking)
    
    Returns:
        batch: Dictionary containing collected rollout data ready for training (all on CPU)
    """
    assert env.train, "Environment must have train=True for collect_rollout_batch"
    
    # Get device from model (model is already on target device)
    model = agent.model
    device = next(model.parameters()).device
    model.eval()  # Set to eval mode for rollout collection
    
    BATCH = {
        "obs_node_feats": [],
        "obs_token_pad": [],
        "obs_token_lengths": [],
        "obs_graph_id": [],
        "obs_base_idx": [],
        "obs_node_lengths": [],
        "obs_worker_pos": [],
        "obs_mask": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "logps": [],
    }
    
    # collect transitions
    steps_collected = 0
    obs, mask = env.reset()
    
    while steps_collected < num_steps_per_update:
        # ========== GPU OPERATIONS: Model Forward Pass ==========
        graph_id = obs["graph_id"]
        graph_data = graph_registry[graph_id]
        base_idx = graph_data["base_idx"]
        adj_aff_t = graph_data["adj_affinity"].unsqueeze(0)
        
        # Convert observations to tensors and move to GPU for model forward pass
        node_feats_t = torch.from_numpy(obs["node_feats"]).unsqueeze(0).float().to(device)
        token_pad_t = torch.from_numpy(obs["schedule_tokens"]).unsqueeze(0).float().to(device)
        token_len_t = torch.from_numpy(obs["token_lengths"]).unsqueeze(0).long().to(device)
        node_lengths_t = torch.from_numpy(obs["node_lengths"]).long().to(device)
        worker_pos_t = torch.LongTensor([obs["worker_pos"]]).to(device)
        mask_t = torch.from_numpy(mask[np.newaxis, :]).to(device)
        
        # Model forward pass (GPU)
        with torch.no_grad():
            logits, value, _ = model(node_feats_t, token_pad_t, token_len_t, adj_aff_t, worker_pos_t, 
                                    base_idx=base_idx, mask=mask_t, node_lengths=node_lengths_t)
        
        # Compute probabilities on GPU
        probs = masked_softmax(logits, mask_t, dim=-1)
        
        # ========== CPU OPERATIONS: Action Sampling with Exploration ==========
        probs_cpu = probs.cpu().numpy().squeeze(0)
        mask_cpu = mask_t.cpu().numpy().squeeze(0)
        allowed_actions = np.where(mask_cpu)[0]
        
        # Exploration strategies: random warmup and epsilon-greedy
        use_random = False
        current_total_steps = total_steps_collected + steps_collected
        
        if current_total_steps < RANDOM_WARMUP_STEPS:
            use_random = True
        elif EPSILON_GREEDY > 0.0 and np.random.random() < EPSILON_GREEDY:
            use_random = True
        
        if use_random:
            action = np.random.choice(allowed_actions)
            uniform_prob = 1.0 / len(allowed_actions)
            logp = np.log(uniform_prob)
        else:
            action = np.random.choice(len(probs_cpu), p=probs_cpu)
            logp = float(torch.log(probs[0, action] + 1e-12).item())
        
        # ========== CPU OPERATIONS: Environment Step ==========
        obs2, reward, done, info, mask2 = env.step(int(action))
        
        # ========== CPU OPERATIONS: Store Rollout Data ==========
        BATCH["obs_node_feats"].append(obs["node_feats"])
        BATCH["obs_token_pad"].append(obs["schedule_tokens"])
        BATCH["obs_token_lengths"].append(obs["token_lengths"])
        BATCH["obs_graph_id"].append(graph_id)
        BATCH["obs_base_idx"].append(base_idx)
        BATCH["obs_node_lengths"].append(obs["node_lengths"][0])
        BATCH["obs_worker_pos"].append(obs["worker_pos"])
        BATCH["obs_mask"].append(mask.astype(np.bool_))
        BATCH["actions"].append(action)
        BATCH["rewards"].append(reward)
        BATCH["dones"].append(done)
        BATCH["values"].append(value.cpu().numpy().squeeze(0))
        BATCH["logps"].append(logp)
        
        obs, mask = obs2, mask2
        steps_collected += 1
        
        if done:
            obs, mask = env.reset()
    
    # Bootstrap value for final observation
    graph_id = obs["graph_id"]
    graph_data = graph_registry[graph_id]
    base_idx = graph_data["base_idx"]
    adj_aff_t = graph_data["adj_affinity"].unsqueeze(0)
    
    node_feats_t = torch.from_numpy(obs["node_feats"]).unsqueeze(0).float().to(device)
    token_pad_t = torch.from_numpy(obs["schedule_tokens"]).unsqueeze(0).float().to(device)
    token_len_t = torch.from_numpy(obs["token_lengths"]).unsqueeze(0).long().to(device)
    node_lengths_t = torch.from_numpy(obs["node_lengths"]).long().to(device)
    worker_pos_t = torch.LongTensor([obs["worker_pos"]]).to(device)
    mask_t = torch.from_numpy(mask[np.newaxis, :]).to(device)
    
    with torch.no_grad():
        _, last_val, _ = model(node_feats_t, token_pad_t, token_len_t, adj_aff_t, worker_pos_t, 
                              base_idx=base_idx, mask=mask_t, node_lengths=node_lengths_t)
    
    last_val = last_val.cpu().numpy().squeeze(0)

    # ========== CPU OPERATIONS: GAE Computation ==========
    rewards = np.array(BATCH["rewards"], dtype=np.float32)
    values = np.array(BATCH["values"] + [last_val], dtype=np.float32)
    dones = np.array(BATCH["dones"], dtype=np.bool_)
    advs, returns = agent.compute_gae(rewards, values, dones)
    
    # ========== CPU OPERATIONS: Pack Batch as CPU Tensors ==========
    torch_batch = {
        "obs_node_feats": torch.from_numpy(np.array(BATCH["obs_node_feats"], dtype=np.float32)),
        "obs_token_pad": torch.from_numpy(np.array(BATCH["obs_token_pad"], dtype=np.float32)),
        "obs_token_lengths": torch.from_numpy(np.array(BATCH["obs_token_lengths"], dtype=np.int64)),
        "obs_graph_id": torch.from_numpy(np.array(BATCH["obs_graph_id"], dtype=np.int64)),
        "obs_base_idx": torch.from_numpy(np.array(BATCH["obs_base_idx"], dtype=np.int64)),
        "obs_node_lengths": torch.from_numpy(np.array(BATCH["obs_node_lengths"], dtype=np.int64)),
        "obs_worker_pos": torch.from_numpy(np.array(BATCH["obs_worker_pos"], dtype=np.int64)),
        "obs_mask": torch.from_numpy(np.array(BATCH["obs_mask"], dtype=np.bool_)),
        "actions": torch.from_numpy(np.array(BATCH["actions"], dtype=np.int64)),
        "old_log_probs": torch.from_numpy(np.array(BATCH["logps"], dtype=np.float32)),
        "returns": torch.from_numpy(returns.astype(np.float32)),
        "advantages": torch.from_numpy(advs.astype(np.float32)),
        "graph_registry": graph_registry
    }
    
    return torch_batch


def train_on_batch(agent: PPOAgent, batch: dict, epochs: int = 4, minibatch_size: int = 64):
    """
    Train the agent on a collected batch of rollouts using PPO.
    
    Args:
        agent: PPOAgent to train
        batch: Dictionary containing rollout data (from collect_rollout_batch)
        epochs: Number of training epochs over the batch
        minibatch_size: Size of minibatches for training
    """
    agent.ppo_update(batch, clip_eps=agent.clip_eps, epochs=epochs, 
                    batch_size=min(minibatch_size, len(batch["actions"])))


def rollout_and_train(env, agent: PPOAgent, device=None,
                      num_updates=None, num_steps_per_update=None,
                      epochs=None, minibatch_size=None):
    """
    Main training loop: collects rollouts and trains the agent.
    
    Args:
        env: The environment to collect from (CPU operations)
        agent: PPOAgent to train
        device: torch.device or str. If None, auto-detects GPU/CPU (prefers GPU).
        num_updates: Number of training updates (rollout + train cycles)
        num_steps_per_update: Number of steps to collect per update
        epochs: Number of training epochs per batch
        minibatch_size: Size of minibatches for training
    """
    # ========== Device Setup (Once at Start) ==========
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Use config defaults if not provided
    num_updates = num_updates if num_updates is not None else NUM_UPDATES
    num_steps_per_update = num_steps_per_update if num_steps_per_update is not None else NUM_STEPS_PER_UPDATE
    epochs = epochs if epochs is not None else EPOCHS
    minibatch_size = minibatch_size if minibatch_size is not None else MINIBATCH_SIZE
    
    print(f"Using device: {device}")
    print(f"Training config: updates={num_updates}, steps_per_update={num_steps_per_update}, epochs={epochs}, minibatch_size={minibatch_size}")
    
    if not env.train:
        env.train = True
        print("Setting env.train=True for training mode")

    # ========== GPU Setup: Model and Graph Registry ==========
    model = agent.model
    model.to(device)
    model.train()

    graph_registry, _ = build_graph_registry(env.graph_dataset, device=device)

    total_steps = 0
    for update in range(num_updates):
        batch = collect_rollout_batch(
            env=env,
            agent=agent,
            graph_registry=graph_registry,
            num_steps_per_update=num_steps_per_update,
            total_steps_collected=total_steps
        )
        
        total_steps += num_steps_per_update
        
        train_on_batch(
            agent=agent,
            batch=batch,
            epochs=epochs,
            minibatch_size=minibatch_size
        )

        if (update + 1) % 20 == 0:
            agent.save("ppo_gnn_model_latest.pth")
            print(f"[update {update+1}] saved model to ppo_gnn_model_latest.pth")

        print(f"Update {update+1} completed. Total steps so far: {total_steps}")


# -------------------------
# Usage example (train)
# -------------------------
if __name__ == "__main__":
    try:
        from env import GraphSweepEnv
        
        from graph_datasets.complex_topologies import get_graph_dataset as get_complex_topologies
        from graph_datasets.complex_topology import get_graph_dataset as get_complex_topology
        from graph_datasets.curriculum_basic import get_graph_dataset as get_curriculum_basic
        from graph_datasets.curriculum_advanced import get_graph_dataset as get_curriculum_advanced
        from graph_datasets.early_late_tasks import get_graph_dataset as get_early_late_tasks
        from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
        from graph_datasets.multi_room import get_graph_dataset as get_multi_room
        from graph_datasets.multiple_rooms import get_graph_dataset as get_multiple_rooms
        from graph_datasets.priority_focused import get_graph_dataset as get_priority_focused 
        from graph_datasets.priority_mixed import get_graph_dataset as get_priority_mixed     
    

        complex_topologies_graphs = get_complex_topologies()
        complex_topology_graphs = get_complex_topology()
        curriculum_basic_graphs = get_curriculum_basic()
        curriculum_advanced_graphs = get_curriculum_advanced()
        early_late_tasks_graphs = get_early_late_tasks()
        long_gap_graphs = get_long_gaps()
        multi_room_graphs = get_multi_room()
        multiple_rooms_graphs = get_multiple_rooms()
        priority_focused_graphs = get_priority_focused()
        priority_mixed_graphs = get_priority_mixed()


        # dataset = long_gap_graphs[:5] + multi_room_graphs[:3]
        # dataset = long_gap_graphs + multi_room_graphs + complex_topologies_graphs + complex_topology_graphs + curriculum_basic_graphs + curriculum_advanced_graphs
        dataset = early_late_tasks_graphs
        print(f"Loaded {len(dataset)} graphs from the combined dataset")
        
        env = GraphSweepEnv(dataset, train=True)

    except Exception as e:
        print("Could not import modules. Ensure env module is available.")
        print(f"Error: {e}")
        raise

    node_feat_dim = NODE_FEATURE_DIM
    token_dim = SCHEDULE_TOKEN_DIM
    
    device = get_device()
    print(f"Auto-detected device: {device}")

    model = ActorCritic(node_feat_dim=node_feat_dim, token_dim=token_dim,
                        sched_d_model=SCHED_D_MODEL, node_emb_dim=NODE_EMB_DIM, 
                        fused_dim=FUSED_DIM, gnn_dim=GNN_DIM, dropout=DROPOUT)

    agent = PPOAgent(model=model, lr=LEARNING_RATE, ent_coef=ENTROPY_COEF)

    agent.load(path = "ppo_gnn_model_latest.pth") # ---------------------------------------------------------------------------------------
    print("Model loaded successfully!")

    rollout_and_train(env, agent, device=device, num_updates=NUM_UPDATES,
                      num_steps_per_update=NUM_STEPS_PER_UPDATE, epochs=EPOCHS, minibatch_size=MINIBATCH_SIZE)

