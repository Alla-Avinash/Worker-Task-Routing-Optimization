"""
Model Testing Script
Loads a trained PPO model and tests it on a sample graph with simplified output.

This script:
1. Loads saved model weights
2. Creates a test graph (same format as rl_env2.py example)
3. Runs the model in evaluation mode
4. Shows action history with action names and timestamps
5. Optionally prints detailed logits for specified steps
"""

from typing import Any, List, Optional
import numpy as np
import torch
from env import MAX_STEPS_PER_EPISODE, GraphData, GraphSweepEnv
from model import (
    ActorCritic, get_device, masked_softmax, build_graph_registry,
    NODE_MAX_LENGTH, SCHEDULE_TOKENS_MAX_LENGTH
)
from config import (
    NODE_FEATURE_DIM,
    SCHEDULE_TOKEN_DIM,
    NODE_EMB_DIM,
    SCHED_D_MODEL,
    FUSED_DIM,
    GNN_DIM,
    DROPOUT,
)


def load_trained_model(model_path: str, device=None):
    """
    Load a trained model from saved weights.
    
    Args:
        model_path: Path to the saved model file (.pth)
        device: torch.device or str. If None, auto-detects GPU/CPU.
    
    Returns:
        model: ActorCritic model with loaded weights
        device: The device the model is on
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Create model with same architecture as training
    node_feat_dim = NODE_FEATURE_DIM
    token_dim = SCHEDULE_TOKEN_DIM
    
    model = ActorCritic(
        node_feat_dim=node_feat_dim,
        token_dim=token_dim,
        sched_d_model=SCHED_D_MODEL,
        node_emb_dim=NODE_EMB_DIM,
        fused_dim=FUSED_DIM,
        gnn_dim=GNN_DIM,
        dropout=0.0  # Disable dropout for evaluation
    )
    
    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    # Move model to device
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, device


def print_logits_for_step(step_num, logits, probs, value, mask, actual_num_nodes):
    """
    Print detailed model predictions for a specific step.
    
    Args:
        step_num: Step number
        logits: Action logits tensor
        probs: Action probabilities tensor
        value: State value
        mask: Action mask
        actual_num_nodes: Actual number of nodes in the graph
    """
    print(f"\n{'='*60}")
    print(f"STEP {step_num} - ACTION LOGITS")
    print(f"{'='*60}")
    
    # Move to CPU for printing
    logits_cpu = logits.cpu().numpy().squeeze(0)
    probs_cpu = probs.cpu().numpy().squeeze(0)
    value_cpu = value.cpu().item()
    mask_cpu = mask.cpu().numpy().squeeze(0) if isinstance(mask, torch.Tensor) else mask
    
    print(f"Predicted Value (V(s)): {value_cpu:.4f}")
    print(f"\nAction Logits and Probabilities:")
    for i, (logit, prob) in enumerate(zip(logits_cpu, probs_cpu)):
        action_name = get_action_name(i, actual_num_nodes)
        mask_status = "✓" if mask_cpu[i] else "✗"
        print(f"  {i:2d} [{action_name:15s}] Logit: {logit:8.4f}  Prob: {prob:6.4f}  {mask_status}")
    
    # Show top allowed actions
    allowed_mask = mask_cpu
    allowed_probs = probs_cpu[allowed_mask]
    allowed_indices = np.where(allowed_mask)[0]
    
    if len(allowed_probs) > 0:
        top_k = min(5, len(allowed_probs))
        top_indices = np.argsort(allowed_probs)[-top_k:][::-1]
        print(f"\nTop {top_k} Allowed Actions:")
        for rank, idx in enumerate(top_indices, 1):
            action_idx = allowed_indices[idx]
            action_name = get_action_name(action_idx, actual_num_nodes)
            print(f"  {rank}. Action {action_idx:2d} [{action_name:15s}] - Prob: {allowed_probs[idx]:.4f}")


def get_action_name(action_idx, actual_num_nodes):
    """
    Get human-readable action name.
    
    Args:
        action_idx: The action index (0 to actual_num_nodes+2)
        actual_num_nodes: The actual number of nodes in the graph (not padded length)
    
    Action space:
    - 0 to (actual_num_nodes-1): GO_TO_NODE_X
    - actual_num_nodes: RETURN_TO_BASE
    - actual_num_nodes+1: REST
    - actual_num_nodes+2: WAIT
    """
    if action_idx < actual_num_nodes:
        return f"GO_TO_NODE_{action_idx}"
    elif action_idx == actual_num_nodes:
        return "RETURN_TO_BASE"
    elif action_idx == actual_num_nodes + 1:
        return "REST"
    elif action_idx == actual_num_nodes + 2:
        return "WAIT"
    else:
        return "UNKNOWN"


def test_model_on_graph(model, device, test_graph, num_episodes=1, max_steps=100, show_logits_for_steps: Optional[List[int]] = None):
    """
    Test the trained model on a sample graph with simplified output.
    
    Args:
        model: ActorCritic model with loaded weights
        device: torch.device where model is located
        test_graph: GraphData instance to test on
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        show_logits_for_steps: List of step numbers to print detailed logits for (e.g., [1, 5, 10])
    """
    if show_logits_for_steps is None:
        show_logits_for_steps = []
    
    print(f"\n{'='*60}")
    print("MODEL TESTING")
    print(f"{'='*60}")
    print(f"Test Graph: {test_graph.num_nodes} nodes, Base Node: {test_graph.base}")
    print(f"Episodes: {num_episodes}, Max Steps: {max_steps}")
    if show_logits_for_steps:
        print(f"Will show logits for steps: {show_logits_for_steps}")
    
    # Create environment with test graph
    dataset = [test_graph]
    env = GraphSweepEnv(dataset, train=False)
    
    # Build graph registry for model inference
    graph_registry, _ = build_graph_registry(dataset, device=device)
    
    # Ensure model is in eval mode
    model.eval()
    
    episode_stats = []
    
    for episode in range(num_episodes):
        print(f"\n{'#'*60}")
        print(f"EPISODE {episode + 1}")
        print(f"{'#'*60}")
        
        obs, mask = env.reset()
        episode_reward = 0.0
        step_num = 0
        action_history = []  # (step_num, action_name, time, reward)
        final_info = {}  # Store final info when episode ends
        final_time = obs['current_time']  # Initialize with start time
        
        while step_num < max_steps:
            step_num += 1
            current_time = obs['current_time']
            
            # Get graph-level data from registry
            graph_id = obs["graph_id"]
            graph_data = graph_registry[graph_id]
            base_idx = graph_data["base_idx"]
            adj_aff_t = graph_data["adj_affinity"].unsqueeze(0)
            
            # Convert observations to tensors (handle both padded and unpadded)
            if env.train:
                node_feats_t = torch.from_numpy(obs["node_feats"]).unsqueeze(0).float().to(device)
                token_pad_t = torch.from_numpy(obs["schedule_tokens"]).unsqueeze(0).float().to(device)
                token_len_t = torch.from_numpy(obs["token_lengths"]).unsqueeze(0).long().to(device)
                node_lengths_t = torch.from_numpy(obs["node_lengths"]).long().to(device)
            else:
                # Unpadded format - need to pad manually for model
                node_feats = obs["node_feats"]
                N = node_feats.shape[0]
                node_feat_dim = node_feats.shape[1]
                node_feats_padded = np.zeros((NODE_MAX_LENGTH, node_feat_dim), dtype=np.float32)
                node_feats_padded[:N, :] = node_feats
                node_feats_t = torch.from_numpy(node_feats_padded).unsqueeze(0).float().to(device)
                node_lengths_t = torch.tensor([N], device=device).long()
                
                # Pad schedule tokens
                schedule_tokens = obs["schedule_tokens"]
                token_dim = SCHEDULE_TOKEN_DIM
                token_pad = np.zeros((NODE_MAX_LENGTH, SCHEDULE_TOKENS_MAX_LENGTH, token_dim), dtype=np.float32)
                token_len = np.zeros((NODE_MAX_LENGTH,), dtype=np.int64)
                for n in range(N):
                    tokens = schedule_tokens[n]
                    if tokens is not None and len(tokens) > 0:
                        k = min(len(tokens), SCHEDULE_TOKENS_MAX_LENGTH)
                        token_pad[n, :k, :] = tokens[:k]
                        token_len[n] = k
                token_pad_t = torch.from_numpy(token_pad).unsqueeze(0).float().to(device)
                token_len_t = torch.from_numpy(token_len).unsqueeze(0).long().to(device)
                
                # Pad mask
                mask_padded = np.zeros(NODE_MAX_LENGTH + 3, dtype=np.bool_)
                mask_padded[:len(mask)] = mask
                mask = mask_padded
            
            worker_pos_t = torch.LongTensor([obs["worker_pos"]]).to(device)
            mask_t = torch.from_numpy(mask[np.newaxis, :]).to(device)
            
            # Model forward pass
            with torch.no_grad():
                logits, value, extra = model(
                    node_feats_t, token_pad_t, token_len_t, adj_aff_t,
                    worker_pos_t, base_idx=base_idx, mask=mask_t,
                    node_lengths=node_lengths_t
                )
            
            # Compute probabilities
            probs = masked_softmax(logits, mask_t, dim=-1)
            
            # Get actual number of nodes from observation
            if 'node_lengths' in obs:
                if isinstance(obs['node_lengths'], np.ndarray):
                    actual_num_nodes = int(obs['node_lengths'][0])
                elif isinstance(obs['node_lengths'], list):
                    actual_num_nodes = int(obs['node_lengths'][0])
                else:
                    actual_num_nodes = int(obs['node_lengths'])
            elif 'node_feats' in obs:
                actual_num_nodes = obs['node_feats'].shape[0]
            else:
                actual_num_nodes = test_graph.num_nodes
            
            # Print logits if this step is requested
            if step_num in show_logits_for_steps:
                print_logits_for_step(step_num, logits, probs, value, mask_t, actual_num_nodes)
            
            # Sample action (deterministic: choose highest probability action)
            probs_cpu = probs.cpu().numpy().squeeze(0)
            allowed_mask = mask_t.cpu().numpy().squeeze(0)
            allowed_probs = probs_cpu.copy()
            allowed_probs[~allowed_mask] = -np.inf
            action = int(np.argmax(allowed_probs))
            
            action_name = get_action_name(action, int(actual_num_nodes))
            
            # Step environment
            obs_next, reward, done, info, mask_next = env.step(int(action))
            episode_reward += reward
            
            # Store action history with time
            action_history.append((step_num, action_name, current_time, reward))
            
            # Store info if episode is done
            if done:
                final_info = info.copy()
                final_time = obs_next['current_time']
                break
            
            # Update for next iteration
            obs, mask = obs_next, mask_next
        
        # Get final time (if not already captured, get from current obs)
        if not done:
            final_time = obs['current_time']
        
        # Check if episode was truncated (max steps reached without done=True)
        was_truncated = (step_num >= max_steps) and not done
        
        # Print action history
        print(f"\n{'='*60}")
        print("ACTION HISTORY")
        print(f"{'='*60}")
        print(f"{'Step':<8} {'Time':<10} {'Action':<20} {'Reward':<10}")
        print("-" * 60)
        for step, act_name, time, rew in action_history:
            print(f"{step:<8} {time:<10.2f} {act_name:<20} {rew:<10.2f}")
        
        episode_stats.append({
            "episode": episode + 1,
            "steps": step_num,
            "reward": episode_reward,
            "final_time": final_time,
            "done": done,
            "truncated": was_truncated,
            "info": final_info,
            "actions": action_history
        })
    
    # Print summary
    print(f"\n{'#'*60}")
    print("TEST SUMMARY")
    print(f"{'#'*60}")
    for stat in episode_stats:
        print(f"\nEpisode {stat['episode']}:")
        print(f"  Steps: {stat['steps']}")
        print(f"  Final Time: {stat['final_time']:.2f}")
        print(f"  Total Reward: {stat['reward']:.4f}")
        
        # Determine status and reason
        if stat['done']:
            if stat['info'].get('finished_all', False):
                status = "COMPLETED"
                reason = "All tasks finished successfully"
            elif stat['info'].get('failed_time', False):
                status = "TERMINATED"
                reason = "Shift time exceeded (failed_time)"
            elif stat['info'].get('max_steps', False):
                status = "TERMINATED"
                reason = "Maximum steps reached (max_steps)"
            else:
                status = "TERMINATED"
                reason = "Episode ended (unknown reason)"
        elif stat['truncated']:
            status = "TRUNCATED"
            reason = "Maximum steps reached (episode not done)"
        else:
            status = "UNKNOWN"
            reason = "Episode status unclear"
        
        print(f"  Status: {status}")
        print(f"  Reason: {reason}")
    
    if len(episode_stats) > 1:
        avg_reward = np.mean([s["reward"] for s in episode_stats])
        avg_steps = np.mean([s["steps"] for s in episode_stats])
        completed_count = sum(1 for s in episode_stats if s['done'] and s['info'].get('finished_all', False))
        terminated_count = sum(1 for s in episode_stats if s['done'] and not s['info'].get('finished_all', False))
        truncated_count = sum(1 for s in episode_stats if s['truncated'])
        
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Completed: {completed_count}/{len(episode_stats)}")
        print(f"Terminated: {terminated_count}/{len(episode_stats)}")
        print(f"Truncated: {truncated_count}/{len(episode_stats)}")


if __name__ == "__main__":

    # Configuration
    MODEL_PATH = "ppo_gnn_model_latest.pth"  # Path to saved model
    NUM_EPISODES = 1  # Number of episodes to test
    MAX_STEPS_PER_EPISODE = 100  # Maximum steps per episode

    SHOW_LOGITS_FOR_STEPS = [27,37]  # Specify step numbers to show logits for, e.g., [1, 5, 10]
    
    # Create test graph (same format as rl_env2.py example)
    test_graph = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],           # 6min and 12min travel times
        base=0,
        classroom_schedules={
            0: [
                {"time": 8.0, "priority": 0, "cleaning_time": 2.0},
                {"time": 13.0, "priority": 1, "cleaning_time": 2.0},
            ],
            1: [
                {"time": 10.0, "priority": 0, "cleaning_time": 1.0},
            ],
            2: [
                {"time": 16.0, "priority": 1, "cleaning_time": 2.0},
            ],
        },
    )

    
    try:
        # Load trained model
        model, device = load_trained_model(MODEL_PATH)
        
        # Test model on graph
        test_model_on_graph(
            model=model,
            device=device,
            test_graph=test_graph,
            num_episodes=NUM_EPISODES,
            max_steps=MAX_STEPS_PER_EPISODE,
            show_logits_for_steps=SHOW_LOGITS_FOR_STEPS
        )
        


    except FileNotFoundError:
        print(f"\nERROR: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first or check the model path.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

