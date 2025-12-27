# Graph-Based Reinforcement Learning for Optimal Workforce Scheduling

## Project Overview

This project implements a **Graph Neural Network (GNN) + Proximal Policy Optimization (PPO)** based Reinforcement Learning system to solve a **realistic workforce scheduling and routing problem**.

The core problem is to **optimally guide a worker across multiple locations (nodes)** to complete scheduled tasks (classroom cleaning) **within strict time constraints**, while:
- Minimizing travel time
- Respecting task priorities and deadlines
- Allowing intelligent waiting and resting decisions
- Maximizing feasibility and efficiency of the overall schedule

The environment is modeled as a **weighted graph**, and the agent learns to make sequential decisions under uncertainty using **deep reinforcement learning**.

---

## Problem Statement

We are given:

- A **graph of locations**
  - Nodes represent physical locations (e.g., buildings or floors)
  - Edges represent walkable paths with travel times
- A **set of time-scheduled tasks** at each location
  - Each task has:
    - Scheduled time
    - Priority level
- A **single worker**
  - Starts at a base location
  - Has a fixed shift duration
  - Can move, clean, wait, or rest

### Objective

> **Complete all tasks on time (or as close as possible) while minimizing travel and idle penalties, and maximizing rest when no work is feasible.**

---

## Key Challenges Addressed

This problem is non-trivial due to:

- Variable number of tasks per location
- Time-dependent constraints
- Trade-offs between traveling now vs waiting
- Graph-structured spatial dependencies
- Dynamic state updates after every action

Traditional shortest-path or greedy heuristics fail to handle:
- Time windows
- Task priorities
- Sequential decision-making under uncertainty

---

## Solution Architecture

### 1. Graph Representation

Each environment instance is represented as a graph:

- **Nodes** → Locations
- **Edges** → Travel distances (weighted)
- **Precomputed shortest paths** → Efficient routing using Dijkstra

GraphData
├── adjacency matrix
├── shortest path matrix
├── base location
└── classroom schedules per node


---

### 2. Environment Design

A custom `gym.Env` called `GraphSweepEnv` is implemented.

#### Action Space

For a graph with `N` locations:

| Action Index |           Meaning           |
|--------------|-----------------------------|
| `0 … N-1`    | Go to location and clean a task |
| `N`          | Return to base              |
| `N+1`        | Rest at base                |
| `N+2`        | Wait at current location    |

Actions are **masked dynamically** based on feasibility and time constraints.

---

### 3. State Representation

Each environment step returns a rich observation:

#### Node Features (Per Location)

Examples:
- Time to next scheduled task
- Task priority
- Number of tasks in next time window
- Remaining tasks
- Distance to base
- Distance from worker

Shape: (B, N, node_feature_dim)

Where:
- `B` = batch size (number of time steps or rollout samples)
- `N` = number of graph nodes (locations)

---

#### Schedule Tokens (Attention-Based Encoding)

Each node contains a **variable-length sequence** of upcoming tasks:  (time_delta, overdue_time, priority)


These sequences are processed using **self-attention with a CLS token**, producing a fixed-size schedule embedding per node.

---

### 4. Neural Network Model

A **single Actor-Critic network** is used.

#### Model Flow

Node Features
↓
Node Embedding MLP
↓
Schedule Attention (CLS token)
↓
Concatenation
↓
Fusion MLP
↓
GNN Message Passing (2 layers)
↓
Final Node Embeddings


#### Action Handling

Special actions (`WAIT`, `REST`, `RETURN_TO_BASE`) are handled using:
- Trainable action embeddings
- Context-aware fusion with node embeddings

Final output:
- Policy logits for all actions
- Value function estimate

---

## Reinforcement Learning Algorithm

### Proximal Policy Optimization (PPO)

The agent is trained using PPO with:

- **Generalized Advantage Estimation (GAE)**
- **Clipped policy updates**
- **Value function loss**
- **Entropy regularization**

The PPO objective ensures **stable and safe policy updates**, avoiding destructive changes during learning.

---

## Why This Approach Matters

### Real-World Relevance

This problem closely matches real-world challenges such as:

- Facility management scheduling
- Logistics and delivery routing
- Warehouse task allocation
- Technician dispatch planning
- Security patrol routing
- Robotics task sequencing


### Industry Usage

Similar graph-based RL approaches are used by:

- **Amazon** – warehouse routing and picking optimization
- **Google** – fleet and logistics planning
- **Uber** – driver dispatch and routing
- **FedEx / DHL** – delivery optimization
- **Tesla** – robot and fleet coordination

---

## Why GNN + Reinforcement Learning?

| Traditional Methods | This Approach |
|--------------------|---------------|
| Static heuristics | Learns adaptive strategies |
| Shortest-path only | Time-aware decisions |
| Rule-based | Policy learned from experience |
| Poor generalization | Works across unseen graphs |

---

## Project Structure
    ├── graph_data.py # Graph and dataset definitions
    ├── environment.py # Custom RL environment
    ├── model.py # GNN + Attention Actor-Critic
    ├── ppo_agent.py # PPO update and GAE
    ├── train.py # Training loop
    ├── utils.py # Masking and helpers
    └── README.md


---

## Implemented Features

- Custom graph dataset
- Time-aware reward shaping
- Dynamic action masking
- Attention over variable schedules
- Graph neural network message passing
- PPO with GAE
- Cached shortest paths
- Explicit REST and WAIT behaviors

---

## Future Extensions

- Multi-agent PPO (multiple workers)
- Curriculum learning (easy → complex graphs)
- Stochastic task durations
- Energy and fatigue modeling
- Hierarchical RL (planner + executor)

---

## Summary

This project demonstrates how **deep reinforcement learning combined with graph neural networks** can solve **complex, constraint-heavy planning problems** that are common in industry.

It is designed to be:
- Research-grade
- Scalable
- Extensible
- Aligned with real-world decision-making systems

---

