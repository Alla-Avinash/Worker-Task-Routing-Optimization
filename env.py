

import gym
import numpy as np
from torch_geometric.data import Data
import torch
import heapq
import math
import copy
from typing import List, Dict, Tuple, Any
import numpy as np
from gym import spaces
import numpy as np
from copy import deepcopy




# ============================================================
# 1. CLASS TO MAKE YOUR CUSTOM GRAPH DATA
# ============================================================



class GraphData:
    """
    Represents a single graph: static graph information
    - Locations = nodes
    - Edge list + edge weights = distances
    - Most used information for the environment state
        - num_nodes: number of nodes
        - Base location
        - adj_matrix: precomputed adjacency matrix
        - shortest_paths: precomputed all-pairs shortest paths in a matrix format
        - Classroom schedules stored per location
    """

    def __init__(self, num_nodes, edge_list, edge_weights, base, classroom_schedules):
        """
        Parameters:
        - num_nodes: int
        - edge_list: list of tuples (u, v)
        - edge_weights: list of float distances
        - classroom_schedules: dict:
              {
                node_id : [
                    {"time": float, "priority": int, "cleaned": False},                # we don't use the cleaned flag in the environment, just here for reference    
                    {"time": float, "priority": int, "cleaned": False},
                    ...
                ]
              }
        """
        self.num_nodes = num_nodes
        self.edge_list = edge_list
        self.edge_weights = edge_weights

        # Build adjacency matrix
        self.adj_matrix = self.build_adj_matrix()

        self.base = base

        # pack edges for algorithms
        self.edges_weighted = [(u, v, w) for (u, v), w in zip(edge_list, edge_weights)]

        # Precompute all pairs shortest path distances (cached)
        # This is done once per graph initialization to avoid repeated Dijkstra in-step.
        self.shortest_paths = self.all_pairs_shortest_paths(num_nodes, self.edges_weighted)

        # Deepcopy so environment can MODIFY state safely
        self.classroom_schedules = deepcopy(classroom_schedules)

    def build_adj_matrix(self):
        mat = np.full((self.num_nodes, self.num_nodes), np.inf)
        np.fill_diagonal(mat, 0)

        for (u, v), w in zip(self.edge_list, self.edge_weights):
            mat[u, v] = w
            mat[v, u] = w
        return mat
        
    # -----------------------------------------------------
    # Utility: Dijkstra (single-source)
    # -----------------------------------------------------
    def dijkstra(self, n_nodes: int, edges: List[Tuple[int, int, float]], src: int):
        """
        Compute shortest distances from src to all nodes using adjacency list + heap.
        Returns an array of shape (n_nodes,) with distances (np.inf if unreachable).
        """
        adj = [[] for _ in range(n_nodes)]
        for (u, v, w) in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))            # assume undirected

        dist = [math.inf] * n_nodes
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return np.array(dist, dtype=np.float32)


    def all_pairs_shortest_paths(self, n_nodes: int, edges: List[Tuple[int, int, float]]):
        """
        Compute all-pairs shortest paths (returns matrix [n_nodes, n_nodes]).
        Uses Dijkstra from every node (good for sparse graphs and small N).
        """
        mat = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
        for i in range(n_nodes):
            mat[i] = self.dijkstra(n_nodes, edges, i)
        return mat
        
    def copy_schedules(self):
        """Used by environment to get a fresh copy of classroom schedules."""
        return deepcopy(self.classroom_schedules)


# ========================================================================
# 2. DATASET CLASS TO HOLD MULTIPLE GRAPHS
# ========================================================================

class GraphDataset:
    """
    Holds a LIST of GraphData objects.
    Each reset() in the environment can sample a different GraphData.
    Environment will pick one per new episode.
    """

    def __init__(self, graph_data_list):
        self.graphs = graph_data_list

    def __len__(self):
        return len(self.graphs)

    def get(self, index):
        """
        Return a deep copy so environment can modify it safely as a state.
        """
        g = self.graphs[index]
        return GraphData(
            g.num_nodes,
            g.edge_list,
            g.edge_weights,
            g.base,
            deepcopy(g.classroom_schedules)
        )



# ==========================================================================================
#  DEFINING THE CONSTANTS AND HELPER FUNCTIONS FOR THE RL ENVIRONMENT
# ========================================================================================== 

from config import (
    NEXT_X_HOURS,
    SHIFT_END_TIME,
    SHIFT_START_TIME,
    SHIFT_TIME,
    WAIT_TIME,
    REST_TIME,
    WAIT_IN_NODE,
    CLEANING_TIME,
    ACCEPTABLE_RANGE,
    LATE_RANGE,
    NODE_FEATURE_DIM,
    SCHEDULE_TOKEN_DIM,
    MAX_STEPS_PER_EPISODE,
    NODE_MAX_LENGTH,
    SCHEDULE_TOKENS_MAX_LENGTH,
    REWARD_CONFIG,
)


# ===========================================================================================
# PRE PROCESSING THE GRAPH DATA FOR THE ENVIRONMENT
# ===========================================================================================


import bisect

# ----------------------------------------------------------------------
# Helper: filter rooms in time window
# ----------------------------------------------------------------------

def rooms_in_time_window(rooms, start_time, end_time):
    # extract times into a separate list (needed for bisect)
    times = [r["time"] for r in rooms]

    # find the first index where r["time"] > end_time
    right = bisect.bisect_right(times, end_time)

    return rooms[:right]

# ----------------------------------------------------------------------
# Helper: compute node features & schedule tokens
# ----------------------------------------------------------------------

def compute_node_features_and_schedule_tokens(
    graph: GraphData,
    schedules_state: Dict[int, List[Dict[str, Any]]],
    current_time: float,
    worker_pos: int,
    next_hours: float = NEXT_X_HOURS,
):
    """
    Build node-level features

    Returns:
    - node_feats: np.array shape (N, feat_dim)
    
    The node_feats follow this layout:
    [
        time_to_next,
        time_overdue,
        next_priority,
        is_base_location,
        count_xhr,
        count_xhr_priority,
        remaining_classes,
        earliest_deadline (or 0 if no tasks),
        last_room_deadline (or 0 if no tasks),
        distance_to_base,
        distance_to_worker
    ]

    (Can drop/normalize fields later in the model.)
    """

    N = graph.num_nodes
    node_feats = np.zeros((N, NODE_FEATURE_DIM), dtype=np.float32)
    schedule_tokens = [None] * N 

    # get distances from base and worker
    dist_to_base = graph.shortest_paths[:, graph.base].astype(np.float32)  # from each node -> base
    dist_from_worker = graph.shortest_paths[worker_pos].astype(np.float32)  # from worker -> each node

    for node in range(N):
        roomsTotal = schedules_state.get(node, [])  # list of rooms at a particular location

        # filter rooms within the next_hours window
        rooms = rooms_in_time_window(roomsTotal, current_time, current_time + next_hours)

        # filter only future or current rooms (time >= past?), but we keep all remaining rooms
        # compute time-to-next (clamped >= 0)
        if len(roomsTotal) == 0:
            time_to_next = 0.0
            time_overdue = 0.0
            time_to_next_priority = 0.0
            count_xhr = 0
            count_xhr_priority = 0
            remaining = 0
            earliest = 0.0
            last_room_deadline = 0.0
            # for the schedule tokens
            tokens = np.zeros((0, SCHEDULE_TOKEN_DIM), dtype=np.float32)
        else:

            # pick the room with the earliest time among all rooms
            next_room = roomsTotal[0]

            # compute time to that earliest room
            time_to_next = max(next_room["time"] - current_time, 0.0)
            time_overdue = 0.0 if (next_room["time"] - current_time) > 0 else abs(next_room["time"] - current_time)

            # pick the earliest room with prioriy = 1
            for room in roomsTotal:
                if room.get("priority", 0) > 0:
                    time_to_next_priority = float(room.get("time", 0.0))
                else:
                    time_to_next_priority = 0.0

            # next X hours window
            count_xhr = len(rooms)
            count_xhr_priority = sum(int(t.get("priority", 0) > 0) for t in rooms)
            remaining = len(roomsTotal)

            earliest = float(roomsTotal[0]["time"])
            last_room_deadline = float(roomsTotal[-1]["time"])

            """
            schedule tokens: 
                SCHEDULE_TOKEN_DIM = 3  set in the config
                only consider the rooms in the xhr time window
                [
                  time delta 
                  time overdue
                  priority
                ]
            """
            tokens_list = []
            for room in rooms:
                if room["time"] < current_time:
                    # already in past (but remaining because not cleaned) -> treat as immediate (time_to=0)
                    time_delta = 0.0
                    time_overdue = float(current_time - room["time"])
                else:
                    time_delta = float(room["time"] - current_time)
                    time_overdue = 0.0

                tokens_list.append([time_delta, time_overdue, float(room.get("priority", 0))])

            if len(tokens_list) == 0:
                tokens = np.zeros((0, SCHEDULE_TOKEN_DIM), dtype=np.float32)
            else:
                tokens = np.array(tokens_list, dtype=np.float32)

        # build feature vector
        node_feats[node, 0] = float(time_to_next)
        node_feats[node, 1] = float(time_overdue)
        node_feats[node, 2] = float(time_to_next_priority)
        node_feats[node, 3] = 1.0 if node == graph.base else 0.0
        node_feats[node, 4] = float(count_xhr)
        node_feats[node, 5] = float(count_xhr_priority)
        node_feats[node, 6] = float(remaining)
        node_feats[node, 7] = float(earliest)
        node_feats[node, 8] = float(last_room_deadline)

        # distances: use shortest paths (inf if unreachable)
        d_base = dist_to_base[node]
        d_worker = dist_from_worker[node]
        # clamp inf -> large value (so model can learn)
        if not np.isfinite(d_base):
            d_base_val = 1e6
        else:
            d_base_val = float(d_base)
        if not np.isfinite(d_worker):
            d_worker_val = 1e6
        else:
            d_worker_val = float(d_worker)
        node_feats[node, 9] = d_base_val                                   
        # node_feats[node, 10] = d_worker_val                                             # ---------------------------------- change this later you only have the dimnesion of 10 but you specified a dim of 11

        schedule_tokens[node] = tokens

    return node_feats, schedule_tokens



# =================================================================================================================================
# 4.   MAKING THE RL ENVIRONMENT CLASS
# =================================================================================================================================

class GraphSweepEnv(gym.Env):
    """
    Single-worker environment.

    Action space:
      0 .. N-1  => choose location i to work on a specific classroom there
      N         => RETURN_TO_BASE (travel to base location)
      N+1       => REST (only meaningful at base; masked otherwise)   --> When there is a REST action, the worker travels to the base location and then he will take REST for a fixed time block of 15 min
      N+2       => WAIT (stay here for 10 min or x time unit; always allowed)

    Observation (returned as dict):
      {
        "node_feats": np.array (N, feat_dim),
        "schedule_tokens": list length N of np.arrays (k_i, 2)  # for attention processor
        "shortest_paths": np.array (N,N)  # optional helper for model
        "worker_pos": int
        "current_time": float
      }

    The environment will REMOVE cleaned classroom entries from its per-episode schedules_state, everytime the worker cleans a classroom/ an action is taken by the agent.

    """

    def __init__(self, graph_dataset: List[GraphData], shift_end_time: float = SHIFT_END_TIME, shift_start_time: float = SHIFT_START_TIME,
                 next_hours: float = NEXT_X_HOURS, train: bool = False):
        super().__init__()
        assert isinstance(graph_dataset, list) and len(graph_dataset) > 0
        self.graph_dataset = graph_dataset
        self.shift_start_time = float(shift_start_time)
        self.shift_end_time = float(shift_end_time)
        self.shift_time = self.shift_end_time - self.shift_start_time
        self.next_hours = float(next_hours)
        self.train = train  # If True, return padded observations ready for training

        # runtime (set on reset)
        self.current_graph: GraphData = None
        self.current_graph_id: int = None  # dataset index of current graph
        self.schedules_state: Dict[int, List[Dict[str, Any]]] = None
        self.worker_pos: int = 0
        self.current_time: float = 0.0
        self.steps = 0

        # action constants will be set on reset (depends on graph size)
        self.action_space = None
        self.RETURN_TO_BASE_IDX = None
        self.REST_IDX = None
        self.WAIT_IDX = None
        self.N = None

        # We will not set observation_space rigidly (it can be complex). You may set it in your trainer.
        self.observation_space = None

    # -----------------------
    # reset
    # -----------------------
    def reset(self):
        # choose a random graph
        graph_idx = np.random.randint(0, len(self.graph_dataset))
        self.current_graph_id = int(graph_idx)  # store dataset index
        self.current_graph = copy.deepcopy(self.graph_dataset[graph_idx])

        self.N = self.current_graph.num_nodes
        self.RETURN_TO_BASE_IDX = self.N
        self.REST_IDX = self.N + 1
        self.WAIT_IDX = self.N + 2
        # static action space length = N + 3
        self.action_space = spaces.Discrete(self.N + 3)

        self.schedules_state = self.current_graph.copy_schedules()

        # start the worker at base
        self.worker_pos = int(self.current_graph.base)
        self.current_time = SHIFT_START_TIME
        self.steps = 0

        # compute initial node features and schedule tokens
        node_feats, schedule_tokens = compute_node_features_and_schedule_tokens(
            self.current_graph, self.schedules_state, self.current_time, self.worker_pos, self.next_hours
        )

        # Pad if in training mode
        node_feats, node_lengths = self._pad_node_features(node_feats)
        schedule_tokens, token_lengths = self._pad_schedule_tokens(schedule_tokens)

        obs = {
            "node_feats": node_feats,                            # (N, 10) or (NODE_MAX_LENGTH, 10) if train=True
            "schedule_tokens": schedule_tokens,                  # list[N] of arrays or (NODE_MAX_LENGTH, SCHEDULE_TOKENS_MAX_LENGTH, 3) if train=True
            "node_lengths": node_lengths,                        # (1,) - actual number of nodes
            "token_lengths": token_lengths,                      # (N,) or (NODE_MAX_LENGTH,) - tokens per node
            "shortest_paths": self.current_graph.shortest_paths, # --------------------------------------------------- remove this if possible
            "worker_pos": int(self.worker_pos),
            "current_time": float(self.current_time),
            "graph_id": self.current_graph_id,                   # dataset index for graph-level storage
        }

        mask = self._build_mask()                                 # length N+3 or NODE_MAX_LENGTH+3 if train=True
        return obs, mask



    # ------------------------------------------------------
    # internal: compute time-based taper for WAIT/REST/WASTED_ACTION rewards
    # ------------------------------------------------------
    def _get_positive_reward_taper(self, positive_reward_max: float, positive_reward_min: float):
        """
        Returns positive reward that tapers from max (early) to min (late).
        Used when no ready tasks are available.
        Uses time_progress => value between 0 and 1
        """
        time_progress = (self.current_time - self.shift_start_time) / self.shift_time
        time_progress = max(0.0, min(1.0, time_progress))
        taper_range = positive_reward_max - positive_reward_min
        return positive_reward_max - (taper_range * time_progress)
    
    def _get_penalty_multiplier(self, penalty_multiplier_max: float, penalty_multiplier_min: float):
        """
        Returns multiplier for penalties that increases with time.
        Used when ready tasks are available - makes penalties worse over time.
        """
        time_progress = (self.current_time - self.shift_start_time) / self.shift_time
        time_progress = max(0.0, min(1.0, time_progress))
        multiplier_range = penalty_multiplier_max - penalty_multiplier_min
        return penalty_multiplier_min + (multiplier_range * time_progress)
    
    def _get_wasted_action_penalty(self, wasted_action_penalty_max: float, wasted_action_penalty_min: float):
        "This is a reverse taper function for the wasted action penalty, it will return a value that tapers from max (early) to min (late)."
        time_progress = (self.current_time - self.shift_start_time) / self.shift_time
        time_progress = max(0.0, min(1.0, time_progress))
        taper_range = REWARD_CONFIG.WASTED_ACTION_PENALTY_TAPER_RANGE
        return REWARD_CONFIG.WASTED_ACTION_PENALTY_MAX + (taper_range * time_progress)
    
    # ------------------------------------------------------
    # internal: build action mask (True = available)
    # ------------------------------------------------------
    def _build_mask(self):
        """
        Mask length = N+3 (or NODE_MAX_LENGTH+3 if train=True)
        For location i: mask True if there are remaining classes in schedules_state[i]
        RETURN_TO_BASE: always allowed except when the worker is already at base
        REST: allowed True if worker at base (we choose to permit REST only at base)
        WAIT: always allowed
        """
        mask_length = (NODE_MAX_LENGTH + 3) if self.train else (self.N + 3)
        mask = np.zeros(mask_length, dtype=bool)

        # Location availability: True if there are remaining rooms (so allowed)
        for i in range(self.N):
            mask[i] = len(self.schedules_state.get(i, [])) > 0
        
        # Special action indices depend on whether we're in training mode (padded) or not
        if self.train:
            # In training mode, special actions are at the end after padded nodes
            return_to_base_idx = NODE_MAX_LENGTH
            rest_idx = NODE_MAX_LENGTH + 1
            wait_idx = NODE_MAX_LENGTH + 2
        else:
            # In non-training mode, use the actual indices
            return_to_base_idx = self.RETURN_TO_BASE_IDX
            rest_idx = self.REST_IDX
            wait_idx = self.WAIT_IDX
        
        # RETURN_TO_BASE
        mask[return_to_base_idx] = (self.worker_pos != self.current_graph.base)     

        # REST
        mask[rest_idx] = (self.worker_pos == self.current_graph.base)

        # WAIT
        mask[wait_idx] = True

        return mask
    
    # ------------------------------------------------------
    # internal: pad node features for training mode
    # ------------------------------------------------------
    def _pad_node_features(self, node_feats):
        """
        Pad node features to NODE_MAX_LENGTH.
        Returns padded features and actual number of nodes.
        """
        N = node_feats.shape[0]
        node_lengths = np.array([N], dtype=np.int64)
        
        if self.train and N < NODE_MAX_LENGTH:
            padded = np.zeros((NODE_MAX_LENGTH, node_feats.shape[1]), dtype=node_feats.dtype)
            padded[:N, :] = node_feats
            node_feats = padded
        
        return node_feats, node_lengths
    
    # ------------------------------------------------------
    # internal: pad schedule tokens for training mode
    # ------------------------------------------------------
    def _pad_schedule_tokens(self, schedule_tokens):
        """
        Pad schedule tokens to NODE_MAX_LENGTH x SCHEDULE_TOKENS_MAX_LENGTH.
        Returns padded tokens array and token lengths per node.
        """
        N = len(schedule_tokens)
        
        if self.train:
            # Create padded array: (NODE_MAX_LENGTH, SCHEDULE_TOKENS_MAX_LENGTH, SCHEDULE_TOKEN_DIM)
            padded_tokens = np.zeros((NODE_MAX_LENGTH, SCHEDULE_TOKENS_MAX_LENGTH, SCHEDULE_TOKEN_DIM), dtype=np.float32)
            token_lengths = np.zeros(NODE_MAX_LENGTH, dtype=np.int64)
            
            for i in range(min(N, NODE_MAX_LENGTH)):
                tokens = schedule_tokens[i]
                k = min(len(tokens), SCHEDULE_TOKENS_MAX_LENGTH)
                if k > 0:
                    padded_tokens[i, :k, :] = tokens[:k]
                token_lengths[i] = k
            
            return padded_tokens, token_lengths
        else:
            # Return as-is (list of arrays)
            token_lengths = np.array([len(tokens) for tokens in schedule_tokens], dtype=np.int64)
            return schedule_tokens, token_lengths


    # ----------------------------------------------------------------------
    # Internal: pick which classroom within a location to clean
    # ----------------------------------------------------------------------
    def _select_classroom_to_clean(self, location_idx: int):
        """
        Pure software logic:
            First, prefer higher priority classrooms within the next 5-10 minutes if it is feasible or choose any other earliest room
        """
        rooms = self.schedules_state.get(location_idx, [])

        if not rooms:
            return None, None       # nothing to clean

        # Define the 6-minute window
        window_start = self.current_time
        window_end = self.current_time + WAIT_TIME  # 6 minutes

        # Filter rooms that fall within the next 6 minutes
        window_rooms = rooms_in_time_window(rooms, window_start, window_end)

        if not window_rooms:
            return None, None
        
        # Find first Priority rooms if it is present
        for idx, room in enumerate(window_rooms):
            if int(room.get("priority", 0)) > 0:
                return room, idx

        # Find first normal room
        selected_room = window_rooms[0]
        selected_index = 0

        return selected_room, selected_index

    # -----------------------
    # step
    # -----------------------
    def step(self, action: int):
        assert self.action_space is not None,       "Call reset() before step()"
        if self.train:
            # In training mode, actions are padded to NODE_MAX_LENGTH + 3
            assert 0 <= action < (NODE_MAX_LENGTH + 3), f"Action {action} out of range [0, {NODE_MAX_LENGTH + 3}) in training mode"
            # Map padded action indices to actual indices
            if action < self.N:
                # Valid node action
                mapped_action = action
            elif action == NODE_MAX_LENGTH:
                # RETURN_TO_BASE (mapped from padded position)
                mapped_action = self.N
            elif action == NODE_MAX_LENGTH + 1:
                # REST (mapped from padded position)
                mapped_action = self.N + 1
            elif action == NODE_MAX_LENGTH + 2:
                # WAIT (mapped from padded position)
                mapped_action = self.N + 2
            else:
                # Padded node indices (self.N <= action < NODE_MAX_LENGTH) - should be masked out
                # If we get here, the mask failed - treat as invalid
                raise ValueError(f"Invalid padded action {action} - should have been masked out")
            action = mapped_action
        else:
            assert 0 <= action < (self.N + 3)
        self.steps += 1
        """
            for every action:
                update
                    current time
                    travel pos
                    reward/penalty
            update schedules_state if cleaning a classroom
            check terminal conditions after all the update of the state for the actions
            return observation, reward, done, info
        """

        reward = 0.0
        done = False
        info = {}

        previous_time = self.current_time     # numbers are immutable in python so when we change the current time, the self.current_time will not change

        if action == self.WAIT_IDX:
            self.current_time += WAIT_TIME
            
            ready_tasks = 0
            for loc, rooms in self.schedules_state.items():
                for room in rooms:
                    room_time = float(room.get("time", self.current_time))
                    if room_time <= self.current_time:
                        ready_tasks += 1
            
            reward += self._get_positive_reward_taper(REWARD_CONFIG.WAIT_POSITIVE_REWARD_MAX, REWARD_CONFIG.WAIT_POSITIVE_REWARD_MIN)

            if ready_tasks > 0:
                penalty_mult = self._get_penalty_multiplier(REWARD_CONFIG.PENALTY_MULTIPLIER_MAX, REWARD_CONFIG.PENALTY_MULTIPLIER_MIN)
                reward += (REWARD_CONFIG.WAIT_PENALTY_PER_READY_TASK * penalty_mult * ready_tasks)
      
        elif action == self.REST_IDX:
            self.current_time += REST_TIME
            
            ready_tasks = 0
            for loc, rooms in self.schedules_state.items():
                for room in rooms:
                    room_time = float(room.get("time", self.current_time))
                    if room_time <= self.current_time:
                        ready_tasks += 1

            if ready_tasks > 0:
                penalty_mult = self._get_penalty_multiplier()
                reward += REWARD_CONFIG.REST_BASE_REWARD + (-REWARD_CONFIG.REST_PENALTY_PER_READY_TASK * penalty_mult * ready_tasks)
            else:
                reward += self._get_positive_reward_taper(REWARD_CONFIG.REST_POSITIVE_REWARD_MAX, REWARD_CONFIG.REST_POSITIVE_REWARD_MIN)

        elif action == self.RETURN_TO_BASE_IDX:
            base_loc = int(self.current_graph.base)

            travel_t = float(self.current_graph.shortest_paths[self.worker_pos, base_loc])
            self.current_time += travel_t
            
            self.worker_pos = base_loc

            reward += REWARD_CONFIG.RETURN_TO_BASE_REWARD    
        
        else:   # action is a location node
            
            # selecting a location node
            loc = int(action)  
            
            travel_t = float(self.current_graph.shortest_paths[self.worker_pos, loc])
            self.current_time += travel_t
            self.worker_pos = loc

            # reward += -0.1 * travel_t 
            # reward += -0.2 * travel_t        

            # pick which classroom to clean at this loc 
            selected_room, selected_index = self._select_classroom_to_clean(loc)

            if selected_room is None:
                # No rooms at chosen loc -> penalty for wasted action (tapers with time)
                wasted_penalty = self._get_wasted_action_penalty()
                reward += wasted_penalty
                self.current_time += WAIT_IN_NODE
            else:
                # A room is selected, perform cleaning
                
                scheduled_time = float(selected_room.get("time", self.current_time))
                if scheduled_time > self.current_time:
                    # if cleaning before scheduled time, we wait until that time to start cleaning - it will be from 0 - 6 min wait time
                    self.current_time = scheduled_time
                    # reward += -0.05 * (scheduled_time - self.current_time)      #---------------------------------------------------------------------

                self.current_time += CLEANING_TIME

                curr_time = self.current_time
                shift_end_time = self.shift_end_time
                finish_time = scheduled_time + ACCEPTABLE_RANGE
                deadline = curr_time + LATE_RANGE

                if curr_time <= shift_end_time:
                    # reward for finishing at finish_time
                    if (curr_time <= finish_time):
                        reward += REWARD_CONFIG.CLEAN_ON_TIME_REWARD
                    # very less reward for late cleaning within the late range
                    elif curr_time > finish_time and curr_time < deadline:
                        reward += REWARD_CONFIG.CLEAN_LATE_REWARD
                    # no reward for late cleaning beyond late range but within shift time
                    else: 
                        reward += REWARD_CONFIG.CLEAN_VERY_LATE_REWARD
                
                # cost per cleaning time
                # reward += -0.01 * ( CLEANING_TIME )
                # reward += -0.05 * ( CLEANING_TIME )

                # remove the cleaned classroom from schedules_state[loc]
                self.schedules_state[loc].pop(selected_index)
        
        # as time increases, the model should learn that the shift end time is approaching and so it should start cleaning the rooms more efficiently and fastly and not give preference to resting or waiting
        # reward += (-5.0 * (( self.current_time - previous_time)/self.shift_time ) + (-5.0 * ( self.current_time/self.shift_end_time )))
        # something to model the urgency of the task completion and something to remind of the shift end time is approaching
        # reward += (-5.0 * ( self.current_time/self.shift_end_time )) # problem with this is that it does not take into consideration the time spent in this action

        # check terminal conditions
        remaining = sum(len(v) for v in self.schedules_state.values())
        if remaining == 0:
            done = True
            reward += REWARD_CONFIG.FINISH_ALL_BONUS
            info["finished_all"] = True
        elif self.current_time > self.shift_end_time:
            done = True
            reward += REWARD_CONFIG.FAILED_TIME_PENALTY
            info["failed_time"] = True

        # safety max steps
        if self.steps >= MAX_STEPS_PER_EPISODE:
            done = True
            reward += REWARD_CONFIG.MAX_STEPS_PENALTY
            info["max_steps"] = True

        # update observation
        node_feats, schedule_tokens = compute_node_features_and_schedule_tokens(
            self.current_graph, self.schedules_state, self.current_time, self.worker_pos, self.next_hours
        )
        
        # Pad if in training mode
        node_feats, node_lengths = self._pad_node_features(node_feats)
        schedule_tokens, token_lengths = self._pad_schedule_tokens(schedule_tokens)
        
        obs = {
            "node_feats": node_feats,                            # (N, 10) or (NODE_MAX_LENGTH, 10) if train=True
            "schedule_tokens": schedule_tokens,                  # list[N] of arrays or (NODE_MAX_LENGTH, SCHEDULE_TOKENS_MAX_LENGTH, 3) if train=True
            "node_lengths": node_lengths,                        # (1,) - actual number of nodes
            "token_lengths": token_lengths,                      # (N,) or (NODE_MAX_LENGTH,) - tokens per node
            "shortest_paths": self.current_graph.shortest_paths,
            "worker_pos": int(self.worker_pos),
            "current_time": float(self.current_time),
            "graph_id": self.current_graph_id,  # dataset index for graph-level storage
        }
        mask = self._build_mask()

        return obs, float(reward), bool(done), info, mask



# -------------------------------------
# Example usage
# -------------------------------------

if __name__ == "__main__":
    # build a sample GraphData (user-supplied format)
    g1 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],        # 6min and 12min travel times
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

    """
    g1 - contains the properties of GraphData
        num_nodes
        base_node
        adj_matrix
        shortest_paths_matrix
        classroom_schedules for all the nodes
    """
    dataset = [g1]
    env = GraphSweepEnv(dataset, shift_end_time=20.0, next_hours=3.0)



    # initialize environment --------------------------------------------------------
    obs, mask = env.reset()

    step_num = 0
    total_reward = 0.0

    print("================ Episode start ===================")

    while True:
        step_num += 1
        
        # find allowed actions using mask
        allowed_actions = np.where(mask)[0]
        """
            mask = np.array([True, False, True, True, False])
            np.where(mask)   => Give me the indices where mask is True/1
            output of np.where(mask) =>  (array([0, 2, 3]),)  Since mask is 1-D, the tuple has one element
        """

        # if no actions available → failure
        if len(allowed_actions) == 0:
            print(f"\n No allowed actions at step {step_num}. Episode not terminated normally; Episode failed.")
            break

        # choose a random action
        action = np.random.choice(allowed_actions)

        # apply step
        obs, reward, done, info, mask = env.step(action)
        total_reward += reward

        # print step info
        print(f"\nStep {step_num}")
        print(f"  Action chosen: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  New time: {obs['current_time']:.2f}")
        print(f"  Info: {info}")
        print(f"  Done: {done}")
        # print(obs)
        # print(mask)

        # stop if environment signals termination
        if done:
            print("\n Episode terminated normally.")
            print("total reward = ",total_reward )
            break
    

    # code to visualize -------------------------------------------------------
    # from animator import EnvAnimator

    # env = GraphSweepEnv(dataset)
    # obs, mask = env.reset()

    # def rollout():
    #     done = False
    #     while not done:
    #         action = env.action_space.sample()
    #         next_obs, reward, done, info, mask = env.step(action)
    #         yield (next_obs, reward, done, info, mask)

    # anim = EnvAnimator(env, interval=700)
    # anim.animate(rollout())




    
