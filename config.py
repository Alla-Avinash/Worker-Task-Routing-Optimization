"""
Central Configuration File
Contains all trainable parameters, reward structure, and hyperparameters.
This file can be overridden by experiment-specific configs.
"""

import os
import importlib.util

# ==========================================================================
# EXPERIMENT CONFIGURATION
# ==========================================================================
# Set this to the experiment name to load experiment-specific config
# Example: EXPERIMENT_NAME = "exp_001" will load experiments/exp_001/config.py
EXPERIMENT_NAME = None  # Set to None to use default config

# ==========================================================================
# MODEL ARCHITECTURE CONFIG
# ==========================================================================

NODE_MAX_LENGTH = 10                # maximum number of location nodes in the graph               => maximum number of nodes to pad to when env.train=True
SCHEDULE_TOKENS_MAX_LENGTH = 10     # maximum number of rooms per node to clean in the graph      => maximum number of schedule tokens per node when env.train=True

# Model dimensions
NODE_FEATURE_DIM = 10                # number of node features per location node
SCHEDULE_TOKEN_DIM = 3               # number of features in the schedule tokens =>schedule token dimension

# Model architecture hyperparameters
NODE_EMB_DIM = 32                    # node embedding dimension
SCHED_D_MODEL = 16                   # schedule encoder dimension
FUSED_DIM = 64                       # fused embedding dimension
GNN_DIM = 64                         # GNN layer dimension
HIDDEN_V = 128                       # critic hidden dimension

# ==========================================================================
# TRAINING CONFIG
# ==========================================================================

# PPO hyperparameters
LEARNING_RATE = 3e-4
CLIP_EPS = 0.2               # PPO clipping epsilon
VALUE_COEF = 0.5             # value loss coefficient
ENTROPY_COEF = 0.1           # entropy regularization coefficient (higher = more exploration)
GAMMA = 0.99                 # discount factor
LAM = 0.98                   # GAE lambda

DROPOUT = 0.1                # dropout rate

# Training schedule
NUM_UPDATES = 80              # number of training updates
NUM_STEPS_PER_UPDATE = 500    # steps to collect per update
EPOCHS = 3                    # number of training epochs per batch
MINIBATCH_SIZE = 128          # minibatch size for training

# Exploration
RANDOM_WARMUP_STEPS = 3000    # number of steps to use random actions at start
EPSILON_GREEDY = 0.05         # probability of taking random action during training

# Gradient clipping
MAX_GRAD_NORM = 0.5           # maximum gradient norm for clipping during training



# ==========================================================================
# ENVIRONMENT CONFIG
# ==========================================================================

# Time settings
NEXT_X_HOURS = 3.0                              # look-ahead window for schedule attention
SHIFT_END_TIME = 22.0                           # end of shift time limit (24hr format)
SHIFT_START_TIME = 7.50                          # start of shift time (24hr format)
SHIFT_TIME = SHIFT_END_TIME - SHIFT_START_TIME  # total shift time

# Action time constants
WAIT_IN_NODE = 0.10           # time spent when waiting in a node (hours) => 3 minutes
WAIT_TIME = 0.10              # time spent when choosing WAIT action (hours) => 6 minutes
REST_TIME = 0.25              # time spent when choosing REST action (hours) => 15 minutes
CLEANING_TIME = 0.10          # time spent when cleaning a classroom (hours) => 6 minutes

# Reward timing windows
ACCEPTABLE_RANGE = 0.30          # acceptable range to finish cleaning on time (hours)                 => 20 minutes time frame - 15min clearance time window and ~5min cleaning time
LATE_RANGE = 1.50                # time range for late cleaning to still get some reward (hours)       => 1.5 hours time frame
# acceptable range              --> increases when there are mulitple rooms within the same short time frame or when you have muliple rooms with the same scheduled time

# Episode limits
MAX_STEPS_PER_EPISODE = 120     # maximum steps per episode to avoid infinite episodes, value that allows most episodes to complete under normal conditions, with some buffer for exploration

# ==========================================================================
# REWARD STRUCTURE
# ==========================================================================
class RewardConfig:
    """Centralized reward configuration"""
    
    # WAIT action rewards
    #   - when there is no available room
    #       - Positive reward taper: 15.0 - 5.0 (early[max] to later[min] in the shift)
    #   - when there is at least one available room
    #       - Base reward: 10.0
    #       - Penalty per ready task: 2.0   
    WAIT_POSITIVE_REWARD_MAX = 20.0        # early in shift
    WAIT_POSITIVE_REWARD_MIN = 8.0        # late in shift
    WAIT_PENALTY_PER_READY_TASK = -3.0     # multiplied by penalty_multiplier
    
    # REST action rewards
    REST_POSITIVE_REWARD_MAX = 70.0       # early in shift
    REST_POSITIVE_REWARD_MIN = 20.0        # late in shift (via taper)  
    REST_PENALTY_PER_READY_TASK = -2.0     # multiplied by penalty_multiplier
    
    # Penalty multiplier settings
    # penalty multiplier is used to penalize the agent for waiting too long when there are available rooms
    # penalty_multiplier  = (1.0 + (3.0 - 1.0) * time_progress)    => between 1.0 and 3.0, halfway of the shift is 2.0
    # time_progress is a value between 0 and 1, it is the progress of the shift
    # reward/penalty = tapered_reward + (penalty_multiplier * penalty_per_ready_task * ready_tasks)
    PENALTY_MULTIPLIER_MIN = 1.0          # minimum penalty multiplier
    PENALTY_MULTIPLIER_MAX = 3.0          # maximum penalty multiplier (1.0 + 2.0 * time_progress)
    

    # RETURN_TO_BASE action rewards
    RETURN_TO_BASE_REWARD = 1000.0
    
    # Location node action rewards
    # WASTED_ACTION_PENALTY_MIN = -10.0
    WASTED_ACTION_PENALTY_MAX = -30.0
    WASTED_ACTION_PENALTY_TAPER_RANGE = 20.0
    
    # Cleaning rewards (based on timing)
    CLEAN_ON_TIME_REWARD = 120.0          # reward for finishing within acceptable_range
    CLEAN_LATE_REWARD = 60.0              # reward for finishing within late_range
    CLEAN_VERY_LATE_REWARD = 35.0         # reward for finishing beyond late_range but within shift
    
    # Episode completion rewards
    FINISH_ALL_BONUS = 600.0              # bonus for completing all tasks
    FAILED_TIME_PENALTY = -100.0          # penalty for exceeding shift time with rooms remaining
    MAX_STEPS_PENALTY = -50.0             # penalty for hitting max steps
    


# Create reward config instance
REWARD_CONFIG = RewardConfig()

# ==========================================================================
# LOAD EXPERIMENT-SPECIFIC CONFIG (if specified)
# ==========================================================================
def load_experiment_config():
    """
    Load experiment-specific configuration if EXPERIMENT_NAME is set.
    Experiment configs can override any parameter from this file.
    """
    if EXPERIMENT_NAME is None:
        return
    
    exp_config_path = os.path.join("experiments", EXPERIMENT_NAME, "config.py")
    
    if not os.path.exists(exp_config_path):
        print(f"Warning: Experiment config not found at {exp_config_path}")
        print(f"Using default configuration.")
        return
    
    # Load experiment config
    spec = importlib.util.spec_from_file_location("exp_config", exp_config_path)
    exp_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp_config)
    
    # Override global variables with experiment-specific values
    globals_dict = globals()
    for key, value in exp_config.__dict__.items():
        if not key.startswith("_"):
            globals_dict[key] = value
    
    print(f"Loaded experiment config from: {exp_config_path}")


EXPERIMENT_NAME = None  # Set to None to use default config

# Load experiment config on import
load_experiment_config()

