# Graph Datasets for Curriculum Learning

This directory contains various graph datasets organized by learning objectives. Each dataset is designed to teach specific behaviors to the RL agent, enabling curriculum learning where you can progressively increase difficulty.

## Dataset Files

### 1. `single_room.py`
**Purpose**: Simplest graphs with single cleaning task per graph
- **Learning Objective**: Basic task completion, timing awareness
- **Characteristics**: 
  - 3-node graphs (base + hallway + classroom)
  - One task per graph
  - Simple linear paths
- **Use Case**: Initial training, foundational skills

### 2. `long_gaps.py` ⭐ **REST ACTION FOCUS**
**Purpose**: Graphs with significant time gaps between tasks (6-14 hour gaps)
- **Learning Objective**: Learn when to rest during idle periods
- **Characteristics**:
  - Tasks at extreme times (early morning, late evening)
  - Long midday gaps (8-14 hours between tasks)
  - Encourages rest actions during waiting periods
- **Use Case**: Teaching rest behavior, time management across extended periods
- **Example Gap Patterns**: 
  - Early morning (7-8 AM) → Evening (6-8 PM)
  - Morning (9 AM) → Late night (9-10 PM)

### 3. `multi_room.py`
**Purpose**: Multiple rooms requiring cleaning in sequence
- **Learning Objective**: Route planning, sequential task handling
- **Characteristics**:
  - 2-5 cleaning tasks per graph
  - Tasks require optimal ordering
  - Various topologies (linear, star, branching)
- **Use Case**: Multi-task planning, path optimization

### 4. `priority_mixed.py`
**Purpose**: Mix of high and low priority tasks
- **Learning Objective**: Priority-based decision making
- **Characteristics**:
  - Priority conflicts (temporal, spatial)
  - Tasks requiring priority-aware ordering
  - Trade-offs between convenience and urgency
- **Use Case**: Teaching priority awareness, conflict resolution

### 5. `complex_topology.py`
**Purpose**: Complex graph structures (cycles, multiple paths, meshes)
- **Learning Objective**: Path optimization in complex networks
- **Characteristics**:
  - Multiple paths between nodes
  - Cycles and redundant connections
  - Non-linear topologies (grids, meshes, trees)
- **Use Case**: Advanced path finding, handling complex layouts

### 6. `early_late_tasks.py`
**Purpose**: Tasks spread throughout the day (full day coverage)
- **Learning Objective**: Time-based planning across full day
- **Characteristics**:
  - Early morning (6-8 AM)
  - Midday (12-1 PM)
  - Afternoon (3-5 PM)
  - Evening (6-8 PM)
- **Use Case**: Full-day scheduling, varied timing patterns

## Usage Guide

### Basic Import

```python
from graph_datasets.single_room import get_graph_dataset as get_single_room
from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
from graph_datasets.multi_room import get_graph_dataset as get_multi_room
from graph_datasets.priority_mixed import get_graph_dataset as get_priority_mixed
from graph_datasets.complex_topology import get_graph_dataset as get_complex_topology
from graph_datasets.early_late_tasks import get_graph_dataset as get_early_late

# Get dataset
single_room_graphs = get_single_room()
long_gaps_graphs = get_long_gaps()
```

### Mixing Datasets (Proportions)

Create custom training sets with different proportions:

```python
from graph_datasets.single_room import get_graph_dataset as get_single_room
from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
from graph_datasets.multi_room import get_graph_dataset as get_multi_room

def mix_datasets(proportions):
    """
    Mix datasets according to specified proportions.
    
    Args:
        proportions: dict with dataset names and counts
        Example: {
            'single_room': 5,
            'long_gaps': 3,
            'multi_room': 2
        }
    
    Returns:
        List of GraphData objects
    """
    from graph_datasets.single_room import get_graph_dataset as get_single_room
    from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
    from graph_datasets.multi_room import get_graph_dataset as get_multi_room
    from graph_datasets.priority_mixed import get_graph_dataset as get_priority_mixed
    from graph_datasets.complex_topology import get_graph_dataset as get_complex_topology
    from graph_datasets.early_late_tasks import get_graph_dataset as get_early_late
    
    all_graphs = []
    
    dataset_funcs = {
        'single_room': get_single_room,
        'long_gaps': get_long_gaps,
        'multi_room': get_multi_room,
        'priority_mixed': get_priority_mixed,
        'complex_topology': get_complex_topology,
        'early_late_tasks': get_early_late,
    }
    
    for dataset_name, count in proportions.items():
        if dataset_name in dataset_funcs:
            graphs = dataset_funcs[dataset_name]()
            all_graphs.extend(graphs[:min(count, len(graphs))])
    
    return all_graphs

# Usage
training_set = mix_datasets({
    'single_room': 5,
    'long_gaps': 3,
    'multi_room': 2
})
```

### Weighted Mixing (Percentages)

```python
import random

def mix_datasets_weighted(weights):
    """
    Mix datasets using percentage weights.
    
    Args:
        weights: dict with dataset names and percentages (must sum to 1.0)
        Example: {
            'single_room': 0.5,  # 50%
            'long_gaps': 0.3,     # 30%
            'multi_room': 0.2     # 20%
        }
    
    Returns:
        List of GraphData objects
    """
    import random
    from graph_datasets.single_room import get_graph_dataset as get_single_room
    from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
    from graph_datasets.multi_room import get_graph_dataset as get_multi_room
    from graph_datasets.priority_mixed import get_graph_dataset as get_priority_mixed
    from graph_datasets.complex_topology import get_graph_dataset as get_complex_topology
    from graph_datasets.early_late_tasks import get_graph_dataset as get_early_late
    
    datasets = {
        'single_room': get_single_room(),
        'long_gaps': get_long_gaps(),
        'multi_room': get_multi_room(),
        'priority_mixed': get_priority_mixed(),
        'complex_topology': get_complex_topology(),
        'early_late_tasks': get_early_late(),
    }
    
    all_graphs = []
    total_count = 100  # or any desired total
    
    for dataset_name, weight in weights.items():
        if dataset_name in datasets:
            count = int(total_count * weight)
            graphs = datasets[dataset_name]
            all_graphs.extend(random.sample(graphs, min(count, len(graphs))))
    
    return all_graphs

# Usage
training_set = mix_datasets_weighted({
    'single_room': 0.4,
    'long_gaps': 0.3,
    'multi_room': 0.3
})
```

### Curriculum Learning Schedule

Progressive training schedule example:

```python
def get_curriculum_schedule(stage):
    """
    Get dataset for specific curriculum stage.
    
    Stage 1: Simple single tasks
    Stage 2: Add time gaps (rest behavior)
    Stage 3: Multiple rooms
    Stage 4: Priority awareness
    Stage 5: Complex topologies
    Stage 6: Full mix
    """
    from graph_datasets.single_room import get_graph_dataset as get_single_room
    from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
    from graph_datasets.multi_room import get_graph_dataset as get_multi_room
    from graph_datasets.priority_mixed import get_graph_dataset as get_priority_mixed
    from graph_datasets.complex_topology import get_graph_dataset as get_complex_topology
    from graph_datasets.early_late_tasks import get_graph_dataset as get_early_late
    
    if stage == 1:
        # Foundation: single room tasks
        return get_single_room()
    
    elif stage == 2:
        # Add rest behavior with long gaps
        single = get_single_room()
        gaps = get_long_gaps()
        return single + gaps[:3]  # Mix with some long gaps
    
    elif stage == 3:
        # Add multi-room complexity
        single = get_single_room()
        multi = get_multi_room()
        return single[:5] + multi[:5]
    
    elif stage == 4:
        # Add priority challenges
        multi = get_multi_room()
        priority = get_priority_mixed()
        return multi[:5] + priority[:5]
    
    elif stage == 5:
        # Complex topologies
        multi = get_multi_room()
        complex_topo = get_complex_topology()
        return multi[:3] + complex_topo[:7]
    
    elif stage == 6:
        # Full mix for final training
        return (get_single_room()[:2] + 
                get_long_gaps()[:3] +
                get_multi_room()[:3] +
                get_priority_mixed()[:3] +
                get_complex_topology()[:2] +
                get_early_late()[:2])
    
    else:
        return get_single_room()

# Usage in training loop
for epoch in range(num_epochs):
    stage = determine_stage(epoch)  # Your stage determination logic
    graphs = get_curriculum_schedule(stage)
    train_on_graphs(graphs)
```

### Complete Dataset Mixer

```python
def get_all_datasets():
    """Get all available datasets."""
    from graph_datasets.single_room import get_graph_dataset as get_single_room
    from graph_datasets.long_gaps import get_graph_dataset as get_long_gaps
    from graph_datasets.multi_room import get_graph_dataset as get_multi_room
    from graph_datasets.priority_mixed import get_graph_dataset as get_priority_mixed
    from graph_datasets.complex_topology import get_graph_dataset as get_complex_topology
    from graph_datasets.early_late_tasks import get_graph_dataset as get_early_late
    
    return {
        'single_room': get_single_room(),
        'long_gaps': get_long_gaps(),
        'multi_room': get_multi_room(),
        'priority_mixed': get_priority_mixed(),
        'complex_topology': get_complex_topology(),
        'early_late_tasks': get_early_late(),
    }

def create_custom_mix(recipe):
    """
    Create custom dataset mix from recipe.
    
    Args:
        recipe: List of tuples (dataset_name, count or percentage)
        Example: [
            ('single_room', 5),           # Exact count
            ('long_gaps', 0.3),           # Percentage (requires total)
            ('multi_room', 3)
        ]
        OR
        recipe: dict with counts
        {
            'single_room': 5,
            'long_gaps': 3,
            'multi_room': 2
        }
    
    Returns:
        List of GraphData objects
    """
    all_datasets = get_all_datasets()
    result = []
    
    if isinstance(recipe, dict):
        # Simple count-based mixing
        for dataset_name, count in recipe.items():
            if dataset_name in all_datasets:
                graphs = all_datasets[dataset_name]
                result.extend(graphs[:min(count, len(graphs))])
    
    elif isinstance(recipe, list):
        # Handle mixed count/percentage
        counts = {}
        percentages = {}
        total_for_percentages = None
        
        for item in recipe:
            dataset_name, value = item
            if isinstance(value, int):
                counts[dataset_name] = value
            elif isinstance(value, float):
                percentages[dataset_name] = value
        
        # Calculate total for percentages
        if percentages:
            total_percentage = sum(percentages.values())
            if total_percentage <= 1.0:
                # Percentages, need to determine total
                total_for_percentages = 50  # Default, or calculate from counts
                for dataset_name, pct in percentages.items():
                    counts[dataset_name] = int(total_for_percentages * pct)
        
        # Apply counts
        for dataset_name, count in counts.items():
            if dataset_name in all_datasets:
                graphs = all_datasets[dataset_name]
                result.extend(graphs[:min(count, len(graphs))])
    
    return result

# Usage examples
mix1 = create_custom_mix({
    'single_room': 5,
    'long_gaps': 3,
    'multi_room': 2
})

mix2 = create_custom_mix([
    ('single_room', 5),
    ('long_gaps', 0.3),  # 30% of remaining
    ('multi_room', 0.2)  # 20% of remaining
])
```

## Recommended Training Progression

### Phase 1: Foundation (Epochs 0-1000)
- **Focus**: Basic task completion
- **Dataset Mix**: 100% `single_room`
- **Goal**: Learn to complete single tasks, basic timing

### Phase 2: Rest Behavior (Epochs 1000-2000)
- **Focus**: Learning rest actions
- **Dataset Mix**: 50% `single_room`, 50% `long_gaps`
- **Goal**: Understand when to rest during gaps

### Phase 3: Multi-Task Planning (Epochs 2000-3000)
- **Focus**: Multiple tasks, route planning
- **Dataset Mix**: 30% `single_room`, 20% `long_gaps`, 50% `multi_room`
- **Goal**: Handle multiple sequential tasks

### Phase 4: Priority Awareness (Epochs 3000-4000)
- **Focus**: Priority-based decisions
- **Dataset Mix**: 20% `multi_room`, 40% `priority_mixed`, 20% `long_gaps`, 20% `early_late_tasks`
- **Goal**: Make priority-aware choices

### Phase 5: Complex Environments (Epochs 4000-5000)
- **Focus**: Complex topologies and full-day planning
- **Dataset Mix**: 20% `multi_room`, 20% `priority_mixed`, 30% `complex_topology`, 30% `early_late_tasks`
- **Goal**: Handle complex scenarios

### Phase 6: Mastery (Epochs 5000+)
- **Focus**: All scenarios
- **Dataset Mix**: Equal mix of all datasets (except maybe less `single_room`)
- **Goal**: Robust performance across all scenarios

## Notes

- All graphs are designed to be **completable** (no catastrophic failures)
- Each graph has an **optimal solution pattern** that can be learned
- Time values are in hours (0-24 scale, though typically 6-21 for realistic schedules)
- Priority: 0 = low priority, 1 = high priority
- Edge weights represent travel time between locations
- All graphs ensure connectivity (base is reachable from all nodes)

## Dataset Statistics

| Dataset | Graph Count | Avg Nodes | Avg Tasks | Special Feature |
|---------|------------|-----------|-----------|----------------|
| `single_room` | 10 | 3 | 1 | Simplest |
| `long_gaps` | 10 | 3-5 | 1-4 | 6-14hr gaps |
| `multi_room` | 10 | 4-6 | 2-5 | Multiple tasks |
| `priority_mixed` | 10 | 3-5 | 2-4 | Priority conflicts |
| `complex_topology` | 10 | 3-6 | 1-3 | Complex paths |
| `early_late_tasks` | 10 | 3-5 | 1-4 | Full-day spread |

