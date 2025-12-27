"""
Priority-Focused Dataset
Graphs specifically designed to teach priority handling.
These graphs have clear priority vs non-priority task distinctions.

Characteristics:
- Mix of priority 0 and priority 1 tasks
- Scenarios where priority tasks conflict with regular tasks
- Tests understanding of priority importance
- Optimal solutions require prioritizing priority tasks appropriately
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects focused on priority learning.
    
    Returns:
        List[GraphData]: List of graph instances with priority focus
    """
    
    # Simple priority vs non-priority at same time
    pf01 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Priority task earlier, must be done first
    pf02 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.1}],
            2: [{"time": 9.5, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Multiple priorities scattered
    pf03 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.12}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Priority task later but still important
    pf04 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            0: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Two priority tasks, one regular
    pf05 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12, 0.1],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Priority in middle of sequence
    pf06 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.12}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [{"time": 11.0, "priority": 0, "cleaning_time": 0.12}],
        },
    )
    
    # Multiple priorities close together
    pf07 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            0: [{"time": 9.0, "priority": 1, "cleaning_time": 0.1}],
            1: [{"time": 9.5, "priority": 1, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Priority tasks early and late
    pf08 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.1, 0.18],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.12}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [{"time": 11.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 12.0, "priority": 1, "cleaning_time": 0.18}],
        },
    )
    
    # Same time slot, priority wins
    pf09 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [{"time": 11.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Complex priority pattern
    pf10 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            0: [
                {"time": 8.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.12}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [
                {"time": 11.0, "priority": 1, "cleaning_time": 0.12},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.1},
            ],
        },
    )
    
    return [pf01, pf02, pf03, pf04, pf05, pf06, pf07, pf08, pf09, pf10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()
