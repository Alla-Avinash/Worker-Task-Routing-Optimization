"""
Early-Late Tasks Dataset
Graphs with tasks spread throughout the day (early morning, midday, evening)
to teach the agent:
- Time-based planning across the full day
- Handling varied task timing
- Optimal scheduling for tasks at different times
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects with tasks at various times throughout the day.
    
    Returns:
        List[GraphData]: List of graph instances with varied task timing
    """
    
    # Graph 1: Early morning and late evening
    g1 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 7.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 19.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 2: Morning, midday, afternoon spread
    g2 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 12.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 16.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 3: Early morning cluster
    g3 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3)],
        edge_weights=[0.1, 0.12, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 7.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 7.5, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 8.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 4: Late afternoon and evening cluster
    g4 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 16.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 17.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 18.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 5: Midday tasks
    g5 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 13.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 6: Full day spread with gaps
    g6 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 12.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 15.0, "priority": 0, "cleaning_time": 0.15}],
            4: [{"time": 19.0, "priority": 1, "cleaning_time": 0.18}],
        },
    )
    
    # Graph 7: Early morning, late morning, early afternoon
    g7 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 7.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 14.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 8: Late morning and evening
    g8 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 11.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 18.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 9: Very early start
    g9 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 6.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 14.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 10: Evening cluster
    g10 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3)],
        edge_weights=[0.1, 0.12, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 17.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 18.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 19.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

