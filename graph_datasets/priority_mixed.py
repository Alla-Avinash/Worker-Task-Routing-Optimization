"""
Priority Mixed Dataset
Graphs that mix high and low priority tasks to teach the agent:
- Priority-based task ordering
- When to prioritize urgent tasks over convenient ones
- Balancing time efficiency with priority requirements
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects with mixed priority tasks.
    
    Returns:
        List[GraphData]: List of graph instances with priority conflicts
    """
    
    # Graph 1: Low priority early, high priority late (temporal conflict)
    g1 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 2: High priority far, low priority close (spatial conflict)
    g2 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 3: Multiple priorities requiring optimal ordering
    g3 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 4: High priority early but requires travel
    g4 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.5, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 9.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 5: Priority vs proximity trade-off
    g5 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            4: [{"time": 10.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 6: Priority sequence challenge
    g6 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 7: High priority at inconvenient location
    g7 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 9.5, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 8: Multiple high priority tasks
    g8 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 9: Low priority cluster, isolated high priority
    g9 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (0, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 9.5, "priority": 0, "cleaning_time": 0.12}],
            4: [{"time": 10.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 10: Priority ordering with time constraints
    g10 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.15}],
            2: [{"time": 10.5, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 12.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

