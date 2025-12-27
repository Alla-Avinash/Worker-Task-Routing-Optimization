"""
Multi-Room Dataset
Graphs with multiple rooms that need cleaning in a specific sequence.
These graphs teach the agent to:
- Plan optimal routes through multiple locations
- Handle sequential task dependencies
- Manage time across multiple cleaning tasks
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects with multiple rooms to clean.
    
    Returns:
        List[GraphData]: List of graph instances with multiple cleaning tasks
    """
    
    # Graph 1: Linear sequence of 3 rooms, optimal path is clear
    g1 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 2: Star topology with multiple rooms around hub
    g2 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (0, 3), (0, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.18}],
        },
    )
    
    # Graph 3: Multiple rooms with time spacing (allows travel between)
    g3 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3), (0, 2)],
        edge_weights=[0.1, 0.15, 0.12, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.5, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 12.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 4: Four rooms with priority ordering challenge
    g4 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 1, "cleaning_time": 0.1}],
            2: [{"time": 9.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.18}],
        },
    )
    
    # Graph 5: Multiple rooms with overlapping time windows
    g5 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 9.5, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 6: Complex path with 5 rooms
    g6 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.18}],
            5: [{"time": 13.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 7: Branching structure with multiple paths
    g7 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
            4: [{"time": 12.0, "priority": 1, "cleaning_time": 0.18}],
        },
    )
    
    # Graph 8: Rooms with varying cleaning times
    g8 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.15}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.2}],
            3: [{"time": 11.5, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 9: Multiple rooms with rest periods between
    g9 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 12.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 15.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 10: Circular path with 4 rooms
    g10 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.15}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.18}],
        },
    )

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()
