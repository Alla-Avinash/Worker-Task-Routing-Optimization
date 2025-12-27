"""
Long Gaps Dataset
Graphs with significant time gaps between cleaning tasks to encourage 
the model to learn rest actions during idle periods.

These graphs are designed to teach the agent:
- When to rest during long gaps
- Time management across extended periods
- Optimal waiting strategies
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects with long time gaps between tasks.
    
    Returns:
        List[GraphData]: List of graph instances with extended time gaps
    """
    
    # Graph 1: Very early task, then very late task (long midday gap)
    g1 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [
                {"time": 8.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 18.0, "priority": 1, "cleaning_time": 0.1},
            ],
        },
    )
    
    # Graph 2: Early morning task, late afternoon task (10-hour gap)
    g2 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            3: [
                {"time": 7.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 17.0, "priority": 1, "cleaning_time": 0.15},
            ],
        },
    )
    
    # Graph 3: Morning task, evening task (12-hour gap)
    g3 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 20.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 4: Two tasks with very long gap in between
    g4 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 19.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 5: Early task, late task at different locations (encourages rest at base)
    g5 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            2: [{"time": 8.5, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 18.5, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 6: Three tasks spread across the day with large gaps
    g6 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 14.0, "priority": 1, "cleaning_time": 0.12}],
            3: [{"time": 20.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 7: Very early start, long gap, late finish
    g7 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 7.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 19.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 8: Morning cluster, evening cluster with long midday break
    g8 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 17.0, "priority": 1, "cleaning_time": 0.12}],
            4: [{"time": 18.0, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 9: Single early task, single late task (maximize rest opportunity)
    g9 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            3: [
                {"time": 7.5, "priority": 0, "cleaning_time": 0.1},
                {"time": 19.5, "priority": 1, "cleaning_time": 0.1},
            ],
        },
    )
    
    # Graph 10: Extended day with tasks at extreme times
    g10 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 6.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 21.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()
