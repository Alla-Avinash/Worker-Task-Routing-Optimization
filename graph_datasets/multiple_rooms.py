
from env import GraphData


def get_graph_dataset():
    """
    Returns graphs with multiple rooms/tasks to teach coordination.
    
    These graphs have 3+ rooms with tasks, teaching the model to:
    - Plan routes efficiently
    - Handle multiple concurrent tasks
    - Optimize task ordering
    
    Returns:
        List[GraphData]: List of graph instances with multiple rooms
    """

    # Three rooms with different schedules
    mr01 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            3: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Four rooms in a chain
    mr02 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Star topology with 4 rooms
    mr03 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (0, 3), (0, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            3: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Grid-like structure with 5 rooms
    mr04 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        edge_weights=[0.1, 0.12, 0.1, 0.15, 0.1, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            5: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Multiple rooms with overlapping times
    mr05 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3), (0, 3)],
        edge_weights=[0.1, 0.15, 0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.5, "priority": 1, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Rooms with multiple tasks each
    mr06 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (1, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [
                {"time": 10.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            2: [
                {"time": 11.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 15.0, "priority": 0, "cleaning_time": 0.1},
            ],
            3: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Complex routing with 6 rooms
    mr07 = GraphData(
        num_nodes=7,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)],
        edge_weights=[0.1, 0.12, 0.1, 0.15, 0.1, 0.12, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            5: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
            6: [{"time": 14.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )

    # Diamond topology with 4 rooms
    mr08 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.15, 0.1, 0.12, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            3: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Rooms in parallel paths
    mr09 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        edge_weights=[0.1, 0.12, 0.15, 0.1, 0.12, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 9.5, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            4: [{"time": 11.5, "priority": 0, "cleaning_time": 0.1}],
            5: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Hub with many rooms, multiple tasks
    mr10 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (0, 3), (0, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [
                {"time": 10.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 14.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            3: [
                {"time": 12.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 16.0, "priority": 1, "cleaning_time": 0.1},
            ],
            4: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    

    return [mr01, mr02, mr03, mr04, mr05, mr06, mr07, mr08, mr09, mr10]
    

# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

