
from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects for training.
    
    Returns:
        List[GraphData]: List of graph instances
    """

    g001 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g002 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 16.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g003 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 15.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g004 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 18.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g005 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 11.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g006 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 17.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g007 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g01 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 15.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g02 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g03 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.1],
        base=0,
        classroom_schedules={
            0: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )

    g04 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            0: [{"time": 13.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )

    g05 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.1],
        base=0,
        classroom_schedules={
            2: [{"time": 15.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g06 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 18.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )

    g07 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 18.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )

    g08 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.3],
        base=0,
        classroom_schedules={
            1: [{"time": 14.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )

    g09 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.2, 0.2, 0.3],
        base=0,
        classroom_schedules={
            1: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    g10 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.3, 0.2, 0.3],
        base=0,
        classroom_schedules={
            1: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    

    return [g01, g02, g03, g04, g05, g06, g07, g08, g09, g10]
    

# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

