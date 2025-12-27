
from env import GraphData


def get_graph_dataset():
    """
    Returns graphs with complex topologies to teach advanced routing.
    
    These graphs have non-trivial structures (grids, rings, complex connectivity)
    that require the model to understand path optimization in complex spaces.
    
    Returns:
        List[GraphData]: List of graph instances with complex topologies
    """

    # Ring topology
    ct01 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        edge_weights=[0.1, 0.15, 0.12, 0.18, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            3: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Grid 3x3 structure
    ct02 = GraphData(
        num_nodes=9,
        edge_list=[
            (0, 1), (1, 2),  # Row 1
            (3, 4), (4, 5),  # Row 2
            (6, 7), (7, 8),  # Row 3
            (0, 3), (1, 4), (2, 5),  # Columns
            (3, 6), (4, 7), (5, 8),  # Columns
        ],
        edge_weights=[0.1, 0.12, 0.1, 0.15, 0.12, 0.1, 0.18, 0.15, 0.12, 0.2, 0.18, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            7: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            8: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Tree structure with branches
    ct03 = GraphData(
        num_nodes=7,
        edge_list=[
            (0, 1), (1, 2), (1, 3),  # Branch 1
            (0, 4), (4, 5), (4, 6),  # Branch 2
        ],
        edge_weights=[0.1, 0.15, 0.12, 0.18, 0.15, 0.12],
        base=0,
        classroom_schedules={
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            5: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            6: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Hub with multiple levels
    ct04 = GraphData(
        num_nodes=8,
        edge_list=[
            (0, 1), (0, 2),  # Level 1
            (1, 3), (1, 4), (2, 5), (2, 6),  # Level 2
            (3, 7),  # Level 3
        ],
        edge_weights=[0.1, 0.12, 0.15, 0.1, 0.18, 0.15, 0.2],
        base=0,
        classroom_schedules={
            3: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            5: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            7: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Mesh network (highly connected)
    ct05 = GraphData(
        num_nodes=6,
        edge_list=[
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 4),
            (2, 3), (2, 5),
            (3, 4),
            (4, 5),
        ],
        edge_weights=[0.1, 0.12, 0.15, 0.1, 0.18, 0.12, 0.2, 0.15, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            5: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Double diamond
    ct06 = GraphData(
        num_nodes=7,
        edge_list=[
            (0, 1), (0, 2),
            (1, 3), (2, 3),
            (3, 4), (3, 5),
            (4, 6), (5, 6),
        ],
        edge_weights=[0.1, 0.15, 0.12, 0.1, 0.18, 0.15, 0.12, 0.1],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            5: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            6: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Ladder structure
    ct07 = GraphData(
        num_nodes=6,
        edge_list=[
            (0, 1), (2, 3), (4, 5),  # Horizontal
            (0, 2), (1, 3), (2, 4), (3, 5),  # Vertical/rungs
        ],
        edge_weights=[0.1, 0.12, 0.1, 0.15, 0.18, 0.2, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            5: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Complex graph with multiple paths
    ct08 = GraphData(
        num_nodes=8,
        edge_list=[
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 4), (2, 5),
            (3, 6), (4, 6), (5, 7),
            (6, 7),
        ],
        edge_weights=[0.1, 0.12, 0.15, 0.1, 0.18, 0.2, 0.12, 0.15, 0.18, 0.1],
        base=0,
        classroom_schedules={
            3: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            6: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            7: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Cross topology
    ct09 = GraphData(
        num_nodes=5,
        edge_list=[
            (0, 2), (1, 2), (2, 3), (2, 4),  # Center hub with 4 branches
        ],
        edge_weights=[0.1, 0.15, 0.12, 0.18],
        base=2,
        classroom_schedules={
            0: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            1: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            3: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 13.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )

    # Hexagon-like structure
    ct10 = GraphData(
        num_nodes=7,
        edge_list=[
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Outer ring
            (0, 3), (1, 4), (2, 5),  # Diagonals
            (6, 0), (6, 2), (6, 4),  # Center connections
        ],
        edge_weights=[0.1, 0.12, 0.15, 0.1, 0.18, 0.12, 0.2, 0.18, 0.15, 0.1, 0.12, 0.15],
        base=6,
        classroom_schedules={
            1: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 11.0, "priority": 1, "cleaning_time": 0.1}],
            5: [{"time": 12.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    

    return [ct01, ct02, ct03, ct04, ct05, ct06, ct07, ct08, ct09, ct10]
    

# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

