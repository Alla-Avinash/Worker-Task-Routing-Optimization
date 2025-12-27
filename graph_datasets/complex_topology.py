"""
Complex Topology Dataset
Graphs with complex connectivity patterns (multiple paths, cycles, etc.)
to teach the agent:
- Path optimization in complex graphs
- Choosing optimal routes when multiple paths exist
- Handling non-linear topologies
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of GraphData objects with complex graph topologies.
    
    Returns:
        List[GraphData]: List of graph instances with complex connectivity
    """
    
    # Graph 1: Fully connected triangle with multiple paths
    g1 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.1}],
        },
    )
    
    # Graph 2: Grid-like structure (2x2 grid)
    g2 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3)],
        edge_weights=[0.1, 0.12, 0.15, 0.18],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 3: Cycle with 4 nodes
    g3 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 0)],
        edge_weights=[0.1, 0.15, 0.12, 0.18],
        base=0,
        classroom_schedules={
            2: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 4: Hub with multiple connections (star with cross-links)
    g4 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.2, 0.22],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 5: Multiple alternative paths to same destination
    g5 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 6: Diamond structure with multiple routes
    g6 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.18, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 9.5, "priority": 0, "cleaning_time": 0.12}],
            4: [{"time": 10.5, "priority": 1, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 7: Complex network with cycles and branches
    g7 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            5: [{"time": 11.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 8: Double cycle (figure-8 structure)
    g8 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (3, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.18, 0.2, 0.22],
        base=0,
        classroom_schedules={
            2: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
        },
    )
    
    # Graph 9: Mesh network with redundant paths
    g9 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            4: [{"time": 11.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    # Graph 10: Tree with multiple branches and cross-connections
    g10 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 4), (4, 5)],
        edge_weights=[0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25],
        base=0,
        classroom_schedules={
            3: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            4: [{"time": 10.0, "priority": 1, "cleaning_time": 0.12}],
            5: [{"time": 11.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )

    return [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()
