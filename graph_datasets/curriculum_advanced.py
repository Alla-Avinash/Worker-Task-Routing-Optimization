"""
Curriculum Advanced Dataset
Complex graphs for advanced training stages.
These graphs require sophisticated planning and decision-making.

Characteristics:
- Complex topologies (5+ nodes, multiple paths)
- Many tasks (3+ per graph)
- Timing conflicts and trade-offs
- Multiple priorities to balance
- Requires optimal route planning
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of complex GraphData objects for advanced curriculum learning.
    
    Returns:
        List[GraphData]: List of complex graph instances
    """
    
    # 5-node complex layout with many tasks
    ca01 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
        edge_weights=[0.15, 0.2, 0.1, 0.12, 0.18],
        base=0,
        classroom_schedules={
            0: [
                {"time": 8.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            1: [
                {"time": 9.0, "priority": 0, "cleaning_time": 0.12},
                {"time": 15.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.2}],
            3: [
                {"time": 11.0, "priority": 0, "cleaning_time": 0.15},
                {"time": 13.0, "priority": 1, "cleaning_time": 0.1},
            ],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.18}],
        },
    )
    
    # 6-node mesh with overlapping schedules
    ca02 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (1, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.2, 0.1, 0.18, 0.16],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [
                {"time": 9.0, "priority": 0, "cleaning_time": 0.12},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [
                {"time": 11.0, "priority": 1, "cleaning_time": 0.12},
                {"time": 15.0, "priority": 0, "cleaning_time": 0.1},
            ],
            4: [{"time": 12.0, "priority": 0, "cleaning_time": 0.2}],
            5: [
                {"time": 13.0, "priority": 1, "cleaning_time": 0.1},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.15},
            ],
        },
    )
    
    # Dense 5-node with many concurrent tasks
    ca03 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.14, 0.16, 0.13, 0.18],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [
                {"time": 9.0, "priority": 1, "cleaning_time": 0.12},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [
                {"time": 10.0, "priority": 0, "cleaning_time": 0.15},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            3: [
                {"time": 11.0, "priority": 0, "cleaning_time": 0.12},
                {"time": 17.0, "priority": 0, "cleaning_time": 0.13},
            ],
            4: [{"time": 12.0, "priority": 1, "cleaning_time": 0.18}],
        },
    )
    
    # Large star with multiple tasks per room
    ca04 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        edge_weights=[0.1, 0.15, 0.12, 0.18, 0.2],
        base=0,
        classroom_schedules={
            0: [
                {"time": 8.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 15.0, "priority": 1, "cleaning_time": 0.1},
            ],
            1: [
                {"time": 9.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            2: [{"time": 10.0, "priority": 1, "cleaning_time": 0.15}],
            3: [
                {"time": 11.0, "priority": 0, "cleaning_time": 0.12},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.1},
            ],
            4: [
                {"time": 12.0, "priority": 0, "cleaning_time": 0.18},
                {"time": 17.0, "priority": 1, "cleaning_time": 0.1},
            ],
            5: [{"time": 13.0, "priority": 0, "cleaning_time": 0.2}],
        },
    )
    
    # Complex multi-path scenario
    ca05 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
        edge_weights=[0.2, 0.15, 0.12, 0.18, 0.1],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [
                {"time": 9.0, "priority": 1, "cleaning_time": 0.12},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [
                {"time": 10.0, "priority": 0, "cleaning_time": 0.15},
                {"time": 13.0, "priority": 1, "cleaning_time": 0.1},
            ],
            3: [
                {"time": 11.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 17.0, "priority": 0, "cleaning_time": 0.12},
            ],
            4: [{"time": 12.0, "priority": 1, "cleaning_time": 0.18}],
        },
    )
    
    # Long day with many tasks
    ca06 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        edge_weights=[0.1, 0.15, 0.12, 0.2, 0.1, 0.18],
        base=0,
        classroom_schedules={
            0: [
                {"time": 7.0, "priority": 0, "cleaning_time": 0.1},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            1: [
                {"time": 9.0, "priority": 0, "cleaning_time": 0.12},
                {"time": 15.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [
                {"time": 10.0, "priority": 1, "cleaning_time": 0.15},
                {"time": 18.0, "priority": 0, "cleaning_time": 0.1},
            ],
            3: [{"time": 11.0, "priority": 0, "cleaning_time": 0.12}],
            4: [
                {"time": 12.0, "priority": 0, "cleaning_time": 0.2},
                {"time": 16.0, "priority": 1, "cleaning_time": 0.1},
            ],
            5: [
                {"time": 13.0, "priority": 1, "cleaning_time": 0.1},
                {"time": 17.0, "priority": 0, "cleaning_time": 0.15},
            ],
        },
    )
    
    # Dense schedule with tight timing
    ca07 = GraphData(
        num_nodes=5,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)],
        edge_weights=[0.15, 0.2, 0.12, 0.18, 0.1],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [
                {"time": 8.5, "priority": 1, "cleaning_time": 0.12},
                {"time": 15.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [
                {"time": 9.0, "priority": 0, "cleaning_time": 0.2},
                {"time": 14.0, "priority": 1, "cleaning_time": 0.1},
            ],
            3: [
                {"time": 9.5, "priority": 0, "cleaning_time": 0.1},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.12},
            ],
            4: [
                {"time": 10.0, "priority": 1, "cleaning_time": 0.18},
                {"time": 17.0, "priority": 0, "cleaning_time": 0.1},
            ],
        },
    )
    
    # Multiple priorities with complex routing
    ca08 = GraphData(
        num_nodes=6,
        edge_list=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (1, 4)],
        edge_weights=[0.1, 0.15, 0.12, 0.2, 0.1, 0.18, 0.16],
        base=0,
        classroom_schedules={
            0: [{"time": 8.0, "priority": 0, "cleaning_time": 0.1}],
            1: [
                {"time": 9.0, "priority": 1, "cleaning_time": 0.12},
                {"time": 13.0, "priority": 1, "cleaning_time": 0.1},
                {"time": 17.0, "priority": 0, "cleaning_time": 0.1},
            ],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
            3: [
                {"time": 11.0, "priority": 0, "cleaning_time": 0.12},
                {"time": 15.0, "priority": 1, "cleaning_time": 0.1},
            ],
            4: [
                {"time": 12.0, "priority": 1, "cleaning_time": 0.2},
                {"time": 16.0, "priority": 0, "cleaning_time": 0.1},
            ],
            5: [{"time": 14.0, "priority": 0, "cleaning_time": 0.18}],
        },
    )
    
    return [ca01, ca02, ca03, ca04, ca05, ca06, ca07, ca08]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

