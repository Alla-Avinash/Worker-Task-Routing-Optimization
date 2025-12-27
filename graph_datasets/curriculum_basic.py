"""
Curriculum Basic Dataset
Simple, easy graphs for early training stages.
These graphs have straightforward solutions and help establish basic patterns.

Characteristics:
- Simple topologies (linear, small)
- Few nodes (3-4)
- Few tasks (1-2 per graph)
- Clear optimal solutions
- No complex timing conflicts
"""

from env import GraphData


def get_graph_dataset():
    """
    Returns a list of simple GraphData objects for early curriculum learning.
    
    Returns:
        List[GraphData]: List of simple graph instances
    """
    
    # Simplest: single task, single room
    cb01 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Single task, different room
    cb02 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Single task at base
    cb03 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            0: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Two sequential tasks, linear path
    cb04 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Two tasks, one priority
    cb05 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2)],
        edge_weights=[0.1, 0.2],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 1, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Simple triangle topology
    cb06 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # Two tasks, triangle with choice
    cb07 = GraphData(
        num_nodes=3,
        edge_list=[(0, 1), (1, 2), (0, 2)],
        edge_weights=[0.1, 0.2, 0.15],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # 4-node linear, single task
    cb08 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            3: [{"time": 10.0, "priority": 0, "cleaning_time": 0.1}],
        },
    )
    
    # 4-node, two tasks in order
    cb09 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (1, 2), (2, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            1: [{"time": 9.0, "priority": 0, "cleaning_time": 0.1}],
            3: [{"time": 10.0, "priority": 0, "cleaning_time": 0.12}],
        },
    )
    
    # Simple star topology, single task
    cb10 = GraphData(
        num_nodes=4,
        edge_list=[(0, 1), (0, 2), (0, 3)],
        edge_weights=[0.1, 0.15, 0.12],
        base=0,
        classroom_schedules={
            2: [{"time": 10.0, "priority": 0, "cleaning_time": 0.15}],
        },
    )
    
    return [cb01, cb02, cb03, cb04, cb05, cb06, cb07, cb08, cb09, cb10]


# ============================================================================
# For direct import
# ============================================================================

# Pre-computed dataset (call get_graph_dataset() to get fresh instances)
GRAPH_DATASET = get_graph_dataset()

