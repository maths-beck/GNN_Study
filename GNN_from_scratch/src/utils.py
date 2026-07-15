import numpy as np

def test_graph():
    """
    Builds a fictitious graph for tests
    Return:
        X: node feature matrix (N x F)
        A: adjacency matrix (N x N)
    """
    # 4 nodes, each with 3 features (F = 3)
    X = np.array([
        [1.0, 0.0, 1.0], # Node 0
        [2.0, 0.0, 1.0], # Node 1
        [0.0, 1.0, 2.0], # Node 2
        [1.0, 1.0, 0.0]  # Node 3
    ], dtype=np.float32)

    # Simple undirected graph
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=np.float32)

    return X, A