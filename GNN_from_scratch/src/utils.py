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


def add_self_loops(A):
    """
    Adds self-loops to the Adjacency Matrix, i.e. A_til = A + I
    with I being identity matrix
    Args:
        A: Adjacency Matrix
    Return:
        A_til: Adjacency Matrix with self-loops
    """
    I = np.eye(A.shape[0], dtype=A.dtype)

    A_til = A + I

    return A_til


def degree_matrix(A):
    """
    Computes the degree matrix D (diagonal matrix)
    D_ii = sum of line i of A
    Args:
        A: Adjacency Matrix
    Return:
        D: Degree Matrix (N x N)
    """
    degrees = np.sum(A, axis=1)

    D = np.diag(degrees)

    return D

def GCN_normalization(A):
    """
    Computes the symmetric normalization of GCN model
    A_hat = D^{-1/2} * A_til * D{-1/2}
    Args:
        A: Adjacency Matrix
    Return:
        A_hat: Normalized Adjacency Matrix with self-loops 
    """
    # Adding self_loops to the Adjacency Matrix
    A_til = add_self_loops(A)

    # Obtaining the Degree Matrix
    D_til = degree_matrix(A_til)

    # Calculates D_til^{-1/2}
    d_diag = np.diag(D_til)
    d_inv_sqrt = 1.0 / np.sqrt(d_diag)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Calculates A_hat
    A_hat = D_inv_sqrt @ A_til @ D_inv_sqrt

    return A_hat 
