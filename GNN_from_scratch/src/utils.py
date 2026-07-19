import numpy as np


def test_graph():
    """
    Builds a fictitious graph for tests

    Return:
        X: Node feature Matrix (N x F)
        A: Adjacency Matrix (N x N)
        y: Target One-Hot Encoding Matrix (N x n_out)
    """
    # 4 nodes, each with 3 features (F = 3)
    X = np.array([
        [1.0, 0.0, 1.0], # Node 0
        [2.0, 0.0, 1.0], # Node 1
        [0.0, 0.0, 2.0], # Node 2
        [1.0, 1.0, 0.0]  # Node 3
    ], dtype=np.float32)

    # Simple undirected graph
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=np.float32)

    # Target y in One-Hot Encoding format (4 nodes, 2 classes)
    y = np.array([
        [1.0, 0.0],  # Node 0 -> Class 0
        [1.0, 0.0],  # Node 1 -> Class 0
        [0.0, 1.0],  # Node 2 -> Class 1
        [0.0, 1.0]   # Node 3 -> Class 1
    ])

    return X, A, y


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
    D_ii = sum of row i of A

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
    A_hat = D^{-1/2} * A_til * D^{-1/2}

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


def prepare_attention_tensor(Z, np_module=np):
    """
    Constructs the 3D tensor containing the concatenation of the pairs [Z_i || Z_j]

    Args:
        Z: Projected Features by weights Matrix (N x F')
        np_module: Support for NumPy and Autograd.NumPy modules
    """
    # Transforms Z from (N, F') to (N, 1, F') and then to (N, N, F')
    Z_i_temp = np_module.expand_dims(Z, axis=1)
    Z_i = np_module.repeat(Z_i_temp, repeats=Z_i_temp.shape[0], axis=1)
    
    # Transforms Z from (N, F') to (1, N, F') and then to (N, N, F')
    Z_j_temp = np_module.expand_dims(Z, axis=0)
    Z_j = np_module.repeat(Z_j_temp, repeats=Z_j_temp.shape[1], axis=0)
    
    # Concatenating along the last axis
    # Z_concat (N, N, 2 * F')
    Z_concat = np_module.concatenate([Z_i, Z_j], axis=-1)
    
    return Z_concat


def relu(x, np_module=np):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise
    
    Args:
        x: Input matrix
        np_module: Support for NumPy and Autograd.NumPy modules
    Return:
        Activated matrix
    """
    return np_module.where(x > 0, x, 0.0)


def leaky_relu(x, alpha=0.2, np_module=np):
    """
    Applies the Leaky ReLU function element-wise
    
    Args:
        x: Input matrix
        alpha: Slope for negative values
        np_module: Support for NumPy and Autograd.NumPy modules
    Return:
        Activated matrix
    """
    return np_module.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0, np_module=np):
    """
    Applies the Exponential Linear Unit (ELU) function element-wise
    
    Args:
        x: Input matrix
        alpha: Scaling factor for negative values
        np_module: Support for NumPy and Autograd.NumPy modules
    Return:
        Activated matrix
    """
    return np_module.where(x > 0, x, alpha * (np_module.exp(x) - 1.0))


def softmax(x, np_module=np):
        """
        Applies the Softmax activation function element-wise along the last axis,
        safeguarding against numerical overflow

        Args:
            x: Input Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            Activated Matrix
        """
        # Subtracting the max for numerical stability
        shifted_x = x - np_module.max(x, axis=1, keepdims=True)

        exp_x = np_module.exp(shifted_x)
        sum_exp_x = np_module.sum(exp_x, axis=1, keepdims=True)

        return exp_x / sum_exp_x