import numpy as np
from src.utils import prepare_attention_tensor, leaky_relu, elu, softmax


class MessagePassing:
    def __init__(self, aggr="sum"):
        """
        Base class for Message Passing Layers

        aggr: aggregation types ("sum" or "mean")
        """
        self.aggr = aggr
        

    def message(self, X):
        """
        Sends node features to neighbors
        In this implementation, the messages will be the actual node features

        Args:
            X: Node Features Matrix
        Return:
            Msg: Messages passed through neighborhood
        """
        messages = X

        return messages

    def aggregate(self, messages, A_adj, np_module=np):
        """
        Aggregate the messages from neighbors

        Args:
            messages: Messages from neighborhood
            A_adj: Adjacency or Attention Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            aggr_messages: Aggregated Messages from neighborhood
        """
        if self.aggr == "sum":
            # M is the Aggregated Neighborhood Message Matrix,
            # where each row i contains the sum of 
            # the features of node i's neighbors 
            M = A_adj @ messages
            aggr_messages = M
        else:
            M = A_adj @ messages
            degrees = np_module.sum(A_adj, axis=1).reshape(-1,1)
            safe_degrees = np_module.where(degrees > 0, degrees, 1.0)
            aggr_messages = M / safe_degrees

        return aggr_messages

    # Same as the definition of COMBINE function
    def update(self, aggr_messages):
        """
        Updates the nodes with the aggregated messages

        Args:
            aggr_messages: Aggregated Messages from neighborhood
        Return:
            H: Node Representation 
        """
        H = aggr_messages

        return H

    def propagate(self, X, A_adj, np_module=np):
        """
        Forward Propagation

        Args:
            X: Node Features Matrix
            A_adj: Adjacency or Attention Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            H: Node Representation Matrix after updates
        """
        out_messages = self.message(X)

        aggr_out = self.aggregate(out_messages, A_adj, np_module)

        H = self.update(aggr_out)

        return H
    


class GCNLayer(MessagePassing):
    def __init__(self, n_in, n_out, aggr="sum", activation=True):
        """
        Class for Graph Convolution Network (GCN) Layer
        using the Message Passing framework

        n_in: Number of input features per node 
        n_out: Number of input features per node
        aggr: Aggregation type ("sum" or "mean")
        activation: Trigger the activation function in the layer
        W: Weight Matrix (n_in x n_out)
        b: Bias Vector
        """
        super().__init__(aggr=aggr)
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        # Glorot init
        sd = np.sqrt(6.0 / (n_in + n_out))
        self.W = np.random.uniform(-sd, sd, size=(n_in, n_out))

        # Added a Bias vector to mitigate the Dead ReLU phenomenon
        self.b = np.zeros((1, n_out), dtype=np.float32)


    def forward(self, A_hat, X, np_module=np):
        """
        Applies GCN style Forward Passing  

        Args:
            A_hat: Normalized Adjacency Matrix with self-loops
            X: Node Features Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            H_next: Final Node Representation after forward passing
        """
        # In the first layer of a GNN, H^(0) = X
        # Calculating H^(l=0) * W including Bias 
        Z = (X @ self.W) + self.b

        # Message Passing Propagation
        H = self.propagate(Z, A_hat, np_module)

        if self.activation:
            # For this small toy dataset, traditional ReLU did not perform well (Dead ReLU)
            # Decision: ReLU --> LeakyReLU (alpha=0.01)
            H_next = leaky_relu(H, alpha=0.01, np_module=np_module)
        else:
            H_next = H

        return H_next



class GATLayer(MessagePassing):
    def __init__(self, n_in, n_out, leaky_relu_slope=0.2, activation=True):
        """
        Class for Graph Attention Network (GAT) Layer
        using the Message Passing framework

        n_in: Number of input features per node 
        n_out: Number of input features per node
        leaky_relu_slope: Slope for negative values (alpha)
        activation: Trigger the activation function in the layer
        W: Weight Matrix (n_in x n_out)
        a: Attention Vector (2 * F' x 1)
        """
        super().__init__(aggr="sum")
        self.n_in = n_in
        self.n_out = n_out
        self.leaky_relu_slope = leaky_relu_slope
        self.activation = activation

        # Glorot init for Weight Matrix
        sd_W = np.sqrt(6.0 / (n_in + n_out))
        self.W = np.random.uniform(-sd_W, sd_W, size=(n_in, n_out))

        # Glorot init for Attention Vector
        sd_a = np.sqrt(6.0 / ((2 * n_out) + 1))
        self.a = np.random.uniform(-sd_a, sd_a, size=(2 * n_out, 1))

    
    def forward(self, A_til, X, np_module=np):
        """
        Applies GAT style Forward Passing  

        Args:
            A_til: Adjacency Matrix with self-loops
            X: Node Features Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            H_next: Final Node Representation after forward passing
        """
        # Projecting the Node Features Matrix
        Z = X @ self.W

        # Concatenating Z_i and Z_j (i.e. [Z_i || Z_j])
        Z_concat = prepare_attention_tensor(Z, np_module)

        # Calculating the Attention Score Matrix E (N, N)
        E = Z_concat @ self.a
        E = np_module.squeeze(E, axis=-1)

        # Calculating the Raw Attention Coefficient e_ij
        e_ij = leaky_relu(E, alpha=self.leaky_relu_slope, np_module=np_module)
        masked_att = np_module.where(A_til > 0, e_ij, -9e15)

        # Normalizing e_ij to obtain the Attention Coefficient alpha_ij
        alpha_ij = softmax(masked_att, np_module)

        # Message Passing Propagation
        H = self.propagate(Z, alpha_ij, np_module)

        if self.activation:
            # Applying ELU (alpha=1) as the original paper
            H_next = elu(H, alpha=1.0, np_module=np_module)
        else:
            H_next = H

        return H_next