import numpy as np
from src.utils import GCN_normalization


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

    def aggregate(self, messages, A, np_module=np):
        """
        Aggregate the messages from neighbors
        Args:
            messages: Messages from neighborhood
            A: Adjacency Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            aggr_messages: Aggregated Messages from neighborhood
        """
        if self.aggr == "sum":
            # M is the Aggregated Neighborhood Message Matrix,
            # where each row i contains the sum of 
            # the features of node i's neighbors 
            M = A @ messages
            aggr_messages = M
        else:
            M = A @ messages
            degrees = np_module.sum(A, axis=1).reshape(-1,1)
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

    def propagate(self, X, A, np_module=np):
        """
        Forward Propagation
        Args:
            X: Node Features Matrix
            A: Adjacency Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            H: Node Representation Matrix after updates
        """
        out_messages = self.message(X)

        aggr_out = self.aggregate(out_messages, A, np_module)

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
        W: Weight Matrix (n_in X n_out)
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
        X_projected = (X @ self.W) + self.b

        # Message Passing Propagation
        H = self.propagate(X_projected, A_hat, np_module)

        if self.activation:
            # For this small toy dataset, traditional ReLU did not perform well (Dead ReLU)
            # Decision: ReLU --> LeakyReLU (alpha=0.01)
            H_next = np_module.where(H > 0, H, 0.01 * H)
        else:
            H_next = H

        return H_next