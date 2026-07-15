import numpy as np
from layers import GCNLayer


class GCNModel:
    def __init__(self, n_in, n_hidden, n_out):
        """
        Model representing a 2-layer Graph Convolutional Network (GCN)

        n_in: Number of input features (input dimension)
        n_hidden: Number of hidden units (hidden layer dimension)
        n_out: Number of output units (number of classes/target dimension)
        layer1: First layer of the GCN model with aggr="mean" for testing
        layer2: Second layer of the GCN model with default settings
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.layer1 = GCNLayer(self.n_in, self.n_hidden, aggr="mean")
        self.layer2 = GCNLayer(self.n_hidden, self.n_out) # aggr="sum" as default


    def softmax(self, x):
        """
        Applies the Softmax activation function element-wise along the last axis,
        safeguarding against numerical overflow
        Args:
            x: Input matrix
        Return:
            soft_x: Softmax activated matrix
        """
        # Subtracting the max for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)

        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)

        soft_x = exp_x / sum_exp_x

        return soft_x

    def forward(self, A, X):
        """
        Applies a forward pass through the entire 2-layer GCN model
        Args:
            A: Adjacency Matrix
            X: Node Features Matrix
        Return:
            Y_pred: Class probabilities per node (N x n_out)
        """
        # Propagates through the first layer
        layer1_out = self.layer1.forward(A, X)

        # Propagates through the second (hidden) layer
        layer2_out = self.layer2.forward(A, layer1_out)

        # Applies Softmax for the final Node Representations
        y_pred = self.softmax(layer2_out)

        return y_pred