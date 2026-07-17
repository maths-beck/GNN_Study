import numpy as np
from src.layers import GCNLayer


class GCNModel:
    def __init__(self, n_in, n_hidden, n_out):
        """
        Model representing a 2-layer Graph Convolutional Network (GCN)

        n_in: Number of input features (input dimension)
        n_hidden: Number of hidden units (hidden layer dimension)
        n_out: Number of output units (number of classes/target dimension)
        layer1: First layer of the GCN model
        layer2: Second layer of the GCN model
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.layer1 = GCNLayer(self.n_in, self.n_hidden, aggr="sum", activation=True) # LeakyReLU applied
        self.layer2 = GCNLayer(self.n_hidden, self.n_out, aggr="sum", activation=False) # Straight to Softmax (No LeakyReLU)


    def softmax(self, x, np_module=np):
        """
        Applies the Softmax activation function element-wise along the last axis,
        safeguarding against numerical overflow
        Args:
            x: Input Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            soft_x: Softmax Activated Matrix
        """
        # Subtracting the max for numerical stability
        shifted_x = x - np_module.max(x, axis=1, keepdims=True)

        exp_x = np_module.exp(shifted_x)
        sum_exp_x = np_module.sum(exp_x, axis=1, keepdims=True)

        soft_x = exp_x / sum_exp_x

        return soft_x

    def forward(self, A_hat, X, np_module=np):
        """
        Applies a forward pass through the entire 2-layer GCN model
        Args:
            A_hat: Normalized Adjacency Matrix with self-loops
            X: Node Features Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            y_pred: Class probabilities per node (N x n_out)
        """
        # Propagates through the first layer
        layer1_out = self.layer1.forward(A_hat, X, np_module)

        # Propagates through the second layer
        layer2_out = self.layer2.forward(A_hat, layer1_out, np_module)

        # Applies Softmax for the final Node Representations
        y_pred = self.softmax(layer2_out, np_module)

        return y_pred