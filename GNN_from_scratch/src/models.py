import numpy as np
from src.layers import GCNLayer, GATLayer
from src.utils import softmax


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
        y_pred = softmax(layer2_out, np_module)

        return y_pred



class GATModel:
    def __init__(self, n_in, n_hidden, n_out, leaky_relu_slope=0.2):
        """
        Model representing a 2-layer Graph Attention Network (GAT)

        n_in: Number of input features (input dimension)
        n_hidden: Number of hidden units (hidden layer dimension)
        n_out: Number of output units (number of classes/target dimension)
        leaky_relu_slope: Slope for negative values (alpha)
        layer1: First layer of the GAT model
        layer2: Second layer of the GAT model
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.leaky_relu_slope = leaky_relu_slope

        self.layer1 = GATLayer(self.n_in, self.n_hidden, self.leaky_relu_slope, activation=True) # ELU applied
        self.layer2 = GATLayer(self.n_hidden, self.n_out, self.leaky_relu_slope, activation=False) # Straight to Softmax (No ELU)


    def forward(self, A_til, X, np_module=np):
        """
        Applies a forward pass through the entire 2-layer GAT model
        Args:
            A_til: Adjacency Matrix with self-loops
            X: Node Features Matrix
            np_module: Support for NumPy and Autograd.NumPy modules
        Return:
            y_pred: Class probabilities per node (N x n_out)
        """
        # Propagates through the first layer
        layer1_out = self.layer1.forward(A_til, X, np_module)

        # Propagates through the second layer
        layer2_out = self.layer2.forward(A_til, layer1_out, np_module)

        # Applies Softmax for the final Node Representations
        y_pred = softmax(layer2_out, np_module)

        return y_pred