# GNN From Scratch
This study project is dedicated to implementing classic models of GNNs for Node Classification via Message Passing framework using just NumPy and Autograd for weight optimization.

## Graph Convolutional Networks (GCN)
As one of the foundational GNN architectures, it utilizes the theory of Spectral Graph Convolutions to achieve an optimized way to obtain Node Representations (Embeddings), applied here to a Node Classification task, though its underlying logic is highly relevant for various other domains. In the original paper (Kipf & Welling, 2017), they present the following matrix form of the propagation rule for GCNs:

$$H^{(l+1)} = \sigma \left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

In this project we implement this propagation rule under the Message Passing framework (Message, Aggregate, Update/Combine) for a single layer and then build the complete model with 2 layers, cross-entropy loss function and a Softmax normalization. 

## Project Structure

The repository is structured to be modular and scalable. The core logic relies on a base `MessagePassing` class, making it straightforward to expand from Graph Convolutional Networks (GCNs) to Graph Attention Networks (GATs) and other architectures.

```text
.
├── data/
│   └── dataset.npz              # Toy graph dataset generated for testing and validation
├── notebooks/
│   ├── 01_gcn_training.ipynb    # Step-by-step GCN training, visualization, and evaluation
│   └── 02_gat_training.ipynb    # [Upcoming] Graph Attention Network (GAT) implementation
├── src/
│   ├── layers.py                # MessagePassing base class, GCNLayer, and future GATLayer
│   ├── models.py                # Model architectures (e.g., GCNModel) assembling the layers
│   └── utils.py                 # Graph generation, symmetric adjacency normalization, etc.
└── README.md
