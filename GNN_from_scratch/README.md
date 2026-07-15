# GNN From Scratch
This study project is dedicated to implement classic models of GNNs for Node Classification via Message Passing framework using just NumPy (only forward propagation will be considered).

## Graph Convolutional Networks (GCN)
The main architecture of GNNs, it utilizes the theory of Spectral Graph Convolutions to achieve a optimized way to obtain Node Representations (Embeddings), in this particular case, used for the Node Classification role despite being relevant for many other real applications. In the original paper (Kipf and Welling, 2017), they present this matricial form of the propagation rule for GCNs:
$$H^{(l+1)} = \sigma \left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
In this project we implement this propagation rule including the Message Passing framework (Message, Aggregate, Update (or Combine)) for a single layer and then builds the complete model with 2 layers and a Softmax normalization.