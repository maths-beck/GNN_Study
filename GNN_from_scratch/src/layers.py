import numpy as np

class MessagePassing:
    def __init__(self, aggr="sum"):
        """
        Base class for Message Passing Layers
        aggr: aggregation types ("sum", "mean")
        """
        self.aggr = aggr


    def message(self, X):
        """
        Sends node features to neighbors
        In this implementation, the messages will be
        the actual node features
        Args:
            X: Node Features Matrix
        Return:
            Msg: Messages passed through neighborhood
        """
        messages = X

        return messages
    

    def aggregate(self, messages, A):
        """
        Aggregate the messages from neighbors
        Args:
            messages: Messages from neighborhood
            A: Adjacency Matrix
        Return:
            aggr_messages
        """
        if self.aggr == "sum":
            # M is the Aggregated Neighborhood Message Matrix,
            # where each row i contains the sum of 
            # the features of node i's neighbors 
            M = A @ messages
            aggr_messages = M
        else:
            M = A @ messages
            degrees = np.sum(A, axis=1).reshape(-1,1)
            safe_degrees = np.where(degrees > 0, degrees, 1.0)
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
    

    def propagate(self, X, A):
        """
        Forward Propagation
        Args:
            X: Node Features Matrix
            A: Adjacency Matrix
        Return:
            H: Node Representation Matrix after updates
        """
        out_messages = self.message(X)

        aggr_out = self.aggregate(out_messages, A)
        
        H = self.update(aggr_out)

        return H