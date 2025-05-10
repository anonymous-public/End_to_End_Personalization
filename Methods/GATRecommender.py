

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
#
#
# class GATRecommender(nn.Module):
#     """
#     A scalable GAT Autoencoder for recommender systems.
#
#     This implementation separates the encoder and decoder so that you can
#     experiment with different configurations (e.g. number of layers, dimensions,
#     attention heads, dropout rates, etc.) easily.
#
#     The encoder processes the input node features into a (typically lower-
#     dimensional) bottleneck embedding, and the decoder reconstructs the original
#     features from the bottleneck.
#
#     Parameters
#     ----------
#     input_dim : int
#         Dimensionality of the input node features.
#     encoder_dims : list of int
#         Output dimensions for each encoder layer.
#     decoder_dims : list of int
#         Output dimensions for each decoder layer.
#     encoder_heads : list of int
#         Number of attention heads for each encoder layer.
#     decoder_heads : list of int
#         Number of attention heads for each decoder layer.
#     encoder_dropouts : list of float
#         Dropout rates for each encoder layer.
#     decoder_dropouts : list of float
#         Dropout rates for each decoder layer.
#     edge_dim : int
#         Dimension of edge features (passed to GATv2Conv).
#     initial_node_features : torch.Tensor
#         Initial node features. Registered as a learnable parameter.
#     bottleneck_dim : int
#         Desired dimension for the bottleneck embedding. If this value differs from
#         the output of the last encoder layer, a linear projection is applied.
#     activation : callable, optional
#         Activation function to use (default: F.elu).
#     """
#
#     def __init__(self,
#                  input_dim=128,
#                  encoder_dims=[64, 32],
#                  decoder_dims=[64, 128],
#                  encoder_heads=[4, 4],
#                  decoder_heads=[4, 1],
#                  encoder_dropouts=[0.3, 0.3],
#                  decoder_dropouts=[0.2, 0.1],
#                  edge_dim=1,
#                  initial_node_features=None,
#                  bottleneck_dim=32,
#                  activation=F.elu):
#         super(GATRecommender, self).__init__()
#
#         if initial_node_features is None:
#             raise ValueError("initial_node_features must be provided")
#         self.activation = activation
#
#         # Register the initial node features as a learnable parameter.
#         self.node_embeddings = nn.Parameter(initial_node_features.clone())
#
#         # --- Build Encoder ---
#         # (The encoder compresses the input to a lower-dimensional bottleneck.)
#         assert len(encoder_dims) == len(encoder_heads) == len(encoder_dropouts), \
#             "Encoder configuration lists must have the same length"
#         self.encoder_layers = nn.ModuleList()
#         self.encoder_norms = nn.ModuleList()
#         self.encoder_residuals = nn.ModuleList()
#         in_channels = input_dim
#
#         for i, out_channels in enumerate(encoder_dims):
#             heads = encoder_heads[i]
#             dropout = encoder_dropouts[i]
#             # For all encoder layers except the last one we use concatenation.
#             concat = True if i < len(encoder_dims) - 1 else False
#             if concat:
#                 # Ensure that the output dimension is divisible by the number of heads.
#                 assert out_channels % heads == 0, \
#                     f"Encoder layer {i}: out_channels must be divisible by heads"
#             conv = GATv2Conv(in_channels,
#                              out_channels // heads if concat else out_channels,
#                              heads=heads,
#                              concat=concat,
#                              dropout=dropout,
#                              edge_dim=edge_dim)
#             self.encoder_layers.append(conv)
#             self.encoder_norms.append(nn.LayerNorm(out_channels))
#             # Add a residual connection – if the input and output dimensions differ,
#             # add a linear projection.
#             if in_channels != out_channels:
#                 self.encoder_residuals.append(nn.Linear(in_channels, out_channels))
#             else:
#                 self.encoder_residuals.append(nn.Identity())
#             in_channels = out_channels
#
#         # If a bottleneck dimension different from the last encoder output is desired,
#         # project to that dimension.
#         self.bottleneck_dim = bottleneck_dim
#         if bottleneck_dim != encoder_dims[-1]:
#             self.bottleneck_projection = nn.Linear(encoder_dims[-1], bottleneck_dim)
#         else:
#             self.bottleneck_projection = nn.Identity()
#
#         # --- Build Decoder ---
#         # (The decoder reconstructs the original input from the bottleneck embedding.)
#         assert len(decoder_dims) == len(decoder_heads) == len(decoder_dropouts), \
#             "Decoder configuration lists must have the same length"
#         self.decoder_layers = nn.ModuleList()
#         self.decoder_norms = nn.ModuleList()
#         self.decoder_residuals = nn.ModuleList()
#         in_channels = self.bottleneck_dim
#
#         for i, out_channels in enumerate(decoder_dims):
#             heads = decoder_heads[i]
#             dropout = decoder_dropouts[i]
#             concat = True if i < len(decoder_dims) - 1 else False
#             if concat:
#                 assert out_channels % heads == 0, \
#                     f"Decoder layer {i}: out_channels must be divisible by heads"
#             conv = GATv2Conv(in_channels,
#                              out_channels // heads if concat else out_channels,
#                              heads=heads,
#                              concat=concat,
#                              dropout=dropout,
#                              edge_dim=edge_dim)
#             self.decoder_layers.append(conv)
#             self.decoder_norms.append(nn.LayerNorm(out_channels))
#             if in_channels != out_channels:
#                 self.decoder_residuals.append(nn.Linear(in_channels, out_channels))
#             else:
#                 self.decoder_residuals.append(nn.Identity())
#             in_channels = out_channels
#
#     def encode(self, edge_index, edge_attr):
#         """Passes the input through the encoder to obtain the bottleneck embedding."""
#         x = self.node_embeddings
#         for i, conv in enumerate(self.encoder_layers):
#             x_residual = x
#             x = conv(x, edge_index, edge_attr)
#             x = self.encoder_norms[i](x)
#             x = self.activation(x)
#             x_residual = self.encoder_residuals[i](x_residual)
#             x = x + x_residual
#         # Project (if needed) to the specified bottleneck dimension.
#         z = self.bottleneck_projection(x)
#         return z
#
#     def decode(self, z, edge_index, edge_attr):
#         """Reconstructs the node features from the bottleneck embedding."""
#         x = z
#         for i, conv in enumerate(self.decoder_layers):
#             x_residual = x
#             x = conv(x, edge_index, edge_attr)
#             x = self.decoder_norms[i](x)
#             x = self.activation(x)
#             x_residual = self.decoder_residuals[i](x_residual)
#             x = x + x_residual
#         return x
#
#     def forward(self, data):
#         """
#         Performs a forward pass through the autoencoder.
#
#         Parameters
#         ----------
#         data : torch_geometric.data.Data
#             A PyTorch Geometric data object containing at least:
#               - data.edge_index: Graph connectivity.
#               - data.edge_attr: Edge features.
#
#         Returns
#         -------
#         recon : torch.Tensor
#             The reconstructed node features.
#         embeddings : torch.Tensor
#             The bottleneck embeddings.
#         """
#         edge_index, edge_attr = data.edge_index, data.edge_attr
#         embeddings = self.encode(edge_index, edge_attr)
#         recon = self.decode(embeddings, edge_index, edge_attr)
#         return recon, embeddings

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATRecommender(nn.Module):
    """
    A flexible GAT-based recommender system module that allows:
      - Variable number of layers
      - Different dimensionality in each layer
      - Different dropout/head configurations per layer

    Usage Example and default value:
    --------------
    model = GATRecommender(
        input_dim=768,
        layer_dims=[64, 64, 64],
        heads=[4, 4, 1],
        dropouts=[0.1, 0.1, 0.1],
        initial_node_features=some_tensor_of_shape_(num_nodes, 768)
    )
    """
    def __init__(
        self,
        input_dim: int,
        layer_dims: list,
        heads: list,
        dropouts: list,
        initial_node_features: torch.Tensor
    ):
        """
        :param input_dim:  Dimension of the input node embeddings.
        :param layer_dims: List of output dimensions for each layer, e.g. [128, 64, 32].
        :param heads:      List of number of heads for each layer, e.g. [4, 4, 1].
        :param dropouts:   List of dropout probabilities for each layer, e.g. [0.3, 0.2, 0.1].
        :param initial_node_features: A Tensor of shape (num_nodes, input_dim) containing
                                      the initial embedding for each node.
        """
        super(GATRecommender, self).__init__()

        # Validate arguments
        assert (
            len(layer_dims) == len(heads) == len(dropouts)
        ), "layer_dims, heads, and dropouts must all be the same length."

        # Turn initial node features into a trainable parameter
        self.node_embeddings = nn.Parameter(initial_node_features.clone())

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # Build GAT layers one by one
        for i in range(len(layer_dims)):
            # Determine in/out dimensions for GAT
            if i == 0:
                in_channels = input_dim
            else:
                # If the previous layer used multiple heads (concat=True), the in_channels
                # become (layer_dims[i-1] * heads[i-1]).
                # If the previous layer used 1 head (concat=False), it's just layer_dims[i-1].
                prev_heads = heads[i - 1]
                if prev_heads > 1:
                    in_channels = layer_dims[i - 1] * prev_heads
                else:
                    in_channels = layer_dims[i - 1]

            out_channels = layer_dims[i]
            n_heads = heads[i]
            dropout_p = dropouts[i]

            # If we have multiple heads, we usually set concat=True. If heads=1, typically concat=False.
            concat = (n_heads > 1)

            # Create a GATv2Conv layer
            gat_layer = GATv2Conv(
                in_channels,
                out_channels,
                heads=n_heads,
                concat=concat,
                dropout=dropout_p,
                edge_dim=1
            )
            self.layers.append(gat_layer)

            # LayerNorm dimension depends on whether heads>1 and concat=True
            if concat:
                norm_dim = out_channels * n_heads
            else:
                norm_dim = out_channels

            self.layer_norms.append(nn.LayerNorm(norm_dim))

            # Residual projection: if input channels != output channels, project
            if in_channels != norm_dim:
                self.residual_projections.append(nn.Linear(in_channels, norm_dim))
            else:
                self.residual_projections.append(nn.Identity())

    def forward(self, data):
        """
        :param data: A Data object (from PyG) with attributes:
                     data.edge_index: shape (2, E)
                     data.edge_attr:  shape (E,) or (E, dim)
        :return: (bottleneck_embedding, final_embeddings)
                 - bottleneck_embedding can be None or an intermediate representation
                   if you want to extract it for analysis.
                 - final_embeddings is the final node embedding matrix of shape (num_nodes, last_layer_dim).
        """
        x = self.node_embeddings
        edge_index, edge_attr = data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x_residual = x
            x = layer(x, edge_index, edge_attr)
            x = self.layer_norms[i](x)
            x = F.leaky_relu(x)

            # Residual connection
            x_residual = self.residual_projections[i](x_residual)
            x = x + x_residual

        # You can choose to return an intermediate layer's output as a "bottleneck"
        # if you need it for some special usage, or just return None.
        bottleneck_embedding = None
        return bottleneck_embedding, x


    def forward_minibatch(self, n_id, adjs):
        """
        Mini-batch forward pass using neighbor sampling subgraphs.
          n_id:  1D Tensor of node indices in this mini-batch subgraph
          adjs:  list of (edge_index, e_id, size) for each GAT layer
        Returns:
          x      final embeddings for all nodes in the subgraph (aligned with n_id).
        """
        # x has shape [n_id.size(0), embedding_dim]
        x = self.node_embeddings[n_id]

        for i, (edge_index, _, size) in enumerate(adjs):
            x_src, x_dst = x, x[:size[1]]

            x_updated = self.layers[i]((x_src, x_dst), edge_index)
            x_updated = self.layer_norms[i](x_updated)
            x_updated = F.leaky_relu(x_updated)

            # IMPORTANT FIX:
            # Use x_dst (not x) in the residual projection so that x_residual
            # matches x_updated in shape:
            x_residual = self.residual_projections[i](x_dst)

            x = x_updated + x_residual

        return x


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATBase(nn.Module):
    """
    A flexible GAT-based recommender system module that allows:
      - Variable number of layers
      - Different dimensionality in each layer
      - Different dropout/head configurations per layer

    Usage Example:
    --------------
    model = GATRecommender(
        input_dim=768,
        num_nodes=total_nodes,  # total number of nodes (users + items)
        layer_dims=[64, 64, 64],
        heads=[4, 4, 1],
        dropouts=[0.1, 0.1, 0.1],
    )
    """
    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        layer_dims: list,
        heads: list,
        dropouts: list
    ):
        """
        :param input_dim:  Dimension of the input node embeddings.
        :param num_nodes:  Total number of nodes (users + items) in the graph.
        :param layer_dims: List of output dimensions for each layer, e.g. [128, 64, 32].
        :param heads:      List of number of heads for each layer, e.g. [4, 4, 1].
        :param dropouts:   List of dropout probabilities for each layer, e.g. [0.3, 0.2, 0.1].
        """
        super(GATBase, self).__init__()

        # Validate arguments
        assert (
            len(layer_dims) == len(heads) == len(dropouts)
        ), "layer_dims, heads, and dropouts must all be the same length."

        # Initialize node embeddings randomly and make them trainable.
        # Instead of using pre-calculated embeddings, we create a random tensor.
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, input_dim))

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        # Build GAT layers one by one
        for i in range(len(layer_dims)):
            # Determine in/out dimensions for the GAT layer.
            if i == 0:
                in_channels = input_dim
            else:
                # If the previous layer used multiple heads (with concat=True), then
                # in_channels = (layer_dims[i-1] * heads[i-1]). Otherwise, it's just layer_dims[i-1].
                prev_heads = heads[i - 1]
                if prev_heads > 1:
                    in_channels = layer_dims[i - 1] * prev_heads
                else:
                    in_channels = layer_dims[i - 1]

            out_channels = layer_dims[i]
            n_heads = heads[i]
            dropout_p = dropouts[i]

            # Use concatenation when more than one head is used.
            concat = (n_heads > 1)

            # Create a GATv2Conv layer.
            gat_layer = GATv2Conv(
                in_channels,
                out_channels,
                heads=n_heads,
                concat=concat,
                dropout=dropout_p,
                edge_dim=1  # Assumes that edge attributes have dimension 1 (e.g., ratings)
            )
            self.layers.append(gat_layer)

            # Set up layer normalization.
            if concat:
                norm_dim = out_channels * n_heads
            else:
                norm_dim = out_channels
            self.layer_norms.append(nn.LayerNorm(norm_dim))

            # Create a residual projection if the dimensions do not match.
            if in_channels != norm_dim:
                self.residual_projections.append(nn.Linear(in_channels, norm_dim))
            else:
                self.residual_projections.append(nn.Identity())

    def forward(self, data):
        """
        :param data: A Data object (from PyG) with attributes:
                     - data.edge_index: Tensor of shape (2, E)
                     - data.edge_attr:  Tensor of shape (E,) or (E, dim)
        :return: A tuple (bottleneck_embedding, final_embeddings) where:
                 - bottleneck_embedding: Optionally an intermediate representation (or None)
                 - final_embeddings: The final node embeddings, shape (num_nodes, last_layer_dim)
        """
        x = self.node_embeddings
        edge_index, edge_attr = data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x_residual = x
            x = layer(x, edge_index, edge_attr)
            x = self.layer_norms[i](x)
            x = F.leaky_relu(x)

            # Apply the residual connection.
            x_residual = self.residual_projections[i](x_residual)
            x = x + x_residual

        # Optionally, you can extract an intermediate ("bottleneck") embedding.
        bottleneck_embedding = None
        return bottleneck_embedding, x
