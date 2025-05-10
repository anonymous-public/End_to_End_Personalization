import math
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from GATRecommender import GATRecommender
from Recommender import *
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
import torch
from collections import Counter
from sklearn.decomposition import PCA
import umap.umap_ as umap
import pickle


class VGCF(Recommender):
    def __init__(self, content_df, interactions_df, users_df, device=None, model_path=None):
        super().__init__(content_df, interactions_df, users_df)
        # Mappings from IDs to indices
        self.user_mapping = None
        self.item_mapping = None
        # Paths for mappings and embeddings
        self.user_mapping_path = '../model_files/100k/user_mapping.pkl'
        self.item_mapping_path = '../model_files/100k/item_mapping.pkl'
        self.initial_node_features_path = '../model_files/100k/initial_node_features.pt'
        # PyTorch Geometric data object
        self.graph_data = None
        # GNN model
        self.model = None
        # Device (CPU or GPU)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Model path
        self.model_path = model_path if model_path else '../model_files/best_model.pt'

    def build_graph(self):
        # Load or build user and item mappings
        if os.path.exists(self.user_mapping_path) and os.path.exists(self.item_mapping_path):
            with open(self.user_mapping_path, 'rb') as f:
                self.user_mapping = pickle.load(f)
            with open(self.item_mapping_path, 'rb') as f:
                self.item_mapping = pickle.load(f)
            print("Loaded user and item mappings.")
        else:
            # Build mappings as before
            user_ids = self.users_df[self.user_id_column].unique()
            item_ids = self.content_df[self.content_id_column].unique()
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
            self.item_mapping = {item_id: idx + len(user_ids) for idx, item_id in enumerate(item_ids)}

        # Build edge index using vectorized operations
        user_indices = self.interactions_df[self.user_id_column].map(self.user_mapping)
        item_indices = self.interactions_df[self.content_id_column].map(self.item_mapping)

        # Remove any interactions where the mapping failed (i.e., NaNs in user_indices or item_indices)
        valid_indices = user_indices.notna() & item_indices.notna()
        user_indices = user_indices[valid_indices].astype(int)
        item_indices = item_indices[valid_indices].astype(int)

        edge_index = np.vstack((np.concatenate([user_indices, item_indices]),
                                np.concatenate([item_indices, user_indices])))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Build node features
        if os.path.exists(self.initial_node_features_path):
            node_features = torch.load(self.initial_node_features_path, map_location=self.device)
            print("Loaded initial node features.")
        else:
            user_ids = self.users_df[self.user_id_column].unique()
            item_ids = self.content_df[self.content_id_column].unique()
            user_embeddings = self._get_embeddings(self.users_df, self.user_id_column, self.user_attribute_column,
                                                   user_ids)
            item_embeddings = self._get_embeddings(self.content_df, self.content_id_column,
                                                   self.content_attribute_column,
                                                   item_ids)

            node_features = torch.cat([user_embeddings, item_embeddings], dim=0)

        # Edge attributes (ratings)
        ratings = self.interactions_df['rating']
        ratings = ratings[valid_indices]

        # Duplicate ratings for bidirectional edges
        edge_attr = np.concatenate([ratings, ratings])
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1).to(self.device)

        # Create PyTorch Geometric data object
        self.graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(self.device)

    # --------------------------------------------------------------------------

    def _get_embeddings(self, df, id_column, attribute_column, ids):
        """
        Helper function to extract embeddings in the correct order.
        """
        embeddings_df = df[[id_column, attribute_column]].drop_duplicates(subset=id_column)
        embeddings_df = embeddings_df.set_index(id_column)
        embeddings_df = embeddings_df.reindex(ids)
        embeddings = torch.tensor(np.vstack(embeddings_df[attribute_column].values), dtype=torch.float)
        return embeddings

    # --------------------------------------------------------------------------
    def build_model(self):
        # Load initial node features if they exist
        if os.path.exists(self.initial_node_features_path):
            initial_node_features = torch.load(self.initial_node_features_path, map_location=self.device)
            print("Loaded initial node features.")
        else:
            initial_node_features = self.graph_data.x.to(self.device)

        # Define the GAT model
        input_dim = self.graph_data.num_node_features

        # Create the data object with edge_index and edge_attr (from PyTorch Geometric).
        # Initialize the model with your desired configuration.
        # Suppose we want a 3-layer GAT:
        layer_dims = [64, 64, 64]  # out_channels for each of 4 layers
        heads = [4, 4, 1]  # number of heads for each layer
        dropouts = [0.0, 0.0, 0.0]  # dropout for each layer

        self.model = GATRecommender(
            input_dim=input_dim,
            layer_dims=layer_dims,
            heads=heads,
            dropouts=dropouts,
            initial_node_features=initial_node_features
        ).to(self.device)

    # --------------------------------------------------------------------------
    def sample_negative_edges(self, num_negatives=1):
        # Map interactions to indices
        user_indices = self.interactions_df[self.user_id_column].map(self.user_mapping)
        item_indices = self.interactions_df[self.content_id_column].map(self.item_mapping)

        # Remove NaNs
        valid_indices = user_indices.notna() & item_indices.notna()
        user_indices = user_indices[valid_indices].astype(int)
        item_indices = item_indices[valid_indices].astype(int)

        positive_edges = list(zip(user_indices, item_indices))
        positive_edge_set = set(positive_edges)

        negative_edges = []
        np.random.seed(42)

        # For each user index
        unique_user_indices = user_indices.unique()
        for user_idx in unique_user_indices:
            # Get items the user has interacted with
            interacted_items = set(item_indices[user_indices == user_idx])
            # Get available items
            available_items = set(self.item_mapping.values()) - interacted_items
            available_items = list(available_items)
            if len(available_items) >= num_negatives:
                neg_items = np.random.choice(available_items, size=num_negatives, replace=False)
            else:
                neg_items = available_items
            negative_edges.extend([(user_idx, neg_item) for neg_item in neg_items])

        return positive_edges, negative_edges

    # ----------------------------------------------------------------------------

    # # --------------------------------------------------------------------------
    def train_model(self, epochs=300, lr=0.001):

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=5)

        best_loss = float('inf')
        patience_counter = 0
        patience = 15  # Early stopping patience

        # Map interactions to indices
        user_indices = self.interactions_df[self.user_id_column].map(self.user_mapping)
        item_indices = self.interactions_df[self.content_id_column].map(self.item_mapping)
        ratings = self.interactions_df['rating']

        # Remove NaNs
        valid_indices = user_indices.notna() & item_indices.notna() & ratings.notna()
        user_indices = user_indices[valid_indices].astype(int).values
        item_indices = item_indices[valid_indices].astype(int).values
        ratings = ratings[valid_indices].values

        # Separate positive and negative interactions
        positive_mask = ratings >= 4
        negative_mask = ratings <= 2

        pos_user_indices = user_indices[positive_mask]
        pos_item_indices = item_indices[positive_mask]

        neg_user_indices = user_indices[negative_mask]
        neg_item_indices = item_indices[negative_mask]

        # Build user-item interaction mappings
        user_pos_items = {}
        for u_idx, i_idx in zip(pos_user_indices, pos_item_indices):
            user_pos_items.setdefault(u_idx, set()).add(i_idx)

        user_neg_items = {}
        for u_idx, i_idx in zip(neg_user_indices, neg_item_indices):
            user_neg_items.setdefault(u_idx, set()).add(i_idx)

        all_item_indices = np.array(list(self.item_mapping.values()))

        for epoch in range(epochs):
            optimizer.zero_grad()
            bottleneck_embedding, user_embeddings = self.model(self.graph_data)

            # Positive samples
            pos_user_tensor = torch.tensor(pos_user_indices, dtype=torch.long, device=self.device)
            pos_item_tensor = torch.tensor(pos_item_indices, dtype=torch.long, device=self.device)
            pos_user_emb = user_embeddings[pos_user_tensor]
            pos_item_emb = user_embeddings[pos_item_tensor]
            pos_scores = (pos_user_emb * pos_item_emb).sum(dim=1)

            # Negative samples
            num_negatives = 1  # Number of negatives per positive
            neg_user_list = []
            neg_item_list = []
            for u_idx in pos_user_indices:
                # Avoid positive and negative items
                excluded_items = user_pos_items.get(u_idx, set()) | user_neg_items.get(u_idx, set())
                available_items = list(set(all_item_indices) - excluded_items)
                neg_items = np.random.choice(available_items, size=num_negatives, replace=False)
                neg_user_list.extend([u_idx] * num_negatives)
                neg_item_list.extend(neg_items)

            neg_user_tensor = torch.tensor(neg_user_list, dtype=torch.long, device=self.device)
            neg_item_tensor = torch.tensor(neg_item_list, dtype=torch.long, device=self.device)
            neg_user_emb = user_embeddings[neg_user_tensor]
            neg_item_emb = user_embeddings[neg_item_tensor]
            neg_scores = (neg_user_emb * neg_item_emb).sum(dim=1)

            # Compute loss
            pos_scores_expanded = pos_scores.repeat_interleave(num_negatives)

            # Compute BPR loss
            loss_bpr = self.bpr_loss(pos_scores_expanded, neg_scores)

            # Cosine similarity loss
            cos_sim = F.cosine_similarity(pos_user_emb, pos_item_emb)
            loss_cosine = 1 - cos_sim.mean()

            # Total loss
            loss = 5 * loss_bpr + 2 * loss_cosine  # Weight the cosine loss as needed
            # loss = loss_bpr  # Weight the cosine loss as needed

            loss.backward()
            optimizer.step()

            # Update scheduler
            scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), self.model_path)
                # Save mappings
                with open(self.user_mapping_path, 'wb') as f:
                    pickle.dump(self.user_mapping, f)
                with open(self.item_mapping_path, 'wb') as f:
                    pickle.dump(self.item_mapping, f)
                # Save initial node features
                torch.save(self.graph_data.x, self.initial_node_features_path)
                print(f"New Best Loss: Saving the model to {self.model_path}")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def bpr_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()


    def get_recommendations(self, N=50):
        # Build graph and model if not already done
        if self.graph_data is None:
            self.build_graph()

        if self.model is None:
            self.build_model()
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"Loaded model from {self.model_path}")
            else:
                self.train_model()

        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            bottleneck_embedding, node_embeddings = self.model(self.graph_data)
            node_embeddings = node_embeddings.cpu()
        recommendations_list = []
        num_users = len(self.user_mapping)
        num_items = len(self.item_mapping)
        user_indices = list(self.user_mapping.values())
        item_indices = list(self.item_mapping.values())

        user_embeddings = node_embeddings[user_indices]
        item_embeddings = node_embeddings[item_indices]

        # Compute scores
        scores = torch.matmul(user_embeddings, item_embeddings.t()).numpy()  # Shape: [num_users, num_items]

        # Map indices back to IDs
        idx_to_user_id = {idx: user_id for user_id, idx in self.user_mapping.items()}
        idx_to_item_id = {idx: item_id for item_id, idx in self.item_mapping.items()}

        for i, user_idx in enumerate(user_indices):
            user_id = idx_to_user_id[user_idx]
            user_scores = scores[i]

            # Exclude items already interacted with
            interacted_items = self.interactions_df[self.interactions_df[self.user_id_column] == user_id][
                self.content_id_column].unique()
            interacted_item_indices = [self.item_mapping[item_id] - num_users for item_id in interacted_items if
                                       item_id in self.item_mapping]
            user_scores[interacted_item_indices] = -np.inf  # Exclude interacted items

            # Get top N item indices
            top_item_indices = np.argsort(-user_scores)[:N]
            recommended_item_ids = [idx_to_item_id[idx + num_users] for idx in top_item_indices]

            # Build the recommendation list with ranks and module source
            for rank, item_id in enumerate(recommended_item_ids):
                recommendations_list.append({
                    self.user_id_column: user_id,
                    self.content_id_column: item_id,
                    'recommendation_rank': rank + 1,  # Rank starts from 1
                    'module_source': 'VGCF'
                })

        # Create DataFrame
        self.recommendations_df = pd.DataFrame(recommendations_list)

        # Convert data types of the columns to appropriate types
        self.recommendations_df = self.recommendations_df.astype(
            {self.user_id_column: 'int', self.content_id_column: 'int', 'recommendation_rank': 'int'}
        )

        return self.recommendations_df
