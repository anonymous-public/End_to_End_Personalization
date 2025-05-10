"""
This class will convert users and movies to vector embeddings
"""

# Importing the necessary libraries
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F


class EmbeddingLayer:

    def __init__(self):
        self.model, self.tokenizer = self.get_model()

    def convert_to_embeddings(self, df, col, batch_size=32):
        """
        Process the dataframe in batches and track the progress.
        """

        # Filter out rows where the column is missing or empty
        df = df[df[col].notna() & (df[col].str.strip() != "")]

        n_batches = (len(df) + batch_size - 1) // batch_size  # Calculate the number of batches

        all_embeddings = []

        for i in tqdm(range(n_batches), desc=f"Processing Embeddings ..."):
            batch_texts = df[col].iloc[i * batch_size: (i + 1) * batch_size].tolist()
            embeddings_batch = self.get_embedding(batch_texts)
            all_embeddings.extend(embeddings_batch)

        embeddings_array = np.array(all_embeddings)

        # creating a new column for embeddings
        df[f"{col}_vector"] = list(embeddings_array)

        return df

    def get_embedding(self, texts):
        """
        Get embeddings for batch of texts
        """
        # Filtering out None values, replacing them with None in the final result
        valid_texts_indices = [i for i, text in enumerate(texts) if text is not None]
        valid_texts = ["clustering: " + texts[i] for i in valid_texts_indices]

        # Running the preprocessing on GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        if valid_texts:
            inputs = self.tokenizer(valid_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                model_output = self.model(**inputs)

            embeddings = self.mean_pooling(model_output, attention_mask)
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            embeddings = np.around(embeddings, decimals=5).astype(np.float32)

            # Create the result list with None for invalid indices
            result = [None] * len(texts)
            for idx, embedding in zip(valid_texts_indices, embeddings):
                result[idx] = embedding
        else:
            result = [None] * len(texts)

        return result

    # ----------------------------------------------------------------------------------------------------
    def mean_pooling(self, model_output, attention_mask):
        """
        Apply pooling on the result of the embedding model
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # ----------------------------------------------------------------------------------------------------
    def get_model(self):

        # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True,
        #                                   safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)

        return model, tokenizer

