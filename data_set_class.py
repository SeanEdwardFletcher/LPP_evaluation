import json
import torch
from torch.utils.data import Dataset
import os
from linear_pyramid_pooling import PyramidPooling1D


class LegalTextEmbeddingDataset(Dataset):
    def __init__(self, positive_file, negative_file, embedding_folder, embedding_type="document_embedding", lpp=False, lpp_levels=[16, 32, 64, 128, 256], lpp_pooling="avg"):
        """
        Dataset for legal text embeddings.

        Args:
            positive_file (str): Path to the JSON file with positive examples.
            negative_file (str): Path to the JSON file with negative examples.
            embedding_folder (str): Path to the folder containing embedding subfolders.
            embedding_type (str): Type of embedding to use, either "document_embedding" or "paragraph_embeddings".
            lpp (bool): Whether to apply Linear Pyramid Pooling (LPP) to embeddings.
            lpp_levels (list): Levels for LPP.
            lpp_pooling (str): Pooling method for LPP ("avg" or "max").
        """
        self.data = []
        self.embedding_folder = embedding_folder
        self.embedding_type = embedding_type
        self.lpp = lpp

        # Initialize LPP if enabled
        if self.lpp:
            self.lpp_module = PyramidPooling1D(levels=lpp_levels, pooling=lpp_pooling)

        # Validate embedding type
        if embedding_type not in {"document_embedding", "paragraph_embeddings"}:
            raise ValueError("embedding_type must be either 'document_embedding' or 'paragraph_embeddings'")

        # Load positive and negative examples
        positive_examples = self._load_json(positive_file)
        negative_examples = self._load_json(negative_file)

        # Ensure symmetry between positive and negative examples
        assert set(positive_examples.keys()) == set(
            negative_examples.keys()), "Query IDs must match in positive and negative JSON files."

        # Parse data
        for query_id in positive_examples:
            positive_ids = positive_examples[query_id]
            negative_ids = negative_examples[query_id]

            # Add positive examples (label = 1)
            for doc_id in positive_ids:
                self.data.append((query_id, doc_id, 1))

            # Add negative examples (label = 0)
            for doc_id in negative_ids:
                self.data.append((query_id, doc_id, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_id, doc_id, label = self.data[idx]

        # Load precomputed embeddings
        query_embedding = self._load_embedding(query_id)
        doc_embedding = self._load_embedding(doc_id)

        # Apply LPP if enabled
        if self.lpp:
            query_embedding = self.lpp_module(query_embedding.unsqueeze(0).unsqueeze(0)).squeeze(0)
            doc_embedding = self.lpp_module(doc_embedding.unsqueeze(0).unsqueeze(0)).squeeze(0)

        return query_embedding, doc_embedding, torch.tensor(label, dtype=torch.float32)

    def _load_json(self, file_path):
        """
        Load a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Loaded JSON data.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def _load_embedding(self, file_id):
        """
        Load the embedding from the folder structure.

        Args:
            file_id (str): ID of the embedding folder.

        Returns:
            torch.Tensor: Concatenated embedding for paragraph embeddings or single embedding for document embedding.
        """
        # Remove any file extension from the ID
        folder_id = os.path.splitext(file_id)[0]

        # Construct the path to the embedding file
        embedding_path = f"{self.embedding_folder}/{folder_id}/{self.embedding_type}.pt"

        try:
            embedding = torch.load(embedding_path)
            if self.embedding_type == "paragraph_embeddings":
                # If multiple embeddings are present, concatenate them
                concatenated_embedding = embedding.view(-1)  # Flatten all paragraph embeddings into a 1D vector
                return concatenated_embedding
            return embedding  # Return the single document embedding as is
        except Exception as e:
            raise ValueError(f"Error loading embedding for {file_id}: {e}")

    # def _load_embedding(self, file_id):
    #     """
    #     Load the embedding from the folder structure.
    #
    #     Args:
    #         file_id (str): ID of the embedding folder.
    #
    #     Returns:
    #         torch.Tensor: Loaded embedding.
    #     """
    #     # Remove any file extension from the ID
    #     folder_id = os.path.splitext(file_id)[0]
    #
    #     # Construct the path to the embedding
    #     embedding_path = f"{self.embedding_folder}/{folder_id}/{self.embedding_type}.pt"
    #     return torch.load(embedding_path)  # Assume embeddings are saved as PyTorch tensors
