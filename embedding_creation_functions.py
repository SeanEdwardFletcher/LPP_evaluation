from tqdm import tqdm
import torch
from transformers import (AutoTokenizer, AutoModel)
import os
from preprocessing_functions import remove_french
import re


model_name = "nlpaueb/legal-bert-base-uncased"  # LegalBERT model name
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def create_sliding_windows(tokens, the_tokenizer, max_len, stride=128):
    """
    Create sliding windows of tokens with a fixed overlap.

    Args:
        tokens (list): List of token IDs to process.
        the_tokenizer: Tokenizer instance with a pad_token_id attribute.
        max_len (int): Maximum length of each window (including padding if necessary).
        stride (int): Number of tokens to overlap between consecutive windows.

    Returns:
        list: List of token windows with padding if needed.
    """
    cls_token_id = the_tokenizer.cls_token_id  # Get CLS token ID from tokenizer
    windows = []
    for i in range(0, len(tokens), max_len - stride):
        window = tokens[i:i + max_len - 1]  # Leave space for CLS token
        if i == 0 and tokens[0] == cls_token_id:  # Avoid duplicating CLS for the first window
            window = tokens[:max_len]  # Use original tokens directly
        else:
            window = [cls_token_id] + window  # Prepend CLS token
        if len(window) < max_len:
            window += [the_tokenizer.pad_token_id] * (max_len - len(window))  # Pad to max_len
        windows.append(window)
        if len(window) < max_len:
            break
    return windows


def tokenize_and_process(text, the_model, the_tokenizer, max_len=512, stride=128):
    """
    Tokenizes the input text and processes it based on its token length.
    - If token length < max_len: Manually pads tokens and attention mask.
    - If token length > max_len: Uses sliding windows to break into 512-sized sections.

    Args:
        text (str): The input text to process.
        the_model: Pre-trained model for inference.
        the_tokenizer: Tokenizer corresponding to the model.
        max_len (int): Maximum token length for the model (default is 512).
        stride (int): Overlap between consecutive windows (default is 128).

    Returns:
        List of model outputs for each processed segment.
    """
    # Tokenize the input without padding or truncation
    tokenized = the_tokenizer(text, add_special_tokens=True, return_attention_mask=True, truncation=False)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    paragraph_embeddings = []

    if len(input_ids) <= max_len:
        # If token length is less than or equal to max_len, manually pad
        pad_length = max_len - len(input_ids)
        input_ids += [the_tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length

        # Convert to PyTorch tensors
        input_ids_tensor = torch.tensor([input_ids])
        attention_mask_tensor = torch.tensor([attention_mask])

        # Get model output
        with torch.no_grad():
            output = the_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            cls_embedding = output.last_hidden_state[:, 0, :].squeeze(0)  # Extract CLS embedding
            return cls_embedding
    else:
        # If token length exceeds max_len, use sliding windows
        windows = create_sliding_windows(input_ids, the_tokenizer, max_len, stride)
        averaged_embedding = average_window_embeddings_from_windows(windows, the_model, the_tokenizer)
        return averaged_embedding


def average_window_embeddings_from_windows(windows, the_model, the_tokenizer):
    """
    Averages the CLS embeddings produced for a list of tokenized windows.

    Args:
        windows (list of lists): List of tokenized windows (each window is a list of token IDs).
        the_model: Pre-trained model for inference.
        the_tokenizer: Tokenizer corresponding to the model.

    Returns:
        torch.Tensor: A single embedding representing the input text.
    """
    cls_embeddings = []  # List to store CLS embeddings

    # Process each window
    for window in windows:
        # Generate the attention mask for the current window
        attention_mask = [1 if token != the_tokenizer.pad_token_id else 0 for token in window]

        # Convert window and attention mask to PyTorch tensors
        input_ids_tensor = torch.tensor([window])
        attention_mask_tensor = torch.tensor([attention_mask])

        # Get model output
        with torch.no_grad():
            output = the_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            cls_embedding = output.last_hidden_state[:, 0, :]  # Extract CLS token embedding
            cls_embeddings.append(cls_embedding.squeeze(0))  # Remove batch dimension

    # Stack all CLS embeddings into a single tensor and compute the average
    stacked_embeddings = torch.stack(cls_embeddings, dim=0)  # Shape: (num_windows, hidden_size)
    averaged_embedding = torch.mean(stacked_embeddings, dim=0)  # Shape: (hidden_size)

    return averaged_embedding


def save_document_embedding(embedding, output_dir, file_name):
    """Save the averaged document embedding."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(embedding, os.path.join(output_dir, f"{file_name}.pt"))


def average_cls_embeddings(cls_embeddings):
    """
    Computes the average of a list of CLS embeddings.

    Args:
        cls_embeddings (list of torch.Tensor): List of CLS embeddings, where each embedding
                                               is a 1D tensor of shape [hidden_size].

    Returns:
        torch.Tensor: A single average embedding of shape [hidden_size].
    """
    # Stack embeddings into a single tensor of shape [num_embeddings, hidden_size]
    stacked_embeddings = torch.stack(cls_embeddings, dim=0)

    # Compute the mean along the first dimension (num_embeddings)
    average_embedding = torch.mean(stacked_embeddings, dim=0)

    return average_embedding

