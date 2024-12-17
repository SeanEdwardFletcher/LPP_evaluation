from linear_pyramid_pooling import *
import torch
import torch.nn as nn


class TwoLayerBiLSTMCrossEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.3, pooling="max"):  #, llp_pooling="avg", lpp=False, lpp_levels=[16, 32, 64, 128, 256]):
        super(TwoLayerBiLSTMCrossEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.pooling = pooling
        self.embedding_dim = embedding_dim
        # self.lpp = lpp

        # # Adjust embedding dimension based on LPP
        # if self.lpp:
        #     self.lpp_module = PyramidPooling1D(levels=lpp_levels, pooling=llp_pooling)  # Apply LPP if enabled
        #     self.embedding_dim = sum(lpp_levels)  # New dimension after LPP
        # else:
        #     self.embedding_dim = embedding_dim

        # Two-layer Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim * 2,  # Adjusted input size based on LPP
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout applies only if num_layers > 1
            batch_first=True
        )

        # Batch Normalization for LSTM output
        self.lstm_bn = nn.BatchNorm1d(hidden_dim * self.num_directions)

        # LeakyReLU layer added after LSTM output
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

        # Fully connected scoring layer with Batch Normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 128),
            nn.BatchNorm1d(128),  # Batch normalization before LeakyReLU
            nn.LeakyReLU(),
            nn.Linear(128, 1)  # Final similarity score
        )

        # Apply Kaiming initialization to non-BiLSTM weights and Orthogonal initialization to BiLSTM weights
        self._initialize_weights()

    def forward(self, query_embedding, candidate_embedding):

        # # Ensure embeddings have a sequence dimension, required for LSTMs
        # if query_embedding.dim() == 2:
        #     query_embedding = query_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
        # if candidate_embedding.dim() == 2:
        #     candidate_embedding = candidate_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
        #
        # # Apply Linear Pyramid Pooling if enabled
        # if self.lpp:
        #     query_embedding = self.lpp_module(query_embedding)
        #     candidate_embedding = self.lpp_module(candidate_embedding)
        #     # print(f"LPP Query Embedding Shape: {query_embedding.shape}")  # Debugging
        #     # print(f"LPP Candidate Embedding Shape: {candidate_embedding.shape}")  # Debugging

        # Ensure embeddings have a sequence dimension, required for LSTMs
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]
        if candidate_embedding.dim() == 2:
            candidate_embedding = candidate_embedding.unsqueeze(1)  # [batch, 1, embedding_dim]

        # Concatenate query and candidate embeddings along the sequence dimension
        combined_input = torch.cat([query_embedding, candidate_embedding], dim=-1)  # [batch, seq_len, embedding_dim * 2]
        # print(f"Combined Input Shape: {combined_input.shape}")  # Debugging

        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(combined_input)  # [batch, seq_len, hidden_dim * num_directions]
        # print(f"LSTM Output Shape: {lstm_out.shape}")  # Debugging

        # Reshape for BatchNorm: [batch, seq_len, hidden_dim * num_directions] -> [batch, hidden_dim * num_directions, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)

        # Apply BatchNorm along the feature dimension
        lstm_out = self.lstm_bn(lstm_out)  # Normalize before activation

        # Reshape back: [batch, hidden_dim * num_directions, seq_len] -> [batch, seq_len, hidden_dim * num_directions]
        lstm_out = lstm_out.permute(0, 2, 1)

        # Apply LeakyReLU after BatchNorm
        lstm_out = self.leakyrelu(lstm_out)

        # Pooling: Max pooling or average pooling over the sequence dimension
        if self.pooling == "max":
            pooled_output = torch.max(lstm_out, dim=1).values  # [batch, hidden_dim * num_directions]
        elif self.pooling == "avg":
            pooled_output = torch.mean(lstm_out, dim=1)  # [batch, hidden_dim * num_directions]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # Compute similarity score
        logits = self.fc(pooled_output)  # [batch, 1]
        return logits

    def _initialize_weights(self):
        """
        Initialize weights for the model using Kaiming initialization for non-LSTM layers
        and Orthogonal initialization for LSTM weights.
        """
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name or 'weight_hh' in name:
                    # Orthogonal initialization for LSTM weights
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # Initialize biases to zeros, except forget gate bias
                    nn.init.zeros_(param)
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)  # Forget gate bias to 1
            else:
                # Apply Kaiming initialization only to parameters with 2 or more dimensions
                if param.dim() >= 2:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.zeros_(param)
