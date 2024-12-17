import torch
import torch.nn.functional as F
from torch import nn


class PyramidPooling1D(nn.Module):
    def __init__(self, levels, pooling="avg"):
        """
        :param levels: List of pyramid levels (e.g., [1, 2, 4]).
        :param pooling: Pooling method, either "avg" or "max".
        """
        super(PyramidPooling1D, self).__init__()
        self.levels = levels
        self.pooling = pooling

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, channels, length)
        :return: Fixed-length feature vector for each input in the batch
        """
        batch_size, channels, length = x.size()
        pooled_features = []

        for level in self.levels:
            if self.pooling == "max":
                # Apply adaptive max pooling
                pooled = F.adaptive_max_pool1d(x, output_size=level)
            elif self.pooling == "avg":
                # Apply adaptive average pooling
                pooled = F.adaptive_avg_pool1d(x, output_size=level)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling}")

            # Flatten each level
            pooled_features.append(pooled.reshape(batch_size, -1))

        # Concatenate features from all levels
        return torch.cat(pooled_features, dim=1)


# # Example usage
# input_vector = torch.rand(2, 1, 4096)  # Batch size 2, single channel, length 4096
# pyramid_pooling = PyramidPooling1D(levels=[16, 32, 64, 128, 256])
# output = pyramid_pooling(input_vector)
# print(output.shape)  # Output shape: (2, 1 * (1 + 2 + 4))
