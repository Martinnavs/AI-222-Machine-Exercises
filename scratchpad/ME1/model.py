import math

import torch
import torch.nn as nn
from einops import rearrange

from layers import conv2d, max_pool2d, linear


def _kaiming_conv(out_channels, in_channels, kernel_size):
    fan_in = in_channels * kernel_size * kernel_size
    std = math.sqrt(2.0 / fan_in)
    return torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std


def _kaiming_linear(out_features, in_features):
    std = math.sqrt(2.0 / in_features)
    return torch.randn(out_features, in_features) * std


class EinopsCNN(nn.Module):
    """3 learnable layers: Conv1 -> Conv2 -> Linear(classifier head).

    nn.Module/nn.Parameter are used only as storage (for autograd registration
    and .parameters()) -- forward() contains only our own einops/einsum math,
    no nn.Conv2d/nn.Linear/nn.Sequential.
    """

    def __init__(self):
        super().__init__()
        self.conv1_weight = nn.Parameter(_kaiming_conv(8, 1, 3))
        self.conv1_bias = nn.Parameter(torch.zeros(8))
        self.conv2_weight = nn.Parameter(_kaiming_conv(16, 8, 3))
        self.conv2_bias = nn.Parameter(torch.zeros(16))
        self.classifier_weight = nn.Parameter(_kaiming_linear(10, 16 * 7 * 7))
        self.classifier_bias = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = conv2d(x, self.conv1_weight, self.conv1_bias, padding=1)
        x = x.relu()
        x = max_pool2d(x, kernel_size=2)

        x = conv2d(x, self.conv2_weight, self.conv2_bias, padding=1)
        x = x.relu()
        x = max_pool2d(x, kernel_size=2)

        x = rearrange(x, "batch channels height width -> batch (channels height width)")
        return linear(x, self.classifier_weight, self.classifier_bias)


if __name__ == "__main__":
    model = EinopsCNN()
    dummy_batch = torch.randn(5, 1, 28, 28)
    logits = model(dummy_batch)
    print("output shape:", logits.shape)
    print("num learnable tensors (params):", sum(1 for _ in model.parameters()))
    print("total param count:", sum(p.numel() for p in model.parameters()))

    loss = logits.sum()
    loss.backward()
    print("grad flows to conv1_weight:", model.conv1_weight.grad is not None)
