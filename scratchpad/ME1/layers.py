import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce


def extract_patches(x, kernel_size, padding):
    """(B, C, H, W) -> (B, H_out, W_out, C*kernel_size*kernel_size).

    Builds the sliding-window indices ourselves with plain arithmetic
    (torch.arange + broadcasting), then gathers with plain tensor indexing —
    no function whose purpose is specifically "extract conv/sliding windows"
    (e.g. Tensor.unfold or F.unfold) is used anywhere here.
    """
    x = F.pad(x, (padding, padding, padding, padding))
    height_out = x.shape[2] - kernel_size + 1
    width_out = x.shape[3] - kernel_size + 1

    # row_index[h, kh] = input row read by output row h, kernel offset kh
    row_index = torch.arange(height_out)[:, None] + torch.arange(kernel_size)[None, :]
    col_index = torch.arange(width_out)[:, None] + torch.arange(kernel_size)[None, :]

    windows = x[:, :, row_index, :]  # (B, C, H_out, kh, W_pad)
    windows = windows[:, :, :, :, col_index]  # (B, C, H_out, kh, W_out, kw)

    return rearrange(
        windows,
        "batch channels height kernel_h width kernel_w "
        "-> batch height width (channels kernel_h kernel_w)",
    )


def conv2d(x, weight, bias, padding):
    """weight: (C_out, C_in, kh, kw), bias: (C_out,)."""
    kernel_size = weight.shape[-1]
    patches = extract_patches(x, kernel_size, padding)
    weight_flat = rearrange(
        weight,
        "out_channels in_channels kernel_h kernel_w "
        "-> out_channels (in_channels kernel_h kernel_w)",
    )
    out = einsum(
        patches,
        weight_flat,
        "batch height width patch, out_channels patch -> batch out_channels height width",
    )
    return out + bias[None, :, None, None]


def max_pool2d(x, kernel_size):
    """(B, C, H, W) -> (B, C, H/kernel_size, W/kernel_size)."""
    return reduce(
        x,
        "batch channels (height pool_h) (width pool_w) -> batch channels height width",
        "max",
        pool_h=kernel_size,
        pool_w=kernel_size,
    )


def linear(x, weight, bias):
    """x: (B, in_features), weight: (out_features, in_features)."""
    return einsum(x, weight, "batch in_features, out_features in_features -> batch out_features") + bias


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- conv2d correctness check against F.conv2d ---
    x = torch.randn(4, 3, 10, 10)
    weight = torch.randn(6, 3, 3, 3)
    bias = torch.randn(6)
    ours = conv2d(x, weight, bias, padding=1)
    reference = F.conv2d(x, weight, bias, stride=1, padding=1)
    print("conv2d shape:", ours.shape, "matches F.conv2d:", torch.allclose(ours, reference, atol=1e-5))

    # --- max_pool2d correctness check against F.max_pool2d ---
    ours_pool = max_pool2d(x, kernel_size=2)
    reference_pool = F.max_pool2d(x, kernel_size=2)
    print("max_pool2d shape:", ours_pool.shape, "matches F.max_pool2d:", torch.allclose(ours_pool, reference_pool))

    # --- linear correctness check against F.linear ---
    x_flat = torch.randn(4, 20)
    w_lin = torch.randn(10, 20)
    b_lin = torch.randn(10)
    ours_lin = linear(x_flat, w_lin, b_lin)
    reference_lin = F.linear(x_flat, w_lin, b_lin)
    print("linear shape:", ours_lin.shape, "matches F.linear:", torch.allclose(ours_lin, reference_lin, atol=1e-5))
