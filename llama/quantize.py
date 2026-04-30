"""
INT4 Weight-Only Quantization for Llama 3.2-1B

This module implements per-group INT4 quantization for nn.Linear layers.
Each group of `group_size` weights shares a scale and zero_point.
Two INT4 values are packed into a single uint8 for storage efficiency.

Student implementation by KPD for EE 508 Phase 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """
    A quantized linear layer that stores weights in INT4 format
    and dequantizes them to FP16 at inference time.

    This replaces nn.Linear, similar to how LoRA replaced Q/V projections last year.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = in_features // group_size

        # Quantized weights: two INT4 values packed into one uint8
        # Shape: (out_features, in_features // 2) because 2 values per byte
        self.register_buffer("packed_weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))

        # Per-group scale and zero_point
        # Shape: (out_features, n_groups)
        self.register_buffer("scale", torch.zeros(out_features, self.n_groups, dtype=torch.float16))
        self.register_buffer("zero_point", torch.zeros(out_features, self.n_groups, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @staticmethod
    def quantize_tensor(weight: torch.Tensor, group_size: int = 128):
        """
        Quantize a FP16 weight tensor to INT4 with per-group scale and zero_point.

        Args:
            weight: FP16 tensor of shape (out_features, in_features)
            group_size: number of weights per group (default: 128)

        Returns:
            packed_weight: uint8 tensor of shape (out_features, in_features // 2)
            scale: FP16 tensor of shape (out_features, n_groups)
            zero_point: FP16 tensor of shape (out_features, n_groups)
        """
        out_features, in_features = weight.shape
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
        n_groups = in_features // group_size

        # Reshape into groups: (out_features, n_groups, group_size). Use FP32
        # for the arithmetic so the round/clamp step is numerically stable.
        w = weight.float().reshape(out_features, n_groups, group_size)

        # Per-group min and max -> shape: (out_features, n_groups)
        w_min = w.amin(dim=-1)
        w_max = w.amax(dim=-1)

        # Per-group scale and zero_point.
        # 15 = 2^4 - 1 is the largest INT4 unsigned value.
        scale = (w_max - w_min) / 15.0
        # Avoid division-by-zero for constant groups: substitute scale=1.
        # The dequantization (q - zp) * scale will then return ~0 for those
        # rare groups; in practice randn weights almost never produce them.
        zero_scale = scale.abs() < 1e-12
        safe_scale = torch.where(zero_scale, torch.ones_like(scale), scale)
        zero_point = torch.round(-w_min / safe_scale).clamp(0.0, 15.0)

        # Quantize each weight to INT4 [0, 15].
        # w_int4 shape: (out_features, n_groups, group_size)
        w_int4 = torch.round(w / safe_scale.unsqueeze(-1) + zero_point.unsqueeze(-1))
        w_int4 = w_int4.clamp_(0.0, 15.0).to(torch.uint8)

        # Flatten the group dim back so we can pack along in_features.
        # Shape: (out_features, in_features)
        w_int4 = w_int4.reshape(out_features, in_features)

        # Pack two INT4 values into one uint8: lower-nibble = even index,
        # upper-nibble = odd index. Resulting shape: (out_features, in_features // 2)
        w_even = w_int4[:, 0::2]
        w_odd = w_int4[:, 1::2]
        packed = (w_even | (w_odd << 4)).to(torch.uint8)

        # Persist the safe scale (so dequant uses the same value we quantized
        # against) and cast to half for storage.
        return packed, safe_scale.half(), zero_point.half()

    @staticmethod
    def dequantize_packed(packed_weight, scale, zero_point, group_size):
        """
        Dequantize packed INT4 weights back to FP16.

        Args:
            packed_weight: uint8 tensor of shape (out_features, in_features // 2)
            scale: FP16 tensor of shape (out_features, n_groups)
            zero_point: FP16 tensor of shape (out_features, n_groups)
            group_size: number of weights per group

        Returns:
            weight: FP16 tensor of shape (out_features, in_features)
        """
        out_features = packed_weight.shape[0]
        in_features = packed_weight.shape[1] * 2
        n_groups = in_features // group_size

        # Unpack the two nibbles. Lower nibble was the even index, upper was odd.
        w_even = (packed_weight & 0x0F).to(torch.float16)
        w_odd = ((packed_weight >> 4) & 0x0F).to(torch.float16)

        # Re-interleave back to (out_features, in_features). torch.stack on the
        # last axis followed by a flatten is the cleanest way; it places
        # w_even at index 2k and w_odd at index 2k+1.
        w_int4 = torch.stack((w_even, w_odd), dim=-1).reshape(out_features, in_features)

        # Reshape to per-group, dequantize, then flatten.
        w_int4 = w_int4.reshape(out_features, n_groups, group_size)
        w_fp16 = (w_int4 - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)
        return w_fp16.reshape(out_features, in_features).to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantize weights, then do matmul.

        Args:
            x: input tensor of shape (..., in_features)

        Returns:
            output tensor of shape (..., out_features)
        """
        # Dequantize weights from INT4 to FP16 every forward pass.
        # This is the "naive" weight-only quantization path: the dequantized
        # FP16 weight tensor is materialized in memory before the matmul, so
        # we pay the FP16 footprint as a transient even though the persistent
        # storage is INT4. A fused dequantize+matmul kernel would avoid this.
        weight = self.dequantize_packed(
            self.packed_weight, self.scale, self.zero_point, self.group_size
        )

        # Standard linear operation in FP16.
        output = F.linear(x, weight, self.bias)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128):
        """
        Create a QuantizedLinear from a pretrained nn.Linear layer.

        Args:
            linear: the original nn.Linear module
            group_size: quantization group size

        Returns:
            QuantizedLinear module with quantized weights
        """
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, group_size=group_size, bias=has_bias)

        # Move the empty buffers onto the same device as the source weights so
        # the cross-device copy below works without surprises (and so the
        # caller doesn't have to remember to .to() afterwards).
        ql = ql.to(linear.weight.device)

        # Quantize the weights
        packed, scale, zp = cls.quantize_tensor(linear.weight.data, group_size)
        ql.packed_weight.copy_(packed)
        ql.scale.copy_(scale.half())
        ql.zero_point.copy_(zp.half())

        if has_bias:
            ql.bias.copy_(linear.bias.data.half())

        return ql


def quantize_model(model, group_size=128):
    """
    Replace all nn.Linear layers in the model with QuantizedLinear.

    Args:
        model: the Llama model
        group_size: quantization group size

    Returns:
        The model with quantized linear layers (modified in-place)
    """
    # Collect first, mutate after — we cannot safely modify a module tree
    # while iterating over named_modules().
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features % group_size == 0:
            to_replace.append((name, module))

    for full_name, linear in to_replace:
        ql = QuantizedLinear.from_linear(linear, group_size=group_size)

        # Walk to the parent module so we can setattr the new submodule.
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, ql)

        # Drop the original FP16 weight tensor immediately so the swap is
        # actually a memory win during conversion.
        del linear

    # Best-effort cleanup of any lingering FP16 weight tensors.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def print_model_size(model):
    """Print the model size in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_mb = (param_bytes + buffer_bytes) / (1024 ** 2)
    print(f"Model size: {total_mb:.2f} MB")
    print(f"  Parameters: {param_bytes / (1024**2):.2f} MB")
    print(f"  Buffers: {buffer_bytes / (1024**2):.2f} MB")
    return total_mb
