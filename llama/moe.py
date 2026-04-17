"""
Mixture-of-Experts (MoE) FeedForward layer for Llama 3.2-1B

This module implements two MoE variants:

1. ``slice`` init — split the pretrained dense FFN into N smaller expert FFNs,
   each with hidden_dim / N.  A learned router selects top-K experts per token.

2. ``lora`` init — keep the original dense FFN **frozen** and attach N lightweight
   LoRA adapters as "experts".  The router selects which adapters to activate.
   This is far more parameter-efficient than copying the full FFN.

Students: Complete the classes and functions marked with TODO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Part A: Slice-based MoE
# ===================================================================

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts FeedForward layer (slice variant).

    Each expert has the same SwiGLU structure (w1, w2, w3) as the original FFN,
    but with hidden_dim / num_experts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        """
        Args:
            dim: input/output dimension (model embedding dim)
            hidden_dim: hidden dimension of the original dense FFN
            num_experts: number of experts (N)
            top_k: number of experts activated per token (K)
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_dim = hidden_dim // num_experts

        # Router: maps input to expert scores
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Experts: each is a small SwiGLU FFN (same structure as Llama FeedForward)
        self.expert_w1 = nn.ModuleList([nn.Linear(dim, self.expert_hidden_dim, bias=False) for _ in range(num_experts)])
        self.expert_w2 = nn.ModuleList([nn.Linear(self.expert_hidden_dim, dim, bias=False) for _ in range(num_experts)])
        self.expert_w3 = nn.ModuleList([nn.Linear(dim, self.expert_hidden_dim, bias=False) for _ in range(num_experts)])

        # For tracking expert activations (used in analysis)
        self._last_routing_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, dim)

        Returns:
            output tensor of shape (batch_size, seq_len, dim)
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]

        # TODO Step 1: Compute router scores and select top-K experts
        # - Pass x_flat through self.router to get logits (num_tokens, num_experts)
        # - Apply softmax to get routing probabilities
        # - Use torch.topk to pick the top-K experts and their weights
        # - Renormalize the top-K weights so they sum to 1 per token
        # - Save top_k_indices to self._last_routing_indices (detached)

        # TODO Step 2: Compute the weighted sum of selected expert outputs
        # - Think about what the output shape should be
        # - For each token, for each of its top-K experts:
        #     call self._expert_forward(expert_idx, token_input)
        #     accumulate: weight * expert_output
        # - Hint: _expert_forward is provided below

        # TODO Step 3: Reshape back to original shape and return

        raise NotImplementedError("Complete the MoEFeedForward forward method")

    def _expert_forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Run a single expert's SwiGLU FFN."""
        return self.expert_w2[expert_idx](
            F.silu(self.expert_w1[expert_idx](x)) * self.expert_w3[expert_idx](x)
        )


# ===================================================================
# Part B: LoRA-based MoE
# ===================================================================
#
# Motivation: slice mode (Part A) splits the pretrained FFN into N smaller
# experts. With top_k < N active experts per token and only a few hundred
# fine-tuning samples, each expert loses ~half of its original capacity and
# the router hasn't learned to specialize — quality drops sharply. LoRA MoE
# solves this by keeping the full FFN intact and adding tiny "diff" experts
# on top.
#
# Recall from LoRA (Hu et al., 2021):
#   ΔW = B @ A,  where A ∈ R^{r×d}, B ∈ R^{d×r}, r << d
#
# Here each expert is one such (A, B) pair added to the frozen FFN output:
#   output = base_ffn(x) + Σ_k  weight_k · lora_expert_k(x)
#
# At initialization, B is zero → every expert contributes zero →
# the converted model behaves identically to the original dense model.
#
# Q: How many parameters does each LoRA expert add compared to one full expert?
# Q: With the base FFN frozen, what guarantees the pre-training quality is
#    preserved even before fine-tuning?

class LoRAExpert(nn.Module):
    """
    A single LoRA adapter used as a lightweight expert.

    Architecture:  x → A (down-project to rank r) → B (up-project back to dim)

    TODO: Implement __init__ and forward.

    Hints:
    - You need two linear layers without bias
    - Think about which matrix should be initialized to zero, and why
      (What should the adapter's output be before any training happens?)
    - The base model runs in FP16, but LoRA gradients in FP16 will NaN.
      Use dtype=torch.float32 for LoRA layers to keep training stable.
    - In forward, cast input to float32, compute, then cast output back.
    - Standard LoRA scales output by alpha/rank so that changing rank
      doesn't change the magnitude. Store this as self.scaling.
    """

    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        # TODO: Store the scaling factor alpha / rank
        # TODO: Create lora_A (dim → rank) and lora_B (rank → dim)
        #       Use dtype=torch.float32 for both (training stability)
        # TODO: Initialize one of them to zero — which one ensures
        #       the adapter outputs zero before training?

        raise NotImplementedError("Complete the LoRAExpert __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Cast x to float32, pass through A then B,
        #       multiply by self.scaling, cast back to x.dtype

        raise NotImplementedError("Complete the LoRAExpert forward")


class LoRAMoEFeedForward(nn.Module):
    """
    Frozen shared FFN + routed LoRA experts.

    The base FFN is the original pretrained FeedForward — kept frozen so it
    adds zero extra memory.  Each LoRA expert adds only 2 * dim * rank
    parameters, which is tiny compared to a full FFN copy.

    TODO: Implement __init__ and forward.

    Hints for __init__:
    - Store the original FeedForward as self.base_ff and freeze its parameters
    - Create a router (use dtype=torch.float32 like the LoRA layers)
    - Create N LoRAExpert instances in a ModuleList
    - Don't forget self._last_routing_indices for analysis

    Hints for forward:
    - The forward combines two things:
        (a) the frozen base FFN output (no gradient needed)
        (b) the weighted sum of top-K LoRA expert outputs (has gradient)
    - Route in float32: cast x_flat to float32 before the router
    - The routing logic is the same as MoEFeedForward
    - Final output = base_output + lora_output
    """

    def __init__(self, base_ff: nn.Module, dim: int, num_experts: int = 4,
                 top_k: int = 2, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank

        # TODO: Store and freeze the base FFN
        # TODO: Create the router
        # TODO: Create N LoRA experts

        raise NotImplementedError("Complete the LoRAMoEFeedForward __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # 1. Flatten x to (num_tokens, dim)
        # 2. Compute base FFN output (frozen, use torch.no_grad())
        # 3. Route tokens to top-K LoRA experts (same as MoEFeedForward)
        # 4. Compute weighted sum of LoRA expert outputs
        # 5. Return base_output + lora_output, reshaped to original shape

        raise NotImplementedError("Complete the LoRAMoEFeedForward forward")


# ===================================================================
# Model conversion
# ===================================================================

def convert_to_moe(model, num_experts=4, top_k=2, init_mode="slice",
                   lora_rank=8, lora_alpha=16.0):
    """
    Replace all FeedForward layers in the model with MoE variants.

    Args:
        model: the Llama model
        num_experts: number of experts per layer
        top_k: number of active experts per token
        init_mode: ``"slice"`` or ``"lora"``
        lora_rank: rank for LoRA experts (only used when init_mode="lora")
        lora_alpha: LoRA scaling factor; effective scale = alpha / rank

    Returns:
        (model, trainable_params): the modified model and a flat list of
        parameters that should be passed to the optimizer.
    """
    trainable_params = []

    for layer in model.layers:
        old_ff = layer.feed_forward
        hidden_dim = old_ff.w1.out_features  # original hidden dim

        if init_mode == "lora":
            # TODO: Create a LoRAMoEFeedForward that wraps old_ff
            # - Pass old_ff as the base_ff argument
            # - Move to correct device (but NOT dtype — LoRA params must stay float32)
            # - Replace layer.feed_forward
            # - Collect trainable params: router + all LoRA expert params

            raise NotImplementedError("Complete LoRA MoE conversion")

        elif init_mode == "slice":
            # TODO: Create a MoEFeedForward and initialize expert weights
            # by slicing the original FFN.
            #
            # For expert i (0 ≤ i < num_experts):
            #   slice_start = i * (hidden_dim // num_experts)
            #   slice_end = (i + 1) * (hidden_dim // num_experts)
            #   expert_w1[i] ← old_ff.w1.weight[slice_start:slice_end, :]
            #   expert_w2[i] ← old_ff.w2.weight[:, slice_start:slice_end]
            #   expert_w3[i] ← old_ff.w3.weight[slice_start:slice_end, :]

            raise NotImplementedError("Complete slice MoE conversion")

        else:
            raise ValueError(f"init_mode must be 'slice' or 'lora', got {init_mode!r}")

    return model, trainable_params


# ===================================================================
# Analysis helper
# ===================================================================

def get_expert_load_stats(model, tokenizer, prompts, device="cuda"):
    """
    Run prompts through the model and collect expert activation statistics.

    Returns a dict mapping layer_idx -> tensor of shape (num_experts,)
    with activation counts.
    """

    raise NotImplementedError("Complete the expert load stats collection")
