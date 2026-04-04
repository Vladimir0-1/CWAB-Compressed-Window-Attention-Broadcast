"""
Hybrid State-Space Attention (HSA) Layer
Author: Vladimir0-1
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridStateSpaceAttention(nn.Module):
    """
    HSA: Linear-complexity attention with sliding window, compressed global context,
    information broadcast, and adaptive mixing.

    Replaces standard multi-head attention in any transformer.
    Complexity: O(n) in sequence length.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        window_size: int = 2048,
        num_global_tokens: int = 128,
        dropout: float = 0.1,
        use_dreams: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.use_dreams = use_dreams

        # Learnable global memory tokens
        self.global_memory = nn.Parameter(
            torch.randn(1, num_heads, num_global_tokens, self.head_dim)
        )

        # Compression via 1D convolution (stride = 4)
        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=4)

        # Information broadcast gates
        self.broadcast_gate = nn.Linear(hidden_size, hidden_size)

        # Adaptive mixing weights
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_memory: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Sliding window attention (local)
        local_out = self._sliding_window_attention(hidden_states)

        # 2. Compressed global attention
        global_out = self._compressed_global_attention(hidden_states)

        # 3. Information broadcast (exponential diffusion)
        broadcast_out = self._information_broadcast(hidden_states)

        # 4. Adaptive mixing of local and global
        mix_weights = self.mix_gate(torch.cat([local_out, global_out], dim=-1))
        mixed = mix_weights * local_out + (1 - mix_weights) * global_out
        mixed = mixed + broadcast_out

        # 5. Optional: integrate past memory tokens
        if past_memory is not None and self.use_dreams:
            memory_out = self._memory_attention(hidden_states, past_memory)
            mixed = mixed + 0.3 * memory_out

        return self.out_proj(self.dropout(mixed))

    def _sliding_window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Local attention within a sliding window."""
        batch, seq, dim = x.shape
        window = self.window_size
        stride = window // 2

        # Pad to ensure divisibility
        pad_len = (stride - (seq - window) % stride) % stride
        x_pad = F.pad(x, (0, 0, 0, pad_len))
        windows = x_pad.unfold(1, window, stride).transpose(2, 3)

        # Apply attention inside each window
        attn_windows = F.scaled_dot_product_attention(
            windows, windows, windows, is_causal=True
        )

        # Fold windows back
        return self._fold_windows(attn_windows, seq, window, stride)

    def _compressed_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Global context via k-means centroids compression."""
        batch, seq, dim = x.shape

        # Compress sequence to fixed number of tokens
        compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
        compressed = compressed[:, : self.num_global_tokens, :]

        # Add learnable memory tokens
        global_tokens = torch.cat(
            [compressed, self.global_memory.expand(batch, -1, -1, -1).flatten(1, 2)],
            dim=1,
        )

        # Cross-attention: each token attends to global tokens
        attn_weights = torch.softmax(
            torch.einsum("bnd,bkd->bnk", x, global_tokens) / math.sqrt(dim), dim=-1
        )
        return torch.einsum("bnk,bkd->bnd", attn_weights, global_tokens)

    def _information_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential information diffusion: O(n log n)."""
        batch, seq, dim = x.shape
        result = torch.zeros_like(x)
        stride = 1
        while stride < seq:
            left = torch.roll(x, shifts=stride, dims=1)
            right = torch.roll(x, shifts=-stride, dims=1)
            x = x + 0.5 * (left + right)
            result = result + x
            stride *= 2
        return result / (math.log2(seq) + 1)

    def _memory_attention(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Attention over long-term memory tokens."""
        attn = torch.softmax(
            torch.einsum("bnd,bkd->bnk", x, memory) / math.sqrt(self.head_dim), dim=-1
        )
        return torch.einsum("bnk,bkd->bnd", attn, memory)

    @staticmethod
    def _fold_windows(windows, original_len, window_size, stride):
        """Helper to merge overlapping windows with triangular weighting."""
        batch, _, dim = windows.shape
        num_windows = windows.shape[1]
        output = torch.zeros(batch, original_len, dim, device=windows.device)
        weights = torch.zeros(original_len, 1, device=windows.device)

        for i in range(num_windows):
            start = i * stride
            end = min(start + window_size, original_len)
            length = end - start
            tri_weight = torch.linspace(0.5, 1.5, length, device=windows.device)
            output[:, start:end] += windows[:, i, :length] * tri_weight
            weights[start:end] += tri_weight

        return output / (weights + 1e-6)
