%%writefile /content/HSA_repo/hsa/attention.py
"""
Hybrid State-Space Attention (HSA) - Optimized
Author: Vladimir0-1
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridStateSpaceAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, window_size=512, num_global_tokens=64, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = min(window_size, hidden_size)
        self.num_global_tokens = num_global_tokens
        
        # Optimized: single QKV projection (faster)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # Compression with smaller kernel for speed
        self.compressor = nn.Conv1d(hidden_size, hidden_size, kernel_size=2, stride=2)
        
        # Lightweight mixing
        self.mix_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        batch, seq, dim = x.shape
        
        # Branch based on sequence length for optimal performance
        if seq <= 1024:
            return self._fast_path(x)
        else:
            return self._long_path(x)
    
    def _fast_path(self, x):
        """Optimized for short sequences (no sliding window overhead)"""
        batch, seq, dim = x.shape
        
        # Standard attention but faster
        qkv = self.qkv(x).reshape(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(batch, seq, dim)
        return self.proj(out)
    
    def _long_path(self, x):
        """Optimized for long sequences (with sliding window)"""
        batch, seq, dim = x.shape
        window = self.window_size
        
        # Pad to window size
        pad = (window - seq % window) % window
        if pad > 0:
            x_pad = F.pad(x, (0, 0, 0, pad))
        else:
            x_pad = x
        
        padded_seq = x_pad.shape[1]
        n_windows = padded_seq // window
        
        # Process windows in parallel
        windows = x_pad.reshape(batch, n_windows, window, dim)
        windows = windows.reshape(batch * n_windows, window, dim)
        
        # QKV for all windows at once
        qkv = self.qkv(windows).reshape(batch * n_windows, window, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(batch * n_windows, window, dim)
        out = out.reshape(batch, n_windows, window, dim)
        out = out.reshape(batch, padded_seq, dim)
        
        if pad > 0:
            out = out[:, :seq, :]
        
        # Add global context via compression
        if seq >= 512:
            compressed = self.compressor(x.transpose(1, 2)).transpose(1, 2)
            if compressed.shape[1] > self.num_global_tokens:
                compressed = compressed[:, :self.num_global_tokens, :]
            
            # Cross-attention with compressed tokens
            compressed = compressed.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            x_mh = x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_global = torch.matmul(x_mh, compressed.transpose(-2, -1)) * scale
            attn_global = F.softmax(attn_global, dim=-1)
            global_context = torch.matmul(attn_global, compressed)
            global_context = global_context.transpose(1, 2).reshape(batch, seq, dim)
            
            # Mix local and global
            mix = torch.sigmoid(self.mix_gate(torch.cat([out, global_context], dim=-1)))
            out = mix * out + (1 - mix) * global_context
        
        return self.proj(out)
