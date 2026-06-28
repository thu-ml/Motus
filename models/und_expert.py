# Understanding Expert Model
# Almost identical to Action Expert but:
# 1. Input dim: 2048D (from VLM und queries)
# 2. No registers
# 3. FFN ratio: 1:1 (2048→2048) for parameter reduction
# 4. No decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np
from dataclasses import dataclass
import logging
import sys
import re
from pathlib import Path

# Import WAN's components for consistency
project_root = Path(__file__).parent.parent
bak_root = project_root / "bak"
if str(bak_root.resolve()) not in sys.path:
    sys.path.insert(0, str(bak_root.resolve()))

from wan.modules.attention import flash_attention
from wan.modules.model import WanRMSNorm, WanLayerNorm, sinusoidal_embedding_1d, rope_apply
from utils.common import get_nd_sincos_pos_embed_from_grid

logger = logging.getLogger(__name__)

@dataclass
class UndExpertConfig:
    """Configuration for Understanding Expert model."""
    # Architecture - same naming as ActionExpert for consistency
    dim: int = 512                   # Hidden dimension for understanding expert
    ffn_dim: int = 2048              # FFN dimension (computed from dim * multiplier)
    num_layers: int = 30             # Number of layers (unified with WAN and Action)
    
    # VLM adapter settings - configurable from yaml
    vlm_input_dim: int = 2048        # VLM feature dimension (input)
    vlm_projector_type: str = "mlp3x_silu"  # VLM adapter type

    # Training
    eps: float = 1e-5                # Layer norm epsilon


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return torch.from_numpy(emb).float()


class UndExpertBlock(nn.Module):
    """
    Understanding Expert Block - almost identical to ActionExpertBlock.
    
    Only provides projections for trimodal joint attention with WAN, no registers.
    """
    
    def __init__(self, config: UndExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config
        
        # Layer norms (WAN style) - only need one for joint attention and one for FFN
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)  # For trimodal joint attention
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)  # For FFN
        
        # WAN-side understanding projections and norms (MoT: understanding -> WAN head space for trimodal joint attention)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_und_qkv = nn.Parameter(
            torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim)
            / (config.dim * self.wan_head_dim) ** 0.5
        )
        self.wan_und_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        # normalize Q/K in WAN unified dim
        self.wan_und_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_und_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.ffn_dim, config.dim)
        )

    def enable_wan_und_qkv_lora(self, rank: int, alpha: float, dropout: float = 0.0) -> int:
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if hasattr(self, "wan_und_qkv_lora_A") and hasattr(self, "wan_und_qkv_lora_B"):
            self.wan_und_qkv_lora_A.requires_grad = True
            self.wan_und_qkv_lora_B.requires_grad = True
            return self.wan_und_qkv_lora_A.numel() + self.wan_und_qkv_lora_B.numel()

        dtype = self.wan_und_qkv.dtype
        device = self.wan_und_qkv.device
        a = torch.empty((3, self.config.dim, rank), device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = torch.zeros((3, rank, self.wan_num_heads, self.wan_head_dim), device=device, dtype=torch.float32)
        self.wan_und_qkv_lora_A = nn.Parameter(a.to(dtype=dtype))
        self.wan_und_qkv_lora_B = nn.Parameter(b.to(dtype=dtype))
        self.wan_und_qkv_lora_scaling = float(alpha) / float(rank)
        self.wan_und_qkv_lora_dropout = float(dropout)
        return self.wan_und_qkv_lora_A.numel() + self.wan_und_qkv_lora_B.numel()

    def project_wan_und_qkv(self, norm_und: torch.Tensor) -> torch.Tensor:
        qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, self.wan_und_qkv)
        if not hasattr(self, "wan_und_qkv_lora_A"):
            return qkv
        if getattr(self, "wan_und_qkv_lora_disabled", False) or getattr(self, "lora_disabled", False):
            return qkv
        x = norm_und
        if self.wan_und_qkv_lora_dropout > 0:
            x = F.dropout(x, p=self.wan_und_qkv_lora_dropout, training=self.training)
        hidden = torch.einsum("BTD,KDR->BTKR", x, self.wan_und_qkv_lora_A)
        delta = torch.einsum("BTKR,KRNE->KBTNE", hidden, self.wan_und_qkv_lora_B)
        return qkv + delta * self.wan_und_qkv_lora_scaling


class UndExpert(nn.Module):
    """
    Understanding Expert model.
    
    Key features:
    - VLM adapter: 2048D -> configurable dimension (default 512D)
    - No registers
    - Configurable FFN ratio
    - No decoder
    """
    
    def __init__(self, config: UndExpertConfig, wan_config: dict = None, vlm_config: dict = None):
        super().__init__()
        self.config = config
        self.freq_dim = 256  # Sinusoidal embedding dimension
        
        # VLM adapter - adapts from VLM dimension to understanding expert dimension
        self.vlm_adapter = self.build_condition_adapter(
            config.vlm_projector_type,
            config.vlm_input_dim,
            config.dim
        )

        # Transformer blocks (same number as WAN/Action for 1:1 correspondence)
        if wan_config is not None:
            self.blocks = nn.ModuleList([
                UndExpertBlock(config, wan_config) for _ in range(config.num_layers)
            ])
        else:
            # Fallback: create blocks with default WAN config (for backward compatibility)
            self.blocks = nn.ModuleList([
                UndExpertBlock(config, {'dim': 3072, 'num_heads': 24, 'head_dim': 128}) 
                for _ in range(config.num_layers)
            ])
    
    def build_condition_adapter(self, projector_type, in_features, out_features):
        """Build condition adapter - same as ActionExpert implementation."""
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
