# Action Expert Model
# DiT-style architecture with cross-attention injection from video model

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

logger = logging.getLogger(__name__)

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.
    
    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | tensor): a list of positions to be encoded, size (M,).
    Returns:
        out (tensor): resulting positional embedding, size (M, D).
    """
    import numpy as np
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


@dataclass
class ActionExpertConfig:
    """Configuration for Action Expert model."""
    # Architecture - Updated for UniDiffuser
    dim: int = 1024                   # Hidden dimension (for ~500M params)
    ffn_dim: int = 4096              # FFN dimension (4x hidden dim)
    num_layers: int = 30             # Number of layers (unified with WAN and VLM)
    
    # Input/Output
    state_dim: int = 14              # Robot state dimension
    action_dim: int = 14             # Robot action dimension
    chunk_size: int = 16             # Total sequence length (1 state + 15 actions)
    
    # Video model injection - Updated for WAN 2.2
    video_feature_dim: int = 3072    # Dimension of features from WAN video model
    
    # Attention settings
    causal: bool = False             # Whether to use causal attention mask
    
    # Registers settings
    num_registers: int = 4           # Number of register tokens for global attention (reduced from 8)
    
    # Training
    eps: float = 1e-6

    # Training mode: 'finetune' (state+action) or 'pretrain' (action only)
    training_mode: str = 'finetune'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.chunk_size >= 2, "chunk_size must be at least 2 (1 state + 1 action)"

class StateActionEncoder(nn.Module):
    """Encoder for robot states and actions."""
    
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        
        # Adapters using build_mlp
        self.state_encoder = self.build_mlp(
            'mlp3x_silu',
            in_features=config.state_dim,
            out_features=config.dim
        )
        
        self.action_encoder = self.build_mlp(
            'mlp3x_silu', 
            in_features=config.action_dim,
            out_features=config.dim
        )
        
        # Create fixed sinusoidal positional embeddings (chunk_size + 1 + num_registers)
        max_seq_len = config.chunk_size + 1 + config.num_registers
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            config.dim, 
            np.arange(max_seq_len)
        )
        # Register as buffer (non-trainable)
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))  # [1, chunk_size+1+num_registers, dim]
    
    def build_mlp(self, projector_type, in_features, out_features):
        """Build MLP projector for encoders."""
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
    
    def forward(self, state_tokens: torch.Tensor, action_tokens: torch.Tensor, registers: torch.Tensor = None) -> torch.Tensor:
        """
        Encode state and action tokens separately then concatenate, optionally with registers.
        
        Args:
            state_tokens: [B, 1, state_dim] - initial state
            action_tokens: [B, action_chunk_size, action_dim] - action sequence
            registers: [B, num_registers, dim] - optional register tokens
            
        Returns:
            Encoded sequence [B, chunk_size + num_registers, dim] if registers provided
            Encoded sequence [B, chunk_size, dim] if no registers
        """
        B = state_tokens.shape[0]
        chunk_size = state_tokens.shape[1] + action_tokens.shape[1]
        
        # Encode state tokens: direct encoding without squeeze/unsqueeze
        state_encoded = self.state_encoder(state_tokens)  # [B, 1, dim]
        
        # Encode action tokens: direct encoding
        action_encoded = self.action_encoder(action_tokens)  # [B, action_chunk_size, dim]
        
        # Concatenate state and action encodings
        encoded = torch.cat([state_encoded, action_encoded], dim=1)  # [B, chunk_size, dim]
        
        # Optionally concatenate registers
        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)  # [B, chunk_size + num_registers, dim]
        
        # Add positional embeddings to all tokens (including registers)
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]
        
        return encoded


class ActionEncoder(nn.Module):
    """Encoder for action-only sequences (no state)."""
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config

        self.action_encoder = self.build_mlp(
            'mlp3x_silu',
            in_features=config.action_dim,
            out_features=config.dim
        )

        # Positional embeddings for action tokens + optional registers
        max_seq_len = config.chunk_size + config.num_registers
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            config.dim,
            np.arange(max_seq_len)
        )
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))

    def build_mlp(self, projector_type, in_features, out_features):
        if projector_type == 'linear':
            return nn.Linear(in_features, out_features)
        mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
        if mlp_silu_match:
            mlp_depth = int(mlp_silu_match.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, mlp_depth):
                modules.append(nn.SiLU())
                modules.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*modules)
        raise ValueError(f'Unknown projector type: {projector_type}')

    def forward(self, state_tokens: torch.Tensor, action_tokens: torch.Tensor, registers: torch.Tensor = None) -> torch.Tensor:
        # state_tokens is ignored for action-only mode
        action_encoded = self.action_encoder(action_tokens)  # [B, chunk_size, dim]
        encoded = action_encoded
        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]
        return encoded


class ActionExpertBlock(nn.Module):
    """
    Action Expert Block.

    This block owns action-side parameters only (Q/K/V/O and norms) that map
    action tokens to the head space of external backbones (WAN / VLM).
    The actual attention is executed by the backbone self-attention modules
    via a MoT (mixture-of-tokens) interface; this block provides projections
    and FFN, while higher-level modules orchestrate call order.
    """
    
    def __init__(self, config: ActionExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config
        
        # Layer norms (WAN style) - only need one for joint attention and one for FFN
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)  # For trimodal joint attention
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)  # For FFN
        
        # WAN-side action projections and norms (MoT: action -> WAN head space for trimodal joint attention)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_action_qkv = nn.Parameter(
            torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim)
            / (config.dim * self.wan_head_dim) ** 0.5
        )
        self.wan_action_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        # normalize Q/K in WAN unified dim
        self.wan_action_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_action_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        
        # FFN (Action Expert's own)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.ffn_dim, config.dim)
        )
        
        # Timestep modulation (AdaLN style, 6 parameters)
        # 3 params each for: self-attn residual (WAN-action), FFN (alpha/beta/gamma)
        # self.modulation = nn.Parameter(torch.zeros(1, 6, config.dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim**0.5)

    def enable_wan_action_qkv_lora(self, rank: int, alpha: float, dropout: float = 0.0) -> int:
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if hasattr(self, "wan_action_qkv_lora_A") and hasattr(self, "wan_action_qkv_lora_B"):
            self.wan_action_qkv_lora_A.requires_grad = True
            self.wan_action_qkv_lora_B.requires_grad = True
            return self.wan_action_qkv_lora_A.numel() + self.wan_action_qkv_lora_B.numel()

        dtype = self.wan_action_qkv.dtype
        device = self.wan_action_qkv.device
        a = torch.empty((3, self.config.dim, rank), device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = torch.zeros((3, rank, self.wan_num_heads, self.wan_head_dim), device=device, dtype=torch.float32)
        self.wan_action_qkv_lora_A = nn.Parameter(a.to(dtype=dtype))
        self.wan_action_qkv_lora_B = nn.Parameter(b.to(dtype=dtype))
        self.wan_action_qkv_lora_scaling = float(alpha) / float(rank)
        self.wan_action_qkv_lora_dropout = float(dropout)
        return self.wan_action_qkv_lora_A.numel() + self.wan_action_qkv_lora_B.numel()

    def project_wan_action_qkv(self, norm_action: torch.Tensor) -> torch.Tensor:
        qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, self.wan_action_qkv)
        if not hasattr(self, "wan_action_qkv_lora_A"):
            return qkv
        if getattr(self, "wan_action_qkv_lora_disabled", False) or getattr(self, "lora_disabled", False):
            return qkv
        x = norm_action
        if self.wan_action_qkv_lora_dropout > 0:
            x = F.dropout(x, p=self.wan_action_qkv_lora_dropout, training=self.training)
        hidden = torch.einsum("BTD,KDR->BTKR", x, self.wan_action_qkv_lora_A)
        delta = torch.einsum("BTKR,KRNE->KBTNE", hidden, self.wan_action_qkv_lora_B)
        return qkv + delta * self.wan_action_qkv_lora_scaling


class ActionDecoder(nn.Module):
    """Final layer to decode action predictions."""
    
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        
        self.norm = WanLayerNorm(config.dim, eps=config.eps)

        self.action_head = self.build_mlp('mlp1x_silu', config.dim, config.action_dim)

        # Timestep modulation for head input (WAN Head style: 2-way modulation)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim**0.5)

    def build_mlp(self, projector_type, in_features, out_features):
        if projector_type == 'linear':
            return nn.Linear(in_features, out_features)
        mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
        if mlp_silu_match:
            mlp_depth = int(mlp_silu_match.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, mlp_depth):
                modules.append(nn.SiLU())
                modules.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*modules)
        raise ValueError(f'Unknown projector type: {projector_type}')
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Decode action predictions.
        
        Args:
            x: Features [B, chunk_size, dim]
            time_emb: Time embeddings [B, chunk_size, dim] for head modulation
            
        Returns:
            Action predictions [B, chunk_size, action_dim]
        """
        # WAN Head-style modulation using time_emb
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e0, e1 = (self.modulation.unsqueeze(0) + time_emb.unsqueeze(2)).chunk(2, dim=2)
        z = self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2)
        return self.action_head(z)

class ActionExpert(nn.Module):
    """
    Action Expert model with DiT-style architecture and cross-attention injection from video model.
    
    Attention modes:
    - causal=False: Full sequence diffusion, all tokens can attend to each other
    - causal=True: Causal attention, state+action tokens can only see past tokens
    - Cross-attention with video features for visual grounding
    """
    
    def __init__(self, config: ActionExpertConfig, wan_config: dict = None):
        super().__init__()
        self.config = config
        self.freq_dim = 256  # Sinusoidal embedding dimension (same as WAN)
     
        # Input encoder (mode-dependent)
        if config.training_mode == 'pretrain':
            self.input_encoder = ActionEncoder(config)
        else:
            self.input_encoder = StateActionEncoder(config)
        
        # Timestep embedding (same structure as WAN)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6)  # 6 parameters: 3 for WAN-Action joint attn + 3 for FFN
        )
        
        # Transformer blocks (same number as video model for 1:1 correspondence)
        # If wan_config is provided, create blocks with trimodal joint attention
        if wan_config is not None:
            self.blocks = nn.ModuleList([
                ActionExpertBlock(config, wan_config) for _ in range(config.num_layers)
            ])
        else:
            # Fallback: create blocks with default WAN config (for backward compatibility)
            self.blocks = nn.ModuleList([
                ActionExpertBlock(config, {'dim': 3072, 'num_heads': 24, 'head_dim': 128}) 
                for _ in range(config.num_layers)
            ])
        
        # Register tokens for global attention (optional)
        # When num_registers == 0, do not create parameter to avoid shape issues
        if config.num_registers > 0:
            self.registers = nn.Parameter(
                torch.empty(1, config.num_registers, config.dim).normal_(std=0.02)
            )
        else:
            self.registers = None
        
        # Output decoder
        self.decoder = ActionDecoder(config)
        
        # Initialize weights and set dtype
        self.initialize_weights()
        
        logger.info(f"Action Expert initialized with {self.count_parameters():,} parameters")
    
    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize output layer
        nn.init.zeros_(self.decoder.action_head[-1].weight)
        nn.init.zeros_(self.decoder.action_head[-1].bias)
        
        # Initialize time embedding layers
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
