# WAN Video Diffusion Model
# Provides VAE encoding/decoding and feature extraction for diffusion-pipe I2V training

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import logging
import sys
import os
import json
from pathlib import Path

from wan.modules.model import WanModel, sinusoidal_embedding_1d
from wan.modules.vae2_2 import Wan2_2_VAE
from utils.device import amp_autocast, resolve_device

# Optional safetensors support
try:
    from safetensors.torch import load_file as safe_load_file  # type: ignore
except Exception:  # pragma: no cover
    safe_load_file = None

logger = logging.getLogger(__name__)

def _strip_known_prefixes_for_wan(sd: Dict[str, torch.Tensor], target_model: nn.Module) -> Dict[str, torch.Tensor]:
    """Strip only the 'dit.' prefix from checkpoint keys if present."""
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith('dit.') for k in sd.keys()):
        return sd
    mapped = { (k[4:] if k.startswith('dit.') else k): v for k, v in sd.items() }
    logger.info("Stripped 'dit.' prefix from checkpoint keys")
    return mapped

class WanVideoModel(nn.Module):
    """
    WAN Video Diffusion Model wrapper for TI2V Teacher Forcing training.
    Provides VAE encoding/decoding and feature extraction for joint video-action training.
    Uses Teacher Forcing approach for I2V conditioning (DiffSynth-Studio style).
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        vae_path: str,
        device: str = "auto",
        precision: str = "bfloat16"
    ):
        super().__init__()
        
        self.device = resolve_device(device)
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[precision]
        
        # Initialize WAN model
        self.wan_model = WanModel(**model_config)
        self.wan_model.to(device=self.device, dtype=self.precision)

        # Initialize VAE
        self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device)
        
        logger.info(f"WAN Video Model initialized with {sum(p.numel() for p in self.wan_model.parameters()):,} parameters")
    
    def encode_video(self, video_pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode video pixels to latent space.
        
        Args:
            video_pixels: Video in pixel space [B, C, T, H, W], range [-1, 1]
            
        Returns:
            Video latents [B, C', T', H', W']
        """
        with torch.no_grad():
            vae_parameter = next(self.vae.model.parameters())
            video_pixels = video_pixels.to(device=vae_parameter.device, dtype=vae_parameter.dtype)
            return self.vae.encode(video_pixels)
    
    def decode_video(self, video_latents: torch.Tensor) -> torch.Tensor:
        """
        Decode video latents to pixel space.
        
        Args:
            video_latents: Video latents [B, C, T, H, W]
            
        Returns:
            Video pixels [B, C', T', H', W'], range [-1, 1]
        """
        with torch.no_grad():
            vae_parameter = next(self.vae.model.parameters())
            video_pixels = []
            for i in range(video_latents.shape[0]):
                latent = video_latents[i].to(device=vae_parameter.device, dtype=vae_parameter.dtype)
                pixels = self.vae.decode([latent])[0]
                video_pixels.append(pixels)
            result = torch.stack(video_pixels, dim=0)
            return result
    
    def get_layer_features(
        self,
        video_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract intermediate layer features for cross-attention injection.
        
        Args:
            video_latent: Video latent tensors [B, C, T, H, W]
            timestep: Diffusion timesteps [B]
            text_embeddings: List of text embeddings
            layer_indices: Which layers to extract (None = all layers)
            
        Returns:
            List of feature tensors from specified layers
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.wan_model.blocks)))
        
        # Expect 5D batch input: [B, C, f, h, w] - standard WAN input (48 channels)
        if video_latent.ndim != 5:
            raise ValueError(f"Expected 5D tensor [B, C, f, h, w], got {video_latent.ndim}D with shape {video_latent.shape}")
        
        # Ensure input has correct channel count for WAN 2.2 (48 channels)
        expected_channels = 48
        if video_latent.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels for WAN 2.2, got {video_latent.shape[1]} channels")
        
        # Convert to WAN format (list of tensors)
        video_list = [video_latent[i] for i in range(video_latent.shape[0])]
        seq_len = video_latent.shape[2] * video_latent.shape[3] * video_latent.shape[4] // 4
        
        # Prepare inputs similar to WAN forward
        device = self.wan_model.patch_embedding.weight.device
        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)
        
        # Embeddings
        x = [self.wan_model.patch_embedding(u.unsqueeze(0)) for u in video_list]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long, device=device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) 
            for u in x
        ])
        
        # Time embeddings - handle batch of timesteps [B] -> [B, seq_len]
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).expand(timestep.size(0), seq_len)
        with amp_autocast(self.device, dtype=torch.float32):
            bt = timestep.size(0)
            t_flat = timestep.flatten()
            e = self.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float().to(device)
            )
            e0 = self.wan_model.time_projection(e).unflatten(2, (6, self.wan_model.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        
        # Context
        context = self.wan_model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.wan_model.text_len - u.size(0), u.size(1))])
                for u in text_embeddings
            ])
        )
        
        # Forward through specified layers
        layer_features = []
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.wan_model.freqs,
            context=context,
            context_lens=None
        )
        
        for i, block in enumerate(self.wan_model.blocks):
            x = block(x, **kwargs)
            if i in layer_indices:
                layer_features.append(x.clone())
        
        # Apply head and unpatchify to get final output (like forward method)
        x = self.wan_model.head(x, e)
        x = self.wan_model.unpatchify(x, grid_sizes)
        final_output = torch.stack([u.float() for u in x], dim=0)
        
        # Add final output as last element
        layer_features.append(final_output)
        
        return layer_features

    @classmethod
    def from_config(
        cls,
        config_path: str,
        vae_path: str,
        device: str = "auto",
        precision: str = "bfloat16"
    ) -> 'WanVideoModel':
        """
        Initialize WAN model architecture and VAE only (no WAN weights).
        Useful when model weights will be loaded from a higher-level checkpoint.
        """
        # Load WAN model config
        config_json_path = os.path.join(config_path, 'config.json')
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"WAN config.json not found at {config_json_path}")
        with open(config_json_path, 'r') as f:
            model_config = json.load(f)
        # Create model without loading WAN weights
        model = cls(
            model_config=model_config,
            vae_path=vae_path,
            device=device,
            precision=precision
        )
        logger.info("Initialized WAN model from config only (no WAN weights loaded)")
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        vae_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
        precision: str = "bfloat16"
    ) -> 'WanVideoModel':
        """
        Load pretrained WAN model.
        
        Args:
            checkpoint_path: Path to WAN checkpoint (.pt file or directory)
            vae_path: Path to VAE checkpoint
            config_path: Path to config directory (optional, defaults to checkpoint_path)
            device: Device to load model on
            precision: Model precision
            
        Returns:
            WanVideoModel instance
        """
        # Load WAN model config
        if config_path is None:
            config_path = checkpoint_path
        
        config_json_path = os.path.join(config_path, 'config.json')
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                model_config = json.load(f)
        
        # Create model
        model = cls(
            model_config=model_config,
            vae_path=vae_path,
            device=device,
            precision=precision
        )
        
        # Load WAN weights - support directory and file formats
        try:
            logger.info(f"Loading WAN weights from {checkpoint_path}")
            
            if checkpoint_path.endswith('.pt'):
                # Direct .pt file loading (e.g., from cosmos-predict2 continue training)
                logger.info(f"Loading weights from .pt file: {checkpoint_path}")
                checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different .pt file formats
                if isinstance(checkpoint_state_dict, dict) and 'model' in checkpoint_state_dict:
                    # Standard checkpoint format with 'model' key
                    wan_state_dict = checkpoint_state_dict['model']
                else:
                    # Direct state dict (OrderedDict)
                    wan_state_dict = checkpoint_state_dict
                
                # Load weights into WAN model
                # Strip known prefixes like 'dit.' if present
                try:
                    wan_state_dict = _strip_known_prefixes_for_wan(wan_state_dict, model.wan_model)
                except Exception:
                    pass
                incompatible_keys = model.wan_model.load_state_dict(wan_state_dict, strict=False)
                if incompatible_keys.missing_keys:
                    logger.warning(f"Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    logger.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
                
                logger.info(f"Successfully loaded WAN weights from .pt file")
            elif checkpoint_path.endswith('.bin') or checkpoint_path.endswith('.safetensors'):
                # Single-file HF-style weight
                logger.info(f"Loading weights from weight file: {checkpoint_path}")
                if checkpoint_path.endswith('.safetensors'):
                    if safe_load_file is None:
                        raise RuntimeError("safetensors not available. Please 'pip install safetensors'.")
                    wan_state_dict = safe_load_file(checkpoint_path, device='cpu')
                else:
                    loaded = torch.load(checkpoint_path, map_location='cpu')
                    # If the loaded object is a wrapper dict, try common keys
                    if isinstance(loaded, dict) and ('state_dict' in loaded or 'model' in loaded):
                        wan_state_dict = loaded.get('state_dict', loaded.get('model'))
                    else:
                        wan_state_dict = loaded
                # Strip known prefixes like 'dit.' if present
                try:
                    wan_state_dict = _strip_known_prefixes_for_wan(wan_state_dict, model.wan_model)
                except Exception:
                    pass
                incompatible_keys = model.wan_model.load_state_dict(wan_state_dict, strict=False)
                if incompatible_keys.missing_keys:
                    logger.warning(f"Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    logger.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
                logger.info("Successfully loaded WAN weights from single file")
            else:
                # Directory-based loading (original diffusers format)
                loaded_model = WanModel.from_pretrained(checkpoint_path)
                model.wan_model.load_state_dict(loaded_model.state_dict(), strict=False)
                logger.info(f"Successfully loaded WAN weights from directory")
                
        except Exception as e:
            logger.warning(f"Failed to load WAN checkpoint from {checkpoint_path}: {e}")
            logger.warning("Using random initialization instead")
        
        return model