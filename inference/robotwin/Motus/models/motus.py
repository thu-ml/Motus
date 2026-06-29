# Motus - Modular Architecture
# Three-modal UniDiffuser: Video Model (WAN) + Action Expert + Understanding Expert
# Implements MoT (Mixture of Tokens) architecture with unified attention

import sys
import json
import torch
import logging
import torch.nn as nn
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from utils.common import get_t_distribution
from wan.modules.model import sinusoidal_embedding_1d
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig, get_1d_sincos_pos_embed_from_grid
from .und_expert import UndExpert, UndExpertConfig
from .lora import add_lora_to_linear_modules, mark_only_lora_as_trainable
# Add Flow-Matching schedulers
from wan.utils.fm import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)

@dataclass 
class MotusConfig:
    """Configuration for Motus."""
    # Video model settings
    wan_checkpoint_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"
    vae_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    wan_config_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"
    video_precision: str = "bfloat16"

    # VLM settings
    vlm_checkpoint_path: str = "/share/home/bhz/pretrained_models/Qwen3-VL-2B-Instruct"
    
    # Understanding Expert settings - configurable from yaml
    und_expert_hidden_size: int = 512        # Understanding expert hidden dimension
    und_expert_ffn_dim_multiplier: int = 4   # Understanding expert FFN dimension multiplier
    und_expert_norm_eps: float = 1e-5        # Understanding expert layer norm epsilon
    und_layers_to_extract: List[int] = None  # Which VLM layers to extract from
    
    # VLM adapter settings for understanding expert
    vlm_adapter_input_dim: int = 2048        # VLM feature dimension (input)
    vlm_adapter_projector_type: str = "mlp3x_silu"  # VLM adapter type

    # Action expert settings  
    num_layers: int = 30 
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict
    
    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width
    
    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Control whether to load pretrained WAN/VLM backbones.
    # None = default behavior (load), False = skip loading (init from config only)
    load_pretrained_backbones: Optional[bool] = None

    # LoRA finetuning settings for pipeline adaptation.
    lora_enabled: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_linear: bool = True
    lora_target_qkv: bool = True

    # Pipeline-aware chunk finetuning.  These fields are zero-impact unless
    # extended_chunkwise_finetune.enabled is set in the training config.
    time_distribution: Optional[Dict[str, Any]] = None
    extended_chunkwise_enabled: bool = False
    extended_chunkwise_multiplier: int = 3
    extended_chunkwise_pipeline_depth: int = 3
    extended_chunkwise_chunk_causal_mask: bool = True
    extended_chunkwise_pipeline_embeddings: bool = True
    extended_chunkwise_chunk_weight_0: float = 1.0
    extended_chunkwise_chunk_weight_1: float = 0.7
    extended_chunkwise_chunk_weight_2: float = 0.5
    extended_chunkwise_constant_weight: float = 0.2
    extended_chunkwise_chunkwise_weight: float = 0.8

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size is determined by global downsample rate and frequency ratio
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        
        # Default understanding layers to extract from (if not specified)
        if self.und_layers_to_extract is None:
            # Extract from all layers for comprehensive understanding
            self.und_layers_to_extract = list(range(self.num_layers))


class VideoModule(nn.Module):
    """Video processing module - handles WAN + T5 operations."""

    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video tokens from pre-processed noisy latent."""
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)

        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)

        # Calculate sequence length and padding
        # seq_lens = torch.tensor([u.size(1) for u in video_tokens_list], dtype=torch.long, device=self.device)
        # seq_len = seq_lens.max().item()

        # Concatenate with padding
        # video_tokens = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) 
        #     for u in video_tokens_list
        # ])

        # return video_tokens

        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """Pre-process T5 embeddings once for all layers."""
        # Handle both old format (List[torch.Tensor]) and new format (torch.Tensor)
        if isinstance(language_embeddings, list):
            # Old format: List[torch.Tensor] - do padding
            text_len = self.video_model.wan_model.text_len  # 512
            padded_embeddings = []

            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)

            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            # New format: torch.Tensor [B, seq_len, dim] - already padded by collate_fn
            t5_context_raw = language_embeddings
        
        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)

        return t5_context

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get WAN's time embedding using WAN's own weights."""
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()
            
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32
            
        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor, 
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """Process WAN cross attention with pre-processed T5 context."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None  # WAN uses None for fixed-length context
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out
    
    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for WAN (6 components)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process WAN FFN with proper AdaLN modulation."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        
        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(
        self,
        video_tokens: torch.Tensor,
        video_time_emb: torch.Tensor,
        grid_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply WAN's head + unpatchify for final video output."""
        grid_sizes = self.grid_sizes if grid_sizes is None else grid_sizes
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        und_tokens: torch.Tensor,
        und_block: nn.Module,
        grid_sizes: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trimodal joint self-attention: WAN + Action + Understanding via WAN self-attn (MoT)."""
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)

        # Get dimensions
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n

        # Action heads for WAN space (1024 -> 24*128)
        if hasattr(action_block, "project_wan_action_qkv"):
            a_qkv = action_block.project_wan_action_qkv(norm_action)
        else:
            a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # Understanding Expert processing
        norm_und = und_block.norm1(und_tokens)
        L_u = norm_und.shape[1]
        
        # Understanding Expert heads for WAN space (2048 -> 24*128)
        if hasattr(und_block, "project_wan_und_qkv"):
            u_qkv = und_block.project_wan_und_qkv(norm_und)
        else:
            u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)
        u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
        u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
        u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
        u_v = u_v_h.view(B, L_u, n, d)

        # Meta info for WAN attention
        seq_lens = torch.full((B,), L_v + L_a + L_u, dtype=torch.long, device=self.device)
        grid_sizes = self.grid_sizes if grid_sizes is None else grid_sizes
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        # Call WAN self-attn with trimodal MoT
        y, action_out_h, und_out_h = wan_layer.self_attn(
            norm_video, seq_lens, grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            und_q=u_q, und_k=u_k, und_v=u_v,
            attn_mask=attn_mask,
        )
        
        # Project Understanding Expert output
        und_out = und_block.wan_und_o(und_out_h.flatten(2))

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out  # Regular residual connection

        return video_tokens, action_tokens, und_tokens


class UndModule(nn.Module):
    """Understanding module - handles VLM with understanding queries and Understanding Expert."""

    def __init__(self, vlm_model, und_expert, config, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        
        # VLM model reference
        self.vlm_model = vlm_model
        
        # Understanding Expert reference
        self.und_expert = und_expert
        
    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Extract understanding features from VLM last layer."""
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1]  # [B, seq_len, vlm_dim]

        # [B, seq_len, vlm_dim] -> [B, seq_len, und_dim]
        adapted_features = self.und_expert.vlm_adapter(last_layer_features)

        return adapted_features
        
    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)

        # Insert image embeddings
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    
    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Process Understanding Expert FFN with regular LayerNorm."""
        block = self.und_expert.blocks[layer_idx]
        
        # Pre-norm for FFN (regular LayerNorm)
        ffn_input = block.norm2(und_tokens)
        ffn_output = block.ffn(ffn_input)
        
        # FFN residual connection
        und_tokens = und_tokens + ffn_output
        
        return und_tokens


class ActionModule(nn.Module):
    """Action processing module - handles Action Expert + joint attentions + masks."""
    
    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.vlm_model = vlm_model      # For accessing VLM weights
        self.dtype = dtype
        self.device = device
    
    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action time embedding."""
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            
            # Create sinusoidal embedding (same pattern as VideoModule)
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, seq_len, freq_dim]
            
            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, seq_len, 6, dim]
            
            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """Process Action Expert FFN with AdaLN modulation."""
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: α3, β3, γ3)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class Motus(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM, VLM, and Action modules.
    """

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Decide whether to load pretrained backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        # Initialize video model (WAN)
        logger.info("Initializing WAN video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        # Initialize VLM (frozen)
        logger.info("Initializing VLM (frozen)...")
        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            self.vlm_model.to(device="cuda", dtype=self.dtype)

        # Freeze VLM parameters
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        logger.info("VLM parameters frozen")

        # Keep VLM complete (do not truncate)
        logger.info(f"VLM kept complete with {len(self.vlm_model.model.language_model.layers)} layers")

        # Get WAN and VLM configurations directly
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers  = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D")
        logger.info(f"  VLM: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D")

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads, 
            'head_dim': wan_head_dim
        }
        vlm_config = {
            'hidden_size': vlm_dim,
            'num_attention_heads': vlm_num_heads,
            'num_key_value_heads': vlm_num_kv_heads,
            'head_dim': vlm_head_dim,
            'num_hidden_layers': vlm_num_hidden_layers,
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        self.action_expert = ActionExpert(action_config, wan_config)
        max_pipeline_chunks = max(1, int(config.extended_chunkwise_multiplier))
        max_pipeline_stages = max(64, int(config.extended_chunkwise_pipeline_depth) + 1)
        self.pipeline_chunk_embedding = nn.Embedding(max_pipeline_chunks, config.action_expert_dim)
        self.pipeline_stage_embedding = nn.Embedding(max_pipeline_stages, config.action_expert_dim)
        nn.init.zeros_(self.pipeline_chunk_embedding.weight)
        nn.init.zeros_(self.pipeline_stage_embedding.weight)

        # Initialize Understanding Expert
        logger.info("Initializing Understanding Expert...")
        und_config = UndExpertConfig(
            dim=config.und_expert_hidden_size,
            ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            vlm_input_dim=config.vlm_adapter_input_dim,
            vlm_projector_type=config.vlm_adapter_projector_type,
            eps=config.und_expert_norm_eps,
        )
        
        self.und_expert = UndExpert(und_config, wan_config, vlm_config)

        # Move models to device
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.und_expert.to(device=self.device, dtype=self.dtype)
        
        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W], 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded
        
        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)

        # Initialize t distributions from config
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {
            'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'),
            'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0),
            'min_t': time_dist_config.get('min_t', 0.0),
            'max_t': time_dist_config.get('max_t', 1.0)
        }

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (video)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (action)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        # Log parameter counts
        self.log_parameter_counts()

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        und_params = sum(p.numel() for p in self.und_expert.parameters())

        logger.info(f"Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        logger.info(f"  VLM (frozen): {vlm_params / 1e9:.2f}B")
        logger.info(f"  Und Expert: {und_params / 1e6:.1f}M")

    def enable_action_und_lora_finetune(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_linear: bool = True,
        target_qkv: bool = True,
    ) -> Dict[str, Any]:
        """Freeze the model and train only Action/Understanding Expert LoRA adapters."""
        for param in self.parameters():
            param.requires_grad = False

        stats: Dict[str, Any] = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "target_linear": bool(target_linear),
            "target_qkv": bool(target_qkv),
        }

        if target_linear:
            stats["action_linear"] = add_lora_to_linear_modules(
                self.action_expert, rank=rank, alpha=alpha, dropout=dropout
            )
            stats["und_linear"] = add_lora_to_linear_modules(
                self.und_expert, rank=rank, alpha=alpha, dropout=dropout
            )

        qkv_params = 0
        if target_qkv:
            for block in self.action_expert.blocks:
                qkv_params += block.enable_wan_action_qkv_lora(rank=rank, alpha=alpha, dropout=dropout)
            for block in self.und_expert.blocks:
                qkv_params += block.enable_wan_und_qkv_lora(rank=rank, alpha=alpha, dropout=dropout)
        stats["qkv_lora_params"] = qkv_params
        stats.update(mark_only_lora_as_trainable(self))
        stats["pipeline_adapter_params"] = self._mark_pipeline_adapters_trainable()
        stats["frozen_unused_final_und_lora"] = self._freeze_unused_final_und_lora()
        stats["trainable_lora_params"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stats["trainable_lora_tensors"] = sum(1 for p in self.parameters() if p.requires_grad)
        self.lora_finetune_info = stats
        logger.info("Enabled Action/Understanding LoRA finetuning: %s", stats)
        return stats

    def _mark_pipeline_adapters_trainable(self) -> Dict[str, int]:
        """Keep tiny chunk/stage adapters trainable in LoRA-only finetuning."""
        if not (
            bool(getattr(self.config, "extended_chunkwise_enabled", False))
            and bool(getattr(self.config, "extended_chunkwise_pipeline_embeddings", True))
        ):
            return {"tensors": 0, "params": 0}
        tensors = 0
        params = 0
        for name, param in self.named_parameters():
            if "pipeline_chunk_embedding" in name or "pipeline_stage_embedding" in name:
                param.requires_grad = True
                tensors += 1
                params += param.numel()
        return {"tensors": tensors, "params": params}

    def _freeze_unused_final_und_lora(self) -> Dict[str, int]:
        """Freeze LoRA params that cannot affect action/video losses."""
        if not getattr(self.und_expert, "blocks", None):
            return {"tensors": 0, "params": 0}

        frozen_tensors = 0
        frozen_params = 0
        final_block = self.und_expert.blocks[-1]
        for module_name in ("wan_und_o", "ffn"):
            module = getattr(final_block, module_name, None)
            if module is None:
                continue
            for submodule in module.modules():
                for param_name in ("lora_A", "lora_B"):
                    param = getattr(submodule, param_name, None)
                    if isinstance(param, nn.Parameter) and param.requires_grad:
                        param.requires_grad = False
                        frozen_tensors += 1
                        frozen_params += param.numel()
        return {"tensors": frozen_tensors, "params": frozen_params}

    @contextmanager
    def disable_lora_adapters(self):
        """Temporarily run this model as the frozen base model."""
        saved_states = []
        for module in self.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                saved_states.append((module, "lora_disabled", getattr(module, "lora_disabled", False)))
                module.lora_disabled = True
            if hasattr(module, "wan_action_qkv_lora_A") and hasattr(module, "wan_action_qkv_lora_B"):
                saved_states.append((
                    module,
                    "wan_action_qkv_lora_disabled",
                    getattr(module, "wan_action_qkv_lora_disabled", False),
                ))
                module.wan_action_qkv_lora_disabled = True
            if hasattr(module, "wan_und_qkv_lora_A") and hasattr(module, "wan_und_qkv_lora_B"):
                saved_states.append((
                    module,
                    "wan_und_qkv_lora_disabled",
                    getattr(module, "wan_und_qkv_lora_disabled", False),
                ))
                module.wan_und_qkv_lora_disabled = True
        try:
            yield
        finally:
            for module, attr, value in saved_states:
                setattr(module, attr, value)

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)
    
        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']  
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        
        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items() 
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")
        # Handle directory path (align with load_checkpoint style)
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)

        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)
        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities.
        
        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            return_dict: Whether to return detailed outputs
            
        Returns:
            Dictionary containing losses and metrics
        """
        B = video_frames.shape[0]

        # 1. Video pipeline
        # Normalize/format
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, frames+1, H, W]

        # Encode video using VAE
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))  # [B, 48, latent_frames, H', W']
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']

        # Flow-Matching noise mixture
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        # Scalar timesteps (0..num_train_timesteps) for time embedding
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for noise mixture
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        # Teacher Forcing on the first frame
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        # 2. Action pipeline 
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        # Discrete timesteps for time embedding (0..num_train_timesteps)
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for action noise mixture
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)

        und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, seq_len, und_dim]

        # Time embeddings
        # Use scheduler-provided timesteps (0..num_train_timesteps) for WAN/action time embeddings
        video_head_time_emb, video_adaln_params  = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            # Process through 30 layers - modality-grouped execution
            for layer_idx in range(self.config.num_layers):
                # Compute AdaLN modulation once per layer using pre-computed parameters
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                
                # Trimodal MoT: WAN + Action + Understanding Expert joint attention
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                    video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                    self.action_expert.blocks[layer_idx],
                    und_tokens, self.und_expert.blocks[layer_idx]
                )

                # WAN cross
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)

                # FFNs: WAN, Action, Understanding (each processes their own FFN)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
                
        
            # 4. Heads + Losses
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            # Slice predicted actions depending on mode
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]
            else:
                action_pred = action_pred_full[:, 1:up_len, :]

            # Video loss (mask the first frame)
            video_pred_masked = video_pred.clone()
            video_pred_masked[:, :, 0:1] = 0
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')
        
            # Action loss
            action_loss = torch.nn.functional.mse_loss(action_pred, action_target, reduction='mean')

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )
        
        if return_dict:
            return {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'video_timestep_mean': sigma.float().mean().item(),
                'action_timestep_mean': sigma_action.float().mean().item(),
            }

    def forward(
        self,
        first_frame: torch.Tensor,
        video_frames: torch.Tensor,
        state: torch.Tensor = None,
        actions: torch.Tensor = None,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        extended_chunkwise: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if extended_chunkwise:
            return self.training_step_extended_chunkwise(
                first_frame=first_frame,
                video_frames=video_frames,
                state=state,
                actions=actions,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
                return_dict=return_dict,
            )
        return self.training_step(
            first_frame=first_frame,
            video_frames=video_frames,
            state=state,
            actions=actions,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
            return_dict=return_dict,
        )

    def training_step_extended_chunkwise(
        self,
        first_frame: torch.Tensor,
        video_frames: torch.Tensor,
        state: torch.Tensor = None,
        actions: torch.Tensor = None,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Train on the steady-state rolling pipeline distribution."""
        if actions is None:
            raise ValueError("extended chunkwise training requires actions")

        B = video_frames.shape[0]
        base_action_len = int(self.config.action_chunk_size)
        if actions.dim() != 3 or actions.shape[1] % base_action_len != 0:
            raise ValueError(
                "extended chunkwise actions must have shape [B, multiplier * base_action_len, action_dim], "
                f"got {tuple(actions.shape)} with base_action_len={base_action_len}"
            )
        multiplier = max(1, actions.shape[1] // base_action_len)
        extended_action_len = base_action_len * multiplier

        chunk_stage, action_chunk_t, pipeline_depth = self._extended_pipeline_stage_profile(B, multiplier)

        # Video branch: only the executable first chunk has video supervision in the
        # current dataset, so train it at the same pre-step stage as chunk 0.
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))

        video_t_embed = action_chunk_t[:, 0]
        video_sigma = (video_t_embed.float() / 1000.0).to(self.dtype).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - video_sigma) + video_noise * video_sigma
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))

        clean_actions = actions[:, :extended_action_len].to(self.dtype)
        clean_action_chunks = clean_actions.reshape(
            B,
            multiplier,
            base_action_len,
            self.config.action_dim,
        )
        action_noise_chunks = torch.randn_like(clean_action_chunks, dtype=self.dtype)
        action_sigma = (action_chunk_t.float() / 1000.0).to(self.dtype).view(B, multiplier, 1, 1)
        noisy_action_chunks = clean_action_chunks * (1 - action_sigma) + action_noise_chunks * action_sigma
        action_target_chunks = action_noise_chunks - clean_action_chunks
        noisy_actions = noisy_action_chunks.reshape(B, extended_action_len, self.config.action_dim)
        action_target = action_target_chunks.reshape(B, extended_action_len, self.config.action_dim)

        action_tokens, action_layout = self._encode_extended_action_tokens(
            state=state,
            action_latent=noisy_actions,
            base_action_len=base_action_len,
            chunk_stage=chunk_stage,
        )
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        attn_mask = self._build_extended_training_attn_mask(
            video_token_len=video_tokens.shape[1],
            action_token_len=action_tokens.shape[1],
            und_token_len=und_tokens.shape[1],
            action_layout=action_layout,
            multiplier=multiplier,
        )

        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
            video_t_embed,
            video_tokens.shape[1],
        )
        action_token_t = self._extended_action_token_timesteps(action_chunk_t, action_layout)
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
            action_token_t,
            action_tokens.shape[1],
        )
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                    video_tokens,
                    action_tokens,
                    video_adaln_modulation,
                    action_adaln_modulation,
                    layer_idx,
                    self.action_expert.blocks[layer_idx],
                    und_tokens,
                    self.und_expert.blocks[layer_idx],
                    attn_mask=attn_mask,
                )
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred = self._decode_extended_action_velocity(
                action_tokens,
                action_head_time_emb,
                action_layout,
            )

            video_pred_masked = video_pred.clone()
            video_pred_masked[:, :, 0:1] = 0
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')

            action_error = (action_pred.float() - action_target.float()).pow(2)
            action_error = action_error.reshape(B, multiplier, base_action_len, self.config.action_dim)
            chunk_mse = action_error.mean(dim=(0, 2, 3))
            chunk_weights = self._extended_chunk_loss_weights(multiplier)
            action_loss = (chunk_mse * chunk_weights).sum() / chunk_weights.sum().clamp_min(1e-6)

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )

        if return_dict:
            result = {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'video_timestep_mean': video_sigma.float().mean().item(),
                'action_timestep_mean': action_sigma.float().mean().item(),
                'pipeline_depth': int(pipeline_depth),
                'pipeline_stage_profile': [int(x) for x in chunk_stage[0].detach().cpu().tolist()],
            }
            for idx in range(multiplier):
                result[f'action_loss_chunk_{idx}'] = chunk_mse[idx].detach()
            return result

    def _extended_action_pos_embedding(self, seq_len: int) -> torch.Tensor:
        dim = self.action_expert.config.dim
        positions = torch.arange(seq_len)
        return get_1d_sincos_pos_embed_from_grid(dim, positions).to(
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)

    def _pipeline_embeddings_enabled(self) -> bool:
        return bool(
            getattr(self.config, "extended_chunkwise_enabled", False)
            and getattr(self.config, "extended_chunkwise_pipeline_embeddings", True)
        )

    def _add_pipeline_action_embeddings(
        self,
        action_tokens: torch.Tensor,
        layout: Dict[str, int],
        chunk_stage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self._pipeline_embeddings_enabled():
            return action_tokens
        if chunk_stage is None:
            return action_tokens

        B = action_tokens.shape[0]
        base_len = int(layout["base_action_len"])
        extended_len = int(layout["extended_action_len"])
        multiplier = max(1, extended_len // max(1, base_len))
        max_chunk = int(self.pipeline_chunk_embedding.num_embeddings) - 1
        max_stage = int(self.pipeline_stage_embedding.num_embeddings) - 1
        chunk_stage = chunk_stage.to(device=action_tokens.device, dtype=torch.long)
        if chunk_stage.dim() == 1:
            chunk_stage = chunk_stage.unsqueeze(0).expand(B, -1)

        token_delta = torch.zeros_like(action_tokens)
        for chunk_idx in range(multiplier):
            if chunk_idx == 0:
                start = int(layout["prefix_action_start"])
            else:
                start = int(layout["future_action_start"]) + (chunk_idx - 1) * base_len
            end = start + base_len
            if start >= end or end > action_tokens.shape[1]:
                continue

            chunk_ids = torch.full(
                (B,),
                min(chunk_idx, max_chunk),
                device=action_tokens.device,
                dtype=torch.long,
            )
            stage_ids = chunk_stage[:, min(chunk_idx, chunk_stage.shape[1] - 1)].clamp(0, max_stage)
            emb = (
                self.pipeline_chunk_embedding(chunk_ids)
                + self.pipeline_stage_embedding(stage_ids)
            ).to(dtype=action_tokens.dtype)
            token_delta[:, start:end] = token_delta[:, start:end] + emb.unsqueeze(1)
        return action_tokens + token_delta

    def _extended_pipeline_stage_profile(
        self,
        B: int,
        multiplier: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        depth = max(1, int(getattr(self.config, "extended_chunkwise_pipeline_depth", 3)))
        stage = torch.arange(
            depth - 1,
            depth - multiplier - 1,
            -1,
            device=self.device,
            dtype=torch.long,
        ).clamp(0, depth)
        stage = stage.unsqueeze(0).expand(B, -1).contiguous()
        timestep = ((depth - stage).to(self.dtype) / float(depth)) * 1000.0
        return stage, timestep, depth

    def _extended_chunk_loss_weights(self, multiplier: int) -> torch.Tensor:
        weights = [
            float(getattr(self.config, "extended_chunkwise_chunk_weight_0", 1.0)),
            float(getattr(self.config, "extended_chunkwise_chunk_weight_1", 0.7)),
            float(getattr(self.config, "extended_chunkwise_chunk_weight_2", 0.5)),
        ]
        while len(weights) < multiplier:
            weights.append(weights[-1])
        return torch.tensor(weights[:multiplier], device=self.device, dtype=torch.float32)

    def _build_extended_training_attn_mask(
        self,
        video_token_len: int,
        action_token_len: int,
        und_token_len: int,
        action_layout: Dict[str, int],
        multiplier: int,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self.config, "extended_chunkwise_chunk_causal_mask", True)):
            return None
        base_len = int(action_layout["base_action_len"])
        future_action_abs_start = int(video_token_len + action_layout["future_action_start"])
        future_action_abs_end = int(video_token_len + action_layout["future_action_end"])
        future_chunk_ranges = []
        for future_chunk_idx in range(max(0, multiplier - 1)):
            action_start = future_action_abs_start + future_chunk_idx * base_len
            action_end = action_start + base_len
            future_chunk_ranges.append(
                {
                    "chunk_index": int(future_chunk_idx + 1),
                    "video_start": int(video_token_len),
                    "video_end": int(video_token_len),
                    "action_start": int(action_start),
                    "action_end": int(min(action_end, future_action_abs_end)),
                }
            )
        return {
            "type": "extended_chunk_causal",
            "video_token_len": int(video_token_len),
            "action_token_len": int(action_token_len),
            "und_token_len": int(und_token_len),
            "prefix_video_token_len": int(video_token_len),
            "future_action_start": int(action_layout["future_action_start"]),
            "future_action_end": int(action_layout["future_action_end"]),
            "future_chunk_ranges": future_chunk_ranges,
        }

    def _encode_extended_action_tokens(
        self,
        state: torch.Tensor,
        action_latent: torch.Tensor,
        base_action_len: int,
        chunk_stage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Encode state/current actions/registers/future actions with prefix positions preserved."""
        B, extended_action_len, _ = action_latent.shape
        num_registers = int(self.action_expert.config.num_registers)
        registers = (
            self.action_expert.registers.expand(B, -1, -1)
            if num_registers > 0 and self.action_expert.registers is not None
            else None
        )
        future_action_len = max(0, extended_action_len - base_action_len)
        encoder = self.action_expert.input_encoder

        if self.config.training_mode == 'pretrain':
            prefix_tokens = encoder.action_encoder(action_latent[:, :base_action_len])
            parts = [prefix_tokens]
            prefix_action_start = 0
            prefix_action_end = base_action_len
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            state_encoded = encoder.state_encoder(state_tokens)
            prefix_tokens = encoder.action_encoder(action_latent[:, :base_action_len])
            parts = [state_encoded, prefix_tokens]
            prefix_action_start = 1
            prefix_action_end = 1 + base_action_len

        register_start = sum(part.shape[1] for part in parts)
        if registers is not None:
            parts.append(registers)
        register_end = sum(part.shape[1] for part in parts)

        future_action_start = register_end
        if future_action_len > 0:
            future_tokens = encoder.action_encoder(action_latent[:, base_action_len:])
            parts.append(future_tokens)
        future_action_end = sum(part.shape[1] for part in parts)

        action_tokens = torch.cat(parts, dim=1)
        action_tokens = action_tokens + self._extended_action_pos_embedding(action_tokens.shape[1])
        layout = {
            "prefix_action_start": prefix_action_start,
            "prefix_action_end": prefix_action_end,
            "register_start": register_start,
            "register_end": register_end,
            "future_action_start": future_action_start,
            "future_action_end": future_action_end,
            "extended_action_len": extended_action_len,
            "base_action_len": base_action_len,
        }
        action_tokens = self._add_pipeline_action_embeddings(action_tokens, layout, chunk_stage)
        return action_tokens, layout

    def _decode_extended_action_velocity(
        self,
        action_tokens: torch.Tensor,
        action_head_time_emb: torch.Tensor,
        layout: Dict[str, int],
    ) -> torch.Tensor:
        action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
        B = action_pred_full.shape[0]
        action_velocity = action_pred_full.new_empty(
            B,
            layout["extended_action_len"],
            self.config.action_dim,
        )
        base_len = layout["base_action_len"]
        action_velocity[:, :base_len] = action_pred_full[
            :, layout["prefix_action_start"]:layout["prefix_action_end"], :
        ]
        future_len = layout["extended_action_len"] - base_len
        if future_len > 0:
            action_velocity[:, base_len:] = action_pred_full[
                :, layout["future_action_start"]:layout["future_action_end"], :
            ]
        return action_velocity

    def _build_extended_chunk_causal_attn_mask(
        self,
        B: int,
        video_token_len: int,
        action_token_len: int,
        und_token_len: int,
        prefix_video_token_len: int,
        future_action_start: int,
        future_chunk_ranges: List[Dict[str, int]],
    ) -> Optional[torch.Tensor]:
        total_len = video_token_len + action_token_len + und_token_len
        prefix_query_ranges = [
            (0, prefix_video_token_len),
            (video_token_len, video_token_len + future_action_start),
            (video_token_len + action_token_len, total_len),
        ]
        chunk_query_ranges: List[List[Tuple[int, int]]] = []

        def add_clamped_range(parts: List[Tuple[int, int]], start: int, end: int) -> None:
            start = max(0, min(int(start), total_len))
            end = max(0, min(int(end), total_len))
            if start < end:
                parts.append((start, end))

        for chunk_range in future_chunk_ranges:
            chunk_parts: List[Tuple[int, int]] = []
            add_clamped_range(
                chunk_parts,
                int(chunk_range.get("video_start", 0)),
                int(chunk_range.get("video_end", 0)),
            )
            add_clamped_range(
                chunk_parts,
                int(chunk_range.get("action_start", 0)),
                int(chunk_range.get("action_end", 0)),
            )
            if chunk_parts:
                chunk_query_ranges.append(chunk_parts)
        if not chunk_query_ranges:
            return None

        mask = torch.zeros(
            (B, 1, total_len, total_len),
            device=self.device,
            dtype=torch.float32,
        )

        future_key_ranges = [rng for chunk_parts in chunk_query_ranges for rng in chunk_parts]
        for q_start, q_end in prefix_query_ranges:
            if q_start >= q_end:
                continue
            for k_start, k_end in future_key_ranges:
                mask[:, :, q_start:q_end, k_start:k_end] = -10000.0
        for chunk_idx, chunk_parts in enumerate(chunk_query_ranges):
            future_blocked_ranges = [
                rng
                for later_chunk_parts in chunk_query_ranges[chunk_idx + 1:]
                for rng in later_chunk_parts
            ]
            for q_start, q_end in chunk_parts:
                for k_start, k_end in future_blocked_ranges:
                    mask[:, :, q_start:q_end, k_start:k_end] = -10000.0
        return mask

    def _extended_action_token_timesteps(
        self,
        chunk_timesteps: torch.Tensor,
        layout: Dict[str, int],
    ) -> torch.Tensor:
        """Map per-chunk action timesteps onto state/current/register/future action tokens."""
        B, chunks = chunk_timesteps.shape
        base_len = layout["base_action_len"]
        avg_t = chunk_timesteps.mean(dim=1, keepdim=True)
        parts: List[torch.Tensor] = []
        if self.config.training_mode != 'pretrain':
            parts.append(avg_t)
        parts.append(chunk_timesteps[:, 0:1].expand(B, base_len))
        register_len = layout["register_end"] - layout["register_start"]
        if register_len > 0:
            parts.append(avg_t.expand(B, register_len))
        if chunks > 1:
            parts.append(
                chunk_timesteps[:, 1:]
                .unsqueeze(-1)
                .expand(B, chunks - 1, base_len)
                .reshape(B, (chunks - 1) * base_len)
            )
        return torch.cat(parts, dim=1)

    def _inference_step_extended_horizon(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        num_inference_steps: int,
        language_embeddings: List[torch.Tensor],
        vlm_inputs: Optional[List],
        pipeline_config: Dict[str, Any],
        decode_video: bool = True,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Extended-horizon inference: denoise several action chunks in one pass."""
        B = first_frame.shape[0]
        multiplier = max(1, int(pipeline_config.get("extended_horizon_multiplier", 3)))
        base_action_len = self.config.action_chunk_size
        extended_action_len = base_action_len * multiplier
        base_future_latent_frames = self.config.num_video_frames // 4
        base_total_latent_frames = 1 + base_future_latent_frames
        extended_total_latent_frames = 1 + base_future_latent_frames * multiplier

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))

        _, C_latent, _, H_latent, W_latent = condition_frame_latent.shape
        video_latent = torch.randn(
            (B, C_latent, extended_total_latent_frames, H_latent, W_latent),
            device=self.device,
            dtype=self.dtype,
        )
        video_latent[:, :, 0:1] = condition_frame_latent
        action_latent = torch.randn(
            (B, extended_action_len, self.config.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

        patch_size = self.video_model.wan_model.patch_size
        grid_sizes = torch.tensor(
            [
                extended_total_latent_frames // patch_size[0],
                H_latent // patch_size[1],
                W_latent // patch_size[2],
            ],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0).expand(B, -1)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
        und_context = self.und_module.extract_und_features(vlm_inputs)

        steps_run = 0
        last_prefix_video_token_len = None
        last_action_token_len = None
        last_und_token_len = None
        mask_enabled = bool(pipeline_config.get("extended_prefix_mask", True))
        timesteps = torch.linspace(
            1.0,
            0.0,
            num_inference_steps + 1,
            device=self.device,
            dtype=self.dtype,
        )

        def _predict_velocity(
            video_latent_step: torch.Tensor,
            action_latent_step: torch.Tensor,
            video_timestep: torch.Tensor,
            action_timestep: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
            video_latent_input = video_latent_step.to(self.dtype)
            video_tokens = self.video_module.prepare_input(video_latent_input)
            action_tokens, action_layout = self._encode_extended_action_tokens(
                state=state,
                action_latent=action_latent_step,
                base_action_len=base_action_len,
                chunk_stage=chunk_stage,
            )
            und_tokens = und_context

            tokens_per_latent_frame = video_tokens.shape[1] // extended_total_latent_frames
            if video_tokens.shape[1] % extended_total_latent_frames != 0:
                raise ValueError(
                    "Extended horizon expected video token length divisible by latent frames: "
                    f"tokens={video_tokens.shape[1]}, frames={extended_total_latent_frames}"
                )
            prefix_video_token_len = base_total_latent_frames * tokens_per_latent_frame
            attn_mask = None
            if mask_enabled:
                future_video_tokens_per_chunk = int(base_future_latent_frames * tokens_per_latent_frame)
                future_action_tokens_per_chunk = int(base_action_len)
                future_action_abs_start = int(video_tokens.shape[1] + action_layout["future_action_start"])
                future_action_abs_end = int(video_tokens.shape[1] + action_layout["future_action_end"])
                future_chunk_ranges = []
                for future_chunk_idx in range(max(0, multiplier - 1)):
                    video_start = int(prefix_video_token_len + future_chunk_idx * future_video_tokens_per_chunk)
                    video_end = int(video_start + future_video_tokens_per_chunk)
                    action_start = int(future_action_abs_start + future_chunk_idx * future_action_tokens_per_chunk)
                    action_end = int(action_start + future_action_tokens_per_chunk)
                    future_chunk_ranges.append(
                        {
                            "chunk_index": int(future_chunk_idx + 1),
                            "video_start": video_start,
                            "video_end": min(video_end, int(video_tokens.shape[1])),
                            "action_start": action_start,
                            "action_end": min(action_end, future_action_abs_end),
                        }
                    )
                mask_backend = str(
                    pipeline_config.get("extended_prefix_mask_backend", "chunk_causal")
                ).strip().lower()
                if mask_backend in {
                    "chunk_causal",
                    "chunk-causal",
                    "causal_chunk",
                    "causal-chunk",
                }:
                    attn_mask = {
                        "type": "extended_chunk_causal",
                        "video_token_len": int(video_tokens.shape[1]),
                        "action_token_len": int(action_tokens.shape[1]),
                        "und_token_len": int(und_tokens.shape[1]),
                        "prefix_video_token_len": int(prefix_video_token_len),
                        "future_action_start": int(action_layout["future_action_start"]),
                        "future_action_end": int(action_layout["future_action_end"]),
                        "future_chunk_ranges": future_chunk_ranges,
                    }
                else:
                    attn_mask = self._build_extended_chunk_causal_attn_mask(
                        B=B,
                        video_token_len=video_tokens.shape[1],
                        action_token_len=action_tokens.shape[1],
                        und_token_len=und_tokens.shape[1],
                        prefix_video_token_len=prefix_video_token_len,
                        future_action_start=action_layout["future_action_start"],
                        future_chunk_ranges=future_chunk_ranges,
                    )

            if video_timestep.dim() == 2 and video_timestep.shape[1] == extended_total_latent_frames:
                video_timestep = video_timestep.repeat_interleave(tokens_per_latent_frame, dim=1)
            if action_timestep.dim() == 2 and action_timestep.shape[1] == multiplier:
                action_timestep = self._extended_action_token_timesteps(action_timestep, action_layout)

            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_timestep, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_timestep, action_tokens.shape[1])

                for layer_idx in range(self.config.num_layers):
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens,
                        action_tokens,
                        video_adaln_modulation,
                        action_adaln_modulation,
                        layer_idx,
                        self.action_expert.blocks[layer_idx],
                        und_tokens,
                        self.und_expert.blocks[layer_idx],
                        grid_sizes=grid_sizes,
                        attn_mask=attn_mask,
                    )
                    video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb, grid_sizes=grid_sizes)
                action_velocity = self._decode_extended_action_velocity(
                    action_tokens,
                    action_head_time_emb,
                    action_layout,
                )

            return (
                video_velocity,
                action_velocity,
                int(prefix_video_token_len),
                int(action_tokens.shape[1]),
                int(und_tokens.shape[1]),
            )

        for step_idx in range(num_inference_steps):
            t = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]
            dt = t_next - t
            t_scaled = (t * 1000).expand(B).to(self.dtype)
            video_velocity, action_velocity, prefix_len, action_len, und_len = _predict_velocity(
                video_latent,
                action_latent,
                t_scaled,
                t_scaled,
            )
            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt
            video_latent[:, :, 0:1] = condition_frame_latent
            steps_run += 1
            last_prefix_video_token_len = prefix_len
            last_action_token_len = action_len
            last_und_token_len = und_len

        if decode_video:
            with torch.no_grad():
                decoded_frames = self.video_model.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:1 + self.config.num_video_frames]
                predicted_frames = (predicted_frames + 1.0) / 2.0
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
        else:
            predicted_frames = None

        predicted_actions = action_latent[:, :base_action_len].float()
        self.last_pipeline_condition_frame_latent = condition_frame_latent.detach()
        self.last_pipeline_action_latent = action_latent.detach()
        self.last_pipeline_video_latent = video_latent.detach()
        self.last_pipeline_action_stage = None
        self.last_pipeline_video_stage = None
        self.last_pipeline_denoise_info = {
            "enabled": True,
            "mode": "extended_horizon",
            "extended_horizon_multiplier": int(multiplier),
            "base_action_len": int(base_action_len),
            "extended_action_len": int(extended_action_len),
            "base_total_latent_frames": int(base_total_latent_frames),
            "extended_total_latent_frames": int(extended_total_latent_frames),
            "prefix_video_token_len": last_prefix_video_token_len,
            "action_token_len": last_action_token_len,
            "und_token_len": last_und_token_len,
            "prefix_mask": bool(mask_enabled),
            "prefix_mask_backend": str(pipeline_config.get("extended_prefix_mask_backend", "sdpa")),
            "steps_run": int(steps_run),
            "reuse_allowed": False,
            "reuse_reason": "disabled",
            "chunks": int(multiplier),
            "reused_chunks": 0,
        }
        return predicted_frames, predicted_actions

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        pipeline_config: Optional[Dict[str, Any]] = None,
        decode_video: bool = True,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]
        pipeline_config = pipeline_config or {}
        pipeline_mode = str(pipeline_config.get("mode", "")).strip().lower()
        extended_horizon_modes = {
            "extended_horizon",
            "extended-horizon",
            "extended_prefix_mask",
            "extended-prefix-mask",
            "extended_horizon_pipeline",
            "extended-horizon-pipeline",
            "extended_prefix_pipeline",
            "extended-prefix-pipeline",
        }
        if bool(pipeline_config.get("extended_horizon_enabled", False)) or pipeline_mode in extended_horizon_modes:
            return self._inference_step_extended_horizon(
                first_frame=first_frame,
                state=state,
                num_inference_steps=num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
                pipeline_config=pipeline_config,
                decode_video=decode_video,
            )
        latent_warmstart_modes = {
            "latent_warmstart",
            "latent-warmstart",
            "shape_warmstart",
            "shape-warmstart",
            "warmstart",
        }
        latent_warmstart_enabled = pipeline_mode in latent_warmstart_modes

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_shape = (B, C_latent, num_total_latent_frames, H_latent, W_latent)
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        video_latent = torch.randn(video_shape, device=self.device, dtype=self.dtype)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)
        reuse_allowed = False
        warmstart_reason = str(pipeline_config.get("reuse_reason", "")) if latent_warmstart_enabled else ""
        warmstart_start_t = 1.0
        warmstart_components_text = str(
            pipeline_config.get("latent_warmstart_components", "both")
        ).strip().lower()
        warmstart_components = {
            item.strip()
            for item in warmstart_components_text.replace("+", ",").split(",")
            if item.strip()
        }
        if not warmstart_components or warmstart_components & {"both", "all"}:
            warmstart_components = {"action", "video"}
        use_action_warmstart = "action" in warmstart_components
        use_video_warmstart = "video" in warmstart_components
        if latent_warmstart_enabled and bool(pipeline_config.get("reuse_allowed", True)):
            requested_start_t = float(pipeline_config.get("latent_warmstart_start_t", 0.5))
            requested_start_t = min(1.0, max(0.0, requested_start_t))
            prev_video = pipeline_config.get("prev_video_latent")
            prev_action = pipeline_config.get("prev_action_latent")
            prev_video_valid = torch.is_tensor(prev_video)
            prev_action_valid = torch.is_tensor(prev_action)
            if prev_video_valid:
                prev_video = prev_video.detach().to(device=self.device, dtype=self.dtype)
                prev_video_valid = tuple(prev_video.shape) == video_shape
            if prev_action_valid:
                prev_action = prev_action.detach().to(device=self.device, dtype=self.dtype)
                prev_action_valid = tuple(prev_action.shape) == action_shape
            if (not use_video_warmstart or prev_video_valid) and (not use_action_warmstart or prev_action_valid):
                if use_video_warmstart or use_action_warmstart:
                    if use_video_warmstart:
                        video_noise = video_latent
                        video_latent = prev_video * (1.0 - requested_start_t) + video_noise * requested_start_t
                    if use_action_warmstart:
                        action_noise = action_latent
                        action_latent = prev_action * (1.0 - requested_start_t) + action_noise * requested_start_t
                    reuse_allowed = True
                    warmstart_start_t = requested_start_t
                    warmstart_reason = "reuse:" + ",".join(sorted(warmstart_components))
            else:
                prev_video_shape = tuple(prev_video.shape) if torch.is_tensor(prev_video) else None
                prev_action_shape = tuple(prev_action.shape) if torch.is_tensor(prev_action) else None
                warmstart_reason = (
                    "missing_or_mismatched_prev:"
                    f"video={prev_video_shape} action={prev_action_shape}"
                )
        video_latent[:, :, 0:1] = condition_frame_latent

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop. Warm-started reuse begins from a re-noised previous x0.
        timesteps = torch.linspace(warmstart_start_t, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(num_inference_steps):
            # Timesteps
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)

            # Tokens with Registers
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            state_tokens = state.unsqueeze(1).to(self.dtype)
            # Expand registers for batch
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Note: Understanding tokens already extracted before the loop, will be updated in joint attention
            und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, num_queries * num_layers, und_dim]

            
            # Trimodal MoT forward - joint denoising for WAN, Action, Understanding
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                # Process through all layers - trimodal denoising of WAN, Action, Understanding
                for layer_idx in range(self.config.num_layers):
                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    
                    # Trimodal joint attention: WAN + Action + Understanding
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )

                    # WAN cross-attention with T5 embeddings 
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # FFNs: WAN, Action, Understanding
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Heads (velocities)
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                # Use decoder with all tokens (including registers)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                # Extract middle action chunk (skip first state token and last register tokens)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]

                # Euler integration
                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent

        # 4. Decode outputs
        if decode_video:
            with torch.no_grad():
                decoded_frames = self.video_model.decode_video(video_latent)
                predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
                predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
                predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
        else:
            predicted_frames = None
        
        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]
        self.last_inference_action_latent = action_latent.detach()
        self.last_inference_video_latent = video_latent.detach()
        if latent_warmstart_enabled:
            self.last_pipeline_action_latent = action_latent.detach()
            self.last_pipeline_video_latent = video_latent.detach()
            self.last_pipeline_action_stage = None
            self.last_pipeline_video_stage = None
            self.last_pipeline_condition_frame_latent = condition_frame_latent.detach()
            self.last_pipeline_denoise_info = {
                "mode": "latent_warmstart",
                "reuse_allowed": bool(reuse_allowed),
                "reuse_reason": warmstart_reason,
                "requested_start_t": float(pipeline_config.get("latent_warmstart_start_t", 0.5)),
                "start_t": float(warmstart_start_t),
                "components": ",".join(sorted(warmstart_components)),
                "steps_run": int(num_inference_steps),
                "avg_steps_per_replan": float(num_inference_steps),
                "chunks": 1,
                "reused_chunks": 1 if reuse_allowed else 0,
                "action_shape": tuple(action_latent.shape),
                "video_shape": tuple(video_latent.shape),
            }

        return predicted_frames, predicted_actions

    # Alternative inference (DPM++ solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        solver: Optional[str] = None,
        shift: Optional[float] = None,
        seed: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction with dpm++ solver.
        
        Args:
            first_frame: Initial frame [1, C, H, W] - batch size must be 1 for inference
            state: Initial robot state [1, state_dim]
            num_inference_steps: Number of denoising steps (default: 50)
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: VLM inputs for understanding expert
            solver: Solver type ("dpm++"), defaults to config.inference_solver
            shift: Noise schedule shift, defaults to config.inference_shift
            seed: Random seed for reproducible generation (-1 for random)
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        # Move inputs to device
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)
        
        # Use config defaults if not specified
        if solver is None:
            solver = self.config.inference_solver
        if shift is None:
            shift = self.config.inference_shift

        # Set random seed if specified
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 1. Encode condition frame and initialize latents
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [1, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [1, 48, 1, H', W']

        # Initialize video latent with noise - squeeze batch dimension for WAN format
        _, C_latent, _, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (C_latent, num_total_latent_frames, H_latent, W_latent), 
            device=self.device, 
            dtype=torch.float32,  # Use float32 for sampling numerical stability
            generator=generator
        )
        # Set first frame as condition (teacher forcing)
        video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()
        
        # Initialize action latent with noise
        action_latent = torch.randn(
            (1, self.config.action_chunk_size, self.config.action_dim), 
            device=self.device, 
            dtype=torch.float32,
            generator=generator
        )

        # 2. Prepare understanding features and T5 context (compute once, reuse for all steps)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Setup flow-matching schedulers (separate for video and action due to different tensor shapes)
        if solver == "dpm++":
            # Video scheduler
            video_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,  # Base shift is 1.0
                use_dynamic_shifting=False
            )
            # Action scheduler (independent instance)
            action_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False
            )
            # Get custom sigmas with shift parameter
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                video_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
            # Set same timesteps for action scheduler
            _, _ = retrieve_timesteps(
                action_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Solver '{solver}' not implemented. Currently only 'dpm++' is supported.")

        # 4. Denoising loop with flow-matching solver
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                # Prepare model inputs (add batch dimension back)
                video_latent_input = video_latent.unsqueeze(0).to(self.dtype)  # [1, 48, T, H, W]
                action_latent_input = action_latent.to(self.dtype)  # [1, chunk_size, action_dim]
                
                # Prepare tokens
                video_tokens = self.video_module.prepare_input(video_latent_input)
                state_tokens = state.unsqueeze(1)
                registers = self.action_expert.registers.expand(1, -1, -1)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent_input, registers)
                und_tokens = self.und_module.extract_und_features(vlm_inputs)
                
                # Model forward pass
                with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                    # Time embeddings (t is in [0, 1000] from scheduler)
                    video_t_scaled = t.expand(1).to(self.dtype)
                    action_t_scaled = t.expand(1).to(self.dtype)
                    video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                        video_t_scaled, video_tokens.shape[1]
                    )
                    action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                        action_t_scaled, action_tokens.shape[1]
                    )

                    # Process through all layers - trimodal joint denoising
                    for layer_idx in range(self.config.num_layers):
                        video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                        action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                        
                        # Trimodal joint attention
                        video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                            self.action_expert.blocks[layer_idx],
                            und_tokens, self.und_expert.blocks[layer_idx]
                        )

                        # WAN cross-attention with T5
                        video_tokens = self.video_module.process_cross_attention(
                            video_tokens, video_adaln_params, layer_idx, processed_t5_context
                        )

                        # FFNs for each modality
                        video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                        action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                        und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                    # Prediction heads (predict velocity for flow-matching)
                    video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)  # [1, 48, T, H, W]
                    action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                    action_pred = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]  # [1, chunk_size, action_dim]

                # Update latents using separate schedulers (video and action have different tensor shapes)
                # Video: squeeze batch dim, call video_scheduler, squeeze back
                video_latent = video_scheduler.step(
                    video_pred.squeeze(0).unsqueeze(0),  # Add dummy batch dim for scheduler
                    t,
                    video_latent.unsqueeze(0),  # Add dummy batch dim for scheduler
                    return_dict=False,
                    generator=generator
                )[0].squeeze(0)  # Remove dummy batch dim
                
                # Action: directly use 3D tensor [1, chunk_size, action_dim] with action_scheduler
                # DPM-Solver doesn't require specific dimensions, just consistency
                action_latent = action_scheduler.step(
                    action_pred,  # [1, chunk_size, action_dim]
                    t,
                    action_latent,  # [1, chunk_size, action_dim]
                    return_dict=False,
                    generator=generator
                )[0]
                
                # Teacher forcing: keep first frame as condition
                video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()

        # 5. Decode final outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent.unsqueeze(0).to(self.dtype))  # Add batch dim back
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [1, chunk_size, action_dim]

        return predicted_frames, predicted_actions
    '''

    # Alternative inference (UniPC solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame [B, C, H, W]
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        if self.config.training_mode != 'pretrain':
            state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: use FlowUniPCMultistepScheduler for video and action latents
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        # Use a separate scheduler instance for the action branch to avoid shared internal state
        action_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        action_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        timesteps = scheduler.timesteps  # int64 on device

        for t in timesteps:
            # Tokens (with optional registers)
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
                registers = self.action_expert.registers.expand(B, -1, -1)
            else:
                registers = None
            if self.config.training_mode == 'pretrain':
                action_tokens = self.action_expert.input_encoder(None, action_latent, registers)
            else:
                state_tokens = state.unsqueeze(1).to(self.dtype)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Re-extract understanding features per step (keeps alignment with current pipeline)
            und_tokens = self.und_module.extract_und_features(vlm_inputs)
            
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings: use the current discrete t (0..num_train_timesteps)
                t_scalar = t.to(self.dtype).repeat(B)
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(t_scalar, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(t_scalar, action_tokens.shape[1])

                # Layer stack for joint denoising
                for layer_idx in range(self.config.num_layers):
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )
                    # WAN cross
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Predict velocities (video and action) and take scheduler steps
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_velocity_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                up_len = action_velocity_full.shape[1] - self.action_expert.config.num_registers
                if self.config.training_mode == 'pretrain':
                    action_velocity = action_velocity_full[:, :up_len, :]
                else:
                    action_velocity = action_velocity_full[:, 1:up_len, :]

                # Scheduler steps
                video_latent = scheduler.step(model_output=video_velocity, timestep=t, sample=video_latent, return_dict=False)[0]
                # Teacher Forcing on the first frame (video)
                video_latent[:, :, 0:1] = condition_frame_latent
                action_latent = action_scheduler.step(model_output=action_velocity, timestep=t, sample=action_latent, return_dict=False)[0]

        # 4. Decode outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, 14]

        return predicted_frames, predicted_actions
    '''


def test_motus():
    """Test the complete model."""
    print("Testing Motus...")

    config = MotusConfig()

    try:
        model = Motus(config)
        print("Model created successfully")

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"Model creation failed: {e}")
        print("This is expected without actual pretrained weights")

if __name__ == "__main__":
    test_motus()
