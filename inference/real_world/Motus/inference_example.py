#!/usr/bin/env python3
"""
Real-World Motus Inference Example (No Environment Required)

This script demonstrates how to run Motus inference on a single image without any robot environment.
It supports two modes:
1. With T5: encode instruction text on the fly
2. Without T5: use pre-encoded T5 embeddings

Example usage (similar to RDT2 style):
```python
import torch
import yaml
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.motus import Motus, MotusConfig
from transformers import AutoProcessor
from wan.modules.t5 import T5EncoderModel
from utils.device import resolve_device

# Load config
with open("inference/real_world/Motus/utils/aloha_agilex_2.yml", "r") as f:
    config = yaml.safe_load(f)

# Create model
device = "cuda:0"
model_config = MotusConfig(
    wan_checkpoint_path=config['model']['wan']['checkpoint_path'],
    vae_path=config['model']['wan']['vae_path'],
    wan_config_path=config['model']['wan']['config_path'],
    video_precision=config['model']['wan']['precision'],
    vlm_checkpoint_path=config['model']['vlm']['checkpoint_path'],
    # ... other configs from yaml
    load_pretrained_backbones=False,  # Load from checkpoint instead
)
model = Motus(model_config).to(device).eval()

# Load checkpoint
model.load_checkpoint("/path/to/checkpoint_step_xxxxx", strict=False)

# Prepare inputs
first_frame = Image.open("/path/to/image.png").convert("RGB")
first_frame_tensor = torch.from_numpy(np.array(first_frame.resize((320, 384)))).permute(2,0,1).unsqueeze(0).float() / 255.0
state = torch.zeros((1, config['common']['state_dim']), dtype=torch.bfloat16, device=device)

from utils.device import resolve_device
processor = AutoProcessor.from_pretrained(config['model']['vlm']['checkpoint_path'], trust_remote_code=True)
vlm_inputs = processor(text=["Pick up the cube."], images=[first_frame], return_tensors='pt')
vlm_inputs = {k: v.to(device) for k, v in vlm_inputs.items()}

# Option 1: Encode instruction with T5
t5_encoder = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=device,
    checkpoint_path="/path/to/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="/path/to/Wan2.2-TI2V-5B/google/umt5-xxl",
)
language_embeddings = t5_encoder(["Pick up the cube."], device)

# Option 2: Use pre-encoded T5 embeddings
# language_embeddings = torch.load("/path/to/preencoded_t5.pt", map_location=device)
# if isinstance(language_embeddings, torch.Tensor):
#     language_embeddings = [language_embeddings]

# Run inference
with torch.no_grad():
    predicted_frames, predicted_actions = model.inference_step(
        first_frame=first_frame_tensor.to(device),
        state=state,
        num_inference_steps=config['model']['inference']['num_inference_timesteps'],
        language_embeddings=language_embeddings,
        vlm_inputs=[vlm_inputs],
    )

# predicted_frames: torch.Tensor of shape (B, T, C, H, W) or (B, C, T, H, W)
# predicted_actions: torch.Tensor of shape (B, action_chunk_size, action_dim)
#   - action_chunk_size = num_video_frames * video_action_freq_ratio
#   - action_dim: robot action dimension (e.g., 14 for single arm)
```

This file also provides a command-line interface for convenience.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np
from PIL import Image
import yaml

# Add project root to import model
PROJ_ROOT = str(Path(__file__).resolve().parents[3])
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.motus import Motus, MotusConfig
from transformers import AutoProcessor
from wan.modules.t5 import T5EncoderModel
from utils.device import resolve_device


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_image_as_tensor(image_path: str, size_hw: tuple[int, int]) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size_hw[1], size_hw[0]), Image.BICUBIC)  # (W,H)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    return tensor


def build_vlm_inputs(processor, instruction: str, image: Image.Image, device: torch.device) -> Dict[str, torch.Tensor]:
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image', 'image': image},
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    encoded = processor(text=[text], images=[image], return_tensors='pt')
    vlm_inputs = {
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device),
        'pixel_values': encoded['pixel_values'].to(device),
        'image_grid_thw': encoded.get('image_grid_thw', None)
    }
    if vlm_inputs['image_grid_thw'] is not None:
        vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(device)
    return vlm_inputs


def save_frame_grid(condition_frame: torch.Tensor, predicted_frames: torch.Tensor, save_path: str) -> None:
    # condition_frame: [C,H,W]; predicted_frames: [T,C,H,W]
    cf = (condition_frame.detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    frames = []
    T = predicted_frames.shape[0]
    for i in range(T):
        f = (predicted_frames[i].detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        frames.append(f)
    all_frames = [cf] + frames
    grid = np.concatenate(all_frames, axis=1)
    Image.fromarray(grid).save(save_path)


def create_motus_from_yaml(config_dict: Dict[str, Any], device: torch.device) -> Motus:
    common = config_dict['common']
    model_cfg = config_dict['model']
    mc = MotusConfig(
        device=str(device),
        wan_checkpoint_path=model_cfg['wan']['checkpoint_path'],
        vae_path=model_cfg['wan']['vae_path'],
        wan_config_path=model_cfg['wan']['config_path'],
        video_precision=model_cfg['wan']['precision'],
        vlm_checkpoint_path=model_cfg['vlm']['checkpoint_path'],
        und_expert_hidden_size=model_cfg.get('und_expert', {}).get('hidden_size', 512),
        und_expert_ffn_dim_multiplier=model_cfg.get('und_expert', {}).get('ffn_dim_multiplier', 4),
        und_expert_norm_eps=model_cfg.get('und_expert', {}).get('norm_eps', 1e-5),
        vlm_adapter_input_dim=model_cfg.get('und_expert', {}).get('vlm', {}).get('input_dim', 2048),
        vlm_adapter_projector_type=model_cfg.get('und_expert', {}).get('vlm', {}).get('projector_type', "mlp3x_silu"),
        num_layers=30,
        action_state_dim=common['state_dim'],
        action_dim=common['action_dim'],
        action_expert_dim=model_cfg['action_expert']['hidden_size'],
        action_expert_ffn_dim_multiplier=model_cfg['action_expert']['ffn_dim_multiplier'],
        action_expert_norm_eps=model_cfg['action_expert'].get('norm_eps', 1e-6),
        global_downsample_rate=common['global_downsample_rate'],
        video_action_freq_ratio=common['video_action_freq_ratio'],
        num_video_frames=common['num_video_frames'],
        video_height=common['video_height'],
        video_width=common['video_width'],
        batch_size=1,
        video_loss_weight=model_cfg['loss_weights']['video_loss_weight'],
        action_loss_weight=model_cfg['loss_weights']['action_loss_weight'],
        training_mode='finetune',
        load_pretrained_backbones=False,  # we will load from checkpoint
    )
    model = Motus(mc).to(device)
    return model


def load_checkpoint_into_model(model: Motus, ckpt_path: str) -> None:
    try:
        model.load_checkpoint(ckpt_path, strict=False)
        print(f"Loaded Motus checkpoint from {ckpt_path}")
    except Exception as e:
        print(f"WARNING: failed to load checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Real-World Motus inference sample (no env)")
    parser.add_argument("--model_config", required=True, help="Path to real-world YAML (e.g., inference/real_world/Motus/utils/aloha_agilex_2.yml)")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory (contains mp_rank_00_model_states.pt or state dir)")
    parser.add_argument("--wan_path", required=True, help="Base path to WAN models (to find T5 and VAE)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--instruction", required=True, help="Instruction text")
    parser.add_argument("--output", default="inference_result.png", help="Where to save predicted frames grid")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "xpu", "cpu"), help="PyTorch device backend")
    parser.add_argument("--use_t5", action="store_true", help="Load T5 and encode instruction on the fly")
    parser.add_argument("--t5_embeds", default=None, help="Path to pre-encoded T5 embeddings (.pt) when not using --use_t5")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using PyTorch device: {device}")

    # Load config
    cfg = load_yaml_config(args.model_config)

    # Create model
    model = create_motus_from_yaml(cfg, device)
    model.eval()
    load_checkpoint_into_model(model, args.ckpt_dir)

    # Prepare inputs
    H, W = cfg['common']['video_height'], cfg['common']['video_width']
    first_frame = load_image_as_tensor(args.image, (H, W)).to(device)  # [1,C,H,W]
    state_dim = int(cfg['common']['state_dim'])
    state = torch.zeros((1, state_dim), dtype=torch.bfloat16, device=device)  # no env, zero state

    # Build VLM inputs
    vlm_ckpt = cfg['model']['vlm']['checkpoint_path']
    processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=True)
    first_frame_pil = Image.open(args.image).convert("RGB").resize((W, H), Image.BICUBIC)
    vlm_inputs = build_vlm_inputs(processor, args.instruction, first_frame_pil, device)

    # Build T5 embeddings
    if args.use_t5:
        t5_ckpt = os.path.join(args.wan_path, 'Wan2.2-TI2V-5B', 'models_t5_umt5-xxl-enc-bf16.pth')
        t5_tokenizer = os.path.join(args.wan_path, 'Wan2.2-TI2V-5B', 'google/umt5-xxl')
        t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=str(device),
            checkpoint_path=t5_ckpt,
            tokenizer_path=t5_tokenizer,
        )
        t5_out = t5([args.instruction], device=str(device))
        if isinstance(t5_out, torch.Tensor):
            language_embeddings: List[torch.Tensor] = [t5_out.squeeze(0)]
        else:
            language_embeddings = t5_out
    else:
        if args.t5_embeds is None:
            raise ValueError("Please provide --t5_embeds when not using --use_t5")
        loaded = torch.load(args.t5_embeds, map_location=device)
        # Expect either a tensor [seq_len, dim] or list of tensors
        if isinstance(loaded, torch.Tensor):
            language_embeddings = [loaded.to(device)]
        elif isinstance(loaded, list):
            language_embeddings = [t.to(device) for t in loaded]
        else:
            raise ValueError("Unsupported t5_embeds format, expected Tensor or List[Tensor]")

    # Inference
    with torch.no_grad():
        predicted_frames, predicted_actions = model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=cfg['model']['inference']['num_inference_timesteps'],
            language_embeddings=language_embeddings,
            vlm_inputs=[vlm_inputs],
        )

    # Save frames grid
    # Convert predicted_frames to [T,C,H,W]
    if predicted_frames.dim() == 5 and predicted_frames.shape[1] != 3:
        frames_vis = predicted_frames.squeeze(0)  # [T,C,H,W]
    else:
        frames_vis = predicted_frames.permute(0, 2, 1, 3, 4).squeeze(0)
    save_frame_grid(first_frame.squeeze(0), frames_vis, args.output)
    print(f"Saved predicted frames grid to {args.output}")

    # Print actions
    print("Predicted actions shape:", tuple(predicted_actions.shape))
    print("First 3 actions:\n", predicted_actions.squeeze(0)[:3].float().cpu().numpy())


if __name__ == "__main__":
    main()
