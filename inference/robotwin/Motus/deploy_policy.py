# Motus Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str, device: str = "cuda", log_dir: Optional[str] = None, task_name: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)

        # The model is trained in a specific action space.  Resolve that
        # contract from the checkpoint metadata first, then the deployment
        # config.  Older checkpoints do not carry the field and therefore
        # retain the raw-qpos behavior used by the original RobotWin runs.
        self.action_normalization = self._resolve_action_normalization()

        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_model = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load stats only when the selected model contract requires them.
        if self.action_normalization == "min_max":
            self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _resolve_action_normalization(self) -> str:
        """Resolve and validate the state/action representation.

        A checkpoint's ``config.json`` is authoritative when it records the
        mode.  If both the checkpoint and deployment config specify a mode,
        reject a mismatch instead of silently feeding a model the wrong
        numerical distribution.  Checkpoints produced before this field was
        introduced default to raw qpos (``none``).
        """
        config_common = self.config_dict.get("common", {}) or {}
        configured_mode = config_common.get("action_normalization")
        checkpoint_mode = None

        checkpoint_path = Path(self.checkpoint_path) if self.checkpoint_path else None
        if checkpoint_path is not None:
            metadata_path = (
                checkpoint_path / "config.json"
                if checkpoint_path.is_dir()
                else checkpoint_path.parent / "config.json"
            )
            if metadata_path.exists():
                try:
                    import json

                    with metadata_path.open("r", encoding="utf-8") as metadata_file:
                        metadata = json.load(metadata_file)
                    if not isinstance(metadata, dict):
                        raise ValueError("checkpoint metadata must be a JSON object")
                    checkpoint_common = metadata.get("common", {}) or {}
                    if not isinstance(checkpoint_common, dict):
                        raise ValueError("checkpoint common metadata must be an object")
                    checkpoint_mode = checkpoint_common.get("action_normalization")
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        "Could not read checkpoint action-space metadata from "
                        f"{metadata_path}"
                    ) from exc

        if configured_mode is not None and checkpoint_mode is not None:
            if configured_mode != checkpoint_mode:
                raise ValueError(
                    "Action normalization mismatch: deployment config requests "
                    f"{configured_mode!r}, but checkpoint metadata records "
                    f"{checkpoint_mode!r}"
                )

        mode = checkpoint_mode if checkpoint_mode is not None else configured_mode
        if mode is None:
            mode = "none"
        if mode not in {"none", "min_max"}:
            raise ValueError(
                "action_normalization must be 'none' or 'min_max', "
                f"got {mode!r}"
            )
        logger.info("Action normalization mode: %s", mode)
        return mode

    def _model_state(self) -> torch.Tensor:
        """Return the observation state in the checkpoint's model space."""
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        if self.action_normalization == "min_max":
            return self._normalize_actions(self.current_state)
        return self.current_state

    def _environment_actions(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        """Convert model actions to the qpos space consumed by RoboTwin."""
        if self.action_normalization == "min_max":
            return self._denormalize_actions(predicted_actions)
        return predicted_actions

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        if state_tensor.ndim != 2:
            raise ValueError(
                f"Robot state must have shape [batch, dim], got {tuple(state_tensor.shape)}"
            )
        expected_state_dim = int(
            self.config_dict["common"].get("state_dim", state_tensor.shape[-1])
        )
        if state_tensor.shape[-1] != expected_state_dim:
            raise ValueError(
                f"Robot state dimension {state_tensor.shape[-1]} does not match "
                f"configured state_dim {expected_state_dim}"
            )

        self.current_state = state_tensor.to(self.device)
        self.current_state_model = self._model_state()
        # Preserve the old attribute for integrations that inspected it, but
        # make its meaning explicit: it is populated only for normalized
        # checkpoints and is never an implicit second source of truth.
        self.current_state_norm = (
            self.current_state_model if self.action_normalization == "min_max" else None
        )
    
    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self._model_state(),
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                condition_frame_viz = current_frame.squeeze(0)
                predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        predicted_actions = predicted_actions.squeeze(0)
        if predicted_actions.ndim != 2:
            raise ValueError(
                "Model actions must have shape [time, action_dim] after removing "
                f"the batch dimension, got {tuple(predicted_actions.shape)}"
            )
        actions_real = self._environment_actions(predicted_actions).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load and validate the RobotWin action-space statistics."""
        stats_path = Path(__file__).parent / "utils" / "stat.json"
        try:
            with stats_path.open("r", encoding="utf-8") as stats_file:
                stat_data = (
                    yaml.safe_load(stats_file)
                    if stats_path.suffix in [".yml", ".yaml"]
                    else None
                )
        except Exception:
            stat_data = None
        if stat_data is None:
            import json

            with stats_path.open("r", encoding="utf-8") as stats_file:
                stat_data = json.load(stats_file)

        stats = stat_data.get("robotwin2")
        if stats is None:
            raise ValueError("RobotWin normalization stats not found")
        self.action_min = torch.tensor(stats["min"], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats["max"], dtype=torch.float32, device=self.device)
        if self.action_min.ndim != 1 or self.action_max.shape != self.action_min.shape:
            raise ValueError("RobotWin normalization min/max must be one-dimensional and matching")
        self.action_range = self.action_max - self.action_min
        if not torch.isfinite(self.action_min).all() or not torch.isfinite(self.action_max).all():
            raise ValueError("RobotWin normalization stats must be finite")
        if torch.any(self.action_range <= 0):
            raise ValueError("RobotWin normalization ranges must be positive")

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state/actions to the model's [0, 1] min/max space."""
        if self.action_min is None or self.action_max is None:
            raise RuntimeError("Normalization stats are not loaded")
        if x.shape[-1] != self.action_min.numel():
            raise ValueError(
                f"Expected last dimension {self.action_min.numel()}, got {x.shape[-1]}"
            )
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Convert model-space actions back to RoboTwin qpos coordinates."""
        if self.action_min is None or self.action_max is None:
            raise RuntimeError("Normalization stats are not loaded")
        if y.shape[-1] != self.action_min.numel():
            raise ValueError(
                f"Expected last dimension {self.action_min.numel()}, got {y.shape[-1]}"
            )
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


def encode_obs(observation):
    """Post-Process Observation"""
    return observation


def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name')
    )
    
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')


def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.current_state_model = None
    model.current_state_norm = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
