#!/usr/bin/env python3
# Training script for Motus

import os
import re
import sys
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Set CUDA memory management environment variables to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None
import wandb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
import yaml
from omegaconf import OmegaConf
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.motus import Motus, MotusConfig
from data.dataset import create_dataset, collate_fn
from utils.scheduler import create_scheduler
from sample import evaluate_model, log_evaluation_metrics

logger = logging.getLogger(__name__)


def move_nested_to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_nested_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_nested_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_nested_to_device(v, device) for v in value)
    return value


def _is_lora_key(name: str) -> bool:
    return (
        "lora_A" in name
        or "lora_B" in name
        or "pipeline_chunk_embedding" in name
        or "pipeline_stage_embedding" in name
    )


def collect_lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    try:
        named_params = model.named_parameters(remove_duplicate=True)
    except TypeError:
        named_params = model.named_parameters()

    lora_state = {}
    seen = set()
    for name, param in named_params:
        if not _is_lora_key(name):
            continue
        key = (param.data_ptr(), tuple(param.shape))
        if key in seen:
            continue
        seen.add(key)
        lora_state[name] = param.detach().cpu()
    return lora_state


def get_lora_save_config(config: Optional[Any]) -> Dict[str, Any]:
    if config is None:
        return {}
    try:
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    except Exception:
        return {}
    model_cfg = cfg_dict.get("model", {})
    return {
        "common": cfg_dict.get("common", {}),
        "action_expert": model_cfg.get("action_expert", {}),
        "und_expert": model_cfg.get("und_expert", {}),
        "lora": model_cfg.get("lora", {}),
        "extended_chunkwise_finetune": model_cfg.get("extended_chunkwise_finetune", {}),
        "dataset": cfg_dict.get("dataset", {}),
        "finetune": cfg_dict.get("finetune", {}),
    }


def setup_logging(rank: int = 0, log_level: str = "INFO"):
    """Setup logging configuration."""
    # Temporarily set to DEBUG for NaN debugging
    if log_level == "INFO":
        log_level = "DEBUG"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress specific distributed warnings that are noisy but harmless for our flow
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )

def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Calculate derived parameters
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    
    # Validate dataset configuration
    dataset_config = {
        'dataset_type': config.dataset.type,
        'dataset_dir': config.dataset.dataset_dir if hasattr(config.dataset, 'dataset_dir') else None,
        'global_downsample_rate': config.common.global_downsample_rate,
        'video_action_freq_ratio': config.common.video_action_freq_ratio,
        'num_video_frames': config.common.num_video_frames
    }
    
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Dataset type: {config.dataset.type}")
    if hasattr(config, 'training_mode'):
        logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Action chunk size: {config.common.action_chunk_size}")
    logger.info(f"Video frames: {config.common.num_video_frames}")
    
    return config

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

class UniDiffuserTrainer:
    """Trainer class for Motus."""
    
    def __init__(
        self,
        model: Motus,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: str = "./checkpoints_stage4",
        log_interval: int = 100,
        save_interval: int = 1000,
        val_interval: int = 1000,
        report_to: str = "wandb",
        tb_writer: Optional[Any] = None,
        accelerator: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.report_to = report_to
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config
        if self.config is not None and hasattr(self.config, "training"):
            self.gradient_accumulation_steps = max(
                1,
                int(self.config.training.get("gradient_accumulation_steps", 1)),
            )
        else:
            self.gradient_accumulation_steps = 1
        self.micro_step = 0
        
        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        
        logger.info(f"Motus Trainer initialized on rank {rank}/{world_size}")
        logger.info(f"Logging backends: {report_to}")
        logger.info(
            "Gradient accumulation steps: %d (effective global batch ~= per_device_batch * %d ranks * %d)",
            self.gradient_accumulation_steps,
            world_size,
            self.gradient_accumulation_steps,
        )

    def save_checkpoint(self, suffix: str = ""):
        """Save complete training state using accelerator."""
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"

        save_lora_only = False
        try:
            save_lora_only = bool(getattr(self.config.system, "save_lora_only", False))
        except Exception:
            save_lora_only = False

        if save_lora_only:
            if self.rank == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
                lora_state = collect_lora_state_dict(unwrapped_model)
                torch.save(lora_state, checkpoint_dir / "lora_adapters.pt")
                with open(checkpoint_dir / "config.json", "w") as f:
                    json.dump(get_lora_save_config(self.config), f, indent=2)
                logger.info(f"LoRA checkpoint saved to {checkpoint_dir} ({len(lora_state)} tensors)")
            return

        # Use accelerator to save complete training state
        # This saves model, optimizer, scheduler, dataloader, and RNG states
        self.accelerator.save_state(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        # Also save a config.json alongside weights for reproducibility
        try:
            from omegaconf import OmegaConf as _OmegaConf
            cfg_dict = _OmegaConf.to_container(self.config, resolve=True) if self.config is not None else {}
            # Filter only requested sections
            common = cfg_dict.get("common", {})
            model = cfg_dict.get("model", {})
            filtered = {
                "common": common,
                "action_expert": model.get("action_expert", {}),
                "und_expert": model.get("und_expert", {}),
                "time_distribution": model.get("time_distribution", {}),
                "ema": model.get("ema", {}),
                "lora": model.get("lora", {}),
                "extended_chunkwise_finetune": model.get("extended_chunkwise_finetune", {}),
            }
            import json as _json
            with open(checkpoint_dir / "config.json", "w") as f:
                _json.dump(filtered, f, indent=2)
            logger.info(f"Wrote config.json to {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Failed to write config.json: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = True):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            reset_scheduler: If True, reset scheduler to new config instead of loading from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
            return
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Extract step number from checkpoint path (e.g., checkpoint_step_125000)
        step_match = re.search(r'step_(\d+)', checkpoint_path)
        if step_match:
            self.global_step = int(step_match.group(1))
            logger.info(f"Resuming from step {self.global_step}")
        else:
            logger.warning(f"Could not extract step number from {checkpoint_path}, starting from step 0")

        # Load using accelerator (includes model, optimizer, scheduler states)
        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
        
        # Reset scheduler with new config if requested
        if reset_scheduler and self.config is not None and self.scheduler is not None:
            logger.info("Resetting scheduler to new configuration (not using checkpoint scheduler state)...")
            
            # Unwrap scheduler if it's wrapped by accelerator
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module
            
            # Check if it's our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'warm_up_steps'):
                # Update scheduler parameters with new config
                unwrapped_scheduler.warm_up_steps = self.config.training.warmup_steps
                unwrapped_scheduler.cycle_length = self.config.training.cycle_length
                unwrapped_scheduler.f_max = self.config.training.f_max
                unwrapped_scheduler.f_min = self.config.training.f_min
                # Update base_lrs for all parameter groups
                unwrapped_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                
                # Reset step_count to 0 so scheduler starts warmup from beginning
                unwrapped_scheduler.step_count = 0
                
                logger.info(f"Updated scheduler config: warmup={unwrapped_scheduler.warm_up_steps}, "
                          f"cycle_length={unwrapped_scheduler.cycle_length}, "
                          f"f_max={unwrapped_scheduler.f_max}, f_min={unwrapped_scheduler.f_min}")
                logger.info(f"Base learning rates: {[f'{lr:.2e}' for lr in unwrapped_scheduler.base_lrs]}")
                
                # Don't directly modify optimizer's lr! Let scheduler update it naturally on next step
                # Only log the target lr that scheduler will set
                initial_lrs = [base_lr * unwrapped_scheduler.f_max for base_lr in unwrapped_scheduler.base_lrs]
                logger.info(f"Reset scheduler step_count to 0 (will start warmup from next step)")
                logger.info(f"Target initial learning rates: {[f'{lr:.2e}' for lr in initial_lrs]}")
                logger.info(f"Learning rate will be updated by scheduler on first training step")
            
            # Log current learning rate (from checkpoint)
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (will be overridden by scheduler): {current_lr:.2e}")
        elif self.scheduler is not None:
            # If not resetting scheduler, sync scheduler progress with global_step
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module

            # Case 1: our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'step_count'):
                old_step_count = unwrapped_scheduler.step_count
                unwrapped_scheduler.step_count = self.global_step
                logger.info(f"Synchronized scheduler step_count: {old_step_count} -> {self.global_step}")

            # Case 2: diffusers_cosine wrapper with inner scheduler
            if hasattr(unwrapped_scheduler, 'inner') and hasattr(unwrapped_scheduler.inner, 'last_epoch'):
                try:
                    old_epoch = int(getattr(unwrapped_scheduler.inner, 'last_epoch', -1))
                except Exception:
                    old_epoch = -1
                # Align inner scheduler epoch with current global_step so schedule continues
                unwrapped_scheduler.inner.last_epoch = int(self.global_step)
                logger.info(f"Aligned diffusers scheduler last_epoch: {old_epoch} -> {self.global_step}")

            # Log current optimizer LR (authoritative)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (optimizer): {current_lr:.2e}")
    
    def train_step(self, batch: Dict[str, Any]) -> tuple[Dict[str, float], bool]:
        """Run one micro-batch and return whether an optimizer update happened."""
        self.model.train()
        
        first_frame = batch['first_frame'].to(self.device, dtype=self.dtype)          # [B, C, H, W]
        video_frames = batch['video_frames'].to(self.device, dtype=self.dtype)        # [B, num_video_frames, C, H, W]
        language_embeddings = batch['language_embedding']
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(self.device, dtype=self.dtype)
        state = batch.get('initial_state', None)
        if state is not None:
            state = state.to(self.device, dtype=self.dtype)      # [B, state_dim]
        actions = batch['action_sequence'].to(self.device, dtype=self.dtype)  # [B, action_chunk_size, action_dim]
        # Handle VLM inputs - it's a Dict[str, Tensor] from collate_fn
        vlm_inputs = batch['vlm_inputs']
        if vlm_inputs is not None:
            vlm_inputs = move_nested_to_device(vlm_inputs, self.device)
        
        extended_cfg = getattr(getattr(self.config, 'model', {}), 'extended_chunkwise_finetune', {})
        use_extended_chunkwise = bool(extended_cfg.get('enabled', False))
        if use_extended_chunkwise:
            if "extended_action_sequence" not in batch or batch["extended_action_sequence"] is None:
                raise ValueError(
                    "extended_chunkwise_finetune is enabled but the batch does not contain "
                    "extended_action_sequence. Check dataset horizon and collate settings."
                )
            extended_actions = batch["extended_action_sequence"].to(self.device, dtype=self.dtype)
            loss_dict = self.model(
                first_frame=first_frame,
                video_frames=video_frames,
                state=state,
                actions=extended_actions,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
                extended_chunkwise=True,
                return_dict=True,
            )
        else:
            loss_dict = self.model(
                first_frame=first_frame,
                video_frames=video_frames,
                state=state,
                actions=actions,
                language_embeddings=language_embeddings,  # For WAN cross attention
                vlm_inputs=vlm_inputs,  # Complete VLM inputs from dataset
                return_dict=True
            )
        
        total_loss = loss_dict['total_loss']
        
        # Backward pass (using accelerator if available)
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            with self.accelerator.accumulate(self.model):
                self.accelerator.backward(total_loss)
                did_optimizer_step = bool(self.accelerator.sync_gradients)
                if did_optimizer_step:
                    grad_clip_norm = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
        else:
            scaled_loss = total_loss / self.gradient_accumulation_steps
            scaled_loss.backward()
            self.micro_step += 1
            did_optimizer_step = (self.micro_step % self.gradient_accumulation_steps == 0)
            if did_optimizer_step:
                grad_clip_norm = self.config.training.grad_clip_norm if hasattr(self.config.training, 'grad_clip_norm') else 1.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
        
        # Convert to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        return metrics, did_optimizer_step
    
    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500, reset_scheduler: Optional[bool] = None):
        """
        Main training loop.
        
        Args:
            max_steps: Maximum number of training steps
            resume_from: Path to checkpoint to resume from
            val_interval: Validation interval in steps
            reset_scheduler: If True, reset scheduler to new config. If None, use config.resume.reset_scheduler
        """
        # Load checkpoint if specified
        if resume_from:
            # Determine whether to reset scheduler
            if reset_scheduler is None:
                # Use config value if available, otherwise default to True
                if self.config is not None and hasattr(self.config, 'resume') and hasattr(self.config.resume, 'reset_scheduler'):
                    reset_scheduler = bool(self.config.resume.reset_scheduler)
                else:
                    reset_scheduler = True  # Default behavior
            
            self.load_checkpoint(resume_from, reset_scheduler=reset_scheduler)
        
        logger.info(f"Starting UniDiffuser training for {max_steps} steps")
        
        start_time = time.time()
        
        # Step-based training loop. global_step counts optimizer updates, not micro-batches.
        data_iter = iter(self.train_dataloader)
        epoch = 0
        update_start_time = time.time()
        metric_buffer: Dict[str, float] = {}
        buffered_micro_batches = 0
        
        while self.global_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # End of epoch, restart dataloader
                epoch += 1
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            if batch is None:  # Handle None batches
                continue
                
            if buffered_micro_batches == 0:
                update_start_time = time.time()

            # Training step
            metrics, did_optimizer_step = self.train_step(batch)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_buffer[key] = metric_buffer.get(key, 0.0) + float(value)
            buffered_micro_batches += 1

            if not did_optimizer_step:
                continue
            
            step_time = time.time() - update_start_time
            self.global_step += 1
            metrics = {
                key: value / max(1, buffered_micro_batches)
                for key, value in metric_buffer.items()
            }
            metrics["micro_batches"] = float(buffered_micro_batches)
            metric_buffer = {}
            buffered_micro_batches = 0
            
            # Logging
            if self.global_step % self.log_interval == 0 and self.rank == 0:
                # Log optimizer LR as authoritative (may differ from inner scheduler get_last_lr)
                lrs = [g['lr'] for g in self.optimizer.param_groups]
                lr_main = lrs[0] if len(lrs) > 0 else 0.0
                lr_wan = lrs[1] if len(lrs) > 1 else lr_main
                
                video_metric = metrics.get('video_loss', metrics.get('video_distill_loss', 0.0))
                action_metric = metrics.get('action_loss', metrics.get('action_distill_loss', 0.0))
                extra_metrics = []
                if 'velocity_distill_loss' in metrics:
                    extra_metrics.append(f"Velocity: {metrics['velocity_distill_loss']:.4f}")
                if 'intermediate_distill_loss' in metrics:
                    extra_metrics.append(f"Intermediate: {metrics['intermediate_distill_loss']:.4f}")
                if 'direct_preservation_loss' in metrics:
                    extra_metrics.append(f"Direct: {metrics['direct_preservation_loss']:.4f}")
                extra_metric_str = f", {'; '.join(extra_metrics)}" if extra_metrics else ""
                log_str = (
                    f"Step {self.global_step}/{max_steps}, "
                    f"Loss: {metrics['total_loss']:.4f} "
                    f"(Video: {video_metric:.4f}, Action: {action_metric:.4f}{extra_metric_str}), "
                    f"LR(main/wan): {lr_main:.2e}/{lr_wan:.2e}, Time: {step_time:.2f}s"
                )
                logger.info(log_str)
                
                # Log to WandB
                if "wandb" in self.report_to:
                    wandb.log({
                        **metrics,
                        'learning_rate_main': lr_main,
                        'learning_rate_wan': lr_wan,
                        'step_time': step_time,
                        'epoch': epoch,
                        'global_step': self.global_step,
                        })
                
                # Log to TensorBoard
                if self.tb_writer is not None:
                    for key, value in metrics.items():
                        self.tb_writer.add_scalar(f'train/{key}', value, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_main', lr_main, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_wan', lr_wan, self.global_step)
                    self.tb_writer.add_scalar('train/step_time', step_time, self.global_step)
                    self.tb_writer.add_scalar('train/epoch', epoch, self.global_step)

            # Validation: rank0-only local eval; then synchronize all processes
            if self.global_step % val_interval == 0 and self.val_dataloader is not None:
                if self.rank == 0:
                    val_metrics = evaluate_model(
                        self.model, self.val_dataloader, self.accelerator, self.config,
                        num_eval_batches=2
                    )
                    logger.info(f"Validation - Step {self.global_step}")
                    log_evaluation_metrics(val_metrics, self.tb_writer, self.accelerator, self.global_step)
                # Use explicit barrier with device_ids to avoid NCCL warnings
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier(device_ids=[torch.cuda.current_device()])
                    except TypeError:
                        # Fallback for older PyTorch versions without device_ids argument
                        dist.barrier()
                
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()
        
        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"UniDiffuser training completed in {total_time:.2f}s ({self.global_step} steps)")
            self.save_checkpoint()

def _get_lora_config(config: OmegaConf):
    if hasattr(config.model, "lora"):
        return config.model.lora
    return {}


def _get_extended_chunkwise_config(config: OmegaConf):
    if hasattr(config.model, "extended_chunkwise_finetune"):
        return config.model.extended_chunkwise_finetune
    return {}


def create_model(config: OmegaConf) -> Motus:
    """Create UniDiffuser model from config."""
    lora_cfg = _get_lora_config(config)
    extended_cfg = _get_extended_chunkwise_config(config)
    chunk_loss_weights = list(extended_cfg.get('chunk_loss_weights', [1.0, 0.7, 0.5]))
    while len(chunk_loss_weights) < 3:
        chunk_loss_weights.append(chunk_loss_weights[-1] if chunk_loss_weights else 1.0)
    model_config = MotusConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        # Action Expert configuration from config file
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        # Understanding Expert configuration from config file
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=config.model.und_expert.norm_eps,
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        # Video dimensions from config
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=config.training.batch_size,
        activation_checkpointing=bool(config.model.get('activation_checkpointing', False)),
        activation_checkpointing_use_reentrant=bool(config.model.get('activation_checkpointing_use_reentrant', False)),
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        training_mode=getattr(config, 'training_mode', 'finetune'),
        load_pretrained_backbones=getattr(config.model, 'load_pretrained_backbones', None),
        lora_enabled=bool(lora_cfg.get('enabled', False)),
        lora_rank=int(lora_cfg.get('rank', 8)),
        lora_alpha=float(lora_cfg.get('alpha', 16.0)),
        lora_dropout=float(lora_cfg.get('dropout', 0.0)),
        lora_target_linear=bool(lora_cfg.get('target_linear', True)),
        lora_target_qkv=bool(lora_cfg.get('target_qkv', True)),
        time_distribution=dict(config.model.get('time_distribution', {})),
        extended_chunkwise_enabled=bool(extended_cfg.get('enabled', False)),
        extended_chunkwise_multiplier=int(extended_cfg.get('multiplier', 3)),
        extended_chunkwise_pipeline_depth=int(extended_cfg.get('pipeline_depth', 3)),
        extended_chunkwise_chunk_causal_mask=bool(extended_cfg.get('chunk_causal_mask', True)),
        extended_chunkwise_pipeline_embeddings=bool(extended_cfg.get('pipeline_embeddings', True)),
        extended_chunkwise_chunk_weight_0=float(chunk_loss_weights[0]),
        extended_chunkwise_chunk_weight_1=float(chunk_loss_weights[1]),
        extended_chunkwise_chunk_weight_2=float(chunk_loss_weights[2]),
        extended_chunkwise_constant_weight=float(extended_cfg.get('temporally_constant_weight', 0.2)),
        extended_chunkwise_chunkwise_weight=float(extended_cfg.get('chunk_wise_weight', 0.8)),
    )
    return Motus(model_config)


def apply_lora_finetune_if_enabled(model: Motus, config: OmegaConf):
    lora_cfg = _get_lora_config(config)
    if not bool(lora_cfg.get('enabled', False)):
        return None
    return model.enable_action_und_lora_finetune(
        rank=int(lora_cfg.get('rank', 8)),
        alpha=float(lora_cfg.get('alpha', 16.0)),
        dropout=float(lora_cfg.get('dropout', 0.0)),
        target_linear=bool(lora_cfg.get('target_linear', True)),
        target_qkv=bool(lora_cfg.get('target_qkv', True)),
    )


def create_optimizer_and_scheduler(model: Motus, config: OmegaConf) -> tuple:
    """Create optimizer and scheduler after all finetuning adapters are configured."""
    # Optimizer - parameter groups for separate WAN (video model) learning rate
    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))

    # Collect WAN params explicitly (exclude VAE, we only train diffusion WAN)
    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    wan_param_ids = {id(p) for p in wan_params}
    other_params = [p for p in all_trainable if id(p) not in wan_param_ids]

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'lr': base_lr})
    if len(wan_params) > 0:
        param_groups.append({'params': wan_params, 'lr': wan_lr})
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    scheduler = create_scheduler(optimizer, config)
    
    return optimizer, scheduler


def create_model_and_optimizer(config: OmegaConf) -> tuple:
    """Create UniDiffuser model and optimizer from config."""
    model = create_model(config)
    apply_lora_finetune_if_enabled(model, config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    return model, optimizer, scheduler

def create_dataloaders(config: OmegaConf, rank: int, world_size: int) -> tuple:
    """Create train and validation dataloaders from config."""
    train_dataset = create_dataset(config, val=False)
    val_dataset = create_dataset(config, val=True)

    # Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None
    
    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    return train_dataloader, val_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train Three-Modal UniDiffuser Model")
    
    # Configuration file
    parser.add_argument("--config", type=str, 
                       default="configs/aloha_agilex_2.yaml",
                       help="Path to configuration file")
    
    # System settings
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--save_interval", type=int, default=None, help="Override system.save_interval")
    parser.add_argument("--log_interval", type=int, default=None, help="Override system.log_interval")
    parser.add_argument("--val_interval", type=int, default=None, help="Override system.val_interval")
    parser.add_argument("--num_workers", type=int, default=None, help="Override system.num_workers")
    parser.add_argument("--pin_memory", action="store_true", help="Override system.pin_memory=true")
    parser.add_argument("--no_pin_memory", action="store_true", help="Override system.pin_memory=false")
    
    # Logging settings
    parser.add_argument("--report_to", type=str, default=None, 
                       choices=["wandb", "tensorboard", "all", "none"],
                       help="Logging backends to use")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")
    
    # DeepSpeed settings
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.checkpoint_dir is not None:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.save_interval is not None:
        config.system.save_interval = args.save_interval
    if args.log_interval is not None:
        config.system.log_interval = args.log_interval
    if args.val_interval is not None:
        config.system.val_interval = args.val_interval
    if args.num_workers is not None:
        config.system.num_workers = args.num_workers
    if args.pin_memory and args.no_pin_memory:
        raise ValueError("--pin_memory and --no_pin_memory are mutually exclusive")
    if args.pin_memory:
        config.system.pin_memory = True
    if args.no_pin_memory:
        config.system.pin_memory = False
    if args.report_to is not None:
        config.logging.report_to = args.report_to
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.run_name is not None:
        config.logging.run_name = args.run_name
    # Decide backbone loading policy:
    # If resuming or finetuning from a pretrain checkpoint, skip loading WAN/VLM pretrained weights.
    try:
        if (getattr(config.resume, 'checkpoint_path', None) or
            (hasattr(config, 'finetune') and getattr(config.finetune, 'checkpoint_path', None))):
            config.model.load_pretrained_backbones = False
    except Exception:
        pass
    
    # Extract dataset name from config file path for checkpoint organization
    config_filename = os.path.basename(args.config)  # e.g., "ac_one.yaml"
    dataset_name = os.path.splitext(config_filename)[0]  # e.g., "ac_one"
    
    # Update checkpoint directory to include dataset name
    base_checkpoint_dir = config.system.checkpoint_dir
    config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, dataset_name)
    
    # Create the dataset directory if it doesn't exist
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    
    # Initialize Accelerator with DeepSpeed (if provided)
    accelerator_project_config = ProjectConfiguration(total_limit=20)
    accelerator_log_with = config.logging.get('report_to', 'tensorboard')
    if accelerator_log_with == "none":
        accelerator_log_with = None

    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=config.training.get('gradient_accumulation_steps', 1),
        mixed_precision="bf16",
        log_with=accelerator_log_with,
        project_dir=config.system.checkpoint_dir,
        project_config=accelerator_project_config,
    )
    
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level)
    
    # Handle report_to settings - expand "all" to individual backends
    report_to = config.logging.report_to
    if report_to == "all":
        report_to = ["wandb", "tensorboard"]
    elif report_to == "none":
        report_to = []
    elif isinstance(report_to, str):
        report_to = [report_to]
    
    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.logging.get('run_name', None)
    if not run_name:
        run_name = f"unidiffuser_{config.dataset.type}_bs{config.training.batch_size}_lr{config.training.learning_rate}"
    
    # Update checkpoint directory to include run name
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Checkpoints will be saved to: {config.system.checkpoint_dir}")
    
    # Initialize TensorBoard writer
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        if SummaryWriter is None:
            raise ModuleNotFoundError("tensorboard is required when logging.report_to includes tensorboard")
        tb_log_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        config_dict = OmegaConf.to_container(config, resolve=True)
        tb_writer.add_text('config', yaml.dump(config_dict))
    
    # Initialize WandB
    if rank == 0 and "wandb" in report_to:
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )
    
    try:
        # Create model first. Finetuning adapters are applied after checkpoint load.
        logger.info("Creating UniDiffuser model...")
        model = create_model(config)

        # Optional: load finetune weights before LoRA injection. Full RobotWin
        # checkpoints must not go through load_pretrain_weights because that
        # path intentionally skips action input/decoder layers.
        finetune_ckpt = getattr(config.finetune, 'checkpoint_path', None) if hasattr(config, 'finetune') else None
        if getattr(config, 'training_mode', 'finetune') == 'finetune' and finetune_ckpt:
            load_method = getattr(config.finetune, 'load_method', 'pretrain_partial')
            logger.info(f"Loading finetune weights from {finetune_ckpt} ({load_method})...")
            try:
                unwrapped = model.module if hasattr(model, 'module') else model
                if load_method == 'full':
                    unwrapped.load_checkpoint(finetune_ckpt, strict=False)
                elif load_method == 'pretrain_partial':
                    unwrapped.load_pretrain_weights(finetune_ckpt)
                else:
                    raise ValueError(f"Unknown finetune.load_method: {load_method}")
                logger.info("Finetune weights loaded.")
            except Exception as e:
                logger.error(f"Failed to load finetune weights: {e}")
                raise

        lora_stats = apply_lora_finetune_if_enabled(model, config)
        if lora_stats is not None:
            logger.info(f"LoRA finetuning stats: {lora_stats}")

        logger.info("Creating optimizer and scheduler...")
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(config, rank, world_size)
        
        # Create custom saving hook to avoid NCCL timeout issues
        def save_model_hook(models, weights, output_dir):
            """Custom save hook to save model safely and avoid NCCL timeouts."""
            if accelerator.is_main_process:
                save_lora_only = bool(getattr(config.system, "save_lora_only", False))
                logger.info(f"Saving model to {output_dir}")
                for i, model_to_save in enumerate(models):
                    # Unwrap the model if it's wrapped by DDP/DeepSpeed
                    unwrapped_model = accelerator.unwrap_model(model_to_save)

                    if save_lora_only:
                        model_save_path = os.path.join(output_dir, f"lora_model_{i}.pt")
                        torch.save(collect_lora_state_dict(unwrapped_model), model_save_path)
                    else:
                        # Save using torch.save instead of accelerator's default method
                        model_save_path = os.path.join(output_dir, f"pytorch_model_{i}.bin")
                        torch.save(unwrapped_model.state_dict(), model_save_path)
                    logger.info(f"Model {i} saved to {model_save_path}")
                weights.clear()
        
        # Register the custom save hook
        accelerator.register_save_state_pre_hook(save_model_hook)
        
        # Prepare everything with accelerator (do not prepare val_dataloader to enable rank0-only local eval)
        logger.info("Preparing model, optimizer, and dataloaders with Accelerator...")
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        # Create trainer
        trainer = UniDiffuserTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=accelerator.device,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval,
            save_interval=config.system.save_interval,
            val_interval=config.system.val_interval,
            report_to=report_to,
            tb_writer=tb_writer,
            accelerator=accelerator,
            config=config,
        )
        
        # Start training
        trainer.train(
            max_steps=config.training.max_steps, 
            resume_from=config.resume.checkpoint_path,
            val_interval=config.system.val_interval
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        print(f"[CRITICAL ERROR] Training failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise
    finally:
        # Clean up resources
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()

if __name__ == "__main__":
    main()
