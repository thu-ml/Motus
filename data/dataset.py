# Dataset Factory
# Simple factory to create different types of datasets

from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
import torch


def create_dataset(config: OmegaConf, val: bool = False):
    """
    Create dataset based on config.
    
    Args:
        config: Configuration object
        val: Whether to create validation dataset
        
    Returns:
        Dataset instance
    """
    dataset_type = config.dataset.get('type', 'robotwin')  # Default to robotwin
    
    if dataset_type == 'robotwin':
        from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'randomized_limit_per_task'):
            params['randomized_limit_per_task'] = config.dataset.randomized_limit_per_task
        extended_chunkwise = getattr(config.model, 'extended_chunkwise_finetune', {})
        if bool(extended_chunkwise.get('enabled', False)):
            params['extended_action_multiplier'] = int(extended_chunkwise.get('multiplier', 3))
            params['include_rolling_condition'] = bool(
                extended_chunkwise.get('rolling_action_distill_enabled', False)
                or extended_chunkwise.get('rolled_state_fm_enabled', False)
            )
            params['include_rolling_target_actions'] = bool(
                extended_chunkwise.get('rolled_state_fm_enabled', False)
            )
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return RobotWinTaskDataset(**params)
    
    elif dataset_type == 'ac_one':
        from .ac_one.ac_one_dataset import ACOneDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return ACOneDataset(**params)

    elif dataset_type == 'latent_action':
        from .latent_action.latent_action_dataset import LatentActionDataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        if hasattr(config.dataset, 'dataset_dir'):
            dataset_dir = list(config.dataset.dataset_dir)
            params['dataset_dir'] = [str(p) for p in dataset_dir]
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Optional VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Optional additional params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return LatentActionDataset(**params)

    elif dataset_type == 'aloha_agilex_2':
        from .aloha_agilex_2.aloha_agilex2_dataset import AlohaAgilex2Dataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return AlohaAgilex2Dataset(**params)

    elif dataset_type == 'lerobot':
        from .lerobot.lerobot_dataset import LeRobotMotusDataset

        # Get all parameters from config
        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag
        params['val'] = val
        
        return LeRobotMotusDataset(**params)

    # Example: Add more dataset types here
    # elif dataset_type == 'bridge':
    #     from .bridge_dataset import BridgeDataset  
    #     return BridgeDataset(**params)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: robotwin, aloha_agilex_1, ac_one, aloha_agilex_2, table30")


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    
    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask
            
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    padded_embeddings = []
    
    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def _process_rolling_vlm_inputs_batch(rolling_vlm_inputs: List[List[Dict[str, Any]]]) -> List[Dict[str, torch.Tensor]]:
    steps = len(rolling_vlm_inputs[0])
    batched_steps = []
    for step_idx in range(steps):
        step_inputs = [sample_inputs[step_idx] for sample_inputs in rolling_vlm_inputs]
        batched_steps.append(_process_vlm_inputs_batch(step_inputs))
    return batched_steps


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.
    
    Args:
        batch: List of sample dictionaries (may contain None)
        
    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors（支持无 initial_state 的样本）
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_frames = torch.stack([sample['video_frames'] for sample in batch])
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch])
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None
    
    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)
    
    # Process language embeddings with padding in collate_fn  
    language_embeddings = [sample.get('language_embedding') for sample in batch if 'language_embedding' in sample]
    processed_language_embeddings = None
    if language_embeddings and any(emb is not None for emb in language_embeddings):
        processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)
    
    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W]
        'action_sequence': action_sequences,     # [B, F, D]
        'vlm_inputs': processed_vlm_inputs,
        'language_embedding': processed_language_embeddings,
    }

    if initial_states is not None:
        result['initial_state'] = initial_states

    if all(('extended_action_sequence' in sample and sample['extended_action_sequence'] is not None) for sample in batch):
        result['extended_action_sequence'] = torch.stack([sample['extended_action_sequence'] for sample in batch])
        result['extended_action_indices'] = torch.stack([sample['extended_action_indices'] for sample in batch])

    if all(('rolling_condition_frames' in sample and sample['rolling_condition_frames'] is not None) for sample in batch):
        result['rolling_condition_frames'] = torch.stack([sample['rolling_condition_frames'] for sample in batch])
        result['rolling_initial_states'] = torch.stack([sample['rolling_initial_states'] for sample in batch])
        result['rolling_condition_indices'] = torch.stack([sample['rolling_condition_indices'] for sample in batch])

    if all(('rolling_target_action_chunks' in sample and sample['rolling_target_action_chunks'] is not None) for sample in batch):
        result['rolling_target_action_chunks'] = torch.stack([sample['rolling_target_action_chunks'] for sample in batch])
        result['rolling_target_action_indices'] = torch.stack([sample['rolling_target_action_indices'] for sample in batch])

    rolling_vlm_inputs = [sample.get('rolling_vlm_inputs') for sample in batch]
    if rolling_vlm_inputs and all(vlm_steps is not None for vlm_steps in rolling_vlm_inputs):
        result['rolling_vlm_inputs'] = _process_rolling_vlm_inputs_batch(rolling_vlm_inputs)

    if all(('condition_frame_idx' in sample and sample['condition_frame_idx'] is not None) for sample in batch):
        result['condition_frame_idx'] = torch.stack([sample['condition_frame_idx'] for sample in batch])

    if all(('video_indices' in sample and sample['video_indices'] is not None) for sample in batch):
        result['video_indices'] = torch.stack([sample['video_indices'] for sample in batch])

    if all(('action_indices' in sample and sample['action_indices'] is not None) for sample in batch):
        result['action_indices'] = torch.stack([sample['action_indices'] for sample in batch])
    
    return result
