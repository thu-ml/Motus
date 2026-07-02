# Robotwin2 Dataset Loader for Motus
# Supports Robotwin2 data with video and action data from multiple tasks

import os
import random
import h5py
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
from PIL import Image
import tempfile

# VLM processing imports
from utils.vlm_utils import preprocess_vlm_messages
from transformers import AutoProcessor
try:
    from transformers import Qwen2VLProcessor
except ImportError:
    Qwen2VLProcessor = None

# Import image processing utilities
from data.utils.image_utils import (
    tensor_to_pil, apply_image_augmentation,
    load_video_frames, get_video_frame_count
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)


def _load_vlm_processor(vlm_checkpoint_path: str):
    processor = AutoProcessor.from_pretrained(vlm_checkpoint_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        return processor

    if Qwen2VLProcessor is not None:
        logger.warning(
            "AutoProcessor returned %s without image_processor; falling back to Qwen2VLProcessor",
            processor.__class__.__name__,
        )
        return Qwen2VLProcessor.from_pretrained(vlm_checkpoint_path, trust_remote_code=True)

    return processor

class RobotWinTaskDataset(data.Dataset):
    """
    Dataset for RobotWin data with task-level organization and flexible sampling.
    
    Data structure:
    /share/dataset/preprocess/robotwin2/
    ├── clean/
    │   ├── adjust_bottle/
    │   │   ├── qpos/           # Robot position files (.pt)
    │   │   ├── videos/         # MP4 video files  
    │   │   └── umt5_wan/       # Pre-encoded language embeddings (.pt)
    │   ├── beat_block_hammer/
    │   └── ...
    └── randomized/
        ├── adjust_bottle/
        └── ...
    """
    
    def __init__(
        self,
        dataset_dir: str = "/share/dataset/preprocess/robotwin2/",
        data_mode: str = "clean",  # "clean", "randomized", or "both"
        task_mode: str = "multi",  # "single" or "multi" 
        task_name: Optional[str] = None,  # Required for single task mode
        randomized_limit_per_task: Optional[int] = None,  # Limit randomized episodes per task (take first N)
        
        # Sampling parameters
        global_downsample_rate: int = 3,  # Global downsampling (e.g., 30Hz -> 10Hz)
        video_action_freq_ratio: int = 5,  # Video:Action frequency ratio  
        num_video_frames: int = 3,  # Number of video frames to predict
        
        # Standard parameters
        video_size: Tuple[int, int] = (320, 384),
        max_episodes: Optional[int] = None,
        upsample_rate: int = 1,  # For compatibility with H_RDT
        val: bool = False,
        image_aug: bool = False,
        
        # VLM processing parameters
        vlm_checkpoint_path: Optional[str] = None,  # Path to VLM model
        extended_action_multiplier: int = 1,
        include_rolling_condition: bool = False,
    ):
        """
        Initialize RobotWin dataset with flexible sampling.
        
        Args:
            dataset_dir: Root directory containing clean/ and randomized/ folders
            data_mode: Which data split to use ("clean", "randomized", or "both")
            task_mode: Single task or multi-task ("single" or "multi")
            task_name: Task name for single task mode (e.g., "adjust_bottle")
            
            global_downsample_rate: Global downsampling rate (e.g., 3 for 30Hz->10Hz)
            video_action_freq_ratio: Frequency ratio between video and action
            num_video_frames: Number of video frames to predict
            
            video_size: Target video resolution (H, W)
            max_episodes: Maximum number of episodes to load (for debugging)
            upsample_rate: Temporal data upsampling rate (for H_RDT compatibility)
            val: Whether this is validation set
            image_aug: Whether to apply image augmentation
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_mode = data_mode
        self.task_mode = task_mode
        self.task_name = task_name
        self.randomized_limit_per_task = randomized_limit_per_task
        
        # Sampling parameters
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        
        # Calculate action sequence length
        self.action_chunk_size = num_video_frames * video_action_freq_ratio
        
        # Standard parameters
        self.video_size = video_size
        self.max_episodes = max_episodes
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_aug = image_aug
        self.extended_action_multiplier = max(1, int(extended_action_multiplier or 1))
        self.extended_action_chunk_size = self.action_chunk_size * self.extended_action_multiplier
        self.include_rolling_condition = bool(include_rolling_condition)
        
        # Validate parameters
        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name parameter")
        
        assert data_mode in ["clean", "randomized", "both"], \
            f"data_mode must be 'clean', 'randomized', or 'both', got {data_mode}"
        
        # Initialize data structures
        if task_mode == "single":
            self.episode_files = []  # List of episode files for single task
        else:
            self.task_to_episodes = {}  # Task name -> episode files mapping
            self.task_weights = {}      # Task sampling weights
        
        self.total_episodes = 0
        
        logger.info(f"RobotWin dataset initialized:")
        logger.info(f"  Data mode: {data_mode}")
        logger.info(f"  Task mode: {task_mode}")
        if task_name:
            logger.info(f"  Task name: {task_name}")
        logger.info(f"  Global downsample rate: {global_downsample_rate}")
        logger.info(f"  Video:Action frequency ratio: {video_action_freq_ratio}")
        logger.info(f"  Action chunk size: {self.action_chunk_size}")
        logger.info(f"  Video frames to predict: {num_video_frames}")
        logger.info(f"  Extended action multiplier: {self.extended_action_multiplier}")
        logger.info(f"  Extended action chunk size: {self.extended_action_chunk_size}")
        logger.info(f"  Rolling condition samples: {self.include_rolling_condition}")
        logger.info(f"  Total episodes: {self.total_episodes}")
        
        # Initialize VLM processor for complete VLM processing in dataset
        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            try:
                self.vlm_processor = _load_vlm_processor(vlm_checkpoint_path)
                logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor from {vlm_checkpoint_path}: {e}")
                logger.warning("VLM processing will be disabled for this dataset instance")
        else:
            logger.info("VLM checkpoint path not provided, VLM processing disabled")

        # Load dataset episodes
        self._load_episodes()
    
    def _limit_episodes_first_n(self, episodes: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """
        Limit to the first N episodes based on sorted episode_name.
        Numeric names are sorted numerically; otherwise lexicographically.
        """
        if n is None or n <= 0 or not episodes:
            return episodes
        def _sort_key(ep: Dict[str, Any]):
            name = ep.get('episode_name', '')
            try:
                return (0, int(name))
            except Exception:
                return (1, str(name))
        episodes_sorted = sorted(episodes, key=_sort_key)
        return episodes_sorted[:n]
    
    def _scan_task_folder(self, task_path: Path) -> List[str]:
        """
        Scan a single task folder.
        
        Args:
            task_path: Path to task folder (e.g., .../clean/adjust_bottle)
            
        Returns:
            List of valid episode identifiers
        """
        qpos_dir = task_path / "qpos"
        videos_dir = task_path / "videos"
        umt5_dir = task_path / "umt5_wan"
        
        # Check if all required directories exist
        if not all([qpos_dir.exists(), videos_dir.exists(), umt5_dir.exists()]):
            logger.warning(f"Missing data directories in {task_path}")
            return []
        
        # Find valid episodes (those that have all three data types)
        valid_episodes = []
        
        # Get all qpos files as base (.pt format)
        qpos_files = list(qpos_dir.glob("*.pt"))
        
        for qpos_file in qpos_files:
            episode_name = qpos_file.stem
            
            # Check if corresponding video and language files exist
            video_file = videos_dir / f"{episode_name}.mp4"
            lang_file = umt5_dir / f"{episode_name}.pt"
            
            if video_file.exists() and lang_file.exists():
                # Store full paths
                episode_data = {
                    'episode_name': episode_name,
                    'task_name': task_path.name,
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                }
                valid_episodes.append(episode_data)
        
        logger.info(f"Task {task_path.name} ({task_path.parent.name}): Found {len(valid_episodes)} valid episodes")
        return valid_episodes
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Initialize dataset by scanning folders."""
        logger.info("Initializing dataset...")
        
        # Determine which data splits to scan
        if self.data_mode == "both":
            data_splits = ["clean", "randomized"]
        else:
            data_splits = [self.data_mode]
        
        all_episodes = []
        
        for split in data_splits:
            split_dir = self.dataset_dir / split
            
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            if self.task_mode == "single":
                # Single task mode: scan specific task
                task_dir = split_dir / self.task_name
                if task_dir.exists():
                    episodes = self._scan_task_folder(task_dir)
                    # If scanning randomized split, optionally limit to first N episodes per task
                    if split == "randomized" and self.randomized_limit_per_task is not None:
                        before = len(episodes)
                        episodes = self._limit_episodes_first_n(episodes, self.randomized_limit_per_task)
                        logger.info(f"Applied randomized per-task limit ({self.randomized_limit_per_task}) for {task_dir.name}: {before} -> {len(episodes)}")
                    all_episodes.extend(episodes)
                else:
                    logger.warning(f"Task directory not found: {task_dir}")
            
            else:
                # Multi task mode: scan all tasks
                task_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                
                for task_dir in task_dirs:
                    episodes = self._scan_task_folder(task_dir)
                    
                    # If scanning randomized split, optionally limit to first N episodes per task
                    if split == "randomized" and self.randomized_limit_per_task is not None:
                        before = len(episodes)
                        episodes = self._limit_episodes_first_n(episodes, self.randomized_limit_per_task)
                        logger.info(f"Applied randomized per-task limit ({self.randomized_limit_per_task}) for {task_dir.name}: {before} -> {len(episodes)}")
                    
                    # Group by task for multi-task sampling
                    task_name = task_dir.name
                    if task_name not in self.task_to_episodes:
                        self.task_to_episodes[task_name] = []
                    self.task_to_episodes[task_name].extend(episodes)
        
        if self.task_mode == "single":
            self.episode_files = all_episodes
            
            # Limit episodes if requested
            if self.max_episodes is not None:
                self.episode_files = self.episode_files[:self.max_episodes]
            
            self.total_episodes = len(self.episode_files)
            
            if self.total_episodes == 0:
                raise ValueError(f"No valid episodes found for task {self.task_name}")
        
        else:
            # Multi-task mode: calculate sampling weights
            if not self.task_to_episodes:
                raise ValueError("No valid episodes found for any task")
            
            # Equal weight for all tasks
            num_tasks = len(self.task_to_episodes)
            for task_name in self.task_to_episodes.keys():
                self.task_weights[task_name] = 1.0 / num_tasks
            
            self.total_episodes = sum(len(episodes) for episodes in self.task_to_episodes.values())
            
            logger.info(f"Multi-task dataset with {num_tasks} tasks:")
            for task_name, episodes in self.task_to_episodes.items():
                logger.info(f"  {task_name}: {len(episodes)} episodes")
        
        # Return all episodes for consistency with other datasets
        return all_episodes

    def _load_robot_data(self, qpos_path: str, action_indices: List[int], initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load robot position data.
        
        Args:
            qpos_path: Path to qpos .pt file
            action_indices: List of action frame indices
            initial_state_idx: Index for initial state (should match condition frame)
            
        Returns:
            - initial_state: Robot state at condition frame [state_dim]
            - action_sequence: Actions at specified indices [len(action_indices), action_dim]
        """
        qpos_data = torch.load(qpos_path, map_location='cpu')  # [T, feature_dim]
        
        # Get initial state at the condition frame index
        if initial_state_idx >= len(qpos_data):
            initial_state_idx = len(qpos_data) - 1
        initial_state = qpos_data[initial_state_idx].float()
        
        # Get action sequence at specified indices
        actions = []
        for idx in action_indices:
            if idx >= len(qpos_data):
                raise IndexError(f"Action index {idx} out of bounds for qpos data length {len(qpos_data)}")
            else:
                action = qpos_data[idx]
            actions.append(action)
        
        action_sequence = torch.stack(actions).float()
        
        # Normalize actions and initial state
        # action_sequence = self._normalize_actions(action_sequence)
        # initial_state = self._normalize_actions(initial_state.unsqueeze(0)).squeeze(0)  # Normalize state same way as actions
        
        return initial_state, action_sequence

    def _load_robot_states(self, qpos_path: str, state_indices: List[int]) -> torch.Tensor:
        qpos_data = torch.load(qpos_path, map_location='cpu')
        states = []
        for idx in state_indices:
            safe_idx = min(max(int(idx), 0), len(qpos_data) - 1)
            states.append(qpos_data[safe_idx].float())
        return torch.stack(states).float()

    def _load_language_embedding(self, lang_path: str) -> tuple[torch.Tensor, int]:
        """Load pre-encoded language embedding and return the selected index."""
        try:
            embedding_data = torch.load(lang_path, map_location='cpu')
            
            # RobotWin embedding is always a list of tensors
            selected_idx = random.randint(0, len(embedding_data) - 1)
            embeddings = embedding_data[selected_idx]  # [seq_len, 4096]
            
            # Remove batch dimension if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)
            
            return embeddings, selected_idx
            
        except Exception as e:
            logger.error(f"Error loading language embedding from {lang_path}: {e}")
            raise
    
    def _load_text_instruction(self, task_name: str, episode_name: str, instruction_idx: int = None, split: Optional[str] = None) -> str:
        """Load text instruction for VLM processing.

        Args:
            task_name: Task name (e.g., "adjust_bottle")
            episode_name: Episode identifier (e.g., "429")
            instruction_idx: Optional index to select a deterministic instruction
            split: Dataset split to read from ("clean" or "randomized"). If None,
                   falls back to self.data_mode when it is not "both".
        """
        try:
            # Try to read from meta file first
            # Path structure: dataset_dir/split/task_name/metas/episode_name.txt
            if split is None:
                if self.data_mode in ["clean", "randomized"]:
                    split_to_use = self.data_mode
                else:
                    # Fallback to clean when split cannot be inferred
                    split_to_use = "clean"
            else:
                split_to_use = split

            meta_file = self.dataset_dir / split_to_use / task_name / "metas" / f"{episode_name}.txt"
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    # Filter out empty lines
                    instructions = [line.strip() for line in lines if line.strip()]
                
                if instructions:
                    if instruction_idx is not None and 0 <= instruction_idx < len(instructions):
                        # Use specific index to match language_embedding
                        return instructions[instruction_idx]
                    else:
                        # Random selection (fallback for old behavior)
                        import random
                        return random.choice(instructions)
                else:
                    # No fallback - raise error if no instructions found
                    raise ValueError(f"No instructions found in meta file for {task_name}/{episode_name}")
            else:
                raise FileNotFoundError(f"Meta file not found: {meta_file}")
                
        except Exception as e:
            logger.error(f"Failed to load text instruction for {task_name}/{episode_name}: {e}")
            raise
    
    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """
        Calculate sampling indices.
        
        Args:
            total_frames: Total number of frames in the episode
            
        Returns:
            - condition_frame_idx: Index of condition frame (corresponds to initial state)
            - video_indices: List of video frame indices to predict
            - action_indices: List of action frame indices to predict
        """
        # Ensure the sampled condition frame can provide the full extended action horizon.
        physical_chunk_size = self.extended_action_chunk_size * self.global_downsample_rate
        
        # Sample condition frame directly in physical space
        # Ensure the last action doesn't exceed total_frames - 1
        max_condition_idx = total_frames - physical_chunk_size - 1
        
        if max_condition_idx < 0:
            condition_frame_idx = 0
        else:
            condition_frame_idx = random.randint(0, max_condition_idx)
        
        # Action indices: from condition_frame_idx+1 onwards, with downsampling
        action_indices = []
        for i in range(self.action_chunk_size):
            # Each action is separated by global_downsample_rate frames
            action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))
        
        # Video indices: sample at frequency ratio intervals from action indices
        # Example: ratio=5, frames=[5, 10, 15, 20] for 4 video frames
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])
        
        # Verify spacing
        if len(action_indices) > 1:
            intervals = [action_indices[i+1] - action_indices[i] for i in range(len(action_indices)-1)]
            # print(f"  Action interval verification: {set(intervals)} (should all be {self.global_downsample_rate})")
        
        extended_action_indices = action_indices
        if self.extended_action_multiplier > 1:
            extended_action_indices = []
            for i in range(self.extended_action_chunk_size):
                action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
                extended_action_indices.append(min(action_idx, total_frames - 1))
            action_indices = extended_action_indices[:self.action_chunk_size]

        return condition_frame_idx, video_indices, action_indices, extended_action_indices
    
    def __len__(self) -> int:
        """Return approximate dataset length."""
        return self.total_episodes * 10  # Assume 100 samples per episode
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a training sample.
        
        Args:
            idx: Sample index (not used, random sampling)
            
        Returns:
            Dictionary containing training data
        """
        # Robust sampling with retries to avoid returning None (which breaks DDP sync)
        max_attempts = 8
        for _ in range(max_attempts):
            # Select episode
            if self.task_mode == "single":
                if not self.episode_files:
                    continue
                episode_data = random.choice(self.episode_files)
            else:
                if not self.task_to_episodes:
                    continue
                task_name = random.choices(
                    list(self.task_weights.keys()),
                    weights=list(self.task_weights.values()),
                    k=1
                )[0]
                task_episodes = self.task_to_episodes.get(task_name, [])
                if not task_episodes:
                    continue
                episode_data = random.choice(task_episodes)

            try:
                # Get video frame count (decord-based; robust to ffmpeg timeouts)
                total_frames = get_video_frame_count(episode_data['video_path'])
                if total_frames < 2:
                    continue

                # Calculate sampling indices
                condition_frame_idx, video_indices, action_indices, extended_action_indices = self._calculate_sampling_indices(total_frames)

                # Load frames and aligned robot/action data
                first_frame = load_video_frames(episode_data['video_path'], [condition_frame_idx], self.video_size)
                video_frames = load_video_frames(episode_data['video_path'], video_indices, self.video_size)
                initial_state, action_sequence = self._load_robot_data(episode_data['qpos_path'], action_indices, condition_frame_idx)
                extended_action_sequence = None
                if self.extended_action_multiplier > 1:
                    _, extended_action_sequence = self._load_robot_data(
                        episode_data['qpos_path'],
                        extended_action_indices,
                        condition_frame_idx,
                    )
                language_embedding, instruction_idx = self._load_language_embedding(episode_data['lang_path'])

                # Infer split from paths
                inferred_split = None
                try:
                    for key in ('qpos_path', 'video_path', 'lang_path'):
                        parts = Path(episode_data[key]).parts
                        if 'clean' in parts:
                            inferred_split = 'clean'
                            break
                        if 'randomized' in parts:
                            inferred_split = 'randomized'
                            break
                except Exception:
                    inferred_split = None

                # Load raw text instruction for VLM processing using the same index
                text_instruction = self._load_text_instruction(
                    episode_data['task_name'],
                    episode_data['episode_name'],
                    instruction_idx,
                    split=inferred_split,
                )

                # Complete VLM processing in dataset
                vlm_inputs = None
                if self.vlm_processor is not None:
                    first_frame_pil = tensor_to_pil(first_frame.squeeze(0))
                    vlm_inputs = preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)

                rolling_condition_frames = None
                rolling_initial_states = None
                rolling_condition_indices = None
                rolling_vlm_inputs = None
                if self.include_rolling_condition and self.extended_action_multiplier > 1:
                    rolling_stride = self.action_chunk_size * self.global_downsample_rate
                    rolling_condition_indices_list = [
                        min(condition_frame_idx + step_idx * rolling_stride, total_frames - 1)
                        for step_idx in range(self.extended_action_multiplier)
                    ]
                    rolling_condition_frames = load_video_frames(
                        episode_data['video_path'],
                        rolling_condition_indices_list,
                        self.video_size,
                    )
                    rolling_initial_states = self._load_robot_states(
                        episode_data['qpos_path'],
                        rolling_condition_indices_list,
                    )
                    rolling_condition_indices = torch.tensor(
                        rolling_condition_indices_list,
                        dtype=torch.long,
                    )
                    if self.vlm_processor is not None:
                        rolling_vlm_inputs = [
                            preprocess_vlm_messages(
                                text_instruction,
                                tensor_to_pil(rolling_condition_frames[step_idx]),
                                self.vlm_processor,
                            )
                            for step_idx in range(rolling_condition_frames.shape[0])
                        ]

                sample = {
                    'first_frame': first_frame.squeeze(0),
                    'video_frames': video_frames,
                    'initial_state': initial_state,
                    'action_sequence': action_sequence,
                    'language_embedding': language_embedding,
                    'vlm_inputs': vlm_inputs,
                    'condition_frame_idx': torch.tensor(condition_frame_idx, dtype=torch.long),
                    'video_indices': torch.tensor(video_indices, dtype=torch.long),
                    'action_indices': torch.tensor(action_indices, dtype=torch.long),
                }
                if extended_action_sequence is not None:
                    sample.update(
                        {
                            'extended_action_sequence': extended_action_sequence,
                            'extended_action_indices': torch.tensor(extended_action_indices, dtype=torch.long),
                        }
                    )
                if rolling_condition_frames is not None:
                    sample.update(
                        {
                            'rolling_condition_frames': rolling_condition_frames,
                            'rolling_initial_states': rolling_initial_states,
                            'rolling_condition_indices': rolling_condition_indices,
                            'rolling_vlm_inputs': rolling_vlm_inputs,
                        }
                    )

                return sample

            except Exception as e:
                logger.warning(f"Retry due to sample error ({episode_data.get('episode_name','?')}): {e}")
                continue

        # If all attempts failed, let caller drop this sample (rare)
        return None
