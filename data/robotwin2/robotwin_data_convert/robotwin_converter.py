#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotWin Data Converter

This script converts original RobotWin datasets to our custom format.
Source: Original RobotWin dataset structure
Target: Motus compatible format with metas, videos, and qpos

"""

import os
import sys
import json
import shutil
import argparse
import logging
import multiprocessing as mp

# CRITICAL: Set multiprocessing start method BEFORE any CUDA-related imports
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Now safe to import torch and other CUDA-related modules
import torch
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import yaml
from concurrent.futures import ProcessPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class T5EmbeddingProcessor:
    """T5 embedding processor for generating embeddings in parallel"""
    
    def __init__(self, wan_repo_path: str, t5_max_length: int = 512, device: str = "cuda:0"):
        self.wan_repo_path = wan_repo_path
        self.t5_max_length = t5_max_length
        self.device = device
        self._encoder = None
    
    def _init_encoder(self):
        """Initialize T5 encoder (called in subprocess)"""
        if self._encoder is None:
            # Add WAN module path (relative to this file)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            wan_module_path = os.path.join(script_dir, '..', '..', '..', 'bak')
            wan_module_path = os.path.abspath(wan_module_path)
            
            if wan_module_path not in sys.path:
                sys.path.insert(0, wan_module_path)
            
            try:
                from wan.modules.t5 import T5EncoderModel
                
                # Set device
                torch.cuda.set_device(self.device)
                device_obj = torch.device(self.device)
                
                self._encoder = T5EncoderModel(
                    text_len=self.t5_max_length,
                    dtype=torch.bfloat16,
                    device=device_obj,
                    checkpoint_path=os.path.join(self.wan_repo_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
                    tokenizer_path=os.path.join(self.wan_repo_path, 'google/umt5-xxl'),
                )
                logger.info(f"T5 encoder initialized on {self.device}")
            except Exception as e:
                logger.error(f"Failed to initialize T5 encoder on {self.device}: {e}")
                raise
    
    def process_meta_file(self, meta_path: str, t5_output_path: str) -> bool:
        """Process a single meta file and generate T5 embeddings"""
        try:
            # Initialize encoder if needed
            self._init_encoder()
            
            # Skip if output already exists
            if os.path.exists(t5_output_path):
                return True
            
            # Read meta file
            with open(meta_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content:
                logger.warning(f"Empty meta file: {meta_path}")
                return False
            
            # Process content similar to the reference script
            if content.isspace():
                prompts = [content.rstrip()]
            else:
                lines = content.split('\n')
                prompts = [line for line in lines if line.strip() or line.isspace()]
            
            if not prompts:
                logger.warning(f"No valid prompts in {meta_path}")
                return False
            
            # Generate T5 embeddings
            device_obj = torch.device(self.device)
            encoded_texts = self._encoder(prompts, device_obj)
            
            # Process and save embeddings
            if isinstance(encoded_texts[0], torch.Tensor):
                encoded_list = [enc.cpu() for enc in encoded_texts]
            else:
                encoded_list = [torch.from_numpy(enc) for enc in encoded_texts]
            
            # Create output directory
            os.makedirs(os.path.dirname(t5_output_path), exist_ok=True)
            
            # Save embeddings
            torch.save(encoded_list, t5_output_path)
            logger.debug(f"Saved T5 embeddings to {t5_output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing meta file {meta_path}: {e}")
            return False


def process_t5_batch(args):
    """Function for processing T5 embeddings in parallel"""
    processor, meta_files = args

    # `processor.device` is interpreted in the current torch-visible device space.
    # Do not set CUDA_VISIBLE_DEVICES here: doing so after constructing a
    # processor for e.g. cuda:1 can remap the process to a single visible GPU and
    # turn cuda:1 into an invalid ordinal.
    results = []
    for meta_path, t5_path in meta_files:
        success = processor.process_meta_file(meta_path, t5_path)
        results.append((meta_path, success))
    return results


class RobotWinConverter:
    """
    Converter for RobotWin datasets to Motus format
    """
    
    def __init__(self, config_path: str):
        """
        Initialize converter with configuration file
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_config()
        
        # Meta prefix for instructions
        self.meta_prefix = (
            "The whole scene is in a realistic, industrial art style with three views: "
            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
            "The aloha robot is currently performing the following task: "
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
            raise
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = ['source_root', 'target_root']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Required config key missing: {key}")
        
        # Check if source exists
        if not os.path.exists(self.config['source_root']):
            raise FileNotFoundError(f"Source root not found: {self.config['source_root']}")
        
        # Create target root if not exists
        os.makedirs(self.config['target_root'], exist_ok=True)
        
        logger.info("Configuration validated successfully")
    
    def decode_compressed_image(self, compressed_data: bytes) -> Optional[np.ndarray]:
        """
        Decode compressed image data from HDF5
        
        Args:
            compressed_data: Compressed image bytes from HDF5
            
        Returns:
            BGR image array (OpenCV default format) for direct video writing
        """
        try:
            # Convert bytes to numpy array
            np_array = np.frombuffer(compressed_data, dtype=np.uint8)
            # Decode image using OpenCV (returns BGR) - keep as BGR for video writing
            image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            return image_bgr
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return None
    
    def extract_images_from_hdf5(self, hdf5_path: str) -> Dict[str, List[np.ndarray]]:
        """
        Extract images from all three camera views from HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            Dictionary mapping camera names to lists of decoded images
        """
        images_dict = {"head": [], "left_wrist": [], "right_wrist": []}
        
        # Map our target camera names to actual HDF5 paths
        camera_path_mapping = {
            'head': 'observation/head_camera/rgb',
            'left_wrist': 'observation/left_camera/rgb', 
            'right_wrist': 'observation/right_camera/rgb'
        }
        
        try:
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                for camera_name, hdf5_path_key in camera_path_mapping.items():
                    if hdf5_path_key in hdf5_file:
                        compressed_images = hdf5_file[hdf5_path_key][()]
                        
                        decoded_images = []
                        for compressed_image in compressed_images:
                            decoded_image = self.decode_compressed_image(compressed_image)
                            if decoded_image is not None:
                                decoded_images.append(decoded_image)
                            else:
                                logger.warning(f"Failed to decode image for camera {camera_name}")
                        
                        images_dict[camera_name] = decoded_images
                        logger.debug(f"Extracted {len(decoded_images)} frames from camera {camera_name}")
                    else:
                        logger.warning(f"Camera path not found in HDF5: {hdf5_path_key}")
                        
        except Exception as e:
            logger.error(f"Error processing HDF5 file {hdf5_path}: {e}")
            
        return images_dict
    
    def resize_and_concatenate_frames(
        self, 
        head_img: np.ndarray, 
        left_img: np.ndarray, 
        right_img: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Concatenate three camera views in T-shape layout:
        - Top: Head camera (keep original size)
        - Bottom left: Left wrist camera (resize to half size)
        - Bottom right: Right wrist camera (resize to half size)
        Dynamic output size based on input dimensions
        
        Args:
            head_img: Head camera image (keep original size)
            left_img: Left wrist camera image (resize to half size)  
            right_img: Right wrist camera image (resize to half size)
            
        Returns:
            Concatenated image with dynamic T-shape layout
        """
        try:
            # Get original dimensions
            orig_h, orig_w = head_img.shape[:2]
            
            # Resize wrist cameras to half size
            half_h, half_w = orig_h // 2, orig_w // 2
            left_resized = cv2.resize(left_img, (half_w, half_h))
            right_resized = cv2.resize(right_img, (half_w, half_h))
            
            # Concatenate left and right wrist cameras horizontally for bottom row
            bottom_row = np.hstack([left_resized, right_resized])
            
            # Create final T-shape layout:
            # Top row: head camera (orig_h x orig_w)
            # Bottom row: combined wrist cameras (half_h x orig_w)
            combined = np.vstack([head_img, bottom_row])
            
            return combined
        except Exception as e:
            logger.error(f"Error in frame concatenation: {e}")
            return None
    
    def create_concatenated_video(
        self, 
        images_dict: Dict[str, List[np.ndarray]], 
        output_path: str, 
        fps: int = 30
    ) -> bool:
        """
        Create a concatenated video from three camera views
        
        Args:
            images_dict: Dictionary with 'head', 'left_wrist', 'right_wrist' image lists
            output_path: Output video file path
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if all cameras have the same number of frames
            frame_counts = [len(images_dict[cam]) for cam in images_dict if images_dict[cam]]
            if not frame_counts or len(set(frame_counts)) > 1:
                logger.warning(f"Inconsistent frame counts: {frame_counts}")
                return False
            
            num_frames = min(frame_counts)
            if num_frames == 0:
                logger.warning("No frames to process")
                return False
            
            # Create video writer
            target_width = self.config.get('target_width', 320)
            target_height = self.config.get('target_height', 360)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            if not video_writer.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return False
            
            # Process each frame
            for i in range(num_frames):
                try:
                    head_frame = images_dict['head'][i] if i < len(images_dict['head']) else None
                    left_frame = images_dict['left_wrist'][i] if i < len(images_dict['left_wrist']) else None
                    right_frame = images_dict['right_wrist'][i] if i < len(images_dict['right_wrist']) else None
                    
                    if head_frame is None or left_frame is None or right_frame is None:
                        logger.warning(f"Missing frame data at index {i}")
                        continue
                    
                    combined_frame = self.resize_and_concatenate_frames(head_frame, left_frame, right_frame)
                    if combined_frame is not None:
                        # Ensure frame matches VideoWriter dimensions
                        if combined_frame.shape[:2] != (target_height, target_width):
                            combined_frame = cv2.resize(combined_frame, (target_width, target_height))
                        
                        # Convert BGR to RGB for proper video output
                        combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
                        
                        # Ensure correct data type and memory layout
                        if combined_frame_rgb.dtype != np.uint8:
                            combined_frame_rgb = combined_frame_rgb.astype(np.uint8)
                        
                        if not combined_frame_rgb.flags['C_CONTIGUOUS']:
                            combined_frame_rgb = np.ascontiguousarray(combined_frame_rgb)
                        
                        video_writer.write(combined_frame_rgb)
                    else:
                        logger.warning(f"Failed to create combined frame at index {i}")
                        
                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {e}")
            
            video_writer.release()
            logger.debug(f"Successfully created video: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating video {output_path}: {e}")
            return False
    
    def extract_joint_actions_from_hdf5(self, hdf5_path: str) -> Optional[torch.Tensor]:
        """
        Extract 14-dimensional joint actions from HDF5 file
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            Tensor with shape (num_timesteps, 14) or None if failed
        """
        try:
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                # Extract joint action vector (14-dim)
                if 'joint_action/vector' in hdf5_file:
                    qpos_data = hdf5_file['joint_action/vector'][()]
                    qpos_tensor = torch.from_numpy(qpos_data).float()
                    
                    if qpos_tensor.shape[1] == 14:
                        logger.debug(f"Extracted joint actions with shape: {qpos_tensor.shape}")
                        return qpos_tensor
                    else:
                        logger.warning(f"Unexpected joint action dimensions: {qpos_tensor.shape}")
                
                logger.warning(f"No valid joint_action/vector found in {hdf5_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting joint actions from {hdf5_path}: {e}")
            return None
    
    def process_instructions(self, instruction_path: str) -> List[str]:
        """
        Process instruction JSON file and extract captions
        
        Args:
            instruction_path: Path to instruction JSON file
            
        Returns:
            List of instruction strings
        """
        try:
            with open(instruction_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            instructions = []
            
            # Handle different instruction formats
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'caption' in item:
                        caption = item['caption']
                        if isinstance(caption, list):
                            instructions.extend(caption)
                        else:
                            instructions.append(str(caption))
                    elif isinstance(item, str):
                        instructions.append(item)
            elif isinstance(data, dict):
                if 'captions' in data:
                    captions = data['captions']
                    if isinstance(captions, list):
                        instructions.extend([str(cap) for cap in captions])
                elif 'caption' in data:
                    caption = data['caption']
                    if isinstance(caption, list):
                        instructions.extend([str(cap) for cap in caption])
                    else:
                        instructions.append(str(caption))
                elif 'seen' in data:
                    # Handle RobotWin2 format
                    instructions.extend([str(cap) for cap in data['seen']])
            
            return instructions
            
        except Exception as e:
            logger.error(f"Error processing instructions from {instruction_path}: {e}")
            return []
    
    def create_meta_file(self, meta_path: str, instructions: List[str]) -> bool:
        """
        Create meta file with prefixed instructions
        
        Args:
            meta_path: Output meta file path
            instructions: List of instruction strings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                if instructions:
                    for instruction in instructions:
                        f.write(f"{self.meta_prefix}{instruction}\n")
                else:
                    # Write empty prefix if no instructions
                    f.write(f"{self.meta_prefix}\n")
            return True
        except Exception as e:
            logger.error(f"Error creating meta file {meta_path}: {e}")
            return False
    
    def process_episode(
        self, 
        hdf5_path: str, 
        instruction_path: str, 
        output_dir: Path, 
        episode_id: int
    ) -> bool:
        """
        Process a single episode: extract video, qpos, and instructions
        
        Args:
            hdf5_path: Path to episode HDF5 file
            instruction_path: Path to episode instruction JSON file
            output_dir: Output directory for this episode
            episode_id: Episode ID number
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directories
            videos_dir = output_dir / "videos"
            qpos_dir = output_dir / "qpos"
            metas_dir = output_dir / "metas"
            
            for dir_path in [videos_dir, qpos_dir, metas_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            video_path = videos_dir / f"{episode_id}.mp4"
            qpos_path = qpos_dir / f"{episode_id}.pt"
            meta_path = metas_dir / f"{episode_id}.txt"

            if self.config.get('skip_existing', False) and video_path.exists() and qpos_path.exists() and meta_path.exists():
                logger.debug(f"Skipping existing episode {episode_id}")
                return True
            
            # Extract images and create video
            images_dict = self.extract_images_from_hdf5(hdf5_path)
            if not any(images_dict.values()):
                logger.error(f"No images extracted from {hdf5_path}")
                return False
            
            fps = self.config.get('fps', 30)
            if not self.create_concatenated_video(images_dict, str(video_path), fps):
                logger.error(f"Failed to create video for episode {episode_id}")
                return False
            
            # Extract and validate qpos
            qpos_data = self.extract_joint_actions_from_hdf5(hdf5_path)
            if qpos_data is not None:
                # Validate qpos trajectory - skip if any value exceeds threshold
                if not self.validate_qpos_trajectory(qpos_data):
                    logger.warning(f"Skipping episode {episode_id}: qpos validation failed")
                    return False
                
                torch.save(qpos_data, qpos_path)
                logger.debug(f"Saved qpos to {qpos_path}")
            else:
                logger.warning(f"No qpos data for episode {episode_id}")
                return False  # Skip episodes without qpos data
            
            # Process instructions and create meta file
            instructions = []
            if instruction_path and os.path.exists(instruction_path):
                instructions = self.process_instructions(instruction_path)
            
            if not self.create_meta_file(str(meta_path), instructions):
                logger.error(f"Failed to create meta file for episode {episode_id}")
                return False
            
            logger.info(f"Successfully processed episode {episode_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}")
            return False
    
    def collect_meta_files_for_t5(self) -> List[Tuple[str, str]]:
        """
        Collect all meta files and their corresponding T5 output paths
        
        Returns:
            List of (meta_path, t5_output_path) tuples
        """
        meta_files = []
        target_root = Path(self.config['target_root'])
        
        # Scan all processed directories
        for subset_dir in target_root.iterdir():
            if not subset_dir.is_dir():
                continue
                
            for task_dir in subset_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                metas_dir = task_dir / "metas"
                if not metas_dir.exists():
                    continue
                
                # Create umt5_wan directory
                umt5_dir = task_dir / "umt5_wan"
                umt5_dir.mkdir(exist_ok=True)
                
                # Collect meta files
                for meta_file in metas_dir.glob("*.txt"):
                    t5_file = umt5_dir / f"{meta_file.stem}.pt"
                    meta_files.append((str(meta_file), str(t5_file)))
        
        return meta_files
    
    def process_t5_embeddings_parallel(self):
        """
        Process T5 embeddings using multiple GPUs in parallel
        """
        if not self.config.get('enable_t5_embeddings', False):
            logger.info("T5 embeddings disabled, skipping...")
            return
            
        logger.info("Starting T5 embeddings generation...")
        
        # Get configuration
        wan_repo_path = self.config.get('wan_repo_path', '')
        t5_max_length = self.config.get('t5_max_length', 512)
        cuda_devices = self.config.get('cuda_devices', ['0'])
        
        if not wan_repo_path:
            logger.error("wan_repo_path not configured")
            return
            
        # Collect all meta files
        meta_files = self.collect_meta_files_for_t5()
        if not meta_files:
            logger.warning("No meta files found for T5 processing")
            return
            
        logger.info(f"Found {len(meta_files)} meta files to process")
        logger.info(f"Using {len(cuda_devices)} GPUs: {cuda_devices}")
        
        # Split work across GPUs
        num_devices = len(cuda_devices)
        chunks = [meta_files[i::num_devices] for i in range(num_devices)]
        
        # Create processors for each device
        processors_and_chunks = []
        for i, device_id in enumerate(cuda_devices):
            device = f"cuda:{device_id}"
            processor = T5EmbeddingProcessor(wan_repo_path, t5_max_length, device)
            processors_and_chunks.append((processor, chunks[i]))
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_devices) as executor:
            futures = [
                executor.submit(process_t5_batch, args) 
                for args in processors_and_chunks
            ]
            
            # Collect results
            all_results = []
            for future in tqdm(futures, desc="Processing T5 embeddings"):
                results = future.result()
                all_results.extend(results)
        
        # Report results
        successful = sum(1 for _, success in all_results if success)
        total = len(all_results)
        logger.info(f"T5 embeddings completed: {successful}/{total} successful")
        
        if successful < total:
            failed_files = [path for path, success in all_results if not success]
            logger.warning(f"Failed files: {failed_files[:10]}...")  # Show first 10
    
    def scan_dataset(self, source_root: str) -> Dict[str, Dict[str, List[Path]]]:
        """
        Scan the dataset structure and return organized paths
        
        Args:
            source_root: Root path of the source dataset
            
        Returns:
            Nested dictionary with structure: {subset: {task: [episode_paths]}}
        """
        dataset_structure = {}
        source_path = Path(source_root)
        
        # Look for subset directories (clean, randomized, etc.)
        for subset_path in source_path.iterdir():
            if subset_path.is_dir():
                subset_name = subset_path.name
                dataset_structure[subset_name] = {}
                
                # Look for task directories
                for task_path in subset_path.iterdir():
                    if task_path.is_dir():
                        task_name = task_path.name
                        
                        # Find episode files
                        hdf5_files = list(task_path.glob("*.hdf5"))
                        if not hdf5_files:
                            # Look in data subdirectory
                            data_dir = task_path / "data"
                            if data_dir.exists():
                                hdf5_files = list(data_dir.glob("*.hdf5"))
                        
                        if hdf5_files:
                            dataset_structure[subset_name][task_name] = sorted(hdf5_files)
        
        return dataset_structure
    
    def convert_dataset(self):
        """
        Main conversion function
        """
        logger.info("Starting RobotWin dataset conversion")
        
        # Scan source dataset
        dataset_structure = self.scan_dataset(self.config['source_root'])
        
        if not dataset_structure:
            logger.error("No valid dataset structure found")
            return
        
        logger.info(f"Found {len(dataset_structure)} subsets to process")
        
        # Process each subset
        for subset_name, tasks in dataset_structure.items():
            logger.info(f"Processing subset: {subset_name}")
            
            subset_output_dir = Path(self.config['target_root']) / subset_name
            subset_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each task
            for task_name, episode_files in tasks.items():
                logger.info(f"  Processing task: {task_name} ({len(episode_files)} episodes)")
                
                task_output_dir = subset_output_dir / task_name
                task_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each episode
                for episode_idx, hdf5_path in enumerate(tqdm(episode_files, desc=f"Processing {task_name}")):
                    # Find corresponding instruction file
                    instruction_path = None
                    instruction_base = hdf5_path.parent / "instructions" / f"{hdf5_path.stem}.json"
                    if instruction_base.exists():
                        instruction_path = str(instruction_base)
                    else:
                        # Try alternative paths
                        alt_paths = [
                            hdf5_path.with_suffix('.json'),
                            hdf5_path.parent.parent / "instructions" / f"{hdf5_path.stem}.json"
                        ]
                        for alt_path in alt_paths:
                            if alt_path.exists():
                                instruction_path = str(alt_path)
                                break
                    
                    # Process the episode
                    success = self.process_episode(
                        str(hdf5_path),
                        instruction_path,
                        task_output_dir,
                        episode_idx
                    )
                    
                    if not success:
                        logger.warning(f"Failed to process episode {episode_idx} in task {task_name}")
        
        logger.info("Dataset conversion completed")
        
        # Process T5 embeddings if enabled
        self.process_t5_embeddings_parallel()
    
    def _get_t5_paths(self) -> Dict[str, str]:
        """
        Generate T5 model paths from WAN repo path
        
        Returns:
            Dictionary with T5 model and weights paths
        """
        wan_path = self.config.get('wan_repo_path', '')
        if not wan_path:
            return {}
        
        return {
            't5_model_path': os.path.join(wan_path, 'google', 'umt5-xxl'),
            't5_weights_path': os.path.join(wan_path, 'models_t5_umt5-xxl-enc-bf16.pth')
        }
    
    def validate_qpos_trajectory(self, qpos_tensor: torch.Tensor, threshold: float = 4.0) -> bool:
        """
        Validate qpos trajectory by checking if any value exceeds absolute threshold
        
        Args:
            qpos_tensor: Tensor with shape (num_timesteps, 14)
            threshold: Absolute value threshold (default: 4.0)
            
        Returns:
            True if trajectory is valid (all values within threshold), False otherwise
        """
        try:
            if qpos_tensor is None:
                return False
            
            # Check if any absolute value exceeds threshold
            max_abs_value = torch.max(torch.abs(qpos_tensor)).item()
            
            if max_abs_value > threshold:
                logger.warning(f"Trajectory rejected: max absolute qpos value {max_abs_value:.3f} > {threshold}")
                return False
            
            logger.debug(f"Trajectory valid: max absolute qpos value {max_abs_value:.3f} <= {threshold}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating qpos trajectory: {e}")
            return False
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert RobotWin dataset to Motus format"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        converter = RobotWinConverter(args.config)
        converter.convert_dataset()
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
