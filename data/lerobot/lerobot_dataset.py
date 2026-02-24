"""
LeRobot Dataset Loader for Motus
--------------------------------
This file provides a thin wrapper around `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`(or `lerobot.datasets.lerobot_dataset.LeRobotDataset`)
to match Motus' unified dataset interface (aligned with `Motus/data/dataset.py::collate_fn`).
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

import numpy as np
import torch
import torch.utils.data as data
import warnings

try:
    from transformers import AutoProcessor  # type: ignore
except Exception:  # pragma: no cover
    AutoProcessor = None

from utils.vlm_utils import preprocess_vlm_messages

from data.utils.image_utils import resize_with_padding, tensor_to_pil
from data.utils.norm import normalize_actions, load_normalization_stats

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)


class LeRobotMotusDataset(data.Dataset):
    """
    Motus-compatible dataset wrapper for LeRobotDataset.

    Alignment requirements:
    - Must return: first_frame / video_frames / action_sequence / initial_state / language_embedding / vlm_inputs
    - Uses Motus' `stat.json` for normalization (to stay consistent with AlohaAgilex2Dataset)

    Data structure:
    /home/.cache/huggingface/lerobot/
    ├── repo_id/
    │   ├── meta/
    │   │   ├── info.json
    │   │   ├── episodes.jsonl
    │   │   ├── episodes_stats.jsonl
    |   |   └── tasks.jsonl
    │   ├── data/
    │   |   ├── chunk-000/
    │   |   │   ├── episode_000000.parquet
    │   |   │   ├── episode_000001.parquet
    │   |   │   └── ...
    │   |   ├── chunk-001/
    │   |   │   ├── episode_000000.parquet
    │   |   │   ├── episode_000001.parquet
    │   |   │   └── ...
    │   |   └── ...
    |   ├── t5_embedding/
    |   |   ├── episode_000000.pt
    |   |   ├── episode_000001.pt
    |   |   └── ...
    |   └── videos/
    |   |   ├── chunk-000/
    |   |   │   ├── observation.images.cam_concatenated/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   ├── observation.images.cam_high/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   ├── observation.images.cam_left_wrist/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   ├── observation.images.cam_right_wrist/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   ├── chunk-001/
    |   |   │   ├── observation.images.cam_concatenated/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   ├── observation.images.cam_high/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   ├── observation.images.cam_left_wrist/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   ├── observation.images.cam_right_wrist/
    |   |   │   │   ├── episode_000000.mp4
    |   |   │   │   ├── episode_000001.mp4
    |   |   │   │   └── ...
    |   |   │   └── ...
    |   |   └── ...
    """
    
    def __init__(
        self,
        # Compatibility with `create_dataset(config)`: it passes `dataset_dir`.
        # Here we interpret it as a local LeRobot dataset root (contains meta/data/videos).
        # If `root` is also provided, `root` takes precedence.
        dataset_dir: Optional[str] = None,
        # supports repo_id（HF Hub repo id）
        repo_id: Optional[str] = None,
        # Local dataset root (contains meta/data/videos). If None, LeRobot uses its default cache dir.
        root: Optional[str] = None,
        # Optional split (currently only used for selecting a subset of episodes)
        split: Optional[str] = None,
        
        # Sampling parameters
        global_downsample_rate: int = 1,  # Global downsampling (e.g., 30Hz -> 10Hz)
        video_action_freq_ratio: int = 5,  # Video:Action frequency ratio  
        num_video_frames: int = 8,  # Number of video frames to predict
        video_size: Tuple[int, int] = (736, 640),  # (height, width)
        
        # Episode limits
        max_episodes: int = 10000,
        
        # Data augmentation
        image_aug: bool = False,
        
        # VLM processing
        vlm_checkpoint_path: Optional[str] = None,

        # --- Optional: on-the-fly T5 embedding fallback ---
        # If the dataset does not contain `language_embedding` AND meta/episodes.jsonl has no
        # `t5_embedding_path`, we will encode T5 on-the-fly, cache it under dataset_root, and
        # write back `t5_embedding_path` into episodes.jsonl.
        enable_t5_fallback: bool = False,
        t5_wan_path: Optional[str] = None,
        t5_text_len: int = 512,
        t5_folder_name: str = "t5_embedding",
        t5_device: Optional[str] = None,

        # Video backend: "pyav" (memory efficient) or "torchcodec" (faster but more memory)
        video_backend: Optional[str] = None,

        embodiment_type: str = "aloha_agilex_2", # for loading normalization statistics
        task_mode: str = "single", # "single" or "multi"
        task_name: str = "null",
        **kwargs
    ):
        super().__init__()

        # ---- Resolve repo_id/root for LeRobotDataset ----
        # Compatibility: if only `dataset_dir` is provided, treat it as `root`,
        # and use the directory name as a repo_id identifier.
        if root is None and dataset_dir is not None and os.path.exists(str(dataset_dir)):
            root = str(dataset_dir)
            if repo_id is None:
                repo_id = Path(root).name

        if repo_id is None:
            raise ValueError("repo_id is required (or provide an existing dataset_dir to infer it).")

        # Notes:
        # - repo_id: HF dataset id or a local identifier (prefer 'org/name'-like strings)
        # - root: local dataset root (contains meta/data/videos). If None, LeRobot uses default cache.
        resolved_root: Optional[str] = root
        resolved_repo_id: str = str(repo_id)

        self.repo_id = resolved_repo_id
        self.root = resolved_root

        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.video_size = video_size
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio

        self.max_episodes = max_episodes
        self.image_aug = image_aug # No extra augmentation on LeRobot side for now
        self.task_mode = task_mode
        self.task_name = task_name
        
        # ---- T5 fallback config (lazy init) ----
        self.enable_t5_fallback = bool(enable_t5_fallback)
        self.t5_wan_path = t5_wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
        self.t5_text_len = int(t5_text_len)
        self.t5_folder_name = str(t5_folder_name)
        self.t5_device = t5_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._t5_encoder = None  # lazy-loaded
        
        # VLM processor
        self.vlm_processor = None
        if vlm_checkpoint_path:
            if AutoProcessor is None:
                logger.warning(
                    "transformers is not installed, cannot load VLM processor; will skip VLM processing."
                )
            else:
                try:
                    self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                    logger.info(f"Loaded VLM processor from {vlm_checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to load VLM processor: {e}")

        # ---- Select episode subset (optional) ----
        # Read-only metadata to get total_episodes, avoiding parquet reads while conversion is ongoing.
        if self.task_mode == "single":
            meta = LeRobotDatasetMetadata(self.repo_id, root=self.root)
            total_eps = int(meta.total_episodes)
            
            all_ep_ids = list(range(total_eps))
            rng = random.Random(0)
            rng.shuffle(all_ep_ids)

            if self.max_episodes is not None and self.max_episodes > 0:
                all_ep_ids = all_ep_ids[: min(self.max_episodes, len(all_ep_ids))]

            self.episode_ids = all_ep_ids
        elif self.task_mode == "multi":
            if self.task_name is None:
                self.repo_ids = [task_name for task_name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, task_name))]
            elif isinstance(self.task_name, list):
                self.repo_ids = self.task_name
                for task_name in self.repo_ids:
                    if not os.path.isdir(os.path.join(self.root, task_name)):
                        raise ValueError(f"Task {task_name} not found in {self.root}")
            elif isinstance(self.task_name, str):
                if not os.path.isdir(os.path.join(self.root, self.task_name)):
                    raise ValueError(f"Task {self.task_name} not found in {self.root}")
                self.repo_ids = [self.task_name]
            else:
                raise ValueError(f"Invalid task name: {self.task_name}")
            metas = [LeRobotDatasetMetadata(task_name, root=os.path.join(self.root, task_name)) for task_name in self.repo_ids]
            self.episode_ids = {task_name: list(range(int(meta.total_episodes))) for task_name, meta in zip(self.repo_ids, metas)}

        
        
        # Video backend: use pyav by default (more memory efficient than torchcodec)
        # torchcodec may cause std::bad_alloc errors due to higher memory usage
        resolved_video_backend = video_backend if video_backend is not None else "pyav"
        logger.info(f"Using video backend: {resolved_video_backend} (pyav is more memory efficient)")
        if self.task_mode == "single":
            self.lerobot_dataset = LeRobotDataset(
                repo_id=self.repo_id, 
                root=self.root, 
                episodes=self.episode_ids,
                video_backend=resolved_video_backend
            )
        elif self.task_mode == "multi":
            self.lerobot_dataset = MultiLeRobotDataset(
                repo_ids=self.repo_ids, 
                root=self.root,
                episodes=self.episode_ids,
                video_backend=resolved_video_backend
            )
            self.episode_id_to_task_idx = []
            self.episode_num_accumulated = []
            self.frame_num_accumulated = []
            tmp_episode_cnt = 0
            tmp_frame_cnt = 0
            for idx, task_name in enumerate(self.repo_ids):
                self.episode_id_to_task_idx.extend([idx] * len(self.episode_ids[task_name]))
                tmp_episode_cnt += len(self.episode_ids[task_name])
                self.episode_num_accumulated.append(tmp_episode_cnt)
                
                tmp_frame_cnt += int(self.lerobot_dataset._datasets[idx].num_frames)
                self.frame_num_accumulated.append(tmp_frame_cnt)

        # Episode-level embedding cache (for external t5 embedding files referenced from meta/episodes.jsonl)
        # key: global episode_index (int) ; value: torch.Tensor
        self._episode_embedding_cache: Dict[int, torch.Tensor] = {}
        
        # Pre-compute image feature detection 
        # Priority:
        # 1) If `observation.images.cam_concatenated` exists, use it directly.
        # 2) Else if cam_high + cam_left_wrist + cam_right_wrist exist, stitch them back into a concatenated view
        # 3) Else fall back to other common single-view keys (e.g., "image").
        if self.task_mode == "single":
            features = self.lerobot_dataset.features
        else:
            features = self.lerobot_dataset._datasets[0].features
        self.has_concat = "observation.images.cam_concatenated" in features
        self.has_three_cam = all(
            k in features
            for k in [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        )
        
        # Fallback single-view candidates
        self.single_view_candidates = ["observation.images.main", "observation.image", "image"]
        if not self.has_concat and not self.has_three_cam:
            found_any = any(k in features for k in self.single_view_candidates)
            if not found_any:
                # Last resort: any visual feature (video/image)
                # For MultiLeRobotDataset, features.items() returns datasets.Image/VideoFrame objects (Sequence), not dicts
                # For LeRobotDataset, features is a dict from meta.features
                from lerobot.datasets.video_utils import VideoFrame
                import datasets
                
                any_visual = []
                for k, ft in features.items():
                    # Check if it's a visual feature
                    if isinstance(ft, (datasets.Image, VideoFrame)):
                        # datasets.Image or VideoFrame (for MultiLeRobotDataset)
                        any_visual.append(k)
                    elif isinstance(ft, dict) and ft.get("dtype") in ["video", "image"]:
                        # dict from meta.features (for LeRobotDataset)
                        any_visual.append(k)
                
                if not any_visual:
                    raise ValueError("No image features found in dataset")
                # Use the first visual key deterministically
                self.single_view_candidates = [sorted(any_visual)[0]]
        
        # Load normalization statistics
        current_dir = Path(__file__).parent.parent  # Go up to data directory
        stat_path = current_dir / "utils" / "stat.json"
        self.action_min, self.action_max = load_normalization_stats(str(stat_path), embodiment_type)

        logger.info(f"LeRobot dataset initialized: repo_id={self.repo_id}, root={self.root}")
        logger.info(f"Embodiment type: {embodiment_type} (for normalization statistics)")
        logger.info(f"Image source: {'concatenated' if self.has_concat else ('three_cam' if self.has_three_cam else 'single_view')}")
        if self.task_mode == "single":
            logger.info(f"Selected episodes: {len(self.episode_ids)}/{total_eps}")
        elif self.task_mode == "multi":
            total_selected = sum(len(ep_ids) for ep_ids in self.episode_ids.values())
            logger.info(f"Selected episodes: {total_selected} (across {len(self.repo_ids)} repos)")
        logger.info(f"Video size: {self.video_size}, Frames: {self.num_video_frames}")

    def _episodes_jsonl_path(self) -> Path:
        if self.lerobot_dataset is None:
            raise RuntimeError("LeRobotDataset not initialized")
        return Path(self.lerobot_dataset.root) / "meta" / "episodes.jsonl"

    def _t5_cache_file_path(self, episode_index: int) -> Path:
        """Absolute path: {dataset_root}/{t5_folder_name}/episode_XXXXXX.pt"""
        return Path(self.lerobot_dataset.root) / self.t5_folder_name / f"episode_{episode_index:06d}.pt"

    def _t5_lock_file_path(self, episode_index: int) -> Path:
        return Path(self.lerobot_dataset.root) / self.t5_folder_name / f"episode_{episode_index:06d}.pt.lock"

    def _atomic_update_episodes_jsonl(self, episode_index: int, updates: Dict[str, Any]) -> None:
        """
        Update a given episode entry in meta/episodes.jsonl (jsonlines) in-place.
        We write a temp file and then replace to reduce the chance of corrupting the file.
        """
        path = self._episodes_jsonl_path()
        tmp = path.with_suffix(path.suffix + ".tmp")

        found = False
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if int(obj.get("episode_index", -1)) == int(episode_index):
                    obj.update(updates)
                    found = True
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if not found:
            # If episodes.jsonl doesn't contain that episode yet (can happen during conversion),
            # don't write back to avoid corrupting the file. In this case we still rely on the
            # cached pt file existence as the cache hit signal.
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return

        tmp.replace(path)

    def _ensure_t5_encoder(self):
        if self._t5_encoder is not None:
            return self._t5_encoder

        # Check if we're in a DataLoader worker process (multiprocessing context)
        # In worker processes, initializing T5 encoder can cause memory issues
        # Instead, we should pre-generate T5 embeddings using add_t5_cache_to_lerobot_dataset.py
        import multiprocessing
        current_process = multiprocessing.current_process()
        if current_process.name != "MainProcess":
            raise RuntimeError(
                f"T5 encoder initialization in DataLoader worker process ({current_process.name}) is disabled "
                "to avoid memory issues. Please pre-generate T5 embeddings using:\n"
                "  python -m Motus.data.lerobot.add_t5_cache_to_lerobot_dataset \\\n"
                f"    --dataset_root {self.lerobot_dataset.root} \\\n"
                f"    --t5_wan_path {self.t5_wan_path} \\\n"
                f"    --t5_text_len {self.t5_text_len}"
            )

        # Lazy import WAN T5 encoder to avoid heavy dependencies on all runs
        try:
            # Prefer project-local implementation
            from bak.wan.modules.t5 import T5EncoderModel  # type: ignore
        except Exception:
            # Fallback: add bak path similarly to inference scripts
            import sys
            bak_root = str((Path(__file__).resolve().parents[2] / "bak").resolve())
            if bak_root not in sys.path:
                sys.path.insert(0, bak_root)
            from wan.modules.t5 import T5EncoderModel  # type: ignore

        if not self.t5_wan_path:
            raise ValueError(
                "enable_t5_fallback=True but t5_wan_path is not provided and WAN_PATH/WAN_ROOT is not set."
            )

        ckpt = os.path.join(self.t5_wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
        tok = os.path.join(self.t5_wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
        dtype = torch.bfloat16 if self.t5_device.startswith("cuda") else torch.float32

        logger.info(f"Initializing WAN T5EncoderModel (device={self.t5_device}, text_len={self.t5_text_len})")
        self._t5_encoder = T5EncoderModel(
            text_len=self.t5_text_len,
            dtype=dtype,
            device=self.t5_device,
            checkpoint_path=ckpt,
            tokenizer_path=tok,
        )
        return self._t5_encoder

    def _encode_and_cache_t5_embedding(self, episode_index: int, instruction: str) -> torch.Tensor:
        """
        Encode on-the-fly and cache to disk, returning a tensor (expected shape [S,D] or [V,S,D]).
        - If cache exists, load from disk
        - Use a lock file to avoid duplicate encoding in multi-worker scenarios
        """
        out_pt = self._t5_cache_file_path(episode_index)
        out_pt.parent.mkdir(parents=True, exist_ok=True)

        # Fast path: cache exists
        if out_pt.exists():
            emb = torch.load(out_pt, map_location="cpu")
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            return emb

        # Simple file lock (avoid duplicate work across workers)
        lock_path = self._t5_lock_file_path(episode_index)
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                # Someone else is generating it; wait. If file appears, use it.
                if out_pt.exists():
                    emb = torch.load(out_pt, map_location="cpu")
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    return emb
                if time.time() - start > 600:
                    raise TimeoutError(f"Timeout waiting for T5 embedding lock: {lock_path}")
                time.sleep(0.2)

        try:
            # Double-check after acquiring lock
            if out_pt.exists():
                emb = torch.load(out_pt, map_location="cpu")
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                return emb

            encoder = self._ensure_t5_encoder()
            with torch.no_grad():
                t5_out = encoder([instruction], self.t5_device)

            # Normalize output format:
            # - list[tensor] -> take first
            # - tensor [1,S,D] -> squeeze batch dim
            if isinstance(t5_out, list):
                emb = t5_out[0]
            elif isinstance(t5_out, torch.Tensor):
                emb = t5_out
            else:
                raise ValueError(f"Unexpected T5 encoder output type: {type(t5_out)}")

            if isinstance(emb, torch.Tensor) and emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)

            # Save CPU tensor
            torch.save(emb.detach().cpu(), out_pt)

            # Write back meta/episodes.jsonl (relative path)
            rel = f"{self.t5_folder_name}/episode_{episode_index:06d}.pt"
            try:
                self._atomic_update_episodes_jsonl(episode_index, {"t5_embedding_path": rel})
                # Sync in-memory meta for this process
                if episode_index in self.lerobot_dataset.meta.episodes:
                    self.lerobot_dataset.meta.episodes[episode_index]["t5_embedding_path"] = rel
            except Exception as e:
                logger.warning(f"Failed to update episodes.jsonl for episode {episode_index}: {e}")

            return emb
        finally:
            try:
                lock_path.unlink()
            except Exception:
                pass
    
    def __len__(self):
        """Return number of episodes."""
        return self.lerobot_dataset.num_episodes * 1000
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx: Sample index (not used, random sampling)
            
        Returns:
            Dictionary containing training data
        """
        if not self.lerobot_dataset:
            return None

        episode_idx = random.randint(0, self.lerobot_dataset.num_episodes - 1)
        if self.task_mode == "multi":
            task_idx = self.episode_id_to_task_idx[episode_idx]
            if task_idx > 0:
                episode_idx = episode_idx - self.episode_num_accumulated[task_idx - 1]
            from_idx_t = self.lerobot_dataset._datasets[task_idx].episode_data_index["from"][episode_idx]
            to_idx_t = self.lerobot_dataset._datasets[task_idx].episode_data_index["to"][episode_idx]
        else:
            from_idx_t = self.lerobot_dataset.episode_data_index["from"][episode_idx]
            to_idx_t = self.lerobot_dataset.episode_data_index["to"][episode_idx]
        
        from_idx = int(from_idx_t.item()) if hasattr(from_idx_t, "item") else int(from_idx_t)
        to_idx = int(to_idx_t.item()) if hasattr(to_idx_t, "item") else int(to_idx_t)
        total_frames = int(to_idx - from_idx)

        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

        if self.task_mode == "multi" and task_idx > 0:
            from_idx += self.frame_num_accumulated[task_idx - 1]

        global_cond_idx = int(from_idx + condition_frame_idx) 
        global_video_indices = [int(from_idx + i) for i in video_indices]
        global_action_indices = [int(from_idx + i) for i in action_indices]

        def _to_chw_float(item_data: dict, key: str) -> torch.Tensor:
            """Load a frame from item_data and normalize it to float tensor [C,H,W]."""
            img = item_data[key].float()
            # LeRobot video/image tensors are typically [C,H,W] already, but keep robust
            if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            return img

        def load_concatenated_view(item_data: dict) -> torch.Tensor:
            """
            Build the model input image:
            - If cam_concatenated exists, use it.
            - Else stitch cam_high + left/right wrist into a concatenated view (inverse of split logic).
            """
            if self.has_concat:
                img = _to_chw_float(item_data, "observation.images.cam_concatenated")
                return self._resize_frame_chw(img, self.video_size)

            if self.has_three_cam:
                cam_high = _to_chw_float(item_data, "observation.images.cam_high")
                cam_left = _to_chw_float(item_data, "observation.images.cam_left_wrist")
                cam_right = _to_chw_float(item_data, "observation.images.cam_right_wrist")

                # Inverse of:
                #   split_h = (H//3)*2 ; split_w = W//2
                #   high = frame[:split_h, :] ; left = frame[split_h:, :split_w] ; right = frame[split_h:, split_w:]
                # Here we reconstruct a frame with:
                #   top: cam_high, bottom-left: cam_left, bottom-right: cam_right.
                # We assume all cameras have the same resolution.
                c = cam_high.shape[0]
                top_h = int(cam_high.shape[1])
                target_w = int(cam_high.shape[2])

                # Define bottom region height from wrist cams (use max for robustness)
                bottom_h = int(max(cam_left.shape[1], cam_right.shape[1]))
                split_w = target_w // 2
                right_w = target_w - split_w

                cam_high_r = self._resize_frame_chw(cam_high, (top_h, target_w))
                cam_left_r = self._resize_frame_chw(cam_left, (bottom_h, split_w))
                cam_right_r = self._resize_frame_chw(cam_right, (bottom_h, right_w))

                out = torch.zeros((c, top_h + bottom_h, target_w), dtype=cam_high_r.dtype)
                out[:, :top_h, :target_w] = cam_high_r
                out[:, top_h:, :split_w] = cam_left_r
                out[:, top_h:, split_w:] = cam_right_r

                return self._resize_frame_chw(out, self.video_size)

            # Fall back to a single-view key
            for k in self.single_view_candidates:
                if k in item_data:
                    img = _to_chw_float(item_data, k)
                    return self._resize_frame_chw(img, self.video_size)
            # If we reached here, item_data doesn't contain expected keys
            raise ValueError("No usable image keys found in item_data")

        # ---- Resolve per-task dataset + local indices (multi) ----
        if self.task_mode == "multi":
            base_offset = int(self.frame_num_accumulated[task_idx - 1]) if task_idx > 0 else 0
            ds_media = self.lerobot_dataset._datasets[task_idx]
            local_cond_idx = int(global_cond_idx - base_offset)
            local_video_indices = [int(g - base_offset) for g in global_video_indices]
            local_action_indices = [int(g - base_offset) for g in global_action_indices]
            hf_dataset = ds_media.hf_dataset
        else:
            ds_media = self.lerobot_dataset
            local_cond_idx = int(global_cond_idx)
            local_video_indices = list(global_video_indices)
            local_action_indices = list(global_action_indices)
            hf_dataset = ds_media.hf_dataset

        # ---- Read conditioning row from parquet only (NO video decoding) ----
        item_cond = hf_dataset[local_cond_idx]

        # ---- Decode visuals in ONE shot per video stream (cond + targets) ----
        # This avoids reopening/seeking the mp4 for each frame.
        all_media_indices = [local_cond_idx] + local_video_indices
        ts_vals = hf_dataset[all_media_indices]["timestamp"]
        if isinstance(ts_vals, torch.Tensor):
            timestamps = ts_vals.flatten().tolist()
        elif isinstance(ts_vals, (list, tuple)) and len(ts_vals) > 0 and isinstance(ts_vals[0], torch.Tensor):
            timestamps = torch.stack(ts_vals).flatten().tolist()
        else:
            # last resort
            timestamps = [float(x) for x in list(ts_vals)]

        # Use the true episode_index from parquet to build video file paths.
        # This matters when LeRobotDataset is instantiated with a shuffled/subset `episodes` list (single mode).
        ep_idx_raw = item_cond.get("episode_index", None)
        if ep_idx_raw is None:
            raise KeyError("episode_index not found in hf_dataset row; cannot resolve video file path")
        ep_for_video = int(ep_idx_raw.item()) if hasattr(ep_idx_raw, "item") else int(ep_idx_raw)

        def _decode_key(vid_key: str) -> torch.Tensor:
            video_path = Path(ds_media.root) / ds_media.meta.get_video_file_path(ep_for_video, vid_key)
            frames = decode_video_frames(video_path, timestamps, ds_media.tolerance_s, ds_media.video_backend).squeeze(0)
            return frames  # [T,C,H,W]

        if self.has_concat:
            frames = _decode_key("observation.images.cam_concatenated")
            first_frame = self._resize_frame_chw(frames[0].float(), self.video_size)
            video_frames_sampled = torch.stack(
                [self._resize_frame_chw(frames[i].float(), self.video_size) for i in range(1, frames.shape[0])],
                dim=0,
            )
        elif self.has_three_cam:
            frames_high = _decode_key("observation.images.cam_high")
            frames_left = _decode_key("observation.images.cam_left_wrist")
            frames_right = _decode_key("observation.images.cam_right_wrist")
            stitched = []
            for i in range(frames_high.shape[0]):
                stitched.append(
                    load_concatenated_view(
                        {
                            "observation.images.cam_high": frames_high[i],
                            "observation.images.cam_left_wrist": frames_left[i],
                            "observation.images.cam_right_wrist": frames_right[i],
                        }
                    )
                )
            first_frame = stitched[0]
            video_frames_sampled = torch.stack(stitched[1:], dim=0)
        else:
            # Fall back to a single-view key
            vid_key = None
            for k in self.single_view_candidates:
                if k in ds_media.meta.video_keys or k in hf_dataset.column_names:
                    vid_key = k
                    break
            if vid_key is None:
                vid_key = self.single_view_candidates[0]
            frames = _decode_key(vid_key)
            first_frame = self._resize_frame_chw(frames[0].float(), self.video_size)
            video_frames_sampled = torch.stack(
                [self._resize_frame_chw(frames[i].float(), self.video_size) for i in range(1, frames.shape[0])],
                dim=0,
            )

        # Compatibility: some datasets don't have an explicit state, so use actions as state (e.g., qpos)
        if "observation.state" in item_cond:
            initial_state = torch.as_tensor(item_cond["observation.state"]).float()
        elif "actions" in item_cond:
            initial_state = torch.as_tensor(item_cond["actions"]).float()
        elif "action" in item_cond:
            initial_state = torch.as_tensor(item_cond["action"]).float()
        else:
            raise KeyError("No state found in item (expected observation.state/actions/action)")

        action_key = "action" if "action" in hf_dataset.column_names else None
        if action_key is None and "actions" in hf_dataset.column_names:
            action_key = "actions"
        if action_key is None:
            raise KeyError("No action column found in hf_dataset (expected 'action' or 'actions')")

        # Batch read from parquet; this should not decode video.
        # Using __getitem__ with a list avoids building an intermediate Dataset via .select().
        action_values = hf_dataset[local_action_indices][action_key]
        if isinstance(action_values, torch.Tensor):
            action_sequence = action_values.float()
        elif isinstance(action_values, (list, tuple)) and len(action_values) > 0 and isinstance(action_values[0], torch.Tensor):
            action_sequence = torch.stack([v.float() for v in action_values], dim=0)
        elif isinstance(action_values, (list, tuple)) and len(action_values) > 0 and isinstance(action_values[0], np.ndarray):
            action_sequence = torch.from_numpy(np.stack(action_values, axis=0)).float()
        else:
            action_sequence = torch.tensor(action_values, dtype=torch.float32)
        
        # Language embedding:
        # 1) Prefer parquet (legacy: each frame has `language_embedding`)
        # 2) Otherwise, try meta/episodes.jsonl field `t5_embedding_path` and load external pt by episode_index
        all_embeddings = item_cond.get("language_embedding", None)
        if all_embeddings is None:
            all_embeddings = item_cond.get("observation.feature.language_embedding", None)

        if all_embeddings is None:
            # External episode-level embedding
            ep_index_raw = item_cond.get("episode_index", None)
            if ep_index_raw is None:
                raise KeyError("episode_index not found in item; cannot load external embedding")
            ep_index = int(ep_index_raw.item()) if hasattr(ep_index_raw, "item") else int(ep_index_raw)

            cached = self._episode_embedding_cache.get(ep_index, None)
            if cached is None:
                if self.task_mode == 'single':
                    ep_meta = self.lerobot_dataset.meta.episodes.get(ep_index, None)
                else:
                    ep_meta = self.lerobot_dataset._datasets[task_idx].meta.episodes.get(ep_index, None)
                if ep_meta is None:
                    raise KeyError(f"episode {ep_index} not found in meta.episodes")

                rel_path = ep_meta.get("t5_embedding_path", None)
                if rel_path is None:
                    if not self.enable_t5_fallback:
                        raise KeyError(
                            "language_embedding not found in item and t5_embedding_path not found in meta/episodes.jsonl; "
                            "you can set enable_t5_fallback=True to encode and cache T5 embeddings on-the-fly."
                        )

                    # On-the-fly encoding (use language_instruction, fallback to task)
                    instr = item_cond.get("language_instruction", None)
                    if instr is None or (isinstance(instr, str) and len(instr.strip()) == 0):
                        instr = item_cond.get("task", "")
                    if not isinstance(instr, str):
                        instr = str(instr)
                    emb = self._encode_and_cache_t5_embedding(ep_index, instr)
                    self._episode_embedding_cache[ep_index] = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                    cached = self._episode_embedding_cache[ep_index]
                    all_embeddings = cached
                    # Skip the load-from-disk branch below
                    rel_path = None

                if rel_path is not None:
                    # dataset root is self.lerobot_dataset.root (Path)
                    if self.task_mode == "single":
                        abs_path = Path(self.lerobot_dataset.root) / str(rel_path)
                    else:
                        abs_path = Path(self.lerobot_dataset._datasets[task_idx].root) / str(rel_path)
                    emb = torch.load(abs_path, map_location="cpu")
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb)
                    # normalize shape to [V,S,D]
                    if emb.ndim == 2:
                        emb = emb.unsqueeze(0)
                    self._episode_embedding_cache[ep_index] = emb
                    cached = emb

            all_embeddings = cached

        if not isinstance(all_embeddings, torch.Tensor):
            all_embeddings = torch.tensor(all_embeddings)
        if all_embeddings.ndim == 2:
            all_embeddings = all_embeddings.unsqueeze(0)
        language_embedding = all_embeddings[0].float()

        vlm_tokens = None
        if self.vlm_processor:
            # Prefer dataset-stored text; fallback to `task`
            text_instr = item_cond.get("language_instruction", None)
            if text_instr is None or (isinstance(text_instr, str) and len(text_instr.strip()) == 0):
                text_instr = item_cond.get("task", "")
            first_frame_pil = tensor_to_pil(first_frame)
            vlm_tokens = preprocess_vlm_messages(text_instr, first_frame_pil, self.vlm_processor)

        normalized_actions = normalize_actions(action_sequence, self.action_min, self.action_max)
        normalized_initial_state = normalize_actions(initial_state.unsqueeze(0), self.action_min, self.action_max).squeeze(0)

        return {
            'first_frame': first_frame,
            'video_frames': video_frames_sampled,
            'initial_state': normalized_initial_state,
            'action_sequence': normalized_actions,
            'language_embedding': language_embedding,
            'vlm_inputs': vlm_tokens,
        }

    def _resize_frame_chw(self, frame_chw: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Resize and pad a [C,H,W] torch float frame to target_size=(H,W), keeping [0,1]."""
        if frame_chw.dim() != 3:
            raise ValueError(f"Expected frame [C,H,W], got {tuple(frame_chw.shape)}")
        c, h, w = frame_chw.shape
        th, tw = target_size
        if (h, w) == (th, tw):
            return frame_chw
        frame_hwc = frame_chw.permute(1, 2, 0).cpu().numpy()  # float32 [H,W,C] in [0,1]
        frame_uint8 = np.clip(frame_hwc * 255.0, 0, 255).astype(np.uint8)
        resized = resize_with_padding(frame_uint8, target_size)  # uint8 [th,tw,3]
        out = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return out
    
    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """
        Calculate sampling indices for video and actions (following robotwin's logic).
        
        Args:
            total_frames: Total number of frames in the episode
            
        Returns:
            - condition_frame_idx: Index of condition frame (corresponds to initial state)
            - video_indices: List of video frame indices to predict
            - action_indices: List of action frame indices to predict
        """
        # Calculate physical span of one chunk
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        
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
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])
        
        return condition_frame_idx, video_indices, action_indices
        