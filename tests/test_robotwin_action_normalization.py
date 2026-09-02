"""Regression tests for the RobotWin state/action-space contract.

The policy module bundles simulator and model dependencies that are not needed
for these numerical checks.  Loading the class definition in isolation keeps
the tests CPU-only while still exercising the implementation in the source
file directly.
"""

import ast
import logging
import os
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_class(source_path: Path, class_name: str, namespace: Dict[str, Any]):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[class_name]


def _robotwin_dataset_class():
    source_path = REPO_ROOT / "data" / "robotwin2" / "robotwin_agilex_dataset.py"
    from data.utils.norm import normalize_actions

    return _load_class(
        source_path,
        "RobotWinTaskDataset",
        {
            "__name__": "robotwin_dataset_test",
            "Any": Any,
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "Path": Path,
            "Tuple": Tuple,
            "data": types.SimpleNamespace(Dataset=object),
            "np": np,
            "normalize_actions": normalize_actions,
            "torch": torch,
        },
    )


def _policy_class():
    source_path = REPO_ROOT / "inference" / "robotwin" / "Motus" / "deploy_policy.py"
    return _load_class(
        source_path,
        "MotusPolicy",
        {
            "__name__": "motus_policy_test",
            "Any": Any,
            "AutoProcessor": object,
            "Dict": Dict,
            "Image": types.SimpleNamespace(Image=object),
            "List": List,
            "Motus": object,
            "MotusConfig": object,
            "Optional": Optional,
            "Path": Path,
            "T5EncoderModel": object,
            "cv2": types.SimpleNamespace(),
            "deque": deque,
            "logging": logging,
            "logger": logging.getLogger("motus-policy-test"),
            "nn": torch.nn,
            "np": np,
            "os": os,
            "resize_with_padding": None,
            "sys": sys,
            "torch": torch,
            "yaml": types.SimpleNamespace(),
        },
    )


def test_robotwin_raw_mode_preserves_legacy_qpos(tmp_path):
    dataset_cls = _robotwin_dataset_class()
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.action_normalization = "none"

    qpos = torch.tensor(
        [[-1.0, 0.5, 2.0], [0.0, 1.5, 3.0], [1.0, 2.5, 4.0]],
        dtype=torch.float32,
    )
    qpos_path = tmp_path / "episode.pt"
    torch.save(qpos, qpos_path)

    initial_state, actions = dataset._load_robot_data(str(qpos_path), [1, 2], 0)

    assert torch.equal(initial_state, qpos[0])
    assert torch.equal(actions, qpos[[1, 2]])


def test_robotwin_min_max_mode_normalizes_state_and_actions(tmp_path):
    dataset_cls = _robotwin_dataset_class()
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.action_normalization = "min_max"
    dataset.action_min = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    dataset.action_max = np.array([1.0, 2.0, 5.0], dtype=np.float32)

    qpos = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 1.0, 3.0], [1.0, 2.0, 5.0]],
        dtype=torch.float32,
    )
    qpos_path = tmp_path / "episode.pt"
    torch.save(qpos, qpos_path)

    initial_state, actions = dataset._load_robot_data(str(qpos_path), [1, 2], 0)

    expected = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(initial_state, expected[0])
    assert torch.allclose(actions, expected[[1, 2]])


def test_policy_min_max_state_and_action_round_trip():
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.action_normalization = "min_max"
    policy.action_min = torch.tensor([-2.0, 0.0, 10.0])
    policy.action_max = torch.tensor([2.0, 4.0, 20.0])
    policy.action_range = policy.action_max - policy.action_min
    policy.current_state = torch.tensor([[0.0, 2.0, 15.0]])

    model_state = policy._model_state()
    assert torch.allclose(model_state, torch.tensor([[0.5, 0.5, 0.5]]))

    model_actions = torch.tensor([[0.25, 0.75, 0.1]])
    expected_environment_actions = torch.tensor([[-1.0, 3.0, 11.0]])
    assert torch.allclose(
        policy._environment_actions(model_actions), expected_environment_actions
    )


def test_policy_get_action_uses_model_space_and_returns_qpos():
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.device = "cpu"
    policy.action_normalization = "min_max"
    policy.action_min = torch.tensor([-2.0, 0.0, 10.0])
    policy.action_max = torch.tensor([2.0, 4.0, 20.0])
    policy.action_range = policy.action_max - policy.action_min
    policy.current_state = torch.tensor([[0.0, 2.0, 15.0]])
    policy.current_state_model = None
    policy.current_instruction = "test"
    policy.obs_cache = deque([torch.zeros(1, 3, 2, 2)])
    policy.action_cache = deque()
    policy.prev_action = None
    policy.save_images = False
    policy.config_dict = {"model": {"inference": {"num_inference_timesteps": 2}}}
    policy.t5_encoder = lambda *_args: torch.zeros(1, 1, 1)
    policy._tensor_to_pil_image = lambda _frame: object()
    policy._preprocess_vlm_messages = lambda *_args: {}

    seen_states = []

    class RecordingModel:
        def inference_step(self, **kwargs):
            seen_states.append(kwargs["state"].detach().clone())
            return None, torch.tensor([[[0.25, 0.75, 0.1]]])

    policy.model = RecordingModel()
    actions = policy.get_action()

    assert len(seen_states) == 1
    assert torch.allclose(seen_states[0], torch.tensor([[0.5, 0.5, 0.5]]))
    assert np.allclose(actions, np.array([[-1.0, 3.0, 11.0]]))


def test_policy_raw_mode_does_not_apply_stats():
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.action_normalization = "none"
    policy.action_min = None
    policy.action_max = None
    policy.action_range = None
    policy.current_state = torch.tensor([[2.5, -1.0]])
    model_actions = torch.tensor([[0.25, 0.75]])

    assert torch.equal(policy._model_state(), policy.current_state)
    assert torch.equal(policy._environment_actions(model_actions), model_actions)


def test_policy_rejects_checkpoint_config_mismatch(tmp_path):
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.config_dict = {"common": {"action_normalization": "min_max"}}
    policy.checkpoint_path = str(tmp_path)
    (tmp_path / "config.json").write_text(
        '{"common": {"action_normalization": "none"}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Action normalization mismatch"):
        policy._resolve_action_normalization()


def test_policy_uses_checkpoint_action_space_metadata(tmp_path):
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.config_dict = {"common": {"action_normalization": None}}
    policy.checkpoint_path = str(tmp_path)
    (tmp_path / "config.json").write_text(
        '{"common": {"action_normalization": "min_max"}}', encoding="utf-8"
    )

    assert policy._resolve_action_normalization() == "min_max"


def test_policy_rejects_malformed_checkpoint_metadata(tmp_path):
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.config_dict = {"common": {}}
    policy.checkpoint_path = str(tmp_path)
    (tmp_path / "config.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read checkpoint"):
        policy._resolve_action_normalization()


def test_policy_legacy_checkpoint_defaults_to_raw(tmp_path):
    policy_cls = _policy_class()
    policy = policy_cls.__new__(policy_cls)
    policy.config_dict = {"common": {}}
    policy.checkpoint_path = str(tmp_path / "missing_checkpoint")

    assert policy._resolve_action_normalization() == "none"
