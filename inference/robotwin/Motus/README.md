# Motus Policy Evaluation on RoboTwin Platform

This guide explains how to evaluate Motus policy on the RoboTwin simulation platform.

**Important Note:** RoboTwin evaluation uses a **separate conda environment** from Motus training. You will need two environments:
- **`motus`**: For Motus training (see main README)
- **`RoboTwin`**: For RoboTwin simulation and evaluation (this guide)

## Table of Contents
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Evaluation](#running-evaluation)
- [Viewing Results](#viewing-results)

---

## Environment Setup

### 1. Install RoboTwin Platform

Follow the official RoboTwin installation guide to create the `RoboTwin` conda environment:
```
https://robotwin-platform.github.io/doc/usage/robotwin-install.html
```

This will create a conda environment named `RoboTwin` (or your custom name) with all necessary RoboTwin dependencies.

### 2. System Requirements

- **CUDA >= 12.6**
- **PyTorch >= 2.4.0**
- **torchvision >= 0.19.0**

Install PyTorch with CUDA 12.6:
```bash
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install Additional Dependencies

After installing RoboTwin, **activate the RoboTwin environment** and install Motus-specific requirements:

```bash
conda activate RoboTwin  # Or your RoboTwin environment name
cd /path/to/RoboTwin/policy/Motus
pip install -r requirements.txt
```

**Note:** Make sure you're in the `RoboTwin` environment, not the `motus` training environment.

---

## Installation

### Deploy Motus Policy to RoboTwin

1. **Copy the Motus inference folder** to RoboTwin policy directory:

```bash
# From Motus repository root
cp -r inference/robotwin/Motus /path/to/RoboTwin/policy/
```

2. **Verify directory structure:**

```
RoboTwin/
├── policy/
│   ├── Motus/
│   │   ├── auto_eval.sh
│   │   ├── eval.sh
│   │   ├── per_eval_logs.sh
│   │   ├── paths_config.yml
│   │   ├── deploy_policy.py
│   │   ├── deploy_policy.yml
│   │   ├── requirements.txt
│   │   ├── tasks_all.txt
│   │   ├── models/
│   │   └── README.md (this file)
│   └── ... (other policies)
├── script/
│   └── eval_policy.py
└── ... (other RoboTwin files)
```

---

## Configuration

### Edit `paths_config.yml`

Open `paths_config.yml` and configure all required paths:

```yaml
# ============================================================================
# Core Paths - MUST MODIFY THESE
# ============================================================================
robotwin_root: "/path/to/RoboTwin"
conda_env: "/path/to/conda/envs/RoboTwin-env"
checkpoint_path: 

# Pretrained model paths (for loading configs/tokenizer, not weights)
wan_path: "/path/to/pretrained_models/Wan2.2-TI2V-5B"
vlm_path: "/path/to/pretrained_models/Qwen3-VL-2B-Instruct"

# ============================================================================
# Optional Configuration
# ============================================================================
# GPU IDs (empty means auto-detect)
gpu_ids: []

# Task configuration
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all.txt"
```

**Important Notes:**
- `checkpoint_path`: Must be the **directory** containing `mp_rank_00_model_states.pt`
  - Example structure: `checkpoint_path/mp_rank_00_model_states.pt`
- `wan_path` and `vlm_path`: Only config/tokenizer files are required, not full pretrained weights
- `gpu_ids`: Leave empty `[]` for auto-detection, or specify GPU IDs like `[0, 1, 2, 3]`

### Action-space contract

RobotWin checkpoints trained before the action-space metadata was introduced
use raw qpos values.  The deployment wrapper keeps that behavior for legacy
checkpoints and reads `common.action_normalization` from a checkpoint's
`config.json` when present.

For a new normalized run, set `common.action_normalization: "min_max"` in the
training config.  The data loader then applies the per-dimension statistics to
both the conditioning state and action targets, while deployment normalizes
the observed state and converts predicted actions back to qpos before calling
the environment.  If a deployment config and checkpoint record different
modes, evaluation stops with an explicit mismatch error instead of silently
using an incompatible action distribution.

### Configure Task List

Edit `tasks_all.txt` to specify which tasks to evaluate:

```bash
# One task name per line
PickCube
StackCube
PutSpoon
# ... add more tasks
```

---

## Running Evaluation

### Batch Evaluation (Multiple Tasks)

Evaluate all tasks listed in `tasks_all.txt`:

```bash
cd /path/to/RoboTwin/policy/Motus
bash auto_eval.sh
```

This will:
1. Load configuration from `paths_config.yml`
2. Activate the conda environment
3. Run evaluation for each task in parallel across available GPUs
4. Save logs to `logs_YYYYMMDD_HHMMSS/` directory


### Single Task Evaluation

Evaluate a specific task:

```bash
cd /path/to/RoboTwin/policy/Motus
bash eval.sh <task_name>

# Example:
bash eval.sh PickCube
```

---

## Viewing Results

### Detailed Task Scores

To view detailed per-task success rates, use the `per_eval_logs.sh` script:

1. **Edit `per_eval_logs.sh`** and set the `LOG_DIR` variable to your logs directory:

```bash
# Open per_eval_logs.sh and modify:
LOG_DIR="/path/to/RoboTwin/policy/Motus/logs_YYYYMMDD_HHMMSS"
```

2. **Run the script:**

```bash
cd /path/to/RoboTwin/policy/Motus
bash per_eval_logs.sh
```
