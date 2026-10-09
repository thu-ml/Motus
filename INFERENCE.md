# Motus Inference Guide

This guide covers running inference with Motus models.

## Running Inference

### 1. RoboTwin 2.0 Simulation

For evaluation on [RoboTwin 2.0](https://robotwin-platform.github.io/) benchmark:

**📖 See detailed guide:** [**RoboTwin Inference Guide**](inference/robotwin/Motus/README.md)

**Quick Start:**
```bash
cd inference/robotwin/Motus

# Single task evaluation
bash eval.sh <task_name>

# Multi-task batch evaluation
bash auto_eval.sh
```

### 2. Real-World Inference (No Environment)

**📖 See detailed guide:** [**Real-World Inference Guide**](inference/real_world/Motus/README.md)

We provide a minimal inference script that runs Motus on a single image without any robot environment.

**⚠️ Important:** Input images must be **three-view concatenated** (head + left/right wrist cameras). Use [Multi-Camera Concatenation](data/utils/multi_camera_concat.py) first if you have separate camera views.

**With pre-encoded T5 embeddings (recommended, ~24GB VRAM):**
```bash
# Step 1: Encode instruction to T5 embeddings (do this once)
python inference/real_world/Motus/encode_t5_instruction.py \
  --instruction "Pour water from kettle to flowers" \
  --output t5_embed.pt \
  --wan_path pretrained_models

# Step 2: Run inference with pre-encoded embeddings
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus \
  --wan_path pretrained_models \
  --image examples/first_frame.png \
  --instruction "Pour water from kettle to flowers" \
  --t5_embeds t5_embed.pt \
  --output examples/output_ac_one.png
```

**Without pre-encoded T5 (encode on-the-fly, ~41GB VRAM):**
```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus \
  --wan_path pretrained_models \
  --image examples/first_frame.png \
  --instruction "Pour water from kettle to flowers" \
  --use_t5 \
  --output examples/output_ac_one.png
```

**Intel XPU:**

Install the XPU-enabled PyTorch wheels in the inference environment:
```bash
python -m pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/xpu
```

Run inference with the XPU backend and pre-encoded T5 embeddings:
```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus \
  --wan_path pretrained_models \
  --image examples/first_frame.png \
  --instruction "Pour water from kettle to flowers" \
  --t5_embeds t5_embed.pt \
  --device xpu \
  --output examples/output_ac_one_xpu.png
```

On XPU, Motus uses PyTorch scaled dot-product attention instead of CUDA
FlashAttention. XPU execution currently targets a single device.

**Output:**
- `examples/output_ac_one.png`: Grid of condition frame + predicted future frames
- Console: Predicted action chunk with shape `(action_chunk_size, action_dim)`

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| OOM during inference (~41GB) | T5 encoder loaded at runtime | Use pre-encoded T5 embeddings (`--t5_embeds`) to reduce to 24~25GB |
| Poor action predictions | Checkpoint mismatch | Ensure using correct config for your checkpoint |

---

**📖 Related Guides:**
- [Training Guide](TRAINING.md)
- [Data Format](DATA_FORMAT.md)
- [RoboTwin Inference](inference/robotwin/Motus/README.md)
- [Real-World Inference](inference/real_world/Motus/README.md)
