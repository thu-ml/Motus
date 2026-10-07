#!/usr/bin/env python3
"""
Encode a single text instruction to T5 embeddings for WAN model.
Usage:
    python encode_t5_instruction.py --instruction "pick up the cube" --output t5_embed.pt --wan_path /path/to/wan
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent.parent.parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.device import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Encode a single text instruction to T5 embeddings")
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Text instruction to encode"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path to save T5 embedding (.pt file)"
    )
    parser.add_argument(
        "--wan_path",
        type=str,
        default="/share/home/bhz/pretrained_models",
        help="Path to WAN pretrained models directory"
    )
    parser.add_argument(
        "--text_len",
        type=int,
        default=512,
        help="Maximum text length for T5 encoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run encoding on ('auto', 'cuda', 'xpu', 'cpu')"
    )
    
    args = parser.parse_args()
    device = str(resolve_device(args.device))
    print(f"Using PyTorch device: {device}")
    
    # Initialize T5 encoder
    print(f"Loading T5 encoder from {args.wan_path}...")
    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=os.path.join(args.wan_path, 'Wan2.2-TI2V-5B', 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=os.path.join(args.wan_path, 'Wan2.2-TI2V-5B', 'google/umt5-xxl'),
    )
    
    # Encode instruction
    print(f"Encoding instruction: '{args.instruction}'")
    encoded = encoder([args.instruction], device)
    
    # Handle output format (list of tensors)
    if isinstance(encoded, list):
        if len(encoded) == 1:
            # Single instruction -> save as single tensor
            embedding = encoded[0]
        else:
            # Multiple instructions -> save as list
            embedding = encoded
    elif isinstance(encoded, torch.Tensor):
        embedding = encoded
    else:
        raise ValueError(f"Unexpected encoder output type: {type(encoded)}")
    
    # Save embedding
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(embedding, torch.Tensor):
        torch.save(embedding.cpu(), output_path)
        print(f"Saved T5 embedding to {output_path}")
        print(f"  Shape: {embedding.shape}")
    else:
        # List of tensors
        torch.save([emb.cpu() for emb in embedding], output_path)
        print(f"Saved T5 embeddings (list) to {output_path}")
        print(f"  Number of embeddings: {len(embedding)}")
        if len(embedding) > 0:
            print(f"  First embedding shape: {embedding[0].shape}")


if __name__ == "__main__":
    main()

