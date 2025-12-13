#!/usr/bin/env python3
"""
 ===============================================================
 Authors  : Priestley Fomeche
 Filename : convert_ckpt_to_pt.py
 Purpose  : Converts Nanodet Lightning Checkpoint to Python Model
 Date     : 13.12.2025
 ===============================================================
"""

import argparse
import torch


def convert_checkpoint(ckpt_path, output_path):
    """
    Extract model weights from Lightning checkpoint and save as .pt
    
    Args:
        ckpt_path: Path to the .ckpt file
        output_path: Path where the .pt file will be saved
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Extract the model state_dict
    # Lightning stores it under 'state_dict' key
    if 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
        print(f"Found {len(model_state_dict)} keys in state_dict")
    else:
        print("Warning: 'state_dict' key not found. Available keys:")
        print(checkpoint.keys())
        return
    
    # Save as .pt file
    torch.save(model_state_dict, output_path)
    print(f"Successfully saved model weights to: {output_path}")
    
    # Print some info
    print(f"\nModel info:")
    print(f"  Total parameters: {sum(p.numel() for p in model_state_dict.values() if isinstance(p, torch.Tensor))}")
    if 'epoch' in checkpoint:
        print(f"  Trained epochs: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"  Global step: {checkpoint['global_step']}")


def main():
    parser = argparse.ArgumentParser(description='Convert NanoDet .ckpt to .pt')
    parser.add_argument('ckpt_path', type=str, help='Path to input .ckpt file')
    parser.add_argument('output_path', type=str, help='Path to output .pt file')
    
    args = parser.parse_args()
    
    convert_checkpoint(args.ckpt_path, args.output_path)


if __name__ == '__main__':
    main()
