#!/usr/bin/env python3
"""
Unified MIL Inference & Visualization Script
Supports multiple architectures (LiteMIL, ABMIL, TransMIL, etc.)
"""
import argparse
import sys
from pathlib import Path
import torch
import h5py
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for server use

from MILS.LiteMIL import LiteMIL
from MILS.TransMIL import TransMIL
from MILS.MABMIL import ABMIL, ABMIL_Multihead
from MILS.pool import meanPool, maxPool
from utils.inference import SlidePredictor, extractPredict
from utils.attention_visualizer import AttentionVisualizer

# --- Configurations ---

DATASET_CLASSES = {
    'breast': ['IDC', 'ILC'],
    'lung': ['LUAD', 'LUSC'],
    'kidney': ['PRCC', 'CCRCC', 'CHRCC'],
    'tupac': ['Low', 'High']
}

MIL_CONFIG = {
    "LiteMIL": {
        "class": LiteMIL,
        "default_kwargs": dict(hidden_dim=256, num_heads=4, num_queries=1, dropout=0.25),
    },
    "ABMIL": {
        "class": ABMIL,
        "default_kwargs": dict(hidden_dim=256, dropout=0.25),
    },
    "MADMIL": {
        "class": ABMIL_Multihead,
        "default_kwargs": dict(hidden_dim=256, dropout=0.25),
    },
    "TransMIL": {
        "class": TransMIL,
        "default_kwargs": dict(hidden_dim=256, dropout=0.25),
    },
    "maxPool": {
        "class": meanPool,
        "default_kwargs": {},
    },
    "meanPool": {
        "class": maxPool,
        "default_kwargs": {},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified LiteMIL Inference and Visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Requirements
    parser.add_argument('--mil', type=str, required=True, choices=MIL_CONFIG.keys(), help="MIL architecture")
    parser.add_argument('--checkpoint', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to slide features (.h5/.pt) or raw WSI')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CLASSES.keys())
    parser.add_argument('--output_dir', type=str, default='outputs/inference', help='Directory for results')

    # Model Params
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Inference Mode
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'chunked'])
    parser.add_argument('--chunk_size', type=int, default=1000)

    # Extraction Params (if input is raw WSI)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'phikon-v2', 'uni'])
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--level', type=int, default=0)

    # Visualization Toggle
    parser.add_argument('--visualize', action='store_true', help='Enable attention visualization')
    parser.add_argument('--wsi', type=str, help='Path to original WSI (required if --visualize is set)')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top patches to highlight')
    parser.add_argument('--cmap', type=str, default='jet', choices=['jet', 'hot', 'viridis'])

    return parser.parse_args()


def is_raw_wsi(filepath):
    ext = Path(filepath).suffix.lower()
    return ext in ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs']


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Model Factory
    class_names = DATASET_CLASSES[args.dataset]
    mil_cfg = MIL_CONFIG[args.mil]

    def model_factory():
        return mil_cfg["class"](
            input_dim=args.input_dim,
            num_classes=len(class_names),
            **mil_cfg["default_kwargs"]
        )

    # 2. Initialize Predictor
    needs_extraction = is_raw_wsi(args.input)
    if needs_extraction:
        predictor = extractPredict(
            model_path=args.checkpoint, model_class=model_factory,
            class_names=class_names, backbone=args.backbone,
            patch_size=args.patch_size, level=args.level, device=args.device
        )
    else:
        predictor = SlidePredictor(
            model_path=args.checkpoint, model_class=model_factory,
            class_names=class_names, device=args.device
        )

    # 3. Run Inference
    print(f"Running inference on: {args.input}...")
    result = predictor.predict(args.input, mode=args.mode, chunk_size=args.chunk_size)

    # 4. Print Results
    print(f"\nPREDICTION: {result['predicted_class']} ({result['confidence']:.2%})")

    # 5. Visualization Logic
    if args.visualize:
        if not args.wsi or not Path(args.wsi).exists():
            print("Error: --wsi path required and must exist for visualization.")
            sys.exit(1)

        print("\nGenerating Visualizations...")
        visualizer = AttentionVisualizer(cmap=args.cmap, alpha=0.5)
        slide_name = Path(args.input).stem

        # Extract coordinates and attention
        # Coordinates are either in the result (if raw) or need to be loaded from H5
        if needs_extraction:
            coords = result['coords']
        else:
            with h5py.File(args.input, 'r') as f:
                coords = f['coords'][:]

        # Handle attention mapping
        attention = result.get('patch_attention')
        if attention is None:
            # Fallback for ABMIL/other models that might use a different key
            attention = result.get('attention_weights')

        if attention is not None:
            attention = attention.cpu().numpy() if torch.is_tensor(attention) else attention

            # Save Heatmap
            vis_path = output_dir / f"{slide_name}_heatmap.png"
            visualizer.visualize_full_attention(
                args.wsi, coords, attention,
                save_path=vis_path, patch_size=args.patch_size, level=args.level
            )
            print(f"✓ Heatmap saved to {vis_path}")

            # Save Top Patches
            patch_dir = output_dir / f"{slide_name}_top_patches"
            visualizer.extract_top_patches(
                args.wsi, coords, attention,
                top_k=args.top_k, save_dir=patch_dir
            )
            print(f"✓ Top {args.top_k} patches saved to {patch_dir}")
        else:
            print("Warning: No attention weights found in model output. Skipping visualization.")

    print("\nDone.")


if __name__ == '__main__':
    main()