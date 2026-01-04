#!/usr/bin/env python3
"""
Feature Extraction CLI - Extract features from raw WSI files

Usage:
    # Single file (Auto-names to slide.h5)
    python extractFeatures.py --input slide.svs --backbone resnet50

    # Batch processing
    python extractFeatures.py --input_dir wsi_folder/ --output_dir features/ --backbone uni
"""
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

from utils.feature_extractor import WSIFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract features from whole slide images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                             help='Path to single WSI file')
    input_group.add_argument('--input_dir', type=str,
                             help='Directory containing WSI files')

    # Output - Now optional for single files to allow auto-naming
    parser.add_argument('--output', type=str,
                        help='Output path for single file (defaults to input_stem.format)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for batch processing (required if using --input_dir)')

    # Feature extraction
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'phikon-v2', 'uni'],
                        help='Feature extraction backbone')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size in pixels')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride between patches')
    parser.add_argument('--level', type=int, default=0,
                        help='WSI pyramid level (0 = highest magnification)')
    parser.add_argument('--tissue_threshold', type=float, default=0.5,
                        help='Minimum tissue ratio in patch (0-1)')

    # Processing
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--format', type=str, default='h5',
                        choices=['h5', 'pt'],
                        help='Output format (used for auto-naming and batch mode)')

    # Filtering
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.svs', '.tif', '.tiff', '.ndpi'],
                        help='File extensions to process (batch mode)')

    return parser.parse_args()


def get_wsi_files(input_dir, extensions):
    """Get all WSI files from directory."""
    input_dir = Path(input_dir)
    files = []

    for ext in extensions:
        files.extend(input_dir.glob(f"**/*{ext}"))

    return sorted(files)


def main():
    args = parse_args()

    # Initialize feature extractor
    print("=" * 70)
    print("WSI Feature Extraction")
    print("=" * 70)
    print(f"Backbone: {args.backbone}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Level: {args.level}")
    print(f"Tissue threshold: {args.tissue_threshold}")
    print(f"Device: {args.device}")
    print("=" * 70)

    extractor = WSIFeatureExtractor(
        backbone=args.backbone,
        patch_size=args.patch_size,
        stride=args.stride,
        level=args.level,
        tissue_threshold=args.tissue_threshold,
        batch_size=args.batch_size,
        device=args.device
    )

    # Single file mode
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        # Determine save_path: Use provided --output or generate from input name
        save_path = args.output if args.output else f"{input_path.stem}.{args.format}"

        print(f"\nProcessing single file: {args.input}")
        result = extractor.process(str(input_path), save_path=str(save_path))

        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Output: {save_path}")
        print(f"Patches: {result['metadata']['num_patches']}")
        print(f"Feature dim: {result['features'].shape[1]}")
        print("=" * 70)

    # Batch mode
    else:
        if not args.output_dir:
            print("Error: --output_dir is required when using --input_dir")
            sys.exit(1)

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if not input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all WSI files
        wsi_files = get_wsi_files(input_dir, args.extensions)

        if len(wsi_files) == 0:
            print(f"Error: No WSI files found in {input_dir}")
            print(f"Looking for extensions: {args.extensions}")
            sys.exit(1)

        print(f"\nFound {len(wsi_files)} WSI files")
        print(f"Output directory: {output_dir}\n")

        # Process each file
        results_summary = []

        for wsi_path in tqdm(wsi_files, desc="Processing WSIs"):
            try:
                # Generate output path
                output_path = output_dir / f"{wsi_path.stem}.{args.format}"

                # Skip if already exists
                if output_path.exists():
                    print(f"  Skipping {wsi_path.name} (already exists)")
                    continue

                # Extract features
                result = extractor.process(str(wsi_path), save_path=str(output_path))

                results_summary.append({
                    'file': wsi_path.name,
                    'patches': result['metadata']['num_patches'],
                    'tissue_ratio': result['metadata']['tissue_ratio'],
                    'outputs': output_path.name
                })

            except Exception as e:
                print(f"  Error processing {wsi_path.name}: {e}")
                results_summary.append({
                    'file': wsi_path.name,
                    'error': str(e)
                })

        # Print summary
        print("\n" + "=" * 70)
        print("BATCH EXTRACTION COMPLETE")
        print("=" * 70)

        successful = [r for r in results_summary if 'error' not in r]
        failed = [r for r in results_summary if 'error' in r]

        print(f"Successfully processed: {len(successful)}/{len(wsi_files)}")

        if successful:
            total_patches = sum(r['patches'] for r in successful)
            avg_patches = total_patches / len(successful)
            print(f"Total patches extracted: {total_patches:,}")
            print(f"Average patches per slide: {avg_patches:.0f}")

        if failed:
            print(f"\nFailed: {len(failed)}")
            for r in failed[:5]:  # Show first 5 errors
                print(f"  - {r['file']}: {r['error']}")

        print("=" * 70)


if __name__ == '__main__':
    main()