#!/usr/bin/env python3
"""
LiteMIL Training Script - CLI Interface
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from MILS.LiteMIL import LiteMIL
from MILS.TransMIL import TransMIL
from MILS.MABMIL import ABMIL, ABMIL_Multihead
from MILS.pool import meanPool, maxPool
from utils.mil_dataset import MILDataset
from utils.nested_cv import NestedCrossValidation


DATASETS = {
    'breast': {
        'classes': ['IDC', 'ILC'],
        'features_dir': 'datasets/breast/features',
        'labels_csv': 'datasets/breast/labels.csv',
    },
    'lung': {
        'classes': ['LUAD', 'LUSC'],
        'features_dir': 'datasets/lung/features',
        'labels_csv': 'datasets/lung/labels.csv',
    },
    'kidney': {
        'classes': ['PRCC', 'CCRCC', 'CHRCC'],
        'features_dir': 'datasets/kidney/features',
        'labels_csv': 'datasets/kidney/labels.csv',
    },
    'tupac': {
        'classes': ['Low', 'High'],
        'features_dir': 'datasets/tupac/features',
        'labels_csv': 'datasets/tupac/labels.csv',
    }
}

MIL_REGISTRY = {
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
        description='Train MIL on WSI datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASETS.keys())
    parser.add_argument('--mil', type=str, required=True,
                        choices=MIL_REGISTRY.keys(), help="MIL architecture")
    parser.add_argument('--features_dir', type=str, default=None)
    parser.add_argument('--labels_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--mode', type=str, default='chunked',
                        choices=['full', 'chunked'])
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--feat_type', type=str, default='features')

    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_queries', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.25)

    parser.add_argument('--n_outer', type=int, default=5)
    parser.add_argument('--n_inner', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = DATASETS[args.dataset]
    features_dir = args.features_dir or cfg['features_dir']
    labels_csv = args.labels_csv or cfg['labels_csv']
    output_dir = args.output_dir or f'outputs/{args.mil}/{args.dataset}'

    os.makedirs(output_dir, exist_ok=True)

    class_names = cfg['classes']
    num_classes = len(class_names)

    dataset = MILDataset(
        csv_path=labels_csv,
        features_dir=features_dir,
        class_names=class_names,
        mode=args.mode,
        chunk_size=args.chunk_size,
        feat_type=args.feat_type
    )

    mil_cfg = MIL_REGISTRY[args.mil]
    MILClass = mil_cfg["class"]
    mil_defaults = mil_cfg["default_kwargs"]

    def model_factory():
        return MILClass(
            input_dim=args.input_dim,
            num_classes=num_classes,
            **mil_defaults
        )

    cv = NestedCrossValidation(
        model_factory=model_factory,
        dataset=dataset,
        class_names=class_names,
        output_dir=output_dir,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        use_amp=not args.no_amp
    )

    results = cv.run()

    print("\nTraining complete")
    print(f"Slide accuracy: {results['slide_accuracy_mean']:.3f} ± {results['slide_accuracy_std']:.3f}")


if __name__ == '__main__':
    main()
