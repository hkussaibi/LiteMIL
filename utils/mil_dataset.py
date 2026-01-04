"""MIL Dataset - Handles both dict and tensor formats"""
import torch
import os
import h5py
import pandas as pd
from torch.utils.data import Dataset


class MILDataset(Dataset):
    """Dataset for Multiple Instance Learning with WSI features.

    Supports two feature file formats:
    1. Dictionary: {'features': tensor, 'coords': tensor}
    2. Raw tensor: features only (coords will be None)
    """

    def __init__(self, csv_path, features_dir, class_names, mode='full',
                 chunk_size=1000, feat_type='features'):
        self.features_dir = features_dir
        self.class_names = class_names
        self.cls2idx = {c: i for i, c in enumerate(class_names)}
        self.mode = mode
        self.chunk_size = chunk_size
        self.feat_type = feat_type

        # Load metadata
        df = pd.read_csv(csv_path)
        self.index = []
        self.labels = []
        self.wsi_ids = []
        self.case_ids = []

        print(f"Loading {len(df)} WSIs in {mode} mode...")
        for _, row in df.iterrows():
            wsi_id = str(row['wsi_id'])
            label = self.cls2idx.get(row['label'])
            if label is None:
                continue

            # Find file
            for ext in ['.h5', '.pt', '.pth']:
                path = f"{features_dir}/{wsi_id}{ext}"
                if not os.path.exists(path):
                    continue

                # Get length
                try:
                    if ext == '.h5':
                        with h5py.File(path, 'r') as f:
                            L = f[feat_type].shape[0]
                    else:
                        data = torch.load(path, weights_only=False)
                        # Handle both dict and tensor formats
                        if isinstance(data, dict):
                            L = data[feat_type].shape[0]
                        else:
                            # Raw tensor
                            L = data.shape[0]
                except Exception as e:
                    print(f"Warning: Error loading {path}: {e}")
                    continue

                # Create indices
                if mode == 'full':
                    self.index.append((path, 0, L, ext == '.h5'))
                    self.labels.append(label)
                    self.wsi_ids.append(wsi_id)
                    self.case_ids.append(row.get('case_id', wsi_id))
                else:  # chunked
                    if L < chunk_size:
                        continue
                    n_chunks = L // chunk_size
                    for i in range(n_chunks):
                        offset = i * chunk_size
                        self.index.append((path, offset, chunk_size, ext == '.h5'))
                        self.labels.append(label)
                        self.wsi_ids.append(wsi_id)
                        self.case_ids.append(row.get('case_id', wsi_id))
                break

        print(f"✓ Loaded {len(self)} samples")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, offset, length, is_h5 = self.index[idx]

        if is_h5:
            with h5py.File(path, 'r') as f:
                feat = torch.from_numpy(f[self.feat_type][offset:offset + length]).float()
                coords = torch.from_numpy(f['coords'][offset:offset + length]).float() if 'coords' in f else None
        else:
            data = torch.load(path, weights_only=False)

            # Handle both dict and raw tensor formats
            if isinstance(data, dict):
                feat = data[self.feat_type][offset:offset + length].float()
                coords = data.get('coords', None)
                if coords is not None:
                    coords = coords[offset:offset + length].float()
            else:
                # Raw tensor - features only
                feat = data[offset:offset + length].float()
                coords = None

        # Return dummy coords if None
        if coords is None:
            coords = torch.zeros(feat.shape[0], 2)

        return feat, coords, self.labels[idx]

    def get_wsi_id(self, idx):
        return self.wsi_ids[idx]

    def get_case_id(self, idx):
        return self.case_ids[idx]
