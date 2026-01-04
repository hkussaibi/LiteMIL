"""Inference script for single slides - supports both features and raw WSI"""
import torch
import numpy as np
from pathlib import Path
import h5py
import warnings


class SlidePredictor:
    """Predictor for whole slide images.

    Supports:
    - Pre-extracted features (.h5, .pt, .pth)
    - Raw WSI files (.svs, .tif, .ndpi, etc.) with on-the-fly feature extraction
    
    Inference modes:
    - 'full': Process entire slide at once (default, best for most cases)
    - 'chunked': Split into chunks and aggregate predictions (for very large slides)
    """

    def __init__(self, 
                 model_path, 
                 model_class, 
                 class_names, 
                 device='cuda',
                 feature_extractor=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            model_class: Model factory function
            class_names: List of class names
            device: Device for inference
            feature_extractor: Optional WSIFeatureExtractor instance for raw WSI processing
        """
        self.device = device
        self.class_names = class_names
        self.feature_extractor = feature_extractor

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        self.model = model_class()
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        print(f"✓ Model loaded from {model_path}")

    @torch.no_grad()
    def predict(self, 
                slide_path, 
                mode='full',
                chunk_size=1000,
                extract_kwargs=None):
        """Predict slide class.

        Args:
            slide_path: Path to feature file or raw WSI
            mode: 'full' (entire slide) or 'chunked' (split and aggregate)
            chunk_size: Size of chunks for chunked mode
            extract_kwargs: Optional kwargs for WSI feature extraction
                - patch_size: int (default 256)
                - stride: int (default 256)
                - level: int (default 0)
                - tissue_threshold: float (default 0.5)
        """
        slide_path = Path(slide_path)
        
        # Determine if we need to extract features
        if slide_path.suffix.lower() in ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn']:
            # Raw WSI - need to extract features
            if self.feature_extractor is None:
                raise ValueError(
                    "Raw WSI provided but no feature_extractor configured. "
                    "Create SlidePredictor with feature_extractor argument."
                )
            
            print(f"Detected raw WSI format: {slide_path.suffix}")
            print("Extracting features on-the-fly...")
            
            # Extract features
            extract_kwargs = extract_kwargs or {}
            result = self.feature_extractor.process(str(slide_path))
            features = result['features']
            coords = result['coords']
            
            print(f"Extracted {len(features)} feature vectors")
            
        else:
            # Pre-extracted features
            features, coords = self._load_features(slide_path)
        
        # Validate features
        if torch.isnan(features).any() or torch.isinf(features).any():
            warnings.warn("Features contain NaN or Inf values. Results may be unreliable.")

        print(f"Loaded {features.shape[0]} instances")

        if mode == 'full':
            return self._predict_full(features)
        elif mode == 'chunked':
            return self._predict_chunked(features, chunk_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'full' or 'chunked'")

    def _load_features(self, slide_path):
        """Load pre-extracted features from file."""
        if str(slide_path).endswith('.h5'):
            with h5py.File(slide_path, 'r') as f:
                features = torch.from_numpy(f['features'][:]).float()
                coords = torch.from_numpy(f['coords'][:]).float() if 'coords' in f else None
        else:
            data = torch.load(slide_path, weights_only=False)
            # Handle both dict and tensor formats
            if isinstance(data, dict):
                features = data.get('features', None)
                coords = data.get('coords', None)
                if coords is not None:
                    coords = coords.float()
            else:
                features = data.float()
                coords = None
        
        if coords is None:
            coords = torch.zeros(features.shape[0], 2)
        
        return features, coords

    def _predict_full(self, features):
        """Full mode: Process entire slide at once."""
        features = features.unsqueeze(0).to(self.device)  # (1, N, D)
        
        try:
            outputs, aux = self.model(features)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(dim=1).item()
            
            # Get attention and convert to patch-level scores
            attn = aux.get('attn')  # (1, num_queries, N)
            if attn is not None:
                # Average across queries to get per-patch attention
                patch_attention = attn[0].mean(dim=0).cpu()  # (N,)
            else:
                patch_attention = None

            return {
                'predicted_class': self.class_names[pred_idx],
                'confidence': probs[0, pred_idx].item(),
                'probabilities': {name: float(prob) for name, prob in
                                  zip(self.class_names, probs[0].cpu().numpy())},
                'attention_weights': attn,  # Keep raw attention for advanced use
                'patch_attention': patch_attention,  # Per-patch attention scores
                'num_instances': features.shape[1],
                'mode': 'full'
            }
        except Exception as e:
            raise RuntimeError(f"Error during full-slide inference: {e}")

    def _predict_chunked(self, features, chunk_size):
        """
        Chunked mode: Split slide into chunks and aggregate via attention-weighted aggregation.
        """
        N = features.shape[0]

        if N < chunk_size:
            print(f"Note: Slide has only {N} instances (< chunk_size={chunk_size}), using full mode")
            return self._predict_full(features)

        n_chunks = N // chunk_size
        usable = n_chunks * chunk_size
        chunks = features[:usable].reshape(n_chunks, chunk_size, -1)

        print(f"Processing {n_chunks} chunks with attention-weighted aggregation...")

        chunk_probs = []
        chunk_attn_scores = []

        for i in range(n_chunks):
            try:
                chunk = chunks[i].unsqueeze(0).to(self.device)  # (1, chunk_size, D)
                outputs, aux = self.model(chunk)

                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                chunk_probs.append(probs)

                # Extract attention and compute chunk importance
                attn = aux.get('attn')
                if attn is None:
                    chunk_attn_scores.append(1.0)
                else:
                    # attn from MultiheadAttention: (1, num_queries, chunk_size)
                    attn = attn.detach()
                    attn_max_per_query = attn.max(dim=-1)[0]  # (1, num_queries)
                    max_attn = attn_max_per_query.mean().item()
                    
                    if not np.isfinite(max_attn):
                        warnings.warn(f"Chunk {i}: Non-finite attention score. Using 1.0")
                        max_attn = 1.0
                    
                    chunk_attn_scores.append(max_attn)
                    
            except Exception as e:
                warnings.warn(f"Error processing chunk {i}: {e}. Using uniform weights.")
                if len(chunk_probs) > 0:
                    chunk_probs.append(np.mean(chunk_probs, axis=0))
                else:
                    chunk_probs.append(np.ones(len(self.class_names)) / len(self.class_names))
                chunk_attn_scores.append(1.0)

        chunk_probs = np.stack(chunk_probs)
        chunk_attn_scores = np.array(chunk_attn_scores)

        # Normalize attention scores
        attn_sum = chunk_attn_scores.sum()
        if attn_sum == 0 or not np.isfinite(attn_sum):
            warnings.warn("Attention scores sum to zero or non-finite. Using uniform weights.")
            chunk_attn_scores = np.ones_like(chunk_attn_scores) / len(chunk_attn_scores)
        else:
            chunk_attn_scores = chunk_attn_scores / attn_sum

        # Attention-weighted aggregation
        weighted_probs = np.sum(
            chunk_probs * chunk_attn_scores[:, None],
            axis=0
        )

        pred_idx = int(np.argmax(weighted_probs))
        confidence = float(weighted_probs[pred_idx])

        return {
            'predicted_class': self.class_names[pred_idx],
            'confidence': confidence,
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.class_names, weighted_probs)
            },
            'num_instances': N,
            'num_chunks': n_chunks,
            'chunk_attention': chunk_attn_scores.tolist(),
            'chunk_attention_stats': {
                'min': float(chunk_attn_scores.min()),
                'max': float(chunk_attn_scores.max()),
                'mean': float(chunk_attn_scores.mean()),
                'std': float(chunk_attn_scores.std())
            },
            'mode': 'chunked'
        }


def extractPredict(
    model_path,
    model_class,
    class_names,
    backbone='resnet50',
    patch_size=256,
    stride=256,
    level=0,
    device='cuda'
):
    """
    Convenience function to create a predictor with feature extraction capability.
    
    Args:
        model_path: Path to trained model
        model_class: Model factory
        class_names: List of class names
        backbone: Feature extraction backbone ('resnet50', 'phikon', 'uni')
        patch_size: Patch size for extraction
        stride: Stride between patches
        level: WSI pyramid level
        device: Computation device
    
    Returns:
        SlidePredictor configured with WSI feature extraction
    """
    from .feature_extractor import WSIFeatureExtractor
    
    feature_extractor = WSIFeatureExtractor(
        backbone=backbone,
        patch_size=patch_size,
        stride=stride,
        level=level,
        device=device
    )
    
    return SlidePredictor(
        model_path=model_path,
        model_class=model_class,
        class_names=class_names,
        device=device,
        feature_extractor=feature_extractor
    )
