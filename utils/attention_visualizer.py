"""Attention heatmap visualization for WSI predictions"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import openslide
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import h5py
import warnings


class AttentionVisualizer:
    """Generate attention heatmaps overlaid on WSI thumbnails."""
    
    def __init__(self, 
                 cmap: str = 'jet',
                 alpha: float = 0.5,
                 thumbnail_size: int = 2048):
        """
        Args:
            cmap: Matplotlib colormap for attention ('jet', 'hot', 'viridis', etc.)
            alpha: Transparency of attention overlay (0-1)
            thumbnail_size: Size for WSI thumbnail
        """
        self.cmap_name = cmap
        self.alpha = alpha
        self.thumbnail_size = thumbnail_size
        
        # Create custom colormap (transparent -> red/hot)
        if cmap == 'jet':
            colors = ['blue', 'cyan', 'yellow', 'red']
        elif cmap == 'hot':
            colors = ['black', 'red', 'yellow', 'white']
        elif cmap == 'viridis':
            colors = plt.cm.viridis(np.linspace(0, 1, 256))
        else:
            colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, 256))
        
        self.cmap = plt.cm.get_cmap(cmap)
    
    def visualize_full_attention(self,
                                 slide_path: str,
                                 coords: np.ndarray,
                                 attention: np.ndarray,
                                 save_path: Optional[str] = None,
                                 patch_size: int = 256,
                                 level: int = 0,
                                 title: Optional[str] = None,
                                 show_patches: bool = False,
                                 top_k: Optional[int] = None) -> plt.Figure:
        """
        Visualize attention heatmap for full-slide inference.
        
        Args:
            slide_path: Path to WSI or pre-extracted features
            coords: (N, 2) patch coordinates at level 0
            attention: (N,) attention scores for each patch
            save_path: Optional path to save figure
            patch_size: Patch size in pixels
            level: WSI level for visualization
            title: Custom title for plot
            show_patches: Draw patch boundaries
            top_k: Highlight top-k attended patches
            
        Returns:
            matplotlib Figure
        """
        # Load slide thumbnail
        thumbnail, slide_dims = self._load_thumbnail(slide_path, level)
        
        # Normalize attention to [0, 1]
        attn_norm = self._normalize_attention(attention)
        
        # Create heatmap
        heatmap = self._create_heatmap(
            coords, attn_norm, slide_dims, 
            patch_size, level, thumbnail.shape[:2]
        )
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Show thumbnail
        ax.imshow(thumbnail)
        
        # Overlay heatmap
        heatmap_overlay = self.cmap(heatmap)
        heatmap_overlay[..., 3] = heatmap * self.alpha  # Apply alpha
        ax.imshow(heatmap_overlay)
        
        # Optionally draw patch boundaries
        if show_patches:
            self._draw_patches(ax, coords, patch_size, level, 
                             slide_dims, thumbnail.shape[:2])
        
        # Optionally highlight top-k patches
        if top_k is not None:
            self._highlight_top_k(ax, coords, attn_norm, top_k, 
                                patch_size, level, slide_dims, thumbnail.shape[:2])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score', rotation=270, labelpad=20)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            slide_name = Path(slide_path).stem
            ax.set_title(f'Attention Heatmap: {slide_name}', 
                        fontsize=14, fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
    
    def visualize_chunked_attention(self,
                                    slide_path: str,
                                    coords: np.ndarray,
                                    chunk_attention: List[float],
                                    chunk_size: int,
                                    save_path: Optional[str] = None,
                                    patch_size: int = 256,
                                    level: int = 0,
                                    title: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention for chunked inference (chunk-level attention).
        
        Args:
            slide_path: Path to WSI
            coords: (N, 2) all patch coordinates
            chunk_attention: List of attention scores per chunk
            chunk_size: Number of patches per chunk
            save_path: Optional save path
            patch_size: Patch size in pixels
            level: WSI level
            title: Custom title
            
        Returns:
            matplotlib Figure
        """
        # Expand chunk attention to patch-level
        n_chunks = len(chunk_attention)
        n_patches = n_chunks * chunk_size
        
        # Truncate coords to match chunks
        coords_used = coords[:n_patches]
        
        # Expand chunk attention to patches
        patch_attention = np.repeat(chunk_attention, chunk_size)
        
        # Use full attention visualization
        return self.visualize_full_attention(
            slide_path, coords_used, patch_attention,
            save_path, patch_size, level, title
        )
    
    def visualize_detailed_attention(self,
                                     slide_path: str,
                                     coords: np.ndarray,
                                     attention: np.ndarray,
                                     prediction: Dict,
                                     save_path: Optional[str] = None,
                                     patch_size: int = 256,
                                     level: int = 0,
                                     top_k: int = 10) -> plt.Figure:
        """
        Create detailed visualization with statistics and top patches.
        
        Args:
            slide_path: Path to WSI
            coords: (N, 2) patch coordinates
            attention: (N,) attention scores
            prediction: Prediction dict from SlidePredictor
            save_path: Optional save path
            patch_size: Patch size
            level: WSI level
            top_k: Number of top patches to show
            
        Returns:
            matplotlib Figure with multiple subplots
        """
        # Load thumbnail
        thumbnail, slide_dims = self._load_thumbnail(slide_path, level)
        attn_norm = self._normalize_attention(attention)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main heatmap (large)
        ax_main = fig.add_subplot(gs[:, :2])
        heatmap = self._create_heatmap(coords, attn_norm, slide_dims,
                                      patch_size, level, thumbnail.shape[:2])
        ax_main.imshow(thumbnail)
        heatmap_overlay = self.cmap(heatmap)
        heatmap_overlay[..., 3] = heatmap * self.alpha
        ax_main.imshow(heatmap_overlay)
        
        # Highlight top-k
        self._highlight_top_k(ax_main, coords, attn_norm, top_k,
                            patch_size, level, slide_dims, thumbnail.shape[:2])
        
        slide_name = Path(slide_path).stem
        ax_main.set_title(f'Attention Heatmap: {slide_name}', 
                         fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score', rotation=270, labelpad=20)
        
        # 2. Attention distribution histogram
        ax_hist = fig.add_subplot(gs[0, 2])
        ax_hist.hist(attn_norm, bins=50, color='steelblue', edgecolor='black')
        ax_hist.axvline(attn_norm.mean(), color='red', linestyle='--', 
                       label=f'Mean: {attn_norm.mean():.3f}')
        ax_hist.axvline(np.median(attn_norm), color='green', linestyle='--',
                       label=f'Median: {np.median(attn_norm):.3f}')
        ax_hist.set_xlabel('Attention Score')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Attention Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # 3. Prediction info
        ax_info = fig.add_subplot(gs[1, 2])
        ax_info.axis('off')
        
        info_text = f"""
PREDICTION RESULTS
{'='*30}

Predicted Class: {prediction['predicted_class']}
Confidence: {prediction['confidence']:.2%}

Class Probabilities:
"""
        for cls, prob in prediction['probabilities'].items():
            bar = '█' * int(prob * 20)
            info_text += f"\n  {cls}: {prob:.3f} {bar}"
        
        info_text += f"""

ATTENTION STATISTICS
{'='*30}
Patches: {len(attention)}
Min: {attn_norm.min():.4f}
Max: {attn_norm.max():.4f}
Mean: {attn_norm.mean():.4f}
Std: {attn_norm.std():.4f}
Median: {np.median(attn_norm):.4f}

Top-{top_k} patches shown in yellow
"""
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=10, verticalalignment='top', 
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            # Create parent directories if they don't exist
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved detailed visualization to: {save_path}")
        
        return fig
    
    def extract_top_patches(self,
                           slide_path: str,
                           coords: np.ndarray,
                           attention: np.ndarray,
                           top_k: int = 10,
                           patch_size: int = 256,
                           level: int = 0,
                           save_dir: Optional[str] = None) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Extract and optionally save top-k attended patches.
        
        Args:
            slide_path: Path to WSI
            coords: (N, 2) patch coordinates
            attention: (N,) attention scores
            top_k: Number of top patches to extract
            patch_size: Patch size
            level: WSI level
            save_dir: Optional directory to save patches
            
        Returns:
            patches: List of PIL Images
            top_indices: Indices of top patches
        """
        # Get top-k indices
        top_indices = np.argsort(attention)[-top_k:][::-1]
        top_coords = coords[top_indices]
        top_scores = attention[top_indices]
        
        # Open slide
        slide = openslide.OpenSlide(slide_path)
        
        patches = []
        for i, (idx, (x, y), score) in enumerate(zip(top_indices, top_coords, top_scores)):
            # Read patch
            patch = slide.read_region(
                (int(x), int(y)), level, (patch_size, patch_size)
            ).convert('RGB')
            patches.append(patch)
            
            # Optionally save
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                slide_name = Path(slide_path).stem
                patch_path = save_dir / f"{slide_name}_patch_{i+1}_score_{score:.3f}.png"
                patch.save(patch_path)
        
        slide.close()
        
        return patches, top_indices
    
    def create_patch_grid(self,
                         patches: List[Image.Image],
                         scores: np.ndarray,
                         save_path: Optional[str] = None,
                         cols: int = 5) -> plt.Figure:
        """
        Create a grid visualization of patches with attention scores.
        
        Args:
            patches: List of PIL Images
            scores: Attention scores for each patch
            save_path: Optional save path
            cols: Number of columns in grid
            
        Returns:
            matplotlib Figure
        """
        n_patches = len(patches)
        rows = (n_patches + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten() if n_patches > 1 else [axes]
        
        for i, (patch, score) in enumerate(zip(patches, scores)):
            if i < len(axes):
                axes[i].imshow(patch)
                axes[i].set_title(f'Rank {i+1}\nScore: {score:.4f}', 
                                fontsize=10)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_patches, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Top Attended Patches', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved patch grid to: {save_path}")
        
        return fig
    
    # ==================== Helper Methods ====================
    
    def _load_thumbnail(self, slide_path: str, level: int) -> Tuple[np.ndarray, Tuple]:
        """Load WSI thumbnail or from pre-extracted features."""
        slide_path = Path(slide_path)
        
        if slide_path.suffix == '.h5':
            # Can't load thumbnail from h5, need original WSI
            raise ValueError(
                "Cannot generate thumbnail from .h5 file. "
                "Please provide path to original WSI."
            )
        elif slide_path.suffix in ['.pt', '.pth']:
            raise ValueError(
                "Cannot generate thumbnail from .pt file. "
                "Please provide path to original WSI."
            )
        else:
            # Load from WSI
            slide = openslide.OpenSlide(str(slide_path))
            dims = slide.level_dimensions[level]
            
            # Get thumbnail
            ratio = self.thumbnail_size / max(dims)
            thumb_size = tuple(int(d * ratio) for d in dims)
            thumbnail = slide.get_thumbnail(thumb_size)
            thumbnail = np.array(thumbnail)
            
            slide.close()
            return thumbnail, dims
    
    def _normalize_attention(self, attention: np.ndarray) -> np.ndarray:
        """Normalize attention scores to [0, 1]."""
        attn = np.array(attention).flatten()
        
        # Handle edge cases
        if len(attn) == 0:
            return attn
        
        attn_min = attn.min()
        attn_max = attn.max()
        
        if attn_max == attn_min:
            return np.ones_like(attn) * 0.5
        
        return (attn - attn_min) / (attn_max - attn_min)
    
    def _create_heatmap(self, 
                       coords: np.ndarray,
                       attention: np.ndarray,
                       slide_dims: Tuple,
                       patch_size: int,
                       level: int,
                       thumb_shape: Tuple) -> np.ndarray:
        """Create heatmap from patch coordinates and attention."""
        # Calculate scale factors
        scale_x = thumb_shape[1] / slide_dims[0]
        scale_y = thumb_shape[0] / slide_dims[1]
        
        # Create empty heatmap
        heatmap = np.zeros(thumb_shape[:2], dtype=np.float32)
        
        # Downsample for level
        downsample = 2 ** level
        scaled_patch_size = int(patch_size / downsample)
        
        for (x, y), score in zip(coords, attention):
            # Convert to thumbnail coordinates
            thumb_x = int(x * scale_x)
            thumb_y = int(y * scale_y)
            thumb_patch_size = int(scaled_patch_size * scale_x)
            
            # Draw patch with attention score
            y_end = min(thumb_y + thumb_patch_size, heatmap.shape[0])
            x_end = min(thumb_x + thumb_patch_size, heatmap.shape[1])
            
            if thumb_y < heatmap.shape[0] and thumb_x < heatmap.shape[1]:
                heatmap[thumb_y:y_end, thumb_x:x_end] = np.maximum(
                    heatmap[thumb_y:y_end, thumb_x:x_end], score
                )
        
        # Apply Gaussian blur for smoother appearance
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        return heatmap
    
    def _draw_patches(self, ax, coords, patch_size, level, 
                     slide_dims, thumb_shape):
        """Draw patch boundaries."""
        scale_x = thumb_shape[1] / slide_dims[0]
        scale_y = thumb_shape[0] / slide_dims[1]
        downsample = 2 ** level
        scaled_patch_size = int(patch_size / downsample)
        thumb_patch_size = int(scaled_patch_size * scale_x)
        
        for x, y in coords:
            thumb_x = int(x * scale_x)
            thumb_y = int(y * scale_y)
            
            rect = mpatches.Rectangle(
                (thumb_x, thumb_y), thumb_patch_size, thumb_patch_size,
                linewidth=0.5, edgecolor='white', facecolor='none', alpha=0.3
            )
            ax.add_patch(rect)
    
    def _highlight_top_k(self, ax, coords, attention, top_k,
                        patch_size, level, slide_dims, thumb_shape):
        """Highlight top-k attended patches."""
        top_indices = np.argsort(attention)[-top_k:]
        
        scale_x = thumb_shape[1] / slide_dims[0]
        scale_y = thumb_shape[0] / slide_dims[1]
        downsample = 2 ** level
        scaled_patch_size = int(patch_size / downsample)
        thumb_patch_size = int(scaled_patch_size * scale_x)
        
        for rank, idx in enumerate(reversed(top_indices)):
            x, y = coords[idx]
            thumb_x = int(x * scale_x)
            thumb_y = int(y * scale_y)
            
            rect = mpatches.Rectangle(
                (thumb_x, thumb_y), thumb_patch_size, thumb_patch_size,
                linewidth=3, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add rank number
            ax.text(thumb_x + 5, thumb_y + 15, str(rank + 1),
                   color='yellow', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


def visualize_prediction_with_attention(
    slide_path: str,
    model_path: str,
    model_class,
    class_names: List[str],
    save_dir: Optional[str] = None,
    mode: str = 'full',
    chunk_size: int = 1000,
    patch_size: int = 256,
    level: int = 0,
    top_k: int = 10,
    device: str = 'cuda'
) -> Dict:
    """
    Complete pipeline: predict + visualize attention.
    
    Args:
        slide_path: Path to WSI or features
        model_path: Path to trained model
        model_class: Model factory
        class_names: List of class names
        save_dir: Directory to save visualizations
        mode: 'full' or 'chunked'
        chunk_size: Chunk size for chunked mode
        patch_size: Patch size
        level: WSI level
        top_k: Number of top patches to highlight
        device: Device for inference
        
    Returns:
        dict with prediction results and visualization paths
    """
    from inference import SlidePredictor
    
    # Create predictor
    predictor = SlidePredictor(
        model_path=model_path,
        model_class=model_class,
        class_names=class_names,
        device=device
    )
    
    # Load features and coords
    slide_path = Path(slide_path)
    if slide_path.suffix == '.h5':
        with h5py.File(slide_path, 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()
            coords = f['coords'][:] if 'coords' in f else None
    elif slide_path.suffix in ['.pt', '.pth']:
        data = torch.load(slide_path)
        features = data['features'] if isinstance(data, dict) else data
        coords = data.get('coords', None) if isinstance(data, dict) else None
    else:
        raise ValueError("For visualization, provide pre-extracted features (.h5 or .pt)")
    
    if coords is None:
        raise ValueError("Coordinates not found in features file. Cannot visualize.")
    
    # Run prediction
    result = predictor.predict(slide_path, mode=mode, chunk_size=chunk_size)
    
    # Get attention weights
    if mode == 'full':
        # Extract attention from full mode
        attn = result.get('attention_weights')
        if attn is None:
            raise ValueError("No attention weights returned from model")
        
        # Convert to numpy: (1, num_queries, N) -> (N,)
        attention = attn[0].mean(dim=0).cpu().numpy()
    else:
        # Use chunk attention
        attention = result.get('chunk_attention')
        if attention is None:
            raise ValueError("No chunk attention returned")
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Setup save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        slide_name = slide_path.stem
    
    # Generate visualizations
    print("\nGenerating attention visualizations...")
    
    # 1. Detailed visualization
    save_path = save_dir / f"{slide_name}_detailed.png" if save_dir else None
    fig_detailed = visualizer.visualize_detailed_attention(
        str(slide_path).replace('.h5', '.svs').replace('.pt', '.svs'),
        coords, attention, result,
        save_path=save_path, patch_size=patch_size, level=level, top_k=top_k
    )
    
    # 2. Extract and save top patches
    if save_dir:
        patch_dir = save_dir / f"{slide_name}_top_patches"
        patches, top_indices = visualizer.extract_top_patches(
            str(slide_path).replace('.h5', '.svs').replace('.pt', '.svs'),
            coords, attention, top_k=top_k,
            patch_size=patch_size, level=level, save_dir=patch_dir
        )
        
        # 3. Create patch grid
        save_path = save_dir / f"{slide_name}_patch_grid.png"
        fig_grid = visualizer.create_patch_grid(
            patches, attention[top_indices], save_path=save_path
        )
    
    return {
        'prediction': result,
        'attention': attention,
        'top_patches': top_indices if save_dir else None,
        'visualizations_dir': save_dir
    }
