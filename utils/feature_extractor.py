"""Fast feature extraction from raw WSI files with tissue detection"""
import torch
import torch.nn as nn
import numpy as np
import openslide
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import warnings


class TissueDetector:
    """Fast tissue detection using Otsu thresholding on grayscale thumbnail."""
    
    def __init__(self, 
                 thumbnail_size: int = 2048,
                 min_tissue_area: int = 100,
                 kernel_size: int = 7):
        """
        Args:
            thumbnail_size: Size for thumbnail generation
            min_tissue_area: Minimum contour area to consider as tissue (pixels)
            kernel_size: Morphological operations kernel size
        """
        self.thumbnail_size = thumbnail_size
        self.min_tissue_area = min_tissue_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (kernel_size, kernel_size))
    
    def detect(self, slide: openslide.OpenSlide, level: int = 0) -> np.ndarray:
        """
        Detect tissue regions in WSI.
        
        Returns:
            tissue_mask: Binary mask at thumbnail resolution (H, W)
        """
        # Get thumbnail
        dims = slide.level_dimensions[level]
        ratio = self.thumbnail_size / max(dims)
        thumb_size = tuple(int(d * ratio) for d in dims)
        thumbnail = slide.get_thumbnail(thumb_size)
        
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        
        # Remove small objects
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_tissue_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask


class PatchExtractor:
    """Extract patch coordinates from tissue mask."""
    
    def __init__(self,
                 patch_size: int = 256,
                 stride: int = 256,
                 tissue_threshold: float = 0.5):
        """
        Args:
            patch_size: Size of patches to extract at target level
            stride: Stride between patches (set < patch_size for overlap)
            tissue_threshold: Minimum fraction of tissue pixels in patch
        """
        self.patch_size = patch_size
        self.stride = stride
        self.tissue_threshold = tissue_threshold
    
    def extract_coords(self, 
                       slide: openslide.OpenSlide,
                       tissue_mask: np.ndarray,
                       level: int = 0,
                       target_level: int = 0) -> np.ndarray:
        """
        Extract patch coordinates from tissue mask.
        
        Args:
            slide: OpenSlide object
            tissue_mask: Binary tissue mask at thumbnail resolution
            level: Level at which tissue mask was generated
            target_level: Level to extract patches from
            
        Returns:
            coords: (N, 2) array of (x, y) coordinates at level 0
        """
        # Get dimensions and scaling factors
        mask_h, mask_w = tissue_mask.shape
        slide_w, slide_h = slide.level_dimensions[level]
        
        # Scale factors
        scale_x = slide_w / mask_w
        scale_y = slide_h / mask_h
        
        # Downsample factor between target_level and level 0
        downsample = slide.level_downsamples[target_level]
        
        # Patch size in mask coordinates
        patch_size_mask = int(self.patch_size / scale_x / downsample)
        stride_mask = int(self.stride / scale_x / downsample)
        
        coords = []
        
        # Slide over tissue mask
        for y in range(0, mask_h - patch_size_mask + 1, stride_mask):
            for x in range(0, mask_w - patch_size_mask + 1, stride_mask):
                # Check tissue content
                patch_mask = tissue_mask[y:y+patch_size_mask, x:x+patch_size_mask]
                tissue_ratio = np.sum(patch_mask > 0) / patch_mask.size
                
                if tissue_ratio >= self.tissue_threshold:
                    # Convert to level 0 coordinates
                    x_level0 = int(x * scale_x)
                    y_level0 = int(y * scale_y)
                    coords.append([x_level0, y_level0])
        
        return np.array(coords) if coords else np.zeros((0, 2))


class FeatureExtractor:
    """Extract features using pretrained backbones."""
    
    SUPPORTED_MODELS = {
        'resnet50': {
            'output_dim': 1024,
            'input_size': 224,
            'requires_init': True
        },
        'phikon-v2': {
            'output_dim': 1024,
            'input_size': 224,
            'requires_init': False,
            'model_path': 'owkin/phikon-v2'
        },
        'uni': {
            'output_dim': 1024,
            'input_size': 224,
            'requires_init': False,
            'model_path': 'mahmoodlab/UNI'
        }
    }
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 device: str = 'cuda',
                 batch_size: int = 64):
        """
        Args:
            backbone: One of 'resnet50', 'phikon', 'uni'
            device: Device to run inference on
            batch_size: Batch size for feature extraction
        """
        if backbone not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        self.backbone_name = backbone
        self.device = device
        self.batch_size = batch_size
        self.config = self.SUPPORTED_MODELS[backbone]
        
        # Load model
        print(f"Loading {backbone} backbone...")
        self.model = self._load_model()
        self.model.eval()
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"✓ {backbone} loaded, output dim: {self.config['output_dim']}")
    
    def _load_model(self) -> nn.Module:
        """Load the specified backbone model."""
        if self.backbone_name == 'resnet50':
            # Truncated ResNet50 (remove final FC layer)
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-3],
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(1))
            # Add projection to 1024 if needed (ResNet50 outputs 2048)
            # model = nn.Sequential(model, nn.Flatten(), nn.Linear(2048, 1024))
            return model.to(self.device)


        elif self.backbone_name == 'phikon-v2':
            try:
                from transformers import AutoModel
                # Use AutoModel to automatically detect the correct architecture
                model = AutoModel.from_pretrained(
                    self.config['model_path'],
                    trust_remote_code=True  # May be needed for custom models
                )

                class PhikonWrapper(nn.Module):
                    def __init__(self, vit_model):
                        super().__init__()
                        self.vit = vit_model

                    def forward(self, x):
                        outputs = self.vit(pixel_values=x)
                        # Use CLS token from last_hidden_state
                        features = outputs.last_hidden_state[:, 0]
                        return features

                return PhikonWrapper(model).to(self.device)
            except ImportError:
                raise ImportError("transformers library required for Phikon. "
                                  "Install: uv add transformers")
        
        elif self.backbone_name == 'uni':
            try:
                import timm
                model = timm.create_model(
                    "vit_large_patch16_224", 
                    img_size=224, 
                    patch_size=16, 
                    init_values=1e-5, 
                    num_classes=0,  # Remove classification head
                    dynamic_img_size=True
                )
                # Load pretrained weights from HuggingFace
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(
                    "MahmoodLab/UNI", 
                    filename="pytorch_model.bin"
                )
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=True)
                return model.to(self.device)
            except ImportError as e:
                raise ImportError(f"Required libraries for UNI: {e}. "
                                "Install: pip install timm huggingface_hub")
    
    def _get_transforms(self) -> T.Compose:
        """Get image preprocessing transforms."""
        size = self.config['input_size']
        
        if self.backbone_name in ['phikon', 'uni']:
            # ViT-based MILS typically use different normalization
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            # ResNet ImageNet normalization
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            normalize
        ])
    
    @torch.no_grad()
    def extract_features(self, 
                        slide: openslide.OpenSlide,
                        coords: np.ndarray,
                        level: int = 0,
                        patch_size: int = 256) -> torch.Tensor:
        """
        Extract features from patches at given coordinates.
        
        Args:
            slide: OpenSlide object
            coords: (N, 2) array of (x, y) coordinates at level 0
            level: Level to read patches from
            patch_size: Size of patches to extract
            
        Returns:
            features: (N, feature_dim) tensor
        """
        if len(coords) == 0:
            return torch.zeros((0, self.config['output_dim']))
        
        all_features = []
        n_batches = (len(coords) + self.batch_size - 1) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(coords))
            batch_coords = coords[start_idx:end_idx]
            
            # Read patches
            patches = []
            for x, y in batch_coords:
                try:
                    patch = slide.read_region(
                        (int(x), int(y)), 
                        level, 
                        (patch_size, patch_size)
                    ).convert('RGB')
                    patches.append(self.transform(patch))
                except Exception as e:
                    warnings.warn(f"Error reading patch at ({x}, {y}): {e}")
                    # Use blank patch as fallback
                    blank = Image.new('RGB', (patch_size, patch_size), (255, 255, 255))
                    patches.append(self.transform(blank))
            
            # Stack and extract features
            if patches:
                batch = torch.stack(patches).to(self.device)
                features = self.model(batch)
                all_features.append(features.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {end_idx}/{len(coords)} patches...")
        
        return torch.cat(all_features, dim=0) if all_features else torch.zeros((0, self.config['output_dim']))


class WSIFeatureExtractor:
    """Complete pipeline: WSI → Features with coordinates."""
    
    def __init__(self,
                 backbone: str = 'resnet50',
                 patch_size: int = 256,
                 stride: int = 256,
                 level: int = 0,
                 tissue_threshold: float = 0.5,
                 batch_size: int = 64,
                 device: str = 'cuda'):
        """
        Complete WSI feature extraction pipeline.
        
        Args:
            backbone: Feature extraction backbone ('resnet50', 'phikon', 'uni')
            patch_size: Patch size in pixels
            stride: Stride between patches
            level: Magnification level to extract from (0 = highest)
            tissue_threshold: Minimum tissue ratio in patch
            batch_size: Batch size for feature extraction
            device: Device for computation
        """
        self.patch_size = patch_size
        self.level = level
        
        self.tissue_detector = TissueDetector()
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            stride=stride,
            tissue_threshold=tissue_threshold
        )
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            device=device,
            batch_size=batch_size
        )
    
    def process(self, 
                wsi_path: str,
                save_path: Optional[str] = None) -> dict:
        """
        Process WSI and extract features.
        
        Args:
            wsi_path: Path to WSI file (.svs, .tif, etc.)
            save_path: Optional path to save features (.pt or .h5)
            
        Returns:
            dict with keys:
                - 'features': (N, 1024) tensor
                - 'coords': (N, 2) array
                - 'metadata': dict with processing info
        """
        print(f"\nProcessing: {Path(wsi_path).name}")
        
        # Open slide
        slide = openslide.OpenSlide(wsi_path)
        print(f"  Dimensions: {slide.dimensions}")
        print(f"  Level count: {slide.level_count}")
        
        # Step 1: Tissue detection
        print("  1. Detecting tissue...")
        tissue_mask = self.tissue_detector.detect(slide, level=self.level)
        tissue_ratio = np.sum(tissue_mask > 0) / tissue_mask.size
        print(f"     Tissue ratio: {tissue_ratio:.2%}")
        
        # Step 2: Extract coordinates
        print("  2. Extracting patch coordinates...")
        coords = self.patch_extractor.extract_coords(
            slide, tissue_mask, 
            level=self.level,
            target_level=self.level
        )
        print(f"     Extracted {len(coords)} patches")
        
        if len(coords) == 0:
            warnings.warn("No tissue patches found!")
            return {
                'features': torch.zeros((0, 1024)),
                'coords': np.zeros((0, 2)),
                'metadata': {'num_patches': 0}
            }
        
        # Step 3: Extract features
        print("  3. Extracting features...")
        features = self.feature_extractor.extract_features(
            slide, coords, 
            level=self.level,
            patch_size=self.patch_size
        )
        
        slide.close()
        
        result = {
            'features': features,
            'coords': coords,
            'metadata': {
                'wsi_path': str(wsi_path),
                'num_patches': len(coords),
                'patch_size': self.patch_size,
                'level': self.level,
                'tissue_ratio': float(tissue_ratio),
                'backbone': self.feature_extractor.backbone_name
            }
        }
        
        # Save if requested
        if save_path:
            self._save_features(result, save_path)
        
        print(f"  ✓ Complete: {len(coords)} patches, {features.shape[1]}-dim features")
        return result
    
    def _save_features(self, result: dict, save_path: str):
        """Save features to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.h5':
            import h5py
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('features', data=result['features'].numpy())
                f.create_dataset('coords', data=result['coords'])
                for key, val in result['metadata'].items():
                    f.attrs[key] = val
            print(f"  Saved to: {save_path}")
        
        elif save_path.suffix in ['.pt', '.pth']:
            torch.save(result, save_path)
            print(f"  Saved to: {save_path}")
        
        else:
            raise ValueError(f"Unsupported save format: {save_path.suffix}")
