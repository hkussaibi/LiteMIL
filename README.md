# LiteMIL: A Computationally Efficient Cross-Attention Multiple Instance Learning for Cancer Subtyping on whole-slide images

A lightweight transformer-style cross-attention MIL network for whole slide image classification achieving competitive performance with transformer-based methods while requiring fewer parameters, faster inference, and lower GPU memory.

[![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Medical%20Imaging-blue)](https://doi.org/10.1117/1.JMI.13.1.017501)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
---

## 🎯 Overview

LiteMIL uses learnable query-based multi-head cross-attention to aggregate patch-level features from whole slide images (WSIs) into slide-level predictions.

### Key Features

- ✅ **Query-based Multi-head cross-attention** aggregation with configurable number of queries
- ✅ **Multiple backbones support**: ResNet50, Phikon v2, UNI for feature extraction
- ✅ Support for `.h5`, `.pt`, and `.pth` feature formats
- ✅ **Chunked data loading** for memory-efficient training
- ✅ **Nested cross-validation** with comprehensive metrics
- ✅ **End-to-end inference pipeline**: Raw WSI → Features → Prediction → Visualization
- ✅ **Attention visualization**: Interpretable heatmaps and top-patch extraction
- ✅ **CLI and Jupyter notebook** interfaces
---

## 📦 Installation

### Prerequisites
- **Python 3.13+**
- **[uv](https://github.com/astral-sh/uv)** package manager (recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and setup
git clone https://github.com/hkussaibi/LiteMIL.git
cd LiteMIL
uv sync

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```
---

## 📊 Dataset Preparation

### Directory Structure

```
LiteMIL/
├── datasets/
│   ├── breast/
│   │   ├── wsi/              # .svs files/
│   │   ├── features/          # .h5, .pt, or .pth files
│   │   └── labels.csv
│   ├── ...
```

### Supported Feature File Formats

HDF5 (`.h5`), PyTorch (`.pt`, `.pth`)
### Labels CSV Format

Your `labels.csv` should contain:

- **`case_id`**: Patient/case identifier (for patient-level splitting in CV)
- **`wsi_id`**: Unique slide identifier (must match feature filename without extension)
- **`label`**: Class label (string)
---
### Feature Extraction from Raw WSI

```bash
# Single file
python extractFeatures.py --input testSlides/testSlide.svs --backbone resnet50 --patch_size 256 --stride 256

# Batch processing
python extractFeatures.py --input_dir testSlides/ --output_dir features/ --backbone phikon-v2 --patch_size 256 --level 0 --format h5
```

**Supported Backbones:**

| Backbone | Description | Output Dim | Best For |
|----------|-------------|------------|----------|
| **ResNet50** | ImageNet pretrained, fast | 1024 | Quick prototyping, limited compute |
| **Phikon v2** | 450M pathology images | 1024 | Pathology-specific features |
| **UNI** | 100M+ pathology images | 1024 | Maximum performance |

---
### Pre-extracted Features

You may download pre-extracted features for 4 TCGA datasets from:
- [TCGA Pre-extracted Features (Zenodo)](https://zenodo.org/records/10563985)

**Supported Datasets:**
- **TCGA-BRCA** (Breast): 875 WSIs, 2 classes (IDC vs ILC)
- **TCGA-Kidney**: 906 WSIs, 3 classes (PRCC, CCRCC, CHRCC)
- **TCGA-Lung**: 958 WSIs, 2 classes (LUAD vs LUSC)
- **TUPAC16**: 821 WSIs, 2 classes (Low vs High grade)

Ground truth labels available in: [`datasets/<dataset>/labels.csv`](datasets/breast/labels.csv)

---
### Dataset Loading Modes

**Chunked Mode** (Recommended):
- Splits each slide into fixed-size chunks (e.g., 1000 instances)
- Uniform batch sizes → stable training
- Lower memory footprint
- Better for attention-based models

**Full Mode**:
- Loads entire slide as single bag (variable length)
- Requires more memory
- Better for small slides or models that need global context

---
## 🔧 Model Configuration

### LiteMIL Architecture

```python
from MILS.LiteMIL import LiteMIL

model = LiteMIL(
    input_dim=1024,      # Input feature dimension
    hidden_dim=256,      # Hidden layer dimension
    num_classes=2,       # Number of output classes
    num_heads=4,         # Number of attention heads
    num_queries=1,       # Number of learnable queries (1 or 4)
    dropout=0.25         # Dropout rate
)
```

**Recommended Configurations:**

| Task Complexity | num_queries | hidden_dim | num_heads |
|-----------------|-------------|------------|-----------|
| Simple (2-class) | 1 | 256 | 4 |
| Medium (3-class) | 1-4 | 256-512 | 4-8 |
| Complex (4+ class) | 4 | 512 | 8 |

---

## 📈 Training on Pre-extracted Features with Nested Cross-Validation

```bash
# Train on breast cancer dataset with default settings
python train.py --mil LiteMIL --dataset breast --mode chunked --epochs 1

# Train with custom architecture
python train.py --mil LiteMIL --dataset breast --num_queries 4 --hidden_dim 256 --dropout 0.25 --n_outer 5 --n_inner 4 --epochs 1

# Train with full-slide mode (if memory allows)
python train.py --mil LiteMIL --dataset breast --mode full --batch_size 16 --epochs 1
```

**Key Training Arguments:**
- `--mil`: `LiteMIL`, `ABMIL`, `ABMIL_Multihead`, `TransMIL`,  `meanPool`, or `maxPool` 
- `--dataset`: `breast`, `lung`, `kidney`, or `tupac`
- `--mode`: `chunked` (fixed-size chunks) or `full` (variable-length bags)
- `--num_queries`: Number of learnable queries (1 recommended, 4 for complex tasks)
- `--hidden_dim`: Hidden dimension (default: 256)
- `--num_heads`: Number of attention heads (default: 4)
- `--epochs`: Maximum training epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)

### Training Outputs

Results are saved in `outputs/<mil>/<dataset>/`:

```
outputs/LiteMIL/breast/
├── all_folds_best.pth          # Best model across all folds
├── fold_1_best.pth             # Best model for fold 1
├── fold_2_best.pth
├── ...
└── outputs.json                # Aggregated metrics
```

### Metrics

**outputs.json** contains:
- **Chunk-level Accuracy**: Classification accuracy on chunks
- **Slide-level Accuracy**: Classification accuracy after aggregation
- **F1 Score**: Macro-averaged F1
- **AUC-ROC**: Area under ROC curve

---

## 🏗️ Inference
### Inference on Pre-extracted Features

```bash
# Basic inference
python predict.py --mil LiteMIL --dataset breast --checkpoint outputs/LiteMIL/breast/all_folds_best.pth --input testSlides/testSlide.h5

# Inference with chunked mode (for large slides)
python predict.py --mil LiteMIL --dataset breast --checkpoint outputs/LiteMIL/breast/all_folds_best.pth --input testSlides/testSlide.h5 --mode chunked --chunk_size 1000
```

### Inference on Raw WSI Files

```bash
# Direct inference from SVS file with on-the-fly feature extraction
python predict.py --mil LiteMIL --dataset breast --checkpoint outputs/LiteMIL/breast/all_folds_best.pth --input testSlides/testSlide.svs --backbone resnet50 --patch_size 256
```

## 📖 Attention Visualization

Generate interpretable attention heatmaps:

```bash
# Basic visualization
python predict.py --visualize --mil LiteMIL --dataset breast --checkpoint outputs/LiteMIL/breast/all_folds_best.pth --input testSlides/testSlide.h5 --wsi testSlides/testSlide.svs --output outputs/heatmaps/

# With custom parameters
python predict.py --visualize --mil LiteMIL --dataset breast --checkpoint outputs/LiteMIL/breast/all_folds_best.pth --input testSlides/testSlide.h5 --wsi testSlides/testSlide.h5 --mode chunked --top_k 20 --cmap hot --output outputs/heatmaps/
```

**Visualization Output:**
1. **`slide_detailed.png`** - Comprehensive view with heatmap, histogram, and stats
2. **`slide_heatmap.png`** - Clean attention overlay
3. **`slide_patch_grid.png`** - Grid of top-k patches
4. **`slide_top_patches/`** - Individual high-attention patches
5. **`slide_attention.npz`** - Raw attention data

## Jupyter Notebook

See [`Tutorial.ipynb`](Tutorial.ipynb) for interactive examples including:
- Custom dataset loading
- Feature extraction from raw WSI
- Model training and evaluation
- Attention visualization
- Performance analysis

---

## 🔧 Advanced Usage

### Speed Optimization

```python
# 1. Lower magnification (faster, less detail)
extractor = WSIFeatureExtractor(level=1)  # 20x instead of 40x

# 2. Increase batch size (if GPU memory allows)
extractor = WSIFeatureExtractor(batch_size=128)  # Default: 64

# 3. Increase stride (less overlap, faster)
extractor = WSIFeatureExtractor(stride=512, patch_size=256)
```

### Memory Management

```python
# Solution 1: Reduce batch size
cv = NestedCrossValidation(..., batch_size=8)

# Solution 2: Use chunked mode with smaller chunks
result = predictor.predict('slide.h5', mode='chunked', chunk_size=500)

# Solution 3: Lower magnification for feature extraction
extractor = WSIFeatureExtractor(level=1)
```
---

## 📖 Best Practices

### For Feature Extraction
1. ✅ Always validate on a small subset first
2. ✅ Save intermediate features to avoid re-extraction
3. ✅ Use appropriate backbone for your domain (Phikon/UNI for pathology)
4. ✅ Monitor GPU memory when processing large slides
5. ✅ Verify tissue detection on representative samples

### For Training
1. ✅ Use `chunked` mode for consistent batch sizes
2. ✅ Start with `num_queries=1` for simplicity
3. ✅ Enable mixed precision (`use_amp=True`) to save memory
4. ✅ Monitor both chunk-level and slide-level accuracy
5. ✅ Use patient-level splitting (via `case_id`) to avoid data leakage

### For Inference
1. ✅ Use `full` mode for slides with <10K patches
2. ✅ Use `chunked` mode for slides with >10K patches
3. ✅ Validate predictions with attention heatmaps
4. ✅ Check attention statistics for model confidence

### For Visualization
1. ✅ Use colorblind-friendly colormaps (`viridis`, `plasma`) for publications
2. ✅ Include scale bar or patch size in captions
3. ✅ Show multiple examples (correct, incorrect, uncertain)
4. ✅ Validate attention patterns with domain experts
5. ✅ Save attention data (`.npz`) for reproducibility

---

## 📄 Citation

If you use LiteMIL pipeline in your research, please cite:

```bibtex
@article{Kussaibi2026LiteMIL,
title = {LiteMIL: a computationally efficient cross-attention multiple instance learning for cancer subtyping on whole-slide images},
author = {Kussaibi, Haitham},
journal = {Journal of Medical Imaging},
year = {2026},
volume = {13},
number = {1},
pages = {017501},
doi = {10.1117/1.JMI.13.1.017501}
}
```

---

## 📜 License
Copyright© 2025 Haitham Kussaibi

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TCGA Research Network** for providing datasets
- **MAD-MIL study** for pre-extracted features ([Zenodo](https://zenodo.org/records/10563985))

---

## 📧 Corresponding Author

**Dr. Haitham Kussaibi**  
- 📧 Email: kussaibi@gmail.com
- 🔗 ORCID: [0000-0002-9570-0768](https://orcid.org/0000-0002-9570-0768)
- 💼 LinkedIn: [linkedin.com/in/haithamkussaibi](https://www.linkedin.com/in/haithamkussaibi/)

---

## ⭐ Star History

If you find LiteMIL useful for your research, please consider giving it a star! ⭐

---

**Made with ❤️ for the computational pathology community**
