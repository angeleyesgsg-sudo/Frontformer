# Frontformer

FrontFormer is a multivariate Deformable DETR-based transformer for automated cold- and warm-front detection and segmentation. It combines variable-wise embedding and channel-attention feature fusion with multi-scale deformable attention and a pixel decoder to directly predict frontal categories, bounding boxes, and pixel-level masks.

## Project structure

```text
Frontformer/
├── requirements.txt
├── README.md
└── models/
    ├── backbone.py
    ├── common.py
    ├── transformer.py
    ├── pixel_decoder.py
    └── model.py
```

## Dependency

The deformable-attention operator is provided by the official Deformable-DETR implementation. Clone and compile Deformable-DETR, then set its path before importing Frontformer.

Linux/macOS:

```bash
export DEFORMABLE_DETR_PATH=/path/to/Deformable-DETR
```

Windows PowerShell:

```powershell
$env:DEFORMABLE_DETR_PATH = "D:\path\to\Deformable-DETR"
```

Install the Python dependency with:

```bash
pip install -r requirements.txt
```
