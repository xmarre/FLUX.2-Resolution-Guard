# FLUX.2 Resolution Guard

A research-first correction module and ComfyUI node pack for **FLUX.2 high-resolution drift control**.

This repository is built around one practical observation:

- FLUX.2 edits and generations can become less geometry-stable and less color-stable as effective resolution climbs.
- The symptoms often travel together: slight outward "breathing"/expansion, low-frequency relighting, and washed-out chroma/contrast.
- The problem is not exclusive to inpainting or crop-detailing. It can affect **whole-image FLUX.2 outputs** as well.
- A useful correction strategy is to treat a lower-resolution FLUX.2 result as a more stable anchor manifold, then pull the high-resolution output back toward that manifold in a controlled way.

This repo implements that strategy in two forms:

1. **A trainable correction model** called **SMIC**  
   Stable-Manifold Inward Compander

2. **A training-free analytic fallback**
   usable immediately in ComfyUI before trained weights exist.

The code is deliberately written so the same machinery can be used for:

- whole-image FLUX.2 outputs
- masked regions / FaceDetailer style crops
- anchor-conditioned correction
- anchorless correction using a learned prior or analytic companding

---

## What this repo contains

### Core package

`src/flux2_resolution_guard/`

- `models/smic.py`  
  Small correction network with:
  - low-rank warp head
  - RGB residual head
  - gate/confidence head

- `data/synthetic.py`  
  Synthetic pretraining dataset that creates:
  - outward radial drift
  - anisotropic expansion
  - low-frequency washout
  - mild seam stress
  - whole-image or masked perturbations

- `data/triplets.py`  
  Dataset loader for FLUX.2 self-distillation triplets:
  - original
  - anchor (stable lower-resolution FLUX.2 pass)
  - highres (problematic FLUX.2 pass)
  - mask
  - metadata

- `training/engine.py`  
  Complete training loop with:
  - pixel loss
  - low-frequency loss
  - seam loss
  - OKLab low-frequency loss
  - identity/no-op loss
  - warp smoothness regularization

- `inference/image.py`  
  High-level inference API:
  - load checkpoint
  - correct whole image
  - correct masked image
  - analytic fallback correction

### ComfyUI node pack

Repo root contains a working custom-node package:

- `__init__.py`
- `nodes.py`

Nodes included:

- **FLUX2 RG Load Model**
- **FLUX2 RG Apply Correction**
- **FLUX2 RG Analytic Compand**

These nodes work on standard ComfyUI `IMAGE` and `MASK` types.

### Scripts

- `scripts/train_synthetic.py`
- `scripts/train_triplets.py`

### Tests

- `tests/test_model.py`
- `tests/test_inference.py`
- `tests/test_datasets.py`

---

## Design goals

This project is aimed at **FLUX.2 specifically**, but not limited to one workflow shape.

It is designed to address resolution-linked drift in:

- full-frame FLUX.2 generations
- edited full-frame FLUX.2 outputs
- inpainted/masked regions
- FaceDetailer / crop-and-stitch workflows

The implementation starts in the **image domain** on purpose.

Why?

- It is robust and immediately usable.
- It avoids hard-coding brittle assumptions about specific FLUX.2 runtime internals.
- It still supports FLUX.2-specific training by using **FLUX.2 triplets** as supervision.
- It can later be extended to latent-side correction while preserving the same public interfaces.

---

## Stable-Manifold Inward Companding

The core idea is simple:

1. Use a lower-resolution FLUX.2 result as a more stable low-frequency anchor.
2. Use the high-resolution FLUX.2 result for detail.
3. Predict a smooth inward/companding warp and a residual correction.
4. Restore low-frequency geometry and color stability without erasing high-frequency detail.

Mathematically, the model predicts:

- a low-rank warp field
- an RGB residual
- a confidence gate

The output is:

```text
corrected = warped(edit) + gate * residual
```

with strong regularization to keep the warp broad and controlled.

---

## Installation

### As a Python package

```bash
git clone https://github.com/yourname/flux2-resolution-guard.git
cd flux2-resolution-guard
pip install -e .
```

### For ComfyUI

Clone or copy this repo into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/flux2-resolution-guard.git
```

Restart ComfyUI.

---

## Minimal Python inference example

```python
from PIL import Image
from flux2_resolution_guard.inference.image import correct_image_with_checkpoint

edited = Image.open("highres.png").convert("RGB")
anchor = Image.open("anchor_1mp.png").convert("RGB")

corrected = correct_image_with_checkpoint(
    image=edited,
    checkpoint_path="checkpoints/smic_best.pt",
    anchor_image=anchor,
    mp_ratio=1.8,
    strength=1.0,
)
corrected.save("corrected.png")
```

### Analytic fallback example

```python
from PIL import Image
from flux2_resolution_guard.inference.image import analytic_compand_correction

edited = Image.open("highres.png").convert("RGB")
anchor = Image.open("anchor_1mp.png").convert("RGB")

corrected = analytic_compand_correction(
    image=edited,
    anchor_image=anchor,
    mp_ratio=1.8,
    strength=0.7,
)
corrected.save("analytic_corrected.png")
```

---

## Training

### 1. Synthetic pretraining

Builds a correction prior from synthetic perturbations.

```bash
python scripts/train_synthetic.py \
  --image-dir /path/to/images \
  --output-dir runs/synthetic \
  --epochs 20 \
  --batch-size 4
```

### 2. FLUX.2 triplet training

Train on real FLUX.2 captures with a manifest file.

```bash
python scripts/train_triplets.py \
  --manifest data/flux2_triplets.json \
  --output-dir runs/flux2_triplets \
  --epochs 10 \
  --batch-size 2
```

Manifest format:

```json
[
  {
    "original": "data/originals/example.png",
    "anchor": "data/anchors/example_anchor.png",
    "highres": "data/highres/example_highres.png",
    "mask": "data/masks/example_mask.png",
    "mp_ratio": 1.85
  }
]
```

---

## ComfyUI nodes

### FLUX2 RG Load Model

Inputs:
- `checkpoint_path`
- `device`

Returns:
- `RG_MODEL`

### FLUX2 RG Apply Correction

Inputs:
- `model`
- `image`
- `anchor_image` (optional in practice; pass same image if unavailable)
- `mask` (optional; defaults to full image)
- `mp_ratio`
- `strength`

Returns:
- corrected `IMAGE`

### FLUX2 RG Analytic Compand

Inputs:
- `image`
- `anchor_image`
- `mask`
- `mp_ratio`
- `strength`
- `blur_sigma`

Returns:
- corrected `IMAGE`

---

## Notes on current scope

This repository gives you:

- a real trainable codebase
- a real analytic fallback
- a real ComfyUI node pack
- tests and working interfaces

It does **not** ship with pretrained weights.

For best real-world performance you should train on your own FLUX.2 captures, especially:
- whole-frame pairs across 1MP → highres
- face crops
- portrait work
- the exact denoise/settings you care about

---

## Suggested next experiments

- Add latent-domain backend while keeping current API stable
- Export trained checkpoints to `.safetensors`
- Add face-landmark alignment losses
- Add optional CLIP/identity preservation losses
- Add whole-image low-frequency anchor blending directly inside the model

---

## License

MIT
