# ComfyUI-PuLID-Flux2

[![GitHub stars](https://img.shields.io/github/stars/iFayens/ComfyUI-PuLID-Flux2?style=social)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Flux.2](https://img.shields.io/badge/Flux.2-Klein%20%26%20Dev-green)](https://huggingface.co/black-forest-labs)
[![Version](https://img.shields.io/badge/version-0.3.0-orange)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)

> **First PuLID implementation for FLUX.2 — supports Klein (4B/9B) and Dev (32B)**  
> Auto model detection — just plug and play  
> March 2026

---

## ⚠️ Current State & Honest Assessment

| | Status | Notes |
|---|---|---|
| Node loads and runs | ✅ | All 5 nodes functional |
| InsightFace face detection | ✅ | AntelopeV2 + buffalo_l fallback |
| EVA-CLIP encoding | ✅ | via open_clip |
| Auto model detection | ✅ | Klein 4B / 9B / Dev auto-detected |
| Flux.2 Klein support | ✅ | Best results |
| Flux.2 Dev support | ✅ | Works |
| Green image artifact | ✅ Fixed | Auto-cleanup between runs |
| Half vs BFloat16 crash | ✅ Fixed v0.3.0 | Dynamic cast in PerceiverCA |
| Single blocks injection | ✅ Fixed v0.3.0 | Was only 50% before — now 100% |
| Native Klein/Dev weights | ❌ Not yet | Training scripts included |

**Best results:** Use `weight: 0.6` with **`end_at: 0.5`** to prevent darkening while maintaining strong identity (see Recommended Parameters).

---

## 🎯 What is this?

This custom node brings **PuLID (Pure Identity)** face consistency to **FLUX.2**, supporting both Klein and Dev variants with automatic model detection.

Previous PuLID nodes only support Flux.1 Dev. This project is the **first** to adapt PuLID for the entire FLUX.2 family.

### Supported models

| Model | Double blocks | Single blocks | Hidden dim | Status |
|---|---|---|---|---|
| Flux.1 Dev (lldacing) | 19 | 38 | 4096 | ❌ Not this node |
| **Flux.2 Klein 4B** | 5 | 20 | 3072 | ✅ Best results |
| **Flux.2 Klein 9B** | 8 | 24 | 4096 | ✅ Best results |
| **Flux.2 Dev 32B** | 8 | 48 | 6144 | ✅ Works |

> The node **automatically detects** which model you are using — just select `auto (recommended)`.

---

## 📦 Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git
cd ComfyUI-PuLID-Flux2
```

### Update

```bash
cd ComfyUI/custom_nodes/ComfyUI-PuLID-Flux2
git pull
```

> ⚠️ **If you already have a working ComfyUI**, install only the required packages:
> ```bash
> pip install insightface onnxruntime-gpu open-clip-torch safetensors ml_dtypes==0.3.2
> ```
> Only run `pip install -r requirements.txt` on a **fresh install**.

### 2. EVA-CLIP (automatic)

EVA02-CLIP-L-14-336 downloads automatically on first run via `open_clip` (~800MB).

> ⚠️ **Do NOT** install `eva_clip` from GitHub — that package is broken and not required.  
> `open-clip-torch` is the only package needed.

### 3. Download InsightFace AntelopeV2

Download from: https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2

```
ComfyUI/models/insightface/models/antelopev2/
├── 1k3d68.onnx
├── 2d106det.onnx
├── genderage.onnx
├── glintr100.onnx
└── scrfd_10g_bnkps.onnx
```

> ⚠️ Folder must be named exactly `antelopev2` (lowercase).  
> The node auto-falls back to `buffalo_l` if AntelopeV2 is not found.

### 4. Download PuLID weights

Download `pulid_flux_v0.9.1.safetensors` from https://huggingface.co/guozinan/PuLID  
Place in: `ComfyUI/models/pulid/`

> These are Flux.1 weights used as a starting point. Native Flux.2 Klein weights are not yet available — training scripts are included in this repo.

### 5. Example workflow

Ready-to-use workflow available in the `workflows/` folder — drag & drop into ComfyUI.

---

## 🔌 Available Nodes

| Node | Description |
|---|---|
| `Load InsightFace (PuLID)` | Loads AntelopeV2 (falls back to buffalo_l) |
| `Load EVA-CLIP (PuLID)` | Loads EVA02-CLIP-L-14-336 |
| `Load PuLID ✦ Flux.2` | Loads PuLID weights (.safetensors) |
| **`Apply PuLID ✦ Flux.2`** | Main node — patches Flux.2 model |
| `PuLID — Face Debug Preview` | Visualizes detected faces (debug) |

---

## ⚙️ Recommended Parameters

### 🎯 Optimal Settings (Community-Tested)

**The key discovery:** Setting `end_at` to `0.5` instead of `1.0` prevents image darkening while maintaining strong facial identity.

#### Best Configuration for Klein 9B

```
weight:     0.6
start_at:   0.0
end_at:     0.5  ← Critical — prevents darkening
face_index: 0
```

**What happens during generation:**
- **0–50% (PuLID active):** Facial structure, proportions, and identity are locked in
- **50–100% (PuLID inactive):** Natural refinement of lighting, skin texture, and micro-details

### Parameter Comparison Table

| Parameter | Original (has darkening) | **Optimized** | Dev 32B |
|---|---|---|---|
| `model_variant` | `auto (recommended)` | `auto (recommended)` | `auto (recommended)` |
| `weight` | `0.3` | **`0.6`** | `0.5` |
| `start_at` | `0.0` | `0.0` | `0.0` |
| `end_at` | `0.8–1.0` ⚠️ | **`0.5`** ✅ | `0.5` |

### Fine-Tuning Guide

| Adjustment | Effect |
|---|---|
| `weight: 0.5` | Slightly more creative freedom, less strict identity |
| `weight: 0.7` | Stronger identity match (may reduce variety) |
| `end_at: 0.4` | More subtle influence, maximum natural lighting |
| `end_at: 0.6` | Stronger influence, slight risk of darkening |

> ⚠️ Values of `end_at` above `0.6` tend to introduce darkening artifacts.

---

## 🏗️ How it works

```
Reference image
    │
    ├──► InsightFace AntelopeV2 ──► 512-dim face embedding
    │
    └──► EVA02-CLIP-L-14-336 ─────► 768-dim visual features
                   │
                   ▼
          IDFormer (MLP + PerceiverCA)
                   │
                   ▼
          id_tokens [B, 4, dim]
          auto-projected to model dim (3072 / 4096 / 6144)
                   │
         ┌─────────┴─────────────────────────┐
         │  Injection into Flux.2 model       │
         │                                    │
         │  double_blocks[0, 2, 4, ...]:      │
         │    img += weight * PerceiverCA(...) │
         │                                    │
         │  single_blocks[0, 4, 8, ...]:      │
         │    hidden += weight * PerceiverCA(...)│
         └────────────────────────────────────┘
                   │
                   ▼
          Generated image with consistent identity
```

> **v0.3.0:** Both double_blocks and single_blocks are now patched.  
> Previous versions only patched double_blocks (~50% of injection). This is why identity consistency has improved significantly in v0.3.0.

---

## 🚀 Training Native Weights

Training scripts are included in the `training/` folder. This is the main priority for improving results beyond what's possible with Flux.1 weights.

```bash
# Step 1: Prepare dataset (local photos or CelebA/FFHQ auto-download)
python training/prepare_dataset.py \
    --input_dir ./my_photos \
    --output ./dataset

# Step 2: Phase 1 — IDFormer (no Flux required, fast)
python training/train_pulid.py \
    --dataset ./dataset/filtered \
    --output ./output \
    --phase 1 \
    --epochs 25 \
    --batch_size 4

# Step 3: Phase 2 — IDFormer + PerceiverCA (with frozen Flux.2)
python training/train_pulid.py \
    --dataset ./dataset/filtered \
    --output ./output_phase2 \
    --phase 2 \
    --resume ./output/pulid_klein_phase1_best.safetensors \
    --flux_model_path /path/to/flux-2-klein-9b.safetensors \
    --epochs 15 \
    --batch_size 2

# Validate pipeline locally before renting GPU on Vast.ai
python training/test_pipeline.py --images ./my_photos
```

See [training/README_TRAINING.md](training/README_TRAINING.md) for full details, dataset recommendations, and Vast.ai setup.

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| **Images are too dark** | Set `end_at: 0.5` (see Recommended Parameters) |
| `expected scalar type Half but found BFloat16` | Update to v0.3.0 — fixed in `pulid_flux2.py` |
| `AssertionError: 'detection' not in models` | Install AntelopeV2 correctly (5 .onnx files, section 3) |
| `EVA-CLIP not available` | `pip install open-clip-torch` — do **not** use the GitHub eva_clip package |
| `ml_dtypes has no attribute 'float4_e2m1fn'` | `pip install ml_dtypes==0.3.2` |
| Torch downgraded after install | `pip install insightface --no-deps` |
| Green image after changing weight | Update with `git pull` + restart ComfyUI |
| No visible difference with PuLID | Generate 4–5 images with different seeds and compare faces |
| `Cannot find double_blocks` | Open an issue with your model name — auto-detection may need updating |

---

## 📋 Roadmap

- [x] Flux.2 Klein 4B / 9B support
- [x] Flux.2 Dev 32B support
- [x] Auto model detection (Klein 4B / 9B / Dev)
- [x] Auto dim projection (3072 / 4096 / 6144)
- [x] EVA-CLIP via open_clip
- [x] InsightFace CUDA + buffalo_l fallback
- [x] Green image artifact fix
- [x] Half vs BFloat16 crash fix (v0.3.0)
- [x] Single blocks injection fix — identity now 100% (v0.3.0)
- [x] Training scripts (Phase 1 + Phase 2)
- [x] Dataset preparation script
- [x] Pipeline test script (local validation before Vast.ai)
- [x] Example workflow
- [ ] **Native Klein/Dev trained weights** ← main priority
- [ ] Edit mode (img2img) support
- [ ] Body consistency (LoRA + PuLID)
- [ ] HuggingFace model release

---

## 🙏 Credits

- **PuLID original**: [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) (Apache 2.0)
- **PuLID Flux.1**: [lldacing/ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll)
- **Flux.2**: [Black Forest Labs](https://blackforestlabs.ai)
- **EVA-CLIP**: [BAAI](https://github.com/baaivision/EVA)
- **Adaptation for Flux.2**: [@iFayens](https://github.com/iFayens) — March 2026

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
