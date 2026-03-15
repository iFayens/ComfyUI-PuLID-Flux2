# ComfyUI-PuLID-Flux2

[![GitHub stars](https://img.shields.io/github/stars/iFayens/ComfyUI-PuLID-Flux2?style=social)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Flux.2 Klein](https://img.shields.io/badge/Flux.2-Klein%204B%2F9B-green)](https://huggingface.co/black-forest-labs)

> **First PuLID implementation natively adapted for FLUX.2 Klein (4B & 9B)**  
> Consistent face identity injection without model pollution — March 2026

---

## 🎯 What is this?

This custom node brings **PuLID (Pure Identity)** face consistency to **FLUX.2 Klein**, the latest generation model from Black Forest Labs.

Previous PuLID implementations only support Flux.1 Dev. This project is the **first** to adapt PuLID's architecture specifically for Flux.2 Klein's unique transformer structure.

### Key differences vs existing PuLID nodes

| | PuLID Flux.1 (lldacing) | **ComfyUI-PuLID-Flux2** |
|---|---|---|
| Model | Flux.1 Dev | **Flux.2 Klein 4B / 9B** |
| Double blocks | 19 | 5 (4B) / 8 (9B) |
| Single blocks | 38 | 20 (4B) / 24 (9B) |
| Hidden dim | 4096 | **3072** (4B) / 4096 (9B) |
| Modulation | Per block | **Shared** (Klein-specific) |
| Text encoder | T5 | **Qwen3** |

---

## 📦 Installation

### 1. Clone into ComfyUI custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git ComfyUI-PuLID-Flux2Klein
cd ComfyUI-PuLID-Flux2Klein
pip install -r requirements.txt
```

### 2. EVA-CLIP (automatic)

EVA02-CLIP-L-14-336 downloads automatically on first run via `open_clip` (~800MB).

> ⚠️ **Do NOT** install `eva_clip` from GitHub — that package is broken and not required.  
> `open-clip-torch` is already included in `requirements.txt`.

### 3. Download InsightFace AntelopeV2

Download from: [https://huggingface.co/MonsterMMORPG/InsightFace_AntelopeV2](https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2)

Place the **5 files** in the exact folder structure below:
```
ComfyUI/models/insightface/models/antelopev2/
├── 1k3d68.onnx
├── 2d106det.onnx
├── genderage.onnx
├── glintr100.onnx
└── scrfd_10g_bnkps.onnx
```
> ⚠️ The folder must be named exactly `antelopev2` (lowercase).  
> If missing or misnamed you will get: `AssertionError: 'detection' not in models`

### 4. Download PuLID weights

> ⚠️ No weights officially trained on Klein yet. Use Flux.1 weights as a starting point (partial compatibility with Klein 9B).

Download from: https://huggingface.co/guozinan/PuLID  
File: `pulid_flux_v0.9.1.safetensors`  
Place in: `ComfyUI/models/pulid/`

### 5. Download example workflow

A ready-to-use workflow is available in the `workflows/` folder of this repo.  
Just drag & drop it into ComfyUI.

---

## 🔌 Available Nodes

| Node | Description |
|---|---|
| `Load InsightFace (PuLID Klein)` | Loads AntelopeV2 face detector |
| `Load EVA-CLIP (PuLID Klein)` | Loads EVA02-CLIP-L-14-336 visual encoder |
| `Load PuLID Flux.2 Model` | Loads PuLID weights (.safetensors) |
| **`Apply PuLID ✦ Flux.2`** | Main node — patches Flux.2 Klein model |
| `PuLID Klein — Face Debug Preview` | Visualizes detected faces (debug) |

---

## ⚙️ Recommended Parameters

| Parameter | Value | Notes |
|---|---|---|
| `weight` | `0.7` | Start here, adjust to taste |
| `start_at` | `0.0` | Let PuLID guide from the start |
| `end_at` | `1.0` | Full generation coverage |
| `face_index` | `0` | Use largest detected face |

> ⚠️ Keep `weight` below 0.8 to avoid image quality degradation with non-native weights.

---

## 🏗️ How it works

```
Reference image
    │
    ├──► InsightFace AntelopeV2
    │         └─► 512-dim face embedding
    │
    └──► EVA02-CLIP-L-14-336
              └─► 768-dim visual features
                       │
                       ▼
              IDFormer (MLP + PerceiverCA)
                       │
                       ▼
              id_tokens [B, 4, dim]
                       │
              ┌────────┴──────────────────────┐
              │  Injection into Flux.2 Klein   │
              │                                │
              │  double_blocks[0,2,4,...]:      │
              │    img += w * PerceiverCA(...)  │
              └────────────────────────────────┘
                       │
                       ▼
              Generated image with consistent identity
```

### Klein-specific adaptations

1. **Shared modulation** — Flux.2 Klein shares AdaLayerNorm params across all blocks. PuLID injection acts *after* modulation, independently.
2. **Fused single blocks** — Klein's `single_transformer_blocks` fuse `attn.to_qkv_mlp_proj`. Injection targets output tokens instead of Q/K/V for stability.
3. **Conditioning shape** — Qwen3 produces [B, 512, 12288]. PuLID only interacts with the image stream, not the text stream.

---

## 🚀 Training Native Klein Weights

This repo includes the **first training script for PuLID on Flux.2 Klein**.

```bash
# Step 1: Prepare dataset
python training/prepare_dataset.py --output ./dataset --source celeba --max_images 2000

# Step 2: Train
python training/train_pulid_klein.py \
  --dataset ./dataset/filtered \
  --output ./output \
  --comfyui_path C:/AI/ComfyUI \
  --dim 4096 \
  --epochs 20 \
  --batch_size 4
```

See [training/README_TRAINING.md](training/README_TRAINING.md) for full details.

---

## 🐛 Troubleshooting

**`AssertionError: 'detection' not in models`**  
→ AntelopeV2 not found. Check the folder structure above (section 3).

**`EVA-CLIP not available`**  
→ Run `pip install open-clip-torch` — do NOT use the eva_clip GitHub package.

**`Cannot find double_blocks`**  
→ Your ComfyUI version may name blocks differently. Open an issue with the exact attribute names.

**Noisy / contaminated output images**  
→ Reduce `weight` to 0.5–0.6. This happens with non-native Flux.1 weights on Klein.

**Inconsistent results**  
→ No native Klein weights exist yet. Train using the included script or wait for community weights.

---

## 📋 Roadmap

- [x] Custom node for Flux.2 Klein 4B / 9B
- [x] EVA-CLIP integration via open_clip
- [x] InsightFace CUDA support
- [x] Training dataset preparation script
- [x] Training script (Phase 1 — embedding only)
- [x] Example workflow
- [ ] Native Klein-trained weights (community training)
- [ ] Training script Phase 2 (full pipeline with Flux)
- [ ] HuggingFace model release
- [ ] Body consistency support (LoRA + PuLID combo)

---

## 🙏 Credits

- **PuLID original**: [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) (Apache 2.0)
- **PuLID Flux.1**: [lldacing/ComfyUI_PuLID_Flux_ll](https://github.com/lldacing/ComfyUI_PuLID_Flux_ll)
- **Flux.2 Klein**: [Black Forest Labs](https://blackforestlabs.ai)
- **EVA-CLIP**: [BAAI](https://github.com/baaivision/EVA)
- **Adaptation for Flux.2 Klein**: [@iFayens](https://github.com/iFayens) — March 2026

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
