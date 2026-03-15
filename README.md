# ComfyUI-PuLID-Flux2

[![GitHub stars](https://img.shields.io/github/stars/iFayens/ComfyUI-PuLID-Flux2?style=social)](https://github.com/iFayens/ComfyUI-PuLID-Flux2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-blue)](https://github.com/comfyanonymous/ComfyUI)
[![Flux.2 Klein](https://img.shields.io/badge/Flux.2-Klein%204B%2F9B-green)](https://huggingface.co/black-forest-labs)

> **First PuLID implementation natively adapted for FLUX.2 Klein (4B & 9B)**  
> March 2026

---

## ⚠️ Current State & Honest Assessment

This node is **v0.1 beta**. Here's what works and what doesn't yet:

| | Status | Notes |
|---|---|---|
| Node loads and runs | ✅ | All 5 nodes functional |
| InsightFace face detection | ✅ | AntelopeV2 + buffalo_l fallback |
| EVA-CLIP encoding | ✅ | via open_clip |
| Face consistency | ⚠️ Partial | Limited by Flux.1 weights on Klein |
| Image quality with PuLID | ⚠️ Degraded | Single blocks disabled to avoid artifacts |
| Native Klein weights | ❌ Not yet | Training script included — contributions welcome! |

**Bottom line:** PuLID alone with current Flux.1 weights shows subtle results on Klein. For best face consistency right now, Flux.2 Klein's native **Reference Conditioning** (2x ReferenceLatent nodes) is more reliable. PuLID will shine once native Klein weights are trained.

---

## 🎯 What is this?

This custom node brings **PuLID (Pure Identity)** face consistency to **FLUX.2 Klein**, the latest generation model from Black Forest Labs.

Previous PuLID implementations only support Flux.1 Dev. This project is the **first** to adapt PuLID's architecture specifically for Flux.2 Klein's unique transformer structure — laying the groundwork for native Klein weights.

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
```

> ⚠️ **If you already have a working ComfyUI**, install only the required packages to avoid breaking your torch version:
> ```bash
> pip install insightface onnxruntime-gpu open-clip-torch safetensors ml_dtypes==0.3.2
> ```
> Only run `pip install -r requirements.txt` on a **fresh install**.

### 2. EVA-CLIP (automatic)

EVA02-CLIP-L-14-336 downloads automatically on first run via `open_clip` (~800MB).

> ⚠️ **Do NOT** install `eva_clip` from GitHub — that package is broken and not required.  
> `open-clip-torch` is already included in the install command above.

### 3. Download InsightFace AntelopeV2

Download from: https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2

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
> If missing: `AssertionError: 'detection' not in models`  
> The node will automatically fall back to `buffalo_l` if AntelopeV2 is not found.

### 4. Download PuLID weights

> ⚠️ No weights officially trained on Klein yet. Use Flux.1 weights as a starting point (partial compatibility with Klein 9B).

Download: `pulid_flux_v0.9.1.safetensors` from https://huggingface.co/guozinan/PuLID  
Place in: `ComfyUI/models/pulid/`

### 5. Download example workflow

A ready-to-use workflow is available in the `workflows/` folder of this repo.  
Just drag & drop it into ComfyUI.

---

## 🔌 Available Nodes

| Node | Description |
|---|---|
| `Load InsightFace (PuLID Klein)` | Loads AntelopeV2 face detector (falls back to buffalo_l) |
| `Load EVA-CLIP (PuLID Klein)` | Loads EVA02-CLIP-L-14-336 visual encoder |
| `Load PuLID Flux.2 Model` | Loads PuLID weights (.safetensors) |
| **`Apply PuLID ✦ Flux.2`** | Main node — patches Flux.2 Klein model |
| `PuLID Klein — Face Debug Preview` | Visualizes detected faces (debug) |

---

## ⚙️ Recommended Parameters

| Parameter | Value | Notes |
|---|---|---|
| `weight` | `0.5-0.7` | Keep low with current Flux.1 weights |
| `start_at` | `0.0` | Let PuLID guide from the start |
| `end_at` | `1.0` | Full generation coverage |
| `face_index` | `0` | Use largest detected face |

> ⚠️ Higher weight values (>0.8) will degrade image quality with non-native weights.

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

---

## 🚀 Training Native Klein Weights

This is the **main priority** for improving results. The repo includes the first training script for PuLID on Flux.2 Klein.

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

**If you have a powerful GPU (A100, H100) and want to contribute trained weights, please open an issue!**

---

## 🐛 Troubleshooting

**`AssertionError: 'detection' not in models`**  
→ AntelopeV2 not found. Check the folder structure above (section 3).  
→ The node will auto-fallback to `buffalo_l` — update to latest version with `git pull`.

**`EVA-CLIP not available`**  
→ Run `pip install open-clip-torch` — do NOT use the eva_clip GitHub package.

**`AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'`**  
→ Run `pip install ml_dtypes==0.3.2`

**Noisy / green / contaminated output**  
→ Reduce `weight` to 0.4-0.5. This is a known limitation with Flux.1 weights on Klein.

**No visible difference with PuLID enabled**  
→ PuLID works across multiple generations — generate 4-5 images with different seeds and compare faces.  
→ Check console for: `[PuLID-Flux2Klein] ✅ PuLID applied.`

**`Cannot find double_blocks`**  
→ Open an issue with the exact attribute names from your model.

---

## 📋 Roadmap

- [x] Custom node for Flux.2 Klein 4B / 9B
- [x] EVA-CLIP integration via open_clip
- [x] InsightFace CUDA support + buffalo_l fallback
- [x] Training dataset preparation script
- [x] Training script (Phase 1 — embedding only)
- [x] Example workflow
- [ ] **Native Klein-trained weights** ← main priority
- [ ] Training script Phase 2 (full pipeline with Flux)
- [ ] HuggingFace model release
- [ ] Edit mode (img2img) support
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
