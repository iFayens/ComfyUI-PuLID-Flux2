# ComfyUI-PuLID-Flux2

🔥 Bring consistent identity to FLUX.2 in one node  

> 🚀 **v0.4.0 — Major update** (native weights + improvements)

First working PuLID adaptation for Flux.2 — supports Klein (4B / 9B) and Dev (32B)

---

## 🚀 What's new (V2)

* ✅ Native Klein weights (trained)
* ✅ Safetensors support (plug & play)
* ✅ Full injection (double + single blocks → 100% identity)
* ✅ Improved face recognition
* ✅ Less artifacts
* ✅ Fixed sigma control (`start_at` / `end_at` now fully working)

---

## 🖼️ Results

> Add your BEST before / after images here (very important for impact)

---

## ⚡ Quick Start

1. Load your Flux.2 model
2. Add **Apply PuLID ✦ Flux.2** node
3. Use recommended settings:

   * `weight = 0.8`
   * `start_at = 0.7`
   * `end_at = 0.2`
4. Generate

Done ✅

---

## 🎛️ Recommended Settings

| 🎯 Goal            | ⚖️ Weight | ▶️ Start | ⏹️ End |
| ------------------ | --------- | -------- | ------ |
| Portrait fidelity  | 1.0 – 2.0 | 1.0      | 0.0    |
| Style + face       | 0.5 – 1.0 | 0.7      | 0.3    |
| Subtle inspiration | 0.3 – 0.5 | 1.0      | 0.0    |
| Structure only     | 1.5       | 1.0      | 0.6    |
| Fine details       | 1.0       | 0.4      | 0.0    |

💡 **Best overall:**
`weight = 0.8` + `start_at = 0.7` `end_at = 0.2` → strong identity without darkening

---

## 🧠 How it works (simple)

* Face embedding → InsightFace
* Visual features → EVA-CLIP
* Converted into identity tokens
* Injected into Flux.2 model

➡️ Result: consistent identity across generations

---

## 📦 Installation

### 🟢 Quick Install (recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git
cd ComfyUI-PuLID-Flux2
pip install insightface onnxruntime-gpu open-clip-torch safetensors ml_dtypes==0.3.2
```

---

### 📥 Required models

#### InsightFace (AntelopeV2)

Download:
https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2

Place in:

```
ComfyUI/models/insightface/models/antelopev2/
```

---

#### PuLID weights

Place in:

```
ComfyUI/models/pulid/
```

* Existing PuLID weights supported
* Native Flux.2 weights (recommended, in progress)

---

### ⚠️ Notes

* EVA-CLIP downloads automatically (~800MB)
* Do NOT install `eva_clip` from GitHub
* If ComfyUI already works → don’t use full requirements.txt

---

## 🔌 Available Nodes

* Load InsightFace (PuLID)
* Load EVA-CLIP (PuLID)
* Load PuLID ✦ Flux.2
* Apply PuLID ✦ Flux.2 ⭐ (main node)
* Face Debug Preview

---

## ⚠️ Current Status

| Feature              | Status         |
| -------------------- | -------------- |
| Flux.2 Klein support | ✅ Best         |
| Flux.2 Dev support   | ✅ Working      |
| Identity consistency | ✅ Strong       |
| Native weights       | ⚠️ In progress |

---

## 🚀 Roadmap

* Native trained weights (Klein / Dev)
* Edit mode (img2img)
* Body consistency
* HuggingFace release

---

## 🙏 Credits

* PuLID original: https://github.com/ToTheBeginning/PuLID
* Flux.2: Black Forest Labs
* EVA-CLIP: BAAI
* Adaptation: @iFayens

---

⭐ If this project helps you, consider starring the repo!
