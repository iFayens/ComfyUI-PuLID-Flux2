# ComfyUI-PuLID-Flux2

🔥 Bring consistent identity to FLUX.2 in one node  

> 🚀 **v0.5.0 — improvements**
👉 https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency

First working PuLID adaptation for Flux.2 — supports Klein (4B / 9B) and Dev (32B)

---

## 🚀 What's new (v0.5.0)

* ✅ Improved face recognition
* ✅ Less artifacts
* ✅ Adapted PuLID-Flux2 to Flux2-Klein-9B-Consistency Model
* ⚠️ Sigma control (`start_at` / `end_at` NOT fully working)

---

## 🖼️ Results

> ///

---

## ⚡ Quick Start

1. Load your Flux.2 model
2. Add **Apply PuLID ✦ Flux.2** node
3. Use recommended settings:

   * `weight = 1`
   * `start_at = 0`
   * `end_at = 1`
4. Generate

Done ✅

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

### 📥 Download weights

👉 https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency

Place in:
ComfyUI/models/pulid/

---

### 📥 Required models

#### InsightFace (AntelopeV2)

👉 https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2

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

## ⚠️ Training

Training scripts have been temporarily removed due to instability and bugs.
They will be reintroduced in a future update once fully stable and reliable.
Thanks for your patience.

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
* Weight: dx8152 
* Adaptation: @iFayens

---

## ⭐ Support

If this project helps you, consider giving it a ⭐ on GitHub — it really helps.

You can also support future development:

👉 https://buymeacoffee.com/fayens

Thank you 🙏
