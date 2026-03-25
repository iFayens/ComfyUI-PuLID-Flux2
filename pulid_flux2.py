"""
ComfyUI-PuLID-Flux2
Custom node PuLID for FLUX.2 — Version simplifiée
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import comfy.model_management
import folder_paths

PULID_DIR = os.path.join(folder_paths.models_dir, "pulid")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
os.makedirs(PULID_DIR, exist_ok=True)
os.makedirs(INSIGHTFACE_DIR, exist_ok=True)


class PerceiverAttentionCA(nn.Module):
    def __init__(self, dim: int = 3072, dim_head: int = 64, heads: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        target_dtype = self.norm1.weight.dtype
        
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if context.dtype != target_dtype:
            context = context.to(target_dtype)

        x_n = self.norm1(x)
        ctx = self.norm2(context)

        q = self.to_q(x_n)
        kv = self.to_kv(ctx)
        k, v = kv.chunk(2, dim=-1)

        def reshape(t):
            return t.view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(attn_out)


class IDFormer(nn.Module):
    def __init__(self, dim: int = 4096, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        
        self.proj = nn.Sequential(
            nn.Linear(512 + 768, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )
        self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.layers = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(4)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, id_embed: torch.Tensor, clip_embed: torch.Tensor) -> torch.Tensor:
        B = id_embed.shape[0]
        clip_embed = clip_embed - clip_embed.mean(dim=-1, keepdim=True)
        clip_embed = 1 * clip_embed
        
        combined = torch.cat([id_embed, clip_embed], dim=-1)
        tokens = self.proj(combined).view(B, self.num_tokens, -1)
        
        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = latents + layer(latents, tokens)
        return self.norm(latents)


class PuLIDFlux2(nn.Module):
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.id_former = IDFormer(dim=dim)
        self.dim = dim
        self.double_ca = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(10)])
        self.single_ca = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(30)])

    @classmethod
    def from_pretrained(cls, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        dim = state["id_former.latents"].shape[-1]
        model = cls(dim=dim)
        model.load_state_dict(state, strict=False)
        return model


def get_flux_inner(model):
    if hasattr(model, "model"):
        model = model.model
    if hasattr(model, "diffusion_model"):
        model = model.diffusion_model
    return model


def detect_flux_variant(model):
    double = getattr(model, "transformer_blocks", None) or getattr(model, "double_blocks", [])
    single = getattr(model, "single_transformer_blocks", None) or getattr(model, "single_blocks", [])
    n_double, n_single = len(double), len(single)
    
    if n_double == 5 and n_single == 20:
        return "Klein 4B", 3072
    elif n_double == 8 and n_single == 24:
        return "Klein 9B", 4096
    elif n_double == 19 and n_single == 38:
        return "Dev 32B", 4096
    else:
        return f"Custom ({n_double}D/{n_single}S)", 4096


def load_eva_clip(device):
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k",
        )
        visual = model.visual
        visual.eval().to(device)
        return visual
    except Exception:
        return None


def patch_flux(model, pulid_module, id_tokens, strength):
    dm = get_flux_inner(model)
    
    double_blocks = getattr(dm, "transformer_blocks", None) or getattr(dm, "double_blocks", [])
    single_blocks = getattr(dm, "single_transformer_blocks", None) or getattr(dm, "single_blocks", [])
    
    original_double = {}
    original_single = {}
    
    for idx, block in enumerate(double_blocks):
        original_double[idx] = block.forward
        
        def make_double_patch(block_idx, ca_idx):
            def patched(img, txt, vec, **kwargs):
                out_img, out_txt = original_double[block_idx](img, txt, vec, **kwargs)
                
                if ca_idx < len(pulid_module.double_ca):
                    if block_idx < 3:
                        factor = 8.0
                    elif block_idx < 5:
                        factor = 5.0
                    else:
                        factor = 3.0
                    
                    ca = pulid_module.double_ca[ca_idx]
                    correction = ca(out_img, id_tokens)
                    correction = correction / (correction.norm(dim=-1, keepdim=True) + 1e-6)
                    out_img = out_img + strength * factor * correction
                
                return out_img, out_txt
            return patched
        
        ca_idx = min(idx, len(pulid_module.double_ca) - 1)
        block.forward = make_double_patch(idx, ca_idx)
    
    for idx, block in enumerate(single_blocks):
        original_single[idx] = block.forward
        
        def make_single_patch(block_idx, ca_idx):
            def patched(x, vec, pe, *args, **kwargs):
                try:
                    if len(args) > 0:
                        out = original_single[block_idx](x, vec, pe, args[0], **kwargs)
                    else:
                        out = original_single[block_idx](x, vec, pe, **kwargs)
                except:
                    out = original_single[block_idx](x, vec, pe, **kwargs)
                
                if ca_idx < len(pulid_module.single_ca):
                    if block_idx < 4:
                        factor = 6.0
                    elif block_idx < 8:
                        factor = 4.0
                    else:
                        factor = 2.0
                    
                    if isinstance(out, tuple):
                        hidden = out[0]
                    else:
                        hidden = out
                    
                    ca = pulid_module.single_ca[ca_idx]
                    correction = ca(hidden, id_tokens)
                    correction = correction / (correction.norm(dim=-1, keepdim=True) + 1e-6)
                    hidden = hidden + strength * factor * correction
                    
                    if isinstance(out, tuple):
                        out = (hidden,) + out[1:]
                    else:
                        out = hidden
                
                return out
            return patched
        
        ca_idx = min(idx, len(pulid_module.single_ca) - 1)
        block.forward = make_single_patch(idx, ca_idx)
    
    def unpatch():
        for idx, block in enumerate(double_blocks):
            if idx in original_double:
                block.forward = original_double[idx]
        for idx, block in enumerate(single_blocks):
            if idx in original_single:
                block.forward = original_single[idx]
    
    return unpatch


class PuLIDInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"provider": (["CUDA", "CPU"],)}}
    
    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load"
    CATEGORY = "PuLID-Flux2"
    
    def load(self, provider):
        from insightface.app import FaceAnalysis
        providers = [provider + "ExecutionProvider", "CPUExecutionProvider"]
        
        for name in ["antelopev2", "buffalo_l"]:
            try:
                model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=providers)
                model.prepare(ctx_id=0, det_size=(640, 640))
                return (model,)
            except:
                continue
        raise RuntimeError("Aucun modèle InsightFace trouvé")


class PuLIDEVACLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    
    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load"
    CATEGORY = "PuLID-Flux2"
    
    def load(self):
        device = comfy.model_management.get_torch_device()
        model = load_eva_clip(device)
        if model is None:
            raise RuntimeError("EVA-CLIP non disponible")
        return (model,)


class PuLIDModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        if not os.path.exists(PULID_DIR):
            os.makedirs(PULID_DIR, exist_ok=True)
        
        files = [f for f in os.listdir(PULID_DIR) if f.endswith((".safetensors", ".pt", ".bin"))]

        if not files:
            files = ["__create_new__"]
        
        return {
            "required": {
                "pulid_file": (files,),
            }
        }
    
    RETURN_TYPES = ("PULID_MODEL",)
    FUNCTION = "load"
    CATEGORY = "PuLID-Flux2"
    
    def load(self, pulid_file):
        if pulid_file == "__create_new__":
            return (PuLIDFlux2(dim=4096),)
        
        path = os.path.join(PULID_DIR, pulid_file)
        
        if not os.path.exists(path):
            return (PuLIDFlux2(dim=4096),)
        
        try:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(path, device="cpu")
                dim = state["id_former.latents"].shape[-1]
                model = PuLIDFlux2(dim=dim)
                model.load_state_dict(state, strict=False)
            else:
                model = PuLIDFlux2.from_pretrained(path)
            
            model.eval()
            return (model,)
        except:
            return (PuLIDFlux2(dim=4096),)


class ApplyPuLIDFlux2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_model": ("PULID_MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "eva_clip": ("EVA_CLIP",),
                "face_analysis": ("INSIGHTFACE",),
                "image": ("IMAGE",),
            },
            "optional": {"face_index": ("INT", {"default": 0, "min": 0, "max": 9})}
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "PuLID-Flux2"
    
    def apply(self, model, pulid_model, strength, eva_clip, face_analysis, image, face_index=0):
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16
        
        img_np = (image[0].numpy() * 255).astype(np.uint8)
        faces = face_analysis.get(img_np)
        
        if not faces:
            print("[PuLID] Aucun visage détecté")
            return (model,)
        
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces[min(face_index, len(faces)-1)]
        
        id_embed = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
        id_embed = F.normalize(id_embed, dim=-1)
        
        x1, y1, x2, y2 = face.bbox.astype(int)
        margin = int(max(x2-x1, y2-y1) * 0.2)
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(img_np.shape[1], x2+margin), min(img_np.shape[0], y2+margin)
        
        face_crop = image[:1, y1:y2, x1:x2, :]
        if face_crop.shape[1] == 0 or face_crop.shape[2] == 0:
            face_crop = image[:1]
        
        face_crop = F.interpolate(face_crop.permute(0,3,1,2), size=(336,336), mode="bilinear").to(device)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        face_crop = (face_crop - mean) / std
        
        with torch.no_grad():
            clip_out = eva_clip(face_crop.float())
            if isinstance(clip_out, (list, tuple)):
                clip_out = clip_out[0]
            if clip_out.dim() == 3:
                clip_out = clip_out[:, 0, :]
            clip_embed = clip_out.to(device, dtype=dtype)
        
        pulid_model = pulid_model.to(device, dtype=dtype)
        
        with torch.no_grad():
            id_tokens = pulid_model.id_former(id_embed, clip_embed)
            id_tokens = id_tokens / (id_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        
        work_model = model.clone()
        dm = get_flux_inner(work_model)
        
        if hasattr(dm, "_pulid_unpatcher"):
            try:
                dm._pulid_unpatcher()
            except:
                pass
        
        variant, detected_dim = detect_flux_variant(dm)
        
        if id_tokens.shape[-1] != detected_dim:
            print(f"[PuLID] Projection: {id_tokens.shape[-1]} → {detected_dim}")
            proj = nn.Linear(id_tokens.shape[-1], detected_dim, bias=False).to(device, dtype=dtype)
            nn.init.normal_(proj.weight, std=0.01)
            id_tokens = proj(id_tokens)
        
        unpatch = patch_flux(work_model, pulid_model, id_tokens, strength)
        dm._pulid_unpatcher = unpatch
        
        if strength == 0:
            print("⚪ PuLID: OFF")
        else:
            print(f"🟢 PuLID: ON | {variant} | strength={strength:.2f} | face={face_index}/{len(faces)-1}")
        
        return (work_model,)


class PuLIDFacePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"face_analysis": ("INSIGHTFACE",), "image": ("IMAGE",)}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "PuLID-Flux2"
    OUTPUT_NODE = True
    
    def preview(self, face_analysis, image):
        try:
            import cv2
            img_np = (image[0].numpy() * 255).astype(np.uint8).copy()
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            faces = face_analysis.get(img_np)
            
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img_bgr, f"Face {i}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            return (out,)
        except:
            return (image,)


NODE_CLASS_MAPPINGS = {
    "PuLIDInsightFaceLoader": PuLIDInsightFaceLoader,
    "PuLIDEVACLIPLoader": PuLIDEVACLIPLoader,
    "PuLIDModelLoader": PuLIDModelLoader,
    "ApplyPuLIDFlux2": ApplyPuLIDFlux2,
    "PuLIDFacePreview": PuLIDFacePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLIDInsightFaceLoader": "Load InsightFace (PuLID)",
    "PuLIDEVACLIPLoader": "Load EVA-CLIP (PuLID)",
    "PuLIDModelLoader": "Load PuLID ✦ Flux.2",
    "ApplyPuLIDFlux2": "Apply PuLID ✦ Flux.2",
    "PuLIDFacePreview": "PuLID — Face Preview",
}