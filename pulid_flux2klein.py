"""
ComfyUI-PuLID-Flux2Klein
========================
Custom node PuLID adapté pour FLUX.2 Klein (4B / 9B).

Architecture de FLUX.2 Klein (source: Black Forest Labs / HuggingFace diffusers):
  - 8 double_blocks  (transformer_blocks)   → flux2 dev
  - 5 double_blocks  (transformer_blocks)   → klein 4B
  - 24 single_blocks (single_transformer_blocks) → flux2 dev
  - 20 single_blocks (single_transformer_blocks) → klein 4B
  - dim hidden: 3072 (klein 4B) / 4096 (klein 9B)
  - Shared modulation (double_stream_modulation_img / txt, single_stream_modulation)
  - Text encoder: Qwen3 (4B→ 4B param, 9B→ 8B param)
  - Conditioning shape: [batch, 512, 12288] (Qwen3-8B) ou [batch, 512, 4096]

Différences clés vs Flux.1 qui nécessitent une adaptation :
  1. Shared AdaLayerNorm (une seule modulation partagée vs une par bloc dans Flux.1)
  2. dim=3072 (klein 4B) au lieu de 4096 (flux.1)
  3. single_blocks fusionnent QKV + MLP en une seule projection (to_qkv_mlp_proj)
  4. Ratio double/single très différent: 5/20 (klein4B) vs 19/38 (flux.1)

Strategy d'injection:
  - On injecte les embeddings facials via des Cross-Attention Perceiver (comme PuLID original)
  - On patch le forward des double_blocks et single_blocks via model patching ComfyUI
  - Le perceiver reçoit les id_embedding (512-dim InsightFace + EVA-CLIP features)
    et génère une correction additive sur les image tokens
"""

import os
import logging
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import comfy.model_management
import comfy.utils
import folder_paths

# ──────────────────────────────────────────────────────────────────────────────
# Chemins modèles
# ──────────────────────────────────────────────────────────────────────────────
PULID_DIR = os.path.join(folder_paths.models_dir, "pulid")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
CLIP_DIR = os.path.join(folder_paths.models_dir, "clip")

os.makedirs(PULID_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Perceiver Cross-Attention  (identique à PuLID Flux.1, dim adapté)
# ──────────────────────────────────────────────────────────────────────────────

class PerceiverAttentionCA(nn.Module):
    """
    Cross-attention block Perceiver.
    dim      : dimension cachée du transformer (3072 pour Klein 4B, 4096 pour Klein 9B)
    dim_head : dimension par tête d'attention
    heads    : nombre de têtes
    """
    def __init__(self, dim: int = 3072, dim_head: int = 64, heads: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q   = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv  = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x       : image tokens  [B, N_img, dim]
        context : id embedding  [B, N_id,  dim]
        returns : correction additive [B, N_img, dim]
        """
        B, N, C = x.shape
        x_n = self.norm1(x)
        ctx = self.norm2(context)

        q  = self.to_q(x_n)
        kv = self.to_kv(ctx)
        k, v = kv.chunk(2, dim=-1)

        # reshape pour multi-head
        def reshape(t):
            return t.view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product attention (utilise flash-attention si dispo)
        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.to_out(attn_out)


# ──────────────────────────────────────────────────────────────────────────────
# IDFormer  (projette les embeddings face → dim du transformer)
# ──────────────────────────────────────────────────────────────────────────────

class IDFormer(nn.Module):
    """
    Projette les embeddings id (insightface 512-d + eva_clip 768-d)
    vers l'espace du transformer cible (dim).
    """
    def __init__(self, id_dim: int = 512, clip_dim: int = 768,
                 dim: int = 3072, depth: int = 4,
                 num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        combined = id_dim + clip_dim

        self.proj = nn.Sequential(
            nn.Linear(combined, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )
        # latents apprenables servant de queries pour le perceiver interne
        self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

        self.layers = nn.ModuleList([
            PerceiverAttentionCA(dim=dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, id_embed: torch.Tensor,
                clip_embed: torch.Tensor) -> torch.Tensor:
        """
        id_embed   : [B, 512]
        clip_embed : [B, 768]
        returns    : [B, num_tokens, dim]
        """
        B = id_embed.shape[0]
        combined = torch.cat([id_embed, clip_embed], dim=-1)  # [B, 1280]
        tokens = self.proj(combined).view(B, self.num_tokens, -1)  # [B, T, dim]

        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = latents + layer(latents, tokens)
        return self.norm(latents)


# ──────────────────────────────────────────────────────────────────────────────
# PuLID Flux2Klein  (ensemble poids du module)
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDFlux2Klein(nn.Module):
    """
    Conteneur principal du module PuLID pour Flux.2 Klein.
    Contient :
      - id_former      : IDFormer qui projette face→tokens
      - pulid_ca_double: PerceiverCA injectés dans les double_blocks
      - pulid_ca_single: PerceiverCA injectés dans les single_blocks
    """
    def __init__(self, dim: int = 3072,
                 double_interval: int = 2,
                 single_interval: int = 4):
        super().__init__()
        self.double_interval = double_interval
        self.single_interval  = single_interval

        # IDFormer adapté à la dim Klein
        self.id_former = IDFormer(dim=dim)

        # Calculer le nombre de CA blocks nécessaires
        # Klein 4B : 5 double, 20 single
        # Klein 9B : 8 double, 24 single  (auto-détecté au runtime)
        # On pré-instancie assez de blocs (max 10 double, 30 single)
        max_double = max(1, 10 // double_interval)
        max_single = max(1, 30 // single_interval)

        self.pulid_ca_double = nn.ModuleList([
            PerceiverAttentionCA(dim=dim)
            for _ in range(max_double)
        ])
        self.pulid_ca_single = nn.ModuleList([
            PerceiverAttentionCA(dim=dim)
            for _ in range(max_single)
        ])

    @classmethod
    def from_pretrained(cls, path: str, map_location="cpu") -> "PuLIDFlux2Klein":
        state = torch.load(path, map_location=map_location, weights_only=True)
        # Déterminer la dim depuis les poids
        try:
            dim = state["id_former.latents"].shape[-1]
        except KeyError:
            dim = 3072
        model = cls(dim=dim)
        model.load_state_dict(state, strict=False)
        return model


# ──────────────────────────────────────────────────────────────────────────────
# EVA-CLIP loader via open_clip
# ──────────────────────────────────────────────────────────────────────────────

def load_eva_clip(device):
    """
    Charge EVA02-CLIP-L-14-336 via open_clip (cache HuggingFace).
    Utilise le pretrained 'merged2b_s6b_b61k' — déjà téléchargé dans le cache.
    """
    try:
        import open_clip

        logging.info("[PuLID-Flux2Klein] Chargement EVA02-CLIP-L-14-336 (merged2b_s6b_b61k)...")
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k",
        )
        visual = model.visual
        visual.eval().to(device)
        logging.info("[PuLID-Flux2Klein] EVA02-CLIP-L-14-336 chargé ✅")
        return visual

    except Exception as e:
        logging.warning(f"[PuLID-Flux2Klein] EVA-CLIP non disponible: {e}")
        return None


def encode_image_eva_clip(eva_clip, image: torch.Tensor,
                          device, dtype) -> tuple:
    """
    image : [B, H, W, C] float32 [0,1]
    retourne (id_cond_vit [B, 768], None)
    """
    face_features_image = F.interpolate(
        image.permute(0, 3, 1, 2),
        size=(336, 336), mode="bilinear", align_corners=False
    ).to(device=device, dtype=torch.float32)

    # normalisation EVA02-CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=device, dtype=torch.float32).view(1, 3, 1, 1)
    face_features_image = (face_features_image - mean) / std

    with torch.no_grad():
        # open_clip retourne directement les features [B, 768]
        # EVA-CLIP tourne toujours en float32, on cast après
        id_cond_vit = eva_clip(face_features_image.float())
        if isinstance(id_cond_vit, (list, tuple)):
            id_cond_vit = id_cond_vit[0]

    # Forcer bfloat16 pour compatibilité avec Flux.2 Klein
    target_dtype = torch.bfloat16 if dtype == torch.bfloat16 else dtype
    return id_cond_vit.to(device, dtype=target_dtype), None


# ──────────────────────────────────────────────────────────────────────────────
# Patching du modèle Flux.2 Klein
# ──────────────────────────────────────────────────────────────────────────────

def _get_flux2_inner_model(model):
    """Remonte jusqu'au vrai objet transformer de ComfyUI."""
    # ComfyUI enveloppe le modèle dans ModelPatcher → model.model → diffusion_model
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "diffusion_model"):
        return m.diffusion_model
    return m


def patch_flux2klein_forward(flux_model, pulid_module, id_embedding,
                              weight, sigma_start, sigma_end):
    """
    Injecte les embeddings PuLID dans le forward pass de Flux.2 Klein.
    Utilise le système de patching ComfyUI (set_model_patch / model_options).

    Stratégie :
      - Pour chaque double_block[i] (i % double_interval == 0) :
          img_out += weight * PerceiverCA_double[i // double_interval](img_tokens, id_embed)
      - Pour chaque single_block[i] (i % single_interval == 0) :
          out += weight * PerceiverCA_single[i // single_interval](img_part, id_embed)
    """

    # Stocker les données dans le modèle (pattern PuLID original)
    dm = _get_flux2_inner_model(flux_model)
    dm._pulid_klein_data = {
        "module"     : pulid_module,
        "embedding"  : id_embedding,   # [B, num_tokens, dim]
        "weight"     : weight,
        "sigma_start": sigma_start,
        "sigma_end"  : sigma_end,
    }

    # ── Patch des double_blocks ────────────────────────────────────────────
    original_double_forwards = {}

    def make_double_patch(block_idx, ca_idx):
        def patched_forward(img, txt, vec, **kwargs):
            data = getattr(dm, "_pulid_klein_data", None)
            out_img, out_txt = original_double_forwards[block_idx](
                img, txt, vec, **kwargs
            )
            if data is None:
                return out_img, out_txt

            # Vérifier sigma
            sigma = kwargs.get("timestep", None)
            if sigma is not None:
                s = float(sigma.max())
                if s < data["sigma_end"] or s > data["sigma_start"]:
                    return out_img, out_txt

            embed  = data["embedding"].to(out_img.device, dtype=out_img.dtype)
            ca_mod = data["module"].pulid_ca_double
            if ca_idx < len(ca_mod):
                correction = ca_mod[ca_idx](out_img, embed)
                out_img = out_img + data["weight"] * correction
            return out_img, out_txt
        return patched_forward

    # Récupérer les double_blocks
    if hasattr(dm, "transformer_blocks"):
        blocks = dm.transformer_blocks
    elif hasattr(dm, "double_blocks"):
        blocks = dm.double_blocks
    else:
        blocks = []
        logging.warning("[PuLID-Flux2Klein] Impossible de trouver double_blocks!")

    double_interval = pulid_module.double_interval
    ca_idx = 0
    for i, block in enumerate(blocks):
        if i % double_interval == 0:
            original_double_forwards[i] = block.forward
            block.forward = make_double_patch(i, ca_idx)
            ca_idx += 1

    # ── Patch des single_blocks ────────────────────────────────────────────
    original_single_forwards = {}

    def make_single_patch(block_idx, ca_idx):
        def patched_forward(x, vec, **kwargs):
            data = getattr(dm, "_pulid_klein_data", None)
            out = original_single_forwards[block_idx](x, vec, **kwargs)
            if data is None:
                return out

            sigma = kwargs.get("timestep", None)
            if sigma is not None:
                s = float(sigma.max())
                if s < data["sigma_end"] or s > data["sigma_start"]:
                    return out

            # Dans Flux.2 Klein les single_blocks reçoivent img+txt concaténés
            # On ne corrige que la partie image (premiers N_img tokens)
            embed  = data["embedding"].to(out.device, dtype=out.dtype)
            ca_mod = data["module"].pulid_ca_single
            if ca_idx < len(ca_mod):
                # Estimation: les tokens images = out.shape[1] - 512 (txt tokens)
                n_txt = 512
                n_img = out.shape[1] - n_txt
                img_part = out[:, :n_img, :]
                correction = ca_mod[ca_idx](img_part, embed)
                out = out.clone()
                out[:, :n_img, :] = img_part + data["weight"] * correction
            return out
        return patched_forward

    if hasattr(dm, "single_transformer_blocks"):
        s_blocks = dm.single_transformer_blocks
    elif hasattr(dm, "single_blocks"):
        s_blocks = dm.single_blocks
    else:
        s_blocks = []
        logging.warning("[PuLID-Flux2Klein] Impossible de trouver single_blocks!")

    # Single blocks désactivés pour éviter contamination globale
    # (poids non entraînés sur Klein → artefacts visuels)
    # single_interval = pulid_module.single_interval
    # À réactiver après entraînement des poids natifs Klein
    pass

    # Retourner une fonction de cleanup
    def unpatch():
        for i, fn in original_double_forwards.items():
            blocks[i].forward = fn
        for i, fn in original_single_forwards.items():
            s_blocks[i].forward = fn
        if hasattr(dm, "_pulid_klein_data"):
            del dm._pulid_klein_data

    return unpatch


# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 : Chargeur InsightFace
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDKleinInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["CUDA", "CPU", "ROCM"],),
            }
        }

    RETURN_TYPES  = ("INSIGHTFACE",)
    RETURN_NAMES  = ("face_analysis",)
    FUNCTION      = "load"
    CATEGORY      = "PuLID-Flux2Klein"

    def load(self, provider):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "[PuLID-Flux2Klein] insightface not installed. "
                "Run: pip install insightface onnxruntime-gpu"
            )

        providers = [provider + "ExecutionProvider", "CPUExecutionProvider"]

        # Try antelopev2 first (best accuracy), fall back to buffalo_l
        for model_name in ["antelopev2", "buffalo_l"]:
            try:
                model = FaceAnalysis(
                    name=model_name,
                    root=INSIGHTFACE_DIR,
                    providers=providers,
                )
                model.prepare(ctx_id=0, det_size=(640, 640))
                if model_name == "buffalo_l":
                    logging.warning(
                        "[PuLID-Flux2Klein] AntelopeV2 not found, using buffalo_l as fallback. "
                        "For best results, install AntelopeV2 from: "
                        "https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2"
                    )
                else:
                    logging.info("[PuLID-Flux2Klein] InsightFace AntelopeV2 loaded ✅")
                return (model,)
            except Exception as e:
                logging.warning(f"[PuLID-Flux2Klein] Could not load {model_name}: {e}")
                continue

        raise RuntimeError(
            "[PuLID-Flux2Klein] Could not load any InsightFace model. "
            "Please install AntelopeV2 from: "
            "https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 : Chargeur EVA-CLIP
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDKleinEVACLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES  = ("EVA_CLIP",)
    RETURN_NAMES  = ("eva_clip",)
    FUNCTION      = "load"
    CATEGORY      = "PuLID-Flux2Klein"

    def load(self):
        device = comfy.model_management.get_torch_device()
        model  = load_eva_clip(device)
        if model is None:
            raise RuntimeError(
                "[PuLID-Flux2Klein] EVA-CLIP non disponible. "
                "Installez eva_clip: pip install git+https://github.com/baaivision/EVA.git#subdirectory=EVA-CLIP"
            )
        return (model,)


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 : Chargeur poids PuLID-Flux2Klein
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDKleinModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        pulid_files = [
            f for f in os.listdir(PULID_DIR)
            if f.endswith((".safetensors", ".bin", ".pt"))
        ] if os.path.exists(PULID_DIR) else ["(aucun fichier trouvé)"]

        return {
            "required": {
                "pulid_file": (pulid_files,),
                "model_variant": (["klein_4B (dim=3072)", "klein_9B (dim=4096)"],),
            }
        }

    RETURN_TYPES  = ("PULID_KLEIN",)
    RETURN_NAMES  = ("pulid_model",)
    FUNCTION      = "load"
    CATEGORY      = "PuLID-Flux2Klein"

    def load(self, pulid_file, model_variant):
        path = os.path.join(PULID_DIR, pulid_file)
        dim  = 3072 if "4B" in model_variant else 4096

        if not os.path.exists(path):
            logging.warning(
                f"[PuLID-Flux2Klein] Fichier {path} non trouvé. "
                "Création d'un modèle vierge (non entraîné)."
            )
            model = PuLIDFlux2Klein(dim=dim)
        else:
            # Charger safetensors ou torch
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(path, device="cpu")
                dim_detected = state.get("id_former.latents",
                                         torch.zeros(1, 1, dim)).shape[-1]
                model = PuLIDFlux2Klein(dim=dim_detected)
                model.load_state_dict(state, strict=False)
            else:
                model = PuLIDFlux2Klein.from_pretrained(path)

        model.eval()
        return (model,)


# ──────────────────────────────────────────────────────────────────────────────
# NODE 4 : Application PuLID sur le modèle Flux.2 Klein
# ──────────────────────────────────────────────────────────────────────────────

class ApplyPuLIDFlux2Klein:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"        : ("MODEL",),
                "pulid_model"  : ("PULID_KLEIN",),
                "eva_clip"     : ("EVA_CLIP",),
                "face_analysis": ("INSIGHTFACE",),
                "image"        : ("IMAGE",),
                "weight"       : ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.05}),
                "start_at"     : ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at"       : ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "face_index": ("INT", {"default": 0, "min": 0, "max": 9}),
            }
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "apply"
    CATEGORY      = "PuLID-Flux2Klein"

    def apply(self, model, pulid_model, eva_clip, face_analysis,
              image, weight, start_at, end_at, face_index=0):

        device = comfy.model_management.get_torch_device()
        dtype  = comfy.model_management.unet_dtype()
        # Flux.2 Klein utilise bfloat16 — forcer pour éviter les erreurs de dtype
        dtype = torch.bfloat16

        # ── 1. Détection et embedding InsightFace ─────────────────────────
        img_np = (image[0].numpy() * 255).astype(np.uint8)

        faces = face_analysis.get(img_np)
        if not faces:
            logging.warning("[PuLID-Flux2Klein] Aucun visage détecté. Modèle non modifié.")
            return (model,)

        # Trier par taille de bounding box, choisir face_index
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)
        face_idx = min(face_index, len(faces) - 1)
        face     = faces[face_idx]

        id_embed_raw = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
        # Normaliser l'embedding InsightFace
        id_embed_raw = F.normalize(id_embed_raw, dim=-1)

        # ── 2. EVA-CLIP features ──────────────────────────────────────────
        # Crop du visage pour EVA-CLIP
        x1, y1, x2, y2 = face.bbox.astype(int)
        margin = int(max(x2 - x1, y2 - y1) * 0.2)
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(img_np.shape[1], x2 + margin)
        y2 = min(img_np.shape[0], y2 + margin)

        face_crop = image[:1, y1:y2, x1:x2, :]  # [1, H, W, 3]
        if face_crop.shape[1] == 0 or face_crop.shape[2] == 0:
            face_crop = image[:1]  # fallback image complète

        clip_embed, _ = encode_image_eva_clip(eva_clip, face_crop, device, dtype)
        # clip_embed: [1, 768] (EVA02-CLIP-L-14-336 output dim)

        # ── 3. IDFormer → tokens embedding ───────────────────────────────
        pulid_model = pulid_model.to(device, dtype=dtype)
        with torch.no_grad():
            id_tokens = pulid_model.id_former(id_embed_raw, clip_embed)
            # id_tokens: [1, num_tokens, dim]

        # ── 4. Patcher le modèle via ComfyUI ModelPatcher ─────────────────
        # On clone le ModelPatcher pour ne pas affecter l'original
        work_model = model.clone()

        # Sigma range (Flux.2 Klein utilise timestep [0, 1])
        sigma_start = start_at
        sigma_end   = end_at

        # Cleanup: supprimer les anciens patches pour éviter l'accumulation
        # qui cause l'image verte quand on change le weight entre les runs
        dm = _get_flux2_inner_model(work_model)
        if hasattr(dm, "_pulid_klein_unpatchers"):
            logging.info("[PuLID-Flux2Klein] Cleanup patches précédents...")
            for old_unpatch in dm._pulid_klein_unpatchers:
                try:
                    old_unpatch()
                except Exception:
                    pass
            dm._pulid_klein_unpatchers = []
        if hasattr(dm, "_pulid_klein_data"):
            del dm._pulid_klein_data

        # Appliquer le nouveau patch
        unpatch = patch_flux2klein_forward(
            work_model,
            pulid_model,
            id_tokens,
            weight,
            sigma_start,
            sigma_end,
        )

        # Stocker unpatch pour cleanup au prochain run
        if not hasattr(dm, "_pulid_klein_unpatchers"):
            dm._pulid_klein_unpatchers = []
        dm._pulid_klein_unpatchers.append(unpatch)

        logging.info(
            f"[PuLID-Flux2Klein] ✅ PuLID appliqué. "
            f"weight={weight}, face_idx={face_idx}/{len(faces)-1}, "
            f"sigma=[{sigma_start:.2f},{sigma_end:.2f}]"
        )

        return (work_model,)


# ──────────────────────────────────────────────────────────────────────────────
# NODE 5 (bonus) : Visualisation du visage détecté
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDKleinFacePreview:
    """
    Affiche les bounding boxes détectées sur l'image d'entrée.
    Utile pour débugger la détection de visage.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_analysis": ("INSIGHTFACE",),
                "image"        : ("IMAGE",),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("debug_image",)
    FUNCTION      = "preview"
    CATEGORY      = "PuLID-Flux2Klein"
    OUTPUT_NODE   = True

    def preview(self, face_analysis, image):
        try:
            import cv2
        except ImportError:
            logging.warning("[PuLID-Flux2Klein] cv2 non disponible pour preview.")
            return (image,)

        img_np  = (image[0].numpy() * 255).astype(np.uint8).copy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        faces = face_analysis.get(img_np)
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"Face {i}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)


# ──────────────────────────────────────────────────────────────────────────────
# Enregistrement des nodes ComfyUI
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "PuLIDKleinInsightFaceLoader" : PuLIDKleinInsightFaceLoader,
    "PuLIDKleinEVACLIPLoader"     : PuLIDKleinEVACLIPLoader,
    "PuLIDKleinModelLoader"       : PuLIDKleinModelLoader,
    "ApplyPuLIDFlux2Klein"        : ApplyPuLIDFlux2Klein,
    "PuLIDKleinFacePreview"       : PuLIDKleinFacePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLIDKleinInsightFaceLoader" : "Load InsightFace (PuLID Klein)",
    "PuLIDKleinEVACLIPLoader"     : "Load EVA-CLIP (PuLID Klein)",
    "PuLIDKleinModelLoader"       : "Load PuLID Flux.2 Klein Model",
    "ApplyPuLIDFlux2Klein"        : "Apply PuLID ✦ Flux.2 Klein",
    "PuLIDKleinFacePreview"       : "PuLID Klein — Face Debug Preview",
}
