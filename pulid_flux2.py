"""
ComfyUI-PuLID-Flux2
========================
Custom node PuLID for FLUX.2 — supports Klein 4B, Klein 9B, Dev 32B (distilled & base).

Architecture de FLUX.2 (source: Black Forest Labs / HuggingFace diffusers):
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
  - On patch le forward des double_blocks ET single_blocks via model patching ComfyUI
  - Le perceiver reçoit les id_embedding (512-dim InsightFace + EVA-CLIP features)
    et génère une correction additive sur les image tokens

CHANGELOG:
  v0.3.0:
    - FIX CRITIQUE : Single blocks maintenant patchés (double_blocks seuls = ~50% de l'injection)
    - FIX CRITIQUE : Half vs BFloat16 — cast dynamique dans PerceiverAttentionCA.forward()
    - FIX : Sigma range logique corrigée (était inversée)
    - FIX : EVA-CLIP output 3D géré (extraction CLS token si [B, N, D])
  v0.2.1:
    - Fix dimension mismatch when switching Flux2 Klein ↔ Dev
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

        FIX v0.3.0: Cast dynamique vers le dtype des poids du module.
        Règle le crash "expected scalar type Half but found BFloat16" (issue #4)
        quand le modèle tourne en float16 alors que le module est en bfloat16.
        """
        B, N, C = x.shape

        # ── FIX CRITIQUE : cast vers le dtype des poids pour éviter Half/BFloat16 mismatch ──
        weight_dtype = self.norm1.weight.dtype
        x = x.to(dtype=weight_dtype)
        context = context.to(dtype=weight_dtype)
        # ────────────────────────────────────────────────────────────────────────────────────

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
# PuLID Flux2 (main weights module)
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDFlux2(nn.Module):
    """
    Conteneur principal du module PuLID pour Flux.2 (Klein 4B/9B, Dev 32B).
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
    def from_pretrained(cls, path: str, map_location="cpu") -> "PuLIDFlux2":
        state = torch.load(path, map_location=map_location, weights_only=True)
        # Déterminer la dim depuis les poids
        try:
            dim = state["id_former.latents"].shape[-1]
        except KeyError:
            dim = 3072
        model = cls(dim=dim)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logging.warning(f"[PuLID-Flux2] Poids manquants : {missing[:5]}")
        if unexpected:
            logging.warning(f"[PuLID-Flux2] Poids inattendus : {unexpected[:5]}")
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

        logging.info("[PuLID-Flux2] Chargement EVA02-CLIP-L-14-336 (merged2b_s6b_b61k)...")
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k",
        )
        visual = model.visual
        visual.eval().to(device)
        logging.info("[PuLID-Flux2] EVA02-CLIP-L-14-336 chargé ✅")
        return visual

    except Exception as e:
        logging.warning(f"[PuLID-Flux2] EVA-CLIP non disponible: {e}")
        return None


def encode_image_eva_clip(eva_clip, image: torch.Tensor,
                          device, dtype) -> tuple:
    """
    image : [B, H, W, C] float32 [0,1]
    retourne (id_cond_vit [B, 768], None)

    FIX v0.3.0: Gestion du cas où EVA-CLIP retourne [B, N, D] (tokens de séquence)
    au lieu de [B, D]. On extrait le CLS token (index 0) dans ce cas.
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

    # ── FIX v0.3.0 : EVA-CLIP peut retourner [B, N, D] selon la config ──
    # On extrait le CLS token (premier token) si c'est le cas
    if id_cond_vit.dim() == 3:
        logging.debug(
            f"[PuLID-Flux2] EVA-CLIP output 3D détecté {tuple(id_cond_vit.shape)}, "
            "extraction CLS token (index 0)"
        )
        id_cond_vit = id_cond_vit[:, 0, :]  # [B, N, D] → [B, D]
    # ────────────────────────────────────────────────────────────────────

    # Forcer bfloat16 pour compatibilité avec Flux.2 Klein
    target_dtype = torch.bfloat16 if dtype == torch.bfloat16 else dtype
    return id_cond_vit.to(device, dtype=target_dtype), None


# ──────────────────────────────────────────────────────────────────────────────
# Patching du modèle Flux.2 Klein
# ──────────────────────────────────────────────────────────────────────────────

def _get_flux2_inner_model(model):
    """Remonte jusqu'au vrai objet transformer de ComfyUI."""
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "diffusion_model"):
        return m.diffusion_model
    return m


def _detect_flux2_variant(dm) -> tuple:
    """
    Détecte automatiquement le variant Flux.2 et sa dimension cachée.
    Retourne (variant_name, hidden_dim)
    - Klein 4B  : 5 double,  20 single, dim=3072
    - Klein 9B  : 8 double,  24 single, dim=4096
    - Dev 32B   : 8 double,  48 single, dim=6144
    """
    double_blocks = getattr(dm, "transformer_blocks", None) or getattr(dm, "double_blocks", [])
    single_blocks = getattr(dm, "single_transformer_blocks", None) or getattr(dm, "single_blocks", [])

    n_double = len(double_blocks)
    n_single = len(single_blocks)

    logging.info(f"[PuLID-Flux2] Detected: {n_double} double blocks, {n_single} single blocks")

    if n_double <= 5 and n_single <= 20:
        return "klein_4b", 3072
    elif n_double <= 8 and n_single <= 24:
        return "klein_9b", 4096
    elif n_single >= 40:
        return "flux2_dev", 6144
    else:
        logging.warning(f"[PuLID-Flux2] Unknown variant ({n_double}d/{n_single}s), defaulting to klein_9b")
        return "klein_9b", 4096


def patch_flux2_forward(flux_model, pulid_module, id_embedding,
                        weight, sigma_start, sigma_end):

    # récupérer le vrai modèle flux
    dm = _get_flux2_inner_model(flux_model)

    # détecter automatiquement le variant
    variant, detected_dim = _detect_flux2_variant(dm)
    logging.info(f"[PuLID-Flux2] Model variant: {variant}, dim: {detected_dim}")

    # adapter les modules CA si la dimension change
    current_dim = pulid_module.id_former.latents.shape[-1]

    # Sauvegarder les CA originaux si pas encore fait
    if not hasattr(pulid_module, "_original_pulid_ca_double"):
        pulid_module._original_pulid_ca_double = pulid_module.pulid_ca_double
        pulid_module._original_pulid_ca_single = pulid_module.pulid_ca_single
        pulid_module._original_dim = current_dim

    if detected_dim != getattr(pulid_module, "_current_adapted_dim", current_dim):
        device = pulid_module.id_former.latents.device
        dtype  = pulid_module.id_former.latents.dtype

        if detected_dim == pulid_module._original_dim:
            # Restaurer les CA originaux
            pulid_module.pulid_ca_double = pulid_module._original_pulid_ca_double
            pulid_module.pulid_ca_single = pulid_module._original_pulid_ca_single
            logging.info(f"[PuLID-Flux2] PerceiverCA restored to original dim={detected_dim}")
        else:
            # Créer de nouveaux CA pour la dim cible
            logging.warning(
                f"[PuLID-Flux2] Adapting PerceiverCA {pulid_module._original_dim} → {detected_dim}"
            )
            pulid_module.pulid_ca_double = nn.ModuleList([
                PerceiverAttentionCA(dim=detected_dim).to(device, dtype=dtype)
                for _ in range(len(pulid_module._original_pulid_ca_double))
            ])
            pulid_module.pulid_ca_single = nn.ModuleList([
                PerceiverAttentionCA(dim=detected_dim).to(device, dtype=dtype)
                for _ in range(len(pulid_module._original_pulid_ca_single))
            ])
            logging.info(f"[PuLID-Flux2] PerceiverCA adapted to dim={detected_dim} ✅")

        pulid_module._current_adapted_dim = detected_dim

    # stocker données runtime
    dm._pulid_flux2_data = {
        "module": pulid_module,
        "embedding": id_embedding,
        "weight": weight,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
    }

    # ─────────────────────────────────────────────────────────────────────
    # FIX v0.3.0 : helper pour vérifier la sigma range (logique corrigée)
    # Avant : s < sigma_end or s > sigma_start  → EXCLUAIT la plage [start, end]
    # Après : not (sigma_end <= s <= sigma_start) → EXCLUT en dehors de la plage
    # ─────────────────────────────────────────────────────────────────────
    def _sigma_out_of_range(sigma_tensor, data):
        """Retourne True si on doit skipper ce step (sigma hors plage active)."""
        if sigma_tensor is None:
            return False
        s = float(sigma_tensor.max())
        return not (data["sigma_end"] <= s <= data["sigma_start"])

    # ─────────────────────────────
    # patch double blocks
    # ─────────────────────────────
    original_double_forwards = {}

    def make_double_patch(block_idx, ca_idx):
        def patched_forward(img, txt, vec, **kwargs):

            data = getattr(dm, "_pulid_flux2_data", None)

            out_img, out_txt = original_double_forwards[block_idx](
                img, txt, vec, **kwargs
            )

            if data is None:
                return out_img, out_txt

            if _sigma_out_of_range(kwargs.get("timestep", None), data):
                return out_img, out_txt

            embed = data["embedding"].to(out_img.device, dtype=out_img.dtype)
            ca_mod = data["module"].pulid_ca_double

            if ca_idx < len(ca_mod):
                correction = ca_mod[ca_idx](out_img, embed)
                out_img = out_img + data["weight"] * correction

            return out_img, out_txt

        return patched_forward

    # récupérer blocks flux
    if hasattr(dm, "transformer_blocks"):
        double_blocks = dm.transformer_blocks
    elif hasattr(dm, "double_blocks"):
        double_blocks = dm.double_blocks
    else:
        double_blocks = []

    double_interval = pulid_module.double_interval
    ca_idx = 0

    for i, block in enumerate(double_blocks):
        if i % double_interval == 0:
            original_double_forwards[i] = block.forward
            block.forward = make_double_patch(i, ca_idx)
            ca_idx += 1

    # ─────────────────────────────────────────────────────────────────────
    # FIX v0.3.0 : patch single blocks (MANQUANT dans v0.2.x)
    # Klein 4B = 20 single blocks, Klein 9B = 24 → ~50% de l'injection
    # était complètement ignorée avant ce fix !
    # ─────────────────────────────────────────────────────────────────────
    original_single_forwards = {}

    def make_single_patch(block_idx, ca_idx):
        def patched_forward(hidden_states, temb, **kwargs):

            data = getattr(dm, "_pulid_flux2_data", None)

            out = original_single_forwards[block_idx](
                hidden_states, temb, **kwargs
            )

            if data is None:
                return out

            if _sigma_out_of_range(kwargs.get("timestep", None), data):
                return out

            # out peut être un tuple (hidden_states, residual) selon l'implémentation
            if isinstance(out, tuple):
                out_hidden = out[0]
                embed = data["embedding"].to(out_hidden.device, dtype=out_hidden.dtype)
                ca_mod = data["module"].pulid_ca_single

                if ca_idx < len(ca_mod):
                    correction = ca_mod[ca_idx](out_hidden, embed)
                    out_hidden = out_hidden + data["weight"] * correction

                return (out_hidden,) + out[1:]
            else:
                embed = data["embedding"].to(out.device, dtype=out.dtype)
                ca_mod = data["module"].pulid_ca_single

                if ca_idx < len(ca_mod):
                    correction = ca_mod[ca_idx](out, embed)
                    out = out + data["weight"] * correction

                return out

        return patched_forward

    # récupérer single blocks
    if hasattr(dm, "single_transformer_blocks"):
        single_blocks = dm.single_transformer_blocks
    elif hasattr(dm, "single_blocks"):
        single_blocks = dm.single_blocks
    else:
        single_blocks = []
        logging.warning("[PuLID-Flux2] Aucun single_block trouvé — injection partielle seulement")

    single_interval = pulid_module.single_interval
    ca_idx_single = 0

    for i, block in enumerate(single_blocks):
        if i % single_interval == 0:
            original_single_forwards[i] = block.forward
            block.forward = make_single_patch(i, ca_idx_single)
            ca_idx_single += 1

    logging.info(
        f"[PuLID-Flux2] Patched: {len(original_double_forwards)} double blocks, "
        f"{len(original_single_forwards)} single blocks"
    )

    # ─────────────────────────────
    # cleanup / unpatch
    # ─────────────────────────────
    def unpatch():
        for i, fn in original_double_forwards.items():
            double_blocks[i].forward = fn

        for i, fn in original_single_forwards.items():
            single_blocks[i].forward = fn

        if hasattr(dm, "_pulid_flux2_data"):
            del dm._pulid_flux2_data

    return unpatch


# ──────────────────────────────────────────────────────────────────────────────
# NODE 1 : InsightFace Loader
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDInsightFaceLoader:
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
    CATEGORY      = "PuLID-Flux2"

    def load(self, provider):
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "[PuLID-Flux2] insightface not installed. "
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
                        "[PuLID-Flux2] AntelopeV2 not found, using buffalo_l as fallback. "
                        "For best results, install AntelopeV2 from: "
                        "https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2"
                    )
                else:
                    logging.info("[PuLID-Flux2] InsightFace AntelopeV2 loaded ✅")
                return (model,)
            except Exception as e:
                logging.warning(f"[PuLID-Flux2] Could not load {model_name}: {e}")
                continue

        raise RuntimeError(
            "[PuLID-Flux2] Could not load any InsightFace model. "
            "Please install AntelopeV2 from: "
            "https://huggingface.co/MonsterMMORPG/InstantID_Models/tree/main/models/antelopev2"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NODE 2 : EVA-CLIP Loader
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDEVACLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES  = ("EVA_CLIP",)
    RETURN_NAMES  = ("eva_clip",)
    FUNCTION      = "load"
    CATEGORY      = "PuLID-Flux2"

    def load(self):
        device = comfy.model_management.get_torch_device()
        model  = load_eva_clip(device)
        if model is None:
            raise RuntimeError(
                "[PuLID-Flux2] EVA-CLIP non disponible. "
                "Installez open_clip: pip install open-clip-torch"
            )
        return (model,)


# ──────────────────────────────────────────────────────────────────────────────
# NODE 3 : PuLID Model Loader
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        pulid_files = [
            f for f in os.listdir(PULID_DIR)
            if f.endswith((".safetensors", ".bin", ".pt"))
        ] if os.path.exists(PULID_DIR) else ["(aucun fichier trouvé)"]

        return {
            "required": {
                "pulid_file": (pulid_files,),
                "model_variant": (["auto (recommended)", "klein_4B (dim=3072)", "klein_9B (dim=4096)", "flux2_dev (dim=6144)"],),
            }
        }

    RETURN_TYPES  = ("PULID_KLEIN",)
    RETURN_NAMES  = ("pulid_model",)
    FUNCTION      = "load"
    CATEGORY      = "PuLID-Flux2"

    def load(self, pulid_file, model_variant):
        path = os.path.join(PULID_DIR, pulid_file)
        # Auto = on laisse la détection au runtime, on utilise dim=4096 par défaut
        if "4B" in model_variant:
            dim = 3072
        else:
            dim = 4096  # Klein 9B, Dev 32B, et auto utilisent tous dim=4096

        if not os.path.exists(path):
            logging.warning(
                f"[PuLID-Flux2] Fichier {path} non trouvé. "
                "Création d'un modèle vierge (non entraîné)."
            )
            model = PuLIDFlux2(dim=dim)
        else:
            # Charger safetensors ou torch
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(path, device="cpu")
                dim_detected = state.get("id_former.latents",
                                         torch.zeros(1, 1, dim)).shape[-1]
                model = PuLIDFlux2(dim=dim_detected)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    logging.warning(f"[PuLID-Flux2] Poids manquants ({len(missing)}) : {missing[:3]}")
                if unexpected:
                    logging.warning(f"[PuLID-Flux2] Poids inattendus ({len(unexpected)}) : {unexpected[:3]}")
            else:
                model = PuLIDFlux2.from_pretrained(path)

        model.eval()
        return (model,)


# ──────────────────────────────────────────────────────────────────────────────
# NODE 4 : Apply PuLID on Flux.2 model
# ──────────────────────────────────────────────────────────────────────────────

class ApplyPuLIDFlux2:
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
    CATEGORY      = "PuLID-Flux2"

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
            logging.warning("[PuLID-Flux2] Aucun visage détecté. Modèle non modifié.")
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

        # Utiliser autocast pour gérer les conversions de dtype automatiquement
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type=='cuda')):
                id_tokens = pulid_model.id_former(id_embed_raw, clip_embed)
            # id_tokens: [1, num_tokens, dim]

        # ── 4. Patcher le modèle via ComfyUI ModelPatcher ─────────────────
        # On clone le ModelPatcher pour ne pas affecter l'original
        work_model = model.clone()

        # Sigma range (Flux.2 Klein utilise timestep [0, 1])
        sigma_start = start_at
        sigma_end   = end_at

        # Cleanup: supprimer les anciens patches pour éviter l'accumulation
        dm = _get_flux2_inner_model(work_model)

        if hasattr(dm, "_pulid_flux2_unpatchers"):
            logging.info("[PuLID-Flux2] Cleanup patches précédents...")
            for old_unpatch in dm._pulid_flux2_unpatchers:
                try:
                    old_unpatch()
                except Exception:
                    pass
            dm._pulid_flux2_unpatchers = []

        if hasattr(dm, "_pulid_flux2_data"):
            del dm._pulid_flux2_data

        # Détecter la dim du modèle
        variant, detected_dim = _detect_flux2_variant(dm)

        # Projeter les id_tokens vers la dim du modèle si nécessaire
        id_token_dim = id_tokens.shape[-1]

        if id_token_dim != detected_dim:
            logging.warning(
                f"[PuLID-Flux2] ⚠️ Dim mismatch : id_tokens={id_token_dim} vs modèle={detected_dim}. "
                f"Projection aléatoire — résultats sous-optimaux. "
                f"Entraîner avec --dim {detected_dim} pour de meilleurs résultats."
            )
            proj = nn.Linear(id_token_dim, detected_dim, bias=False).to(device, dtype=dtype)
            torch.nn.init.normal_(proj.weight, std=0.01)

            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type=='cuda')):
                    id_tokens = proj(id_tokens)

        # Appliquer le nouveau patch
        unpatch = patch_flux2_forward(
            work_model,
            pulid_model,
            id_tokens,
            weight,
            sigma_start,
            sigma_end,
        )

        # Stocker unpatch pour cleanup au prochain run
        if not hasattr(dm, "_pulid_flux2_unpatchers"):
            dm._pulid_flux2_unpatchers = []

        dm._pulid_flux2_unpatchers.append(unpatch)

        logging.info(
            f"[PuLID-Flux2] ✅ PuLID applied. "
            f"model={variant} (dim={detected_dim}), weight={weight}, "
            f"face_idx={face_idx}/{len(faces)-1}, "
            f"sigma=[{sigma_start:.2f},{sigma_end:.2f}]"
        )

        return (work_model,)


# ──────────────────────────────────────────────────────────────────────────────
# NODE 5 (bonus) : Visualisation du visage détecté
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDFacePreview:
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
    CATEGORY      = "PuLID-Flux2"
    OUTPUT_NODE   = True

    def preview(self, face_analysis, image):
        try:
            import cv2
        except ImportError:
            logging.warning("[PuLID-Flux2] cv2 non disponible pour preview.")
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
    "PuLIDInsightFaceLoader" : PuLIDInsightFaceLoader,
    "PuLIDEVACLIPLoader"     : PuLIDEVACLIPLoader,
    "PuLIDModelLoader"       : PuLIDModelLoader,
    "ApplyPuLIDFlux2"        : ApplyPuLIDFlux2,
    "PuLIDFacePreview"       : PuLIDFacePreview,
}

# ──────────────────────────────────────────────────────────────────────────────
# Version info
# ──────────────────────────────────────────────────────────────────────────────
__version__ = "0.3.0"
__supported_models__ = ["Flux.2 Klein 4B", "Flux.2 Klein 9B", "Flux.2 Dev 32B"]
__changelog__ = (
    "v0.3.0: "
    "Fix single blocks non patchés (injection à 100% au lieu de 50%) | "
    "Fix Half vs BFloat16 crash (issue #4) | "
    "Fix sigma range logique inversée | "
    "Fix EVA-CLIP output 3D (CLS token extraction)"
)

NODE_DISPLAY_NAME_MAPPINGS = {
    "PuLIDInsightFaceLoader" : "Load InsightFace (PuLID)",
    "PuLIDEVACLIPLoader"     : "Load EVA-CLIP (PuLID)",
    "PuLIDModelLoader"       : "Load PuLID ✦ Flux.2",
    "ApplyPuLIDFlux2"        : "Apply PuLID ✦ Flux.2",
    "PuLIDFacePreview"       : "PuLID — Face Debug Preview",
}
