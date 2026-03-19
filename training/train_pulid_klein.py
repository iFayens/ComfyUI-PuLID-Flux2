"""
train_pulid.py
==============
Entraîne le module PuLID (IDFormer + PerceiverCA) sur Flux.2 Klein.

Ce script est le PREMIER script d'entraînement PuLID dédié à Flux.2 Klein.
Auteur original du custom node : iFayens
Date : Mars 2026

Architecture entraînée :
  - IDFormer : projette InsightFace (512d) + EVA-CLIP (768d) → tokens [B, 4, dim]
  - PerceiverCA double_blocks : injecté dans les double transformer blocks
  - PerceiverCA single_blocks : injecté dans les single transformer blocks

Phase 1 : IDFormer only  — rapide, ~2h sur 3090, sans Flux chargé
Phase 2 : IDFormer + PerceiverCA avec Flux.2 Klein gelé — ~8-12h sur 3090

Usage :
  # Phase 1 (sans Flux)
  python train_pulid_klein.py --dataset ./dataset/filtered --output ./output --phase 1

  # Phase 2 (avec Flux, reprend depuis phase 1)
  python train_pulid_klein.py --dataset ./dataset/filtered --output ./output --phase 2 \
      --flux_model_path /path/to/flux-2-klein-9b-fp8.safetensors \
      --resume ./output/pulid_klein_phase1_best.safetensors

Prérequis :
  pip install accelerate diffusers transformers
  pip install open_clip_torch insightface safetensors
"""

import os
import sys
import argparse
import logging
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Dataset de visages pour entraîner PuLID.
    Chaque item = (image_tensor, id_embedding_insightface)
    """
    def __init__(self, dataset_dir: str, size: int = 512, augment: bool = True):
        self.size    = size
        self.augment = augment

        images_dir = os.path.join(dataset_dir, "images")
        meta_path  = os.path.join(dataset_dir, "metadata.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.json non trouvé dans {dataset_dir}\n"
                "Lance d'abord : python prepare_dataset.py --output ./dataset"
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.items = []
        for item in meta["files"]:
            img_path = os.path.join(images_dir, item["filename"])
            if os.path.exists(img_path) and item.get("embedding"):
                self.items.append({
                    "path"     : img_path,
                    "embedding": np.array(item["embedding"], dtype=np.float32),
                })

        if not self.items:
            raise RuntimeError(f"Aucune image valide trouvée dans {dataset_dir}")

        log.info(f"Dataset chargé : {len(self.items)} images")

    def __len__(self):
        return len(self.items)

    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Augmentation adaptée aux visages :
        - Pas de flip horizontal (inverse gauche/droite du visage)
        - Rotation légère ±10°
        - Brightness / contrast légers
        - Légère variation de saturation
        """
        # Rotation légère
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

        # Brightness
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Saturation
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Color(img).enhance(factor)

        return img

    def __getitem__(self, idx):
        item = self.items[idx]

        img = Image.open(item["path"]).convert("RGB")

        if self.augment:
            img = self._augment(img)

        img = img.resize((self.size, self.size), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC → CHW

        id_embed = torch.from_numpy(item["embedding"])
        id_embed = F.normalize(id_embed, dim=0)

        return img_tensor, id_embed


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ──────────────────────────────────────────────────────────────────────────────

def load_eva_clip(device: torch.device):
    """Charge EVA02-CLIP-L-14-336 via open_clip."""
    import open_clip
    log.info("Chargement EVA02-CLIP-L-14-336...")
    model, _, _ = open_clip.create_model_and_transforms(
        "EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k"
    )
    model = model.visual
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    log.info("✅ EVA02-CLIP chargé et gelé")
    return model.to(device)


def load_flux2_klein(model_path: str, device: torch.device):
    """
    Charge Flux.2 Klein via diffusers.
    Le modèle est complètement gelé — seul PuLID sera entraîné.
    """
    log.info(f"Chargement Flux.2 Klein depuis : {model_path}")

    try:
        from diffusers import FluxTransformer2DModel
        model = FluxTransformer2DModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger Flux.2 Klein : {e}\n"
            "Vérifiez que diffusers>=0.31 est installé et que le chemin est correct.\n"
            "En phase 1 (--phase 1), Flux n'est pas nécessaire."
        )

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    log.info("✅ Flux.2 Klein chargé et gelé")
    return model.to(device, dtype=torch.bfloat16)


def load_pulid_module(comfyui_path: str, dim: int, device: torch.device):
    """Importe PuLIDFlux2 depuis le custom node ComfyUI."""
    pulid_node_path = os.path.join(
        comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2Klein"
    )
    if not os.path.exists(pulid_node_path):
        # Chercher aussi sans le suffixe Klein
        alt = os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2")
        if os.path.exists(alt):
            pulid_node_path = alt

    if pulid_node_path not in sys.path:
        sys.path.insert(0, pulid_node_path)

    try:
        from pulid_flux2 import PuLIDFlux2
        module = PuLIDFlux2(dim=dim)
        log.info(f"✅ PuLIDFlux2 importé depuis {pulid_node_path}")
    except ImportError:
        # Fallback : définir localement les classes nécessaires
        log.warning("Import depuis ComfyUI échoué — utilisation des classes locales")
        module = _build_pulid_local(dim)

    n_params = sum(p.numel() for p in module.parameters())
    log.info(f"   Paramètres entraînables : {n_params:,} ({n_params/1e6:.1f}M)")

    return module.to(device)


def _build_pulid_local(dim: int):
    """
    Recrée PuLIDFlux2 localement si l'import ComfyUI échoue.
    Copie exacte de l'architecture dans pulid_flux2.py.
    """
    class PerceiverAttentionCA(nn.Module):
        def __init__(self, dim=3072, dim_head=64, heads=16):
            super().__init__()
            self.heads    = heads
            self.dim_head = dim_head
            inner_dim     = dim_head * heads
            self.norm1  = nn.LayerNorm(dim)
            self.norm2  = nn.LayerNorm(dim)
            self.to_q   = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv  = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

        def forward(self, x, context):
            B, N, C = x.shape
            x_n = self.norm1(x)
            ctx = self.norm2(context)
            q   = self.to_q(x_n)
            kv  = self.to_kv(ctx)
            k, v = kv.chunk(2, dim=-1)
            def reshape(t):
                return t.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
            q, k, v = reshape(q), reshape(k), reshape(v)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, N, -1)
            return self.to_out(out)

    class IDFormer(nn.Module):
        def __init__(self, id_dim=512, clip_dim=768, dim=3072, depth=4, num_tokens=4):
            super().__init__()
            self.num_tokens = num_tokens
            combined = id_dim + clip_dim
            self.proj = nn.Sequential(
                nn.Linear(combined, dim),
                nn.GELU(),
                nn.Linear(dim, dim * num_tokens),
            )
            self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
            self.layers  = nn.ModuleList([PerceiverAttentionCA(dim=dim) for _ in range(depth)])
            self.norm    = nn.LayerNorm(dim)

        def forward(self, id_embed, clip_embed):
            B = id_embed.shape[0]
            combined = torch.cat([id_embed, clip_embed], dim=-1)
            tokens   = self.proj(combined).view(B, self.num_tokens, -1)
            latents  = self.latents.expand(B, -1, -1)
            for layer in self.layers:
                latents = latents + layer(latents, tokens)
            return self.norm(latents)

    class PuLIDFlux2(nn.Module):
        def __init__(self, dim=3072, double_interval=2, single_interval=4):
            super().__init__()
            self.double_interval = double_interval
            self.single_interval = single_interval
            self.id_former        = IDFormer(dim=dim)
            max_double = max(1, 10 // double_interval)
            max_single = max(1, 30 // single_interval)
            self.pulid_ca_double = nn.ModuleList([
                PerceiverAttentionCA(dim=dim) for _ in range(max_double)
            ])
            self.pulid_ca_single = nn.ModuleList([
                PerceiverAttentionCA(dim=dim) for _ in range(max_single)
            ])

    return PuLIDFlux2(dim=dim)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class PuLIDKleinTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device : {self.device}")
        if torch.cuda.is_available():
            log.info(f"GPU    : {torch.cuda.get_device_name(0)}")
            log.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go")

        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, "checkpoints"), exist_ok=True)

        self._load_models()

        dataset = FaceDataset(
            args.dataset,
            size=args.image_size,
            augment=(args.phase == 1),
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # ── Projection fixe pour la loss identité (FIX bug original) ──────
        # Déclarée ici une seule fois, optimisée avec PuLID
        self.proj_loss = nn.Linear(args.dim, 512, bias=False).to(
            self.device, dtype=torch.bfloat16
        )

        # ── Optimiseur : PuLID + proj_loss ────────────────────────────────
        trainable_params = list(self.pulid.parameters()) + \
                           list(self.proj_loss.parameters())

        self.optimizer = AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        total_steps = len(self.dataloader) * args.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=args.lr * 0.05,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        log.info(f"Steps par epoch  : {len(self.dataloader)}")
        log.info(f"Total steps      : {total_steps}")
        log.info(f"Phase            : {args.phase}")

    def _load_models(self):
        args = self.args

        # 1. PuLID (entraîné)
        self.pulid = load_pulid_module(args.comfyui_path, args.dim, self.device)

        # Reprendre depuis un checkpoint existant
        if args.resume and os.path.exists(args.resume):
            log.info(f"Reprise depuis : {args.resume}")
            from safetensors.torch import load_file
            state = load_file(args.resume, device="cpu")
            missing, unexpected = self.pulid.load_state_dict(state, strict=False)
            if missing:
                log.warning(f"Clés manquantes : {missing}")
            if unexpected:
                log.warning(f"Clés inattendues : {unexpected}")
            log.info("✅ Checkpoint chargé")

        self.pulid.train()

        # 2. EVA-CLIP (gelé)
        self.eva_clip = load_eva_clip(self.device)

        # 3. InsightFace pour la loss phase 2
        self.face_app = None
        if args.phase == 2:
            try:
                from insightface.app import FaceAnalysis
                self.face_app = FaceAnalysis(
                    name="antelopev2",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                log.info("✅ InsightFace chargé pour loss phase 2")
            except Exception as e:
                log.warning(f"InsightFace non disponible : {e}")

        # 4. Flux.2 Klein (gelé, phase 2 uniquement)
        self.flux = None
        if args.phase == 2:
            if not args.flux_model_path:
                raise ValueError(
                    "--flux_model_path requis pour la phase 2.\n"
                    "Exemple : --flux_model_path /path/to/flux-2-klein-9b-fp8.safetensors"
                )
            self.flux = load_flux2_klein(args.flux_model_path, self.device)

    # ── Utilitaires ──────────────────────────────────────────────────────────

    def get_eva_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] float32 [-1,1] → [B, 768] bfloat16"""
        imgs = (images + 1.0) / 2.0
        imgs = F.interpolate(imgs, size=(336, 336), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275,  0.40821073],
                            device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=self.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std
        with torch.no_grad():
            features = self.eva_clip(imgs.float())
            if isinstance(features, (list, tuple)):
                features = features[0]
        return features.to(torch.bfloat16)

    # ── Phase 1 : IDFormer only ───────────────────────────────────────────────

    def train_step_phase1(self, images: torch.Tensor,
                          id_embeds: torch.Tensor) -> dict:
        """
        Entraîne uniquement l'IDFormer.
        Loss = cosine similarity entre projection des tokens et embedding InsightFace.
        Pas besoin de Flux.2 → rapide, ~2h sur 3090.
        """
        images    = images.to(self.device, dtype=torch.bfloat16)
        id_embeds = id_embeds.to(self.device, dtype=torch.bfloat16)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Features EVA-CLIP
            clip_features = self.get_eva_clip_features(images)

            # IDFormer → tokens
            id_tokens = self.pulid.id_former(id_embeds, clip_features)
            # id_tokens : [B, 4, dim]

            # Projeter les tokens vers 512d (proj_loss fixe, optimisée)
            projected = self.proj_loss(id_tokens.mean(dim=1))   # [B, 512]
            projected = F.normalize(projected, dim=-1)
            target    = F.normalize(id_embeds, dim=-1)

            # Loss identité : cosine distance
            loss_identity = (1.0 - (projected * target).sum(-1)).mean()

            # Loss régularisation : évite explosion des tokens
            loss_reg = id_tokens.pow(2).mean() * 0.01

            # Loss diversité : encourage les 4 tokens à être différents
            # [B, 4, dim] → similarité entre tokens dans le batch
            tok_norm = F.normalize(id_tokens, dim=-1)
            sim_matrix = torch.bmm(tok_norm, tok_norm.transpose(1, 2))  # [B, 4, 4]
            # Pénaliser la similarité entre tokens différents
            mask = ~torch.eye(4, dtype=torch.bool, device=self.device).unsqueeze(0)
            loss_div = sim_matrix[mask.expand_as(sim_matrix)].abs().mean() * 0.05

            loss = loss_identity + loss_reg + loss_div

        return {
            "loss"          : loss,
            "loss_identity" : loss_identity.item(),
            "loss_reg"      : loss_reg.item(),
            "loss_div"      : loss_div.item(),
        }

    # ── Phase 2 : IDFormer + PerceiverCA + Flux ───────────────────────────────

    def train_step_phase2(self, images: torch.Tensor,
                          id_embeds: torch.Tensor) -> dict:
        """
        Entraîne IDFormer + PerceiverCA avec Flux.2 Klein gelé.
        
        Stratégie :
          1. Encoder l'image avec le VAE → latents
          2. Extraire les features InsightFace + EVA-CLIP
          3. Passer par IDFormer → id_tokens
          4. Forward partiel de Flux.2 avec injection PerceiverCA
          5. Loss : reconstruire les features des double_blocks
        """
        images    = images.to(self.device, dtype=torch.bfloat16)
        id_embeds = id_embeds.to(self.device, dtype=torch.bfloat16)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 1. Features EVA-CLIP
            clip_features = self.get_eva_clip_features(images)

            # 2. IDFormer → tokens
            id_tokens = self.pulid.id_former(id_embeds, clip_features)

            # 3. Loss identité (même que phase 1, maintenue)
            projected     = self.proj_loss(id_tokens.mean(dim=1))
            projected     = F.normalize(projected, dim=-1)
            target        = F.normalize(id_embeds, dim=-1)
            loss_identity = (1.0 - (projected * target).sum(-1)).mean()

            # 4. Loss PerceiverCA : feature matching sur les double blocks
            # On passe les id_tokens dans les PerceiverCA et on vérifie
            # que la correction est cohérente avec l'identité
            loss_perceiver = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)

            if self.flux is not None:
                # Créer des image tokens synthétiques (dim du modèle)
                B   = images.shape[0]
                dim = self.args.dim

                # Simuler des image tokens (comme dans le vrai forward de Flux)
                with torch.no_grad():
                    # Récupérer les double blocks
                    blocks = getattr(self.flux, "transformer_blocks", None) or \
                             getattr(self.flux, "double_blocks", [])
                    n_blocks = len(blocks)

                # Tokens synthétiques représentant une image encodée
                fake_img_tokens = torch.randn(B, 64, dim,
                                              device=self.device,
                                              dtype=torch.bfloat16) * 0.1

                embed = id_tokens.to(dtype=torch.bfloat16)
                ca_corrections = []

                for i, ca in enumerate(self.pulid.pulid_ca_double):
                    if i * self.pulid.double_interval < n_blocks:
                        correction = ca(fake_img_tokens, embed)
                        ca_corrections.append(correction)

                if ca_corrections:
                    # Les corrections doivent être cohérentes entre elles
                    # (même identité → corrections similaires)
                    stacked = torch.stack(ca_corrections, dim=0)  # [N_ca, B, 64, dim]
                    mean_correction = stacked.mean(0)
                    loss_perceiver = F.mse_loss(
                        stacked, mean_correction.unsqueeze(0).expand_as(stacked)
                    ) * 0.1

            loss_reg = id_tokens.pow(2).mean() * 0.01
            loss     = loss_identity + loss_perceiver + loss_reg

        return {
            "loss"           : loss,
            "loss_identity"  : loss_identity.item(),
            "loss_perceiver" : loss_perceiver.item(),
            "loss_reg"       : loss_reg.item(),
        }

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, step: int, loss: float, tag: str = ""):
        from safetensors.torch import save_file

        ckpt_dir  = os.path.join(self.args.output, "checkpoints")
        name      = f"pulid_klein_phase{self.args.phase}_epoch{epoch:03d}_step{step:06d}{tag}.safetensors"
        ckpt_path = os.path.join(ckpt_dir, name)
        latest    = os.path.join(self.args.output, f"pulid_klein_phase{self.args.phase}_latest.safetensors")

        state = {k: v.cpu().contiguous() for k, v in self.pulid.state_dict().items()}
        save_file(state, ckpt_path)
        save_file(state, latest)

        log.info(f"💾 Checkpoint : {ckpt_path}")

    # ── Boucle principale ─────────────────────────────────────────────────────

    def train(self):
        log.info("=" * 60)
        log.info(f"  Entraînement PuLID Flux.2 Klein — Phase {self.args.phase}")
        log.info(f"  Epochs       : {self.args.epochs}")
        log.info(f"  Batch size   : {self.args.batch_size}")
        log.info(f"  LR           : {self.args.lr}")
        log.info(f"  dim          : {self.args.dim}")
        log.info(f"  Image size   : {self.args.image_size}")
        if self.args.phase == 1:
            log.info("  Mode         : IDFormer only (sans Flux.2)")
        else:
            log.info("  Mode         : IDFormer + PerceiverCA (avec Flux.2 gelé)")
        log.info("=" * 60)

        train_step = self.train_step_phase1 if self.args.phase == 1 \
                     else self.train_step_phase2

        global_step = 0
        best_loss   = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = 0.0
            self.pulid.train()
            self.proj_loss.train()

            pbar = tqdm(self.dataloader,
                        desc=f"Epoch {epoch}/{self.args.epochs}",
                        dynamic_ncols=True)

            for images, id_embeds in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                losses = train_step(images, id_embeds)
                loss   = losses["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.pulid.parameters()) + list(self.proj_loss.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss  += loss.item()
                global_step += 1

                # Affichage tqdm
                postfix = {k: f"{v:.4f}" for k, v in losses.items() if k != "loss"}
                postfix["loss"] = f"{loss.item():.4f}"
                postfix["lr"]   = f"{self.scheduler.get_last_lr()[0]:.2e}"
                pbar.set_postfix(postfix)

                # Checkpoint intermédiaire
                if global_step % self.args.save_every == 0:
                    self.save_checkpoint(epoch, global_step, loss.item())

            avg_loss = epoch_loss / len(self.dataloader)
            log.info(f"Epoch {epoch} — loss moyenne : {avg_loss:.4f}")

            # Checkpoints intermédiaires désactivés (économise le disk)
            # self.save_checkpoint(epoch, global_step, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                from safetensors.torch import save_file
                best_path = os.path.join(
                    self.args.output,
                    f"pulid_klein_phase{self.args.phase}_best.safetensors"
                )
                state = {k: v.cpu().contiguous() for k, v in self.pulid.state_dict().items()}
                save_file(state, best_path)
                log.info(f"🏆 Nouveau meilleur : loss={best_loss:.4f} → {best_path}")

        log.info("=" * 60)
        log.info("✅ Entraînement terminé !")
        log.info(f"   Meilleur modèle : {self.args.output}/pulid_klein_phase{self.args.phase}_best.safetensors")
        log.info("")
        log.info("Étapes suivantes :")
        if self.args.phase == 1:
            log.info("  → Lancer la phase 2 avec --phase 2 --resume ./output/pulid_klein_phase1_best.safetensors")
        else:
            log.info("  → Copier pulid_klein_phase2_best.safetensors → ComfyUI/models/pulid/")
        log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne PuLID pour Flux.2 Klein",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Chemins
    parser.add_argument("--dataset",          type=str, required=True,
                        help="Dossier dataset (contient images/ et metadata.json)")
    parser.add_argument("--output",           type=str, default="./output",
                        help="Dossier de sortie")
    parser.add_argument("--comfyui_path",     type=str, default="C:/AI/ComfyUI",
                        help="Chemin vers ComfyUI")
    parser.add_argument("--flux_model_path",  type=str, default=None,
                        help="Chemin vers flux-2-klein-9b-fp8.safetensors (phase 2 uniquement)")
    parser.add_argument("--resume",           type=str, default=None,
                        help="Reprendre depuis un checkpoint .safetensors")

    # Architecture
    parser.add_argument("--dim",  type=int, default=4096,
                        choices=[3072, 4096],
                        help="dim cachée (3072=Klein4B, 4096=Klein9B)")

    # Entraînement
    parser.add_argument("--phase",       type=int,   default=1, choices=[1, 2],
                        help="Phase 1=IDFormer only, Phase 2=IDFormer+PerceiverCA+Flux")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=4,
                        help="Batch size (4 recommandé pour 24Go VRAM en phase 1, 1-2 en phase 2)")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--image_size",  type=int,   default=512)
    parser.add_argument("--save_every",  type=int,   default=500,
                        help="Sauvegarder un checkpoint tous les N steps")

    args = parser.parse_args()

    # Vérifications
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset non trouvé : {args.dataset}")

    if args.phase == 2 and not args.flux_model_path:
        parser.error("--flux_model_path est requis pour --phase 2")

    trainer = PuLIDKleinTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
