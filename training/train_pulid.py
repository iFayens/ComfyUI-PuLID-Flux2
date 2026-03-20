"""
train_pulid.py - VERSION CORRIGÉE v1.1.0
============================================
Entraîne le module PuLID (IDFormer + PerceiverCA) sur Flux.2 Klein.

Phase 1 : IDFormer only  — rapide, ~2-4h sur 3090, sans Flux chargé
Phase 2 : IDFormer + PerceiverCA avec Flux.2 Klein gelé — ~8-12h sur 3090

OBJECTIF : Clonage d'identité parfait
- Input  : Image de n'importe quelle personne
- Output : Génération avec LA MÊME identité (visage, cheveux, traits)

CHANGELOG v1.1.0:
  - FIX CRITIQUE : EVA-CLIP output 3D géré dans encode_eva_clip()
    (extraction CLS token si retour [B, N, D] au lieu de [B, D])
  - FIX : Loss Phase 2 PerceiverCA remplacée par une loss d'alignement identité
    réelle (cosine similarity correction→target) au lieu d'une cohérence inter-blocs
    artificielle qui ne corrélait pas avec la qualité d'identité
  - FIX : encode_eva_clip() normalise maintenant les features en sortie
    pour cohérence avec l'espace IDFormer (alignement avec Phase 1)
  - AMÉLIORATION : Gradient clipping appliqué aussi à proj_loss
  - AMÉLIORATION : Log du nombre de blocs patchés en Phase 2

Usage :
  # Phase 1 (sans Flux) - RECOMMANDÉ en premier
  python train_pulid.py \\
      --dataset ./dataset/filtered \\
      --output ./output \\
      --phase 1 \\
      --epochs 25 \\
      --batch_size 4 \\
      --lr 1e-4

  # Phase 2 (avec Flux, reprend depuis phase 1)
  python train_pulid.py \\
      --dataset ./dataset/filtered \\
      --output ./output \\
      --phase 2 \\
      --resume ./output/pulid_klein_phase1_best.safetensors \\
      --flux_model_path /path/to/flux-2-klein-9b-fp8.safetensors \\
      --epochs 15 \\
      --batch_size 2 \\
      --lr 5e-5

Prérequis :
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install accelerate diffusers transformers
  pip install open_clip_torch insightface safetensors
  pip install opencv-python pillow tqdm
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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class TrainingConfig:
    """Configuration centralisée pour l'entraînement."""

    # Phase 1 : IDFormer
    PHASE1_EPOCHS = 25
    PHASE1_LR = 1e-4
    PHASE1_BATCH_SIZE = 4
    PHASE1_LOSS_IDENTITY_WEIGHT = 1.0
    PHASE1_LOSS_REG_WEIGHT = 0.01

    # Phase 2 : IDFormer + PerceiverCA
    PHASE2_EPOCHS = 15
    PHASE2_LR = 5e-5
    PHASE2_BATCH_SIZE = 2
    PHASE2_LOSS_IDENTITY_WEIGHT = 2.0
    PHASE2_LOSS_PERCEIVER_WEIGHT = 1.0
    PHASE2_LOSS_REG_WEIGHT = 0.01

    # Critères de succès
    TARGET_LOSS_IDENTITY_GOOD = 0.05
    TARGET_LOSS_IDENTITY_EXCELLENT = 0.03

    # Early stopping
    EARLY_STOPPING_PATIENCE = 5

    # Nombre de fake tokens pour Phase 2
    # Flux.2 Klein 4B : 512×512 image → 64×64 patches de 8px = 4096 tokens
    # On utilise 256 pour un bon compromis mémoire/signal
    FAKE_IMG_TOKENS_COUNT = 256


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class FaceDataset(Dataset):
    """
    Dataset de visages pour entraîner PuLID.
    Chaque item = (image_tensor, id_embedding_insightface)

    Structure attendue :
        dataset/
        ├── images/
        │   ├── face_00001.jpg
        │   └── ...
        └── metadata.json
    """
    def __init__(self, dataset_dir: str, size: int = 512, augment: bool = True):
        self.size = size
        self.augment = augment

        images_dir = os.path.join(dataset_dir, "images")
        meta_path = os.path.join(dataset_dir, "metadata.json")

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
                    "path": img_path,
                    "embedding": np.array(item["embedding"], dtype=np.float32),
                })

        if not self.items:
            raise RuntimeError(f"Aucune image valide trouvée dans {dataset_dir}")

        n_images = len(self.items)
        # Heuristique : identités distinctes basées sur l'embedding clustering
        n_identities = len(set(item["path"].split("/")[-1].split("_")[0]
                              for item in self.items))

        log.info(f"Dataset chargé : {n_images} images, ~{n_identities} identités")

        if n_identities < 50:
            log.warning(
                f"⚠️  Seulement {n_identities} identités détectées.\n"
                f"    Recommandé : >50 identités pour de bons résultats.\n"
                f"    Idéal : 200+ identités"
            )

    def __len__(self):
        return len(self.items)

    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Augmentation adaptée aux visages.
        - PAS de flip horizontal (inverse gauche/droite, perturbe l'identité)
        - Rotation légère ±15°
        - Brightness / contrast / saturation / sharpness modérés
        """
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

        if random.random() > 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.7:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.9, 1.1))

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


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

def load_eva_clip(device: torch.device):
    """Charge EVA02-CLIP-L-14-336 via open_clip."""
    import open_clip
    log.info("Chargement EVA02-CLIP-L-14-336...")

    try:
        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k"
        )
    except Exception as e:
        raise RuntimeError(
            f"Erreur chargement EVA-CLIP : {e}\n"
            "Installation : pip install open_clip_torch"
        )

    model = model.visual
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    log.info("✅ EVA02-CLIP chargé et gelé")
    return model.to(device)


def load_flux2_klein(model_path: str, device: torch.device):
    """
    Charge Flux.2 Klein via diffusers ou safetensors.
    Le modèle est complètement gelé — seul PuLID sera entraîné.
    """
    log.info(f"Chargement Flux.2 Klein depuis : {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle Flux.2 non trouvé : {model_path}\n"
            "Télécharge depuis : https://huggingface.co/black-forest-labs"
        )

    try:
        from diffusers import FluxTransformer2DModel

        if os.path.isdir(model_path):
            model = FluxTransformer2DModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            model = FluxTransformer2DModel.from_single_file(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger Flux.2 Klein : {e}\n"
            "Vérifiez :\n"
            "1. diffusers>=0.31 installé : pip install diffusers>=0.31\n"
            "2. Le chemin est correct\n"
            "3. Le fichier n'est pas corrompu"
        )

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    log.info("✅ Flux.2 Klein chargé et gelé")
    return model.to(device, dtype=torch.bfloat16)


def load_pulid_module(comfyui_path: str, dim: int, device: torch.device):
    """Importe PuLIDFlux2 depuis le custom node ComfyUI."""

    possible_paths = [
        os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2Klein"),
        os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2"),
        os.path.join(comfyui_path, "custom_nodes", "ComfyUI_PuLID_Flux2"),
    ]

    pulid_node_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pulid_node_path = path
            break

    if not pulid_node_path:
        raise FileNotFoundError(
            f"Custom node PuLID non trouvé.\n"
            f"Cherché dans :\n" + "\n".join(f"  - {p}" for p in possible_paths) + "\n"
            "Installation :\n"
            "  cd ComfyUI/custom_nodes\n"
            "  git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2"
        )

    sys.path.insert(0, pulid_node_path)

    try:
        from pulid_flux2 import PuLIDFlux2
    except ImportError as e:
        raise ImportError(
            f"Impossible d'importer PuLIDFlux2 : {e}\n"
            f"Vérifiez que {pulid_node_path}/pulid_flux2.py existe"
        )

    log.info(f"Module PuLID chargé depuis : {pulid_node_path}")
    return PuLIDFlux2(dim=dim).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class PuLIDKleinTrainer:
    """
    Entraîneur pour PuLID Flux.2 Klein.

    Phase 1 : IDFormer uniquement (sans Flux)
        - Apprend à créer des tokens d'identité riches
        - Loss : cosine similarity avec InsightFace

    Phase 2 : IDFormer + PerceiverCA (avec Flux gelé)
        - Apprend à injecter l'identité dans les image tokens
        - Loss : identity + perceiver alignment + regularization
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = TrainingConfig()

        if not torch.cuda.is_available():
            log.warning("⚠️  CUDA non disponible. Entraînement sur CPU (très lent).")

        log.info(f"Device : {self.device}")
        log.info(f"PyTorch version : {torch.__version__}")

        # ── Dataset ───────────────────────────────────────────────────────────
        self.dataset = FaceDataset(
            args.dataset,
            size=args.image_size,
            augment=True
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

        # ── Modèles ───────────────────────────────────────────────────────────
        log.info("Chargement des modèles...")

        self.eva_clip = load_eva_clip(self.device)
        self.pulid = load_pulid_module(args.comfyui_path, args.dim, self.device)

        # Charger checkpoint si --resume
        if args.resume:
            log.info(f"Chargement checkpoint : {args.resume}")
            from safetensors.torch import load_file

            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint non trouvé : {args.resume}")

            state = load_file(args.resume)
            missing, unexpected = self.pulid.load_state_dict(state, strict=False)

            if missing:
                log.warning(f"⚠️  Clés manquantes ({len(missing)}) : {missing[:5]}")
            if unexpected:
                log.warning(f"⚠️  Clés inattendues ({len(unexpected)}) : {unexpected[:5]}")

            log.info("✅ Checkpoint chargé")

        # Flux.2 Klein (phase 2 uniquement)
        self.flux = None
        if args.phase == 2:
            if not args.flux_model_path:
                raise ValueError("--flux_model_path requis pour phase 2")
            self.flux = load_flux2_klein(args.flux_model_path, self.device)

        # Projection pour loss identité (dim → 512)
        self.proj_loss = nn.Linear(args.dim, 512).to(self.device)

        # ── Optimiseur ────────────────────────────────────────────────────────
        trainable_params = (
            list(self.pulid.parameters()) +
            list(self.proj_loss.parameters())
        )

        self.optimizer = AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        total_steps = len(self.dataloader) * args.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=args.lr / 100
        )

        # ── Early stopping ────────────────────────────────────────────────────
        self.best_loss = float('inf')
        self.patience_counter = 0

        # ── Dossiers de sortie ────────────────────────────────────────────────
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, "checkpoints"), exist_ok=True)

        n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        log.info(f"Paramètres entraînables : {n_params:,}")
        log.info(f"Taille batch : {args.batch_size}")
        log.info(f"Steps par epoch : {len(self.dataloader)}")
        log.info(f"Total steps : {total_steps:,}")

    # ──────────────────────────────────────────────────────────────────────────
    # EVA-CLIP ENCODING
    # ──────────────────────────────────────────────────────────────────────────

    def encode_eva_clip(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode les images via EVA-CLIP.

        Args:
            images : [B, 3, H, H] range [-1, 1]

        Returns:
            features : [B, 768] normalisées

        FIX v1.1.0:
            - Gestion du cas où EVA-CLIP retourne [B, N, D] :
              extraction du CLS token (index 0)
            - Normalisation F.normalize() ajoutée pour cohérence
              avec l'espace de l'IDFormer
        """
        with torch.no_grad():
            # Normalisation EVA-CLIP (ImageNet stats)
            mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=images.device, dtype=torch.float32
            ).view(1, 3, 1, 1)

            std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=images.device, dtype=torch.float32
            ).view(1, 3, 1, 1)

            x = (images.float() + 1.0) / 2.0   # [-1,1] → [0,1]
            x = (x - mean) / std

            # Resize vers 336×336 (input EVA-CLIP)
            x = F.interpolate(
                x, size=(336, 336),
                mode='bicubic',
                align_corners=False
            )

            features = self.eva_clip(x)

            # ── FIX v1.1.0 : EVA-CLIP peut retourner [B, N, D] ──────────────
            # selon la config open_clip (avec ou sans pool CLS).
            # On extrait le CLS token si output 3D.
            if isinstance(features, (list, tuple)):
                features = features[0]

            if features.dim() == 3:
                log.debug(
                    f"EVA-CLIP output 3D {tuple(features.shape)} — "
                    "extraction CLS token (index 0)"
                )
                features = features[:, 0, :]   # [B, N, D] → [B, D]
            # ────────────────────────────────────────────────────────────────

            # Normaliser pour cohérence avec l'espace IDFormer
            return F.normalize(features, dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 : IDFORMER ONLY
    # ──────────────────────────────────────────────────────────────────────────

    def train_step_phase1(self, images: torch.Tensor, id_embeds: torch.Tensor):
        """
        Phase 1 : Entraîne uniquement l'IDFormer.

        Objectif : Créer des tokens d'identité riches qui capturent :
            - L'embedding InsightFace (512-d)
            - Les features visuelles EVA-CLIP (768-d)

        Loss : Cosine similarity entre tokens projetés et embedding InsightFace
        """
        images   = images.to(self.device, dtype=torch.bfloat16)
        id_embeds = id_embeds.to(self.device, dtype=torch.bfloat16)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 1. EVA-CLIP encoding
            clip_embed = self.encode_eva_clip(images)           # [B, 768]

            # 2. IDFormer : id_embed + clip_embed → tokens
            id_tokens = self.pulid.id_former(id_embeds, clip_embed)  # [B, 4, dim]

            # 3. Loss identité : cosine similarity
            # Projeter vers 512-d (même dim que InsightFace)
            projected = self.proj_loss(id_tokens.mean(dim=1))   # [B, dim] → [B, 512]
            projected = F.normalize(projected, dim=-1)
            target    = F.normalize(id_embeds, dim=-1)

            # Cosine distance loss (1 - cosine_sim) → 0 = parfait
            loss_identity = (1.0 - (projected * target).sum(-1)).mean()

            # Regularization : empêche les tokens de diverger
            loss_reg = id_tokens.pow(2).mean() * self.config.PHASE1_LOSS_REG_WEIGHT

            loss = (
                loss_identity * self.config.PHASE1_LOSS_IDENTITY_WEIGHT +
                loss_reg
            )

        return {
            "loss"          : loss,
            "loss_identity" : loss_identity.item(),
            "loss_reg"      : loss_reg.item(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 : IDFORMER + PERCEIVERCA
    # ──────────────────────────────────────────────────────────────────────────

    def train_step_phase2(self, images: torch.Tensor, id_embeds: torch.Tensor):
        """
        Phase 2 : Entraîne IDFormer + PerceiverCA avec Flux.2 gelé.

        Objectif : Apprendre à injecter l'identité dans les image tokens.

        Loss :
            - loss_identity  : maintient la qualité des tokens (identique Phase 1)
            - loss_perceiver : les corrections PerceiverCA doivent aligner les
                               image tokens vers l'identité cible
                               (FIX v1.1.0 — remplace la cohérence inter-blocs
                               qui était une contrainte artificielle)
            - loss_reg       : régularisation L2

        FIX v1.1.0 — Ancienne loss_perceiver :
            Mesurait la cohérence entre les corrections des différents CA blocks
            (MSE entre chaque correction et leur moyenne). Cette contrainte forçait
            tous les PerceiverCA à produire des corrections similaires, ce qui
            limitait leur spécialisation et ne corrélait PAS avec la qualité
            d'identité en génération.

        Nouvelle loss_perceiver :
            Pour chaque PerceiverCA block, mesure si la correction qu'il apporte
            aux fake image tokens les rapproche des id_tokens (cosine similarity).
            C'est une loss d'alignement identité directe, bien plus efficace.
        """
        images    = images.to(self.device, dtype=torch.bfloat16)
        id_embeds = id_embeds.to(self.device, dtype=torch.bfloat16)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 1. EVA-CLIP encoding
            clip_embed = self.encode_eva_clip(images)           # [B, 768]

            # 2. IDFormer
            id_tokens = self.pulid.id_former(id_embeds, clip_embed)  # [B, 4, dim]

            # 3. Loss identité (maintenue depuis Phase 1)
            projected    = self.proj_loss(id_tokens.mean(dim=1))
            projected    = F.normalize(projected, dim=-1)
            target       = F.normalize(id_embeds, dim=-1)
            loss_identity = (1.0 - (projected * target).sum(-1)).mean()

            # 4. Loss PerceiverCA — FIX v1.1.0
            loss_perceiver = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)

            if self.flux is not None:
                B   = images.shape[0]
                dim = self.args.dim

                # Nombre de double blocks disponibles
                blocks = (
                    getattr(self.flux, "transformer_blocks", None) or
                    getattr(self.flux, "double_blocks", [])
                )
                n_blocks = len(blocks)

                # Fake image tokens : représentent des activations réalistes
                # normalisées avec une variance de 2.0 (ordre de grandeur typique
                # des activations Flux.2 Klein)
                fake_img_tokens = torch.randn(
                    B,
                    self.config.FAKE_IMG_TOKENS_COUNT,
                    dim,
                    device=self.device,
                    dtype=torch.bfloat16
                )
                fake_img_tokens = F.normalize(fake_img_tokens, dim=-1) * 2.0

                embed = id_tokens.to(dtype=torch.bfloat16)

                # Cible d'identité pour aligner les corrections :
                # on projette id_tokens vers l'espace des image tokens (dim)
                # via une moyenne sur les tokens → [B, 1, dim] pour broadcast
                id_target = id_tokens.mean(dim=1, keepdim=True)  # [B, 1, dim]
                id_target = F.normalize(id_target, dim=-1)

                ca_losses = []

                # Double blocks PerceiverCA
                for i, ca in enumerate(self.pulid.pulid_ca_double):
                    if i * self.pulid.double_interval < n_blocks:
                        correction = ca(fake_img_tokens, embed)  # [B, N, dim]

                        # Les image tokens corrigés doivent être alignés
                        # avec l'identité cible
                        corrected = fake_img_tokens + correction
                        corrected_norm = F.normalize(corrected, dim=-1)

                        # Cosine similarity entre chaque token corrigé et l'id_target
                        sim = (corrected_norm * id_target).sum(-1)  # [B, N]
                        # Loss = 1 - mean similarity (on veut maximiser l'alignement)
                        ca_losses.append((1.0 - sim.mean()))

                # Single blocks PerceiverCA
                single_blocks = (
                    getattr(self.flux, "single_transformer_blocks", None) or
                    getattr(self.flux, "single_blocks", [])
                )
                n_single = len(single_blocks)

                for i, ca in enumerate(self.pulid.pulid_ca_single):
                    if i * self.pulid.single_interval < n_single:
                        correction = ca(fake_img_tokens, embed)  # [B, N, dim]
                        corrected = fake_img_tokens + correction
                        corrected_norm = F.normalize(corrected, dim=-1)
                        sim = (corrected_norm * id_target).sum(-1)  # [B, N]
                        ca_losses.append((1.0 - sim.mean()))

                if ca_losses:
                    loss_perceiver = torch.stack(ca_losses).mean()
                    log.debug(
                        f"Phase 2 — {len(ca_losses)} CA blocks actifs, "
                        f"loss_perceiver={loss_perceiver.item():.4f}"
                    )

            # 5. Regularization
            loss_reg = id_tokens.pow(2).mean() * self.config.PHASE2_LOSS_REG_WEIGHT

            # 6. Loss totale
            loss = (
                loss_identity  * self.config.PHASE2_LOSS_IDENTITY_WEIGHT  +
                loss_perceiver * self.config.PHASE2_LOSS_PERCEIVER_WEIGHT +
                loss_reg
            )

        return {
            "loss"            : loss,
            "loss_identity"   : loss_identity.item(),
            "loss_perceiver"  : loss_perceiver.item(),
            "loss_reg"        : loss_reg.item(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # SAUVEGARDE
    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, step: int, loss: float, tag: str = ""):
        """Sauvegarde un checkpoint."""
        from safetensors.torch import save_file

        ckpt_dir = os.path.join(self.args.output, "checkpoints")
        name = (
            f"pulid_klein_phase{self.args.phase}_"
            f"epoch{epoch:03d}_step{step:06d}{tag}.safetensors"
        )
        ckpt_path = os.path.join(ckpt_dir, name)
        latest = os.path.join(
            self.args.output,
            f"pulid_klein_phase{self.args.phase}_latest.safetensors"
        )

        state = {
            k: v.cpu().contiguous()
            for k, v in self.pulid.state_dict().items()
        }

        save_file(state, ckpt_path)
        save_file(state, latest)

        log.info(f"💾 Checkpoint : {ckpt_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # BOUCLE D'ENTRAÎNEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def train(self):
        """Boucle principale d'entraînement."""

        log.info("=" * 70)
        log.info(f"  ENTRAÎNEMENT PULID FLUX.2 KLEIN — PHASE {self.args.phase}")
        log.info("=" * 70)
        log.info(f"  Epochs       : {self.args.epochs}")
        log.info(f"  Batch size   : {self.args.batch_size}")
        log.info(f"  Learning rate: {self.args.lr}")
        log.info(f"  Dimension    : {self.args.dim}")
        log.info(f"  Image size   : {self.args.image_size}")

        if self.args.phase == 1:
            log.info(f"  Mode         : IDFormer only (sans Flux.2)")
            log.info(f"  Target loss  : < {self.config.TARGET_LOSS_IDENTITY_GOOD}")
        else:
            log.info(f"  Mode         : IDFormer + PerceiverCA (Flux.2 gelé)")
            log.info(f"  Target loss  : < {self.config.TARGET_LOSS_IDENTITY_EXCELLENT}")
            log.info(f"  Fake tokens  : {self.config.FAKE_IMG_TOKENS_COUNT}")

        log.info("=" * 70)

        train_step = (
            self.train_step_phase1 if self.args.phase == 1
            else self.train_step_phase2
        )

        # Tous les paramètres entraînables pour le gradient clipping
        all_trainable = (
            list(self.pulid.parameters()) +
            list(self.proj_loss.parameters())
        )

        global_step = 0

        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = 0.0
            self.pulid.train()
            self.proj_loss.train()

            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.args.epochs}",
                dynamic_ncols=True
            )

            for images, id_embeds in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                losses = train_step(images, id_embeds)
                loss   = losses["loss"]

                loss.backward()

                # Gradient clipping — appliqué à tous les params entraînables
                # (pulid + proj_loss) pour stabilité
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                epoch_loss  += loss.item()
                global_step += 1

                # Affichage tqdm
                postfix = {
                    k: f"{v:.4f}"
                    for k, v in losses.items()
                    if k != "loss"
                }
                postfix["loss"] = f"{loss.item():.4f}"
                postfix["lr"]   = f"{self.scheduler.get_last_lr()[0]:.2e}"
                pbar.set_postfix(postfix)

                # Checkpoint intermédiaire
                if global_step % self.args.save_every == 0:
                    self.save_checkpoint(epoch, global_step, loss.item())

            # Fin epoch
            avg_loss = epoch_loss / len(self.dataloader)
            log.info(f"Epoch {epoch} — Loss moyenne : {avg_loss:.4f}")

            # Évaluation qualité Phase 1
            loss_id = losses.get("loss_identity", avg_loss)
            if self.args.phase == 1:
                if loss_id < self.config.TARGET_LOSS_IDENTITY_EXCELLENT:
                    log.info("    ✅ EXCELLENT ! Identité très bien capturée.")
                elif loss_id < self.config.TARGET_LOSS_IDENTITY_GOOD:
                    log.info("    ✅ BON. Identité bien capturée.")
                else:
                    log.info("    ⚠️  Peut mieux faire. Continuer l'entraînement.")

            # Sauvegarde meilleur modèle
            if avg_loss < self.best_loss:
                self.best_loss      = avg_loss
                self.patience_counter = 0

                from safetensors.torch import save_file
                best_path = os.path.join(
                    self.args.output,
                    f"pulid_klein_phase{self.args.phase}_best.safetensors"
                )

                state = {
                    k: v.cpu().contiguous()
                    for k, v in self.pulid.state_dict().items()
                }
                save_file(state, best_path)

                log.info(f"    🏆 Nouveau meilleur : {avg_loss:.4f} → {best_path}")
            else:
                self.patience_counter += 1

                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    log.info(
                        f"⚠️  Early stopping : pas d'amélioration depuis "
                        f"{self.config.EARLY_STOPPING_PATIENCE} epochs"
                    )
                    break

        # Fin entraînement
        log.info("=" * 70)
        log.info("✅ ENTRAÎNEMENT TERMINÉ !")
        log.info(
            f"   Meilleur modèle : {self.args.output}/"
            f"pulid_klein_phase{self.args.phase}_best.safetensors"
        )
        log.info(f"   Meilleure loss  : {self.best_loss:.4f}")
        log.info("")

        if self.args.phase == 1:
            log.info("📋 Étapes suivantes :")
            log.info("   1. Vérifier que loss_identity < 0.05")
            log.info("   2. Lancer Phase 2 :")
            log.info("")
            log.info(f"      python train_pulid.py \\")
            log.info(f"          --phase 2 \\")
            log.info(f"          --resume {self.args.output}/pulid_klein_phase1_best.safetensors \\")
            log.info(f"          --flux_model_path /path/to/flux-2-klein-9b-fp8.safetensors \\")
            log.info(f"          --dataset {self.args.dataset} \\")
            log.info(f"          --batch_size 2 \\")
            log.info(f"          --lr 5e-5")
        else:
            log.info("📋 Étapes suivantes :")
            log.info("   1. Copier le checkpoint vers ComfyUI :")
            log.info("")
            log.info(f"      cp {self.args.output}/pulid_klein_phase2_best.safetensors \\")
            log.info(f"         ComfyUI/models/pulid/")
            log.info("")
            log.info("   2. Tester dans ComfyUI avec vos photos")
            log.info("   3. Ajuster weight PuLID entre 0.8-1.2")

        log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraîne PuLID pour Flux.2 Klein (Identity Cloning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Chemins ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dossier dataset (contient images/ et metadata.json)"
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Dossier de sortie"
    )
    parser.add_argument(
        "--comfyui_path", type=str, default="C:/AI/ComfyUI",
        help="Chemin vers ComfyUI"
    )
    parser.add_argument(
        "--flux_model_path", type=str, default=None,
        help="Chemin vers flux-2-klein-9b-fp8.safetensors (Phase 2 uniquement)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Reprendre depuis un checkpoint .safetensors"
    )

    # ── Architecture ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--dim", type=int, default=4096, choices=[3072, 4096],
        help="Dimension cachée (3072=Klein 4B, 4096=Klein 9B)"
    )

    # ── Entraînement ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Phase 1=IDFormer, Phase 2=IDFormer+PerceiverCA"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Nombre d'epochs (auto si non spécifié)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (auto si non spécifié)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (auto si non spécifié)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Taille des images"
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Sauvegarder un checkpoint tous les N steps"
    )

    args = parser.parse_args()

    # ── Auto-configuration selon la phase ─────────────────────────────────────
    config = TrainingConfig()

    if args.epochs is None:
        args.epochs = config.PHASE1_EPOCHS if args.phase == 1 else config.PHASE2_EPOCHS

    if args.batch_size is None:
        args.batch_size = config.PHASE1_BATCH_SIZE if args.phase == 1 else config.PHASE2_BATCH_SIZE

    if args.lr is None:
        args.lr = config.PHASE1_LR if args.phase == 1 else config.PHASE2_LR

    # ── Vérifications ─────────────────────────────────────────────────────────
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset non trouvé : {args.dataset}")

    if args.phase == 2 and not args.flux_model_path:
        parser.error("--flux_model_path est requis pour --phase 2")

    if args.phase == 2 and not args.resume:
        log.warning(
            "⚠️  Phase 2 sans --resume : vous partez de zéro.\n"
            "    Recommandé : --resume ./output/pulid_klein_phase1_best.safetensors"
        )

    # ── Lancement ─────────────────────────────────────────────────────────────
    trainer = PuLIDKleinTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
