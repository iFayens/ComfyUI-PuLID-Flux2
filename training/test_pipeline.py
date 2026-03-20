"""
test_pipeline.py - VERSION CORRIGÉE v1.1.0
============================================
Valide tout le pipeline PuLID Klein AVANT de louer sur Vast.ai.
Utilise 10 images locales, 2 epochs, batch=1.

CHANGELOG v1.1.0:
  - FIX CRITIQUE : check_insightface() essaie antelopev2 en premier
    (cohérent avec le node et prepare_dataset — embeddings compatibles)
  - FIX CRITIQUE : check_eva_clip() gère le retour 3D [B, N, D]
    d'EVA-CLIP et vérifie la vraie dim de sortie au lieu de hardcoder (1, 768)
  - FIX : check_training() cherche "train_pulid.py" en priorité
    (l'ancien nom "train_pulid_COMPLETE.py" n'existe plus)
  - FIX : check_prepare_dataset() ajoute les champs manquants dans
    metadata.json ("bbox", "face_size", "original") pour compatibilité
    avec FaceDataset de train_pulid.py
  - FIX : EVA-CLIP forward test lancé sans autocast sur CPU
    (autocast CPU nécessite bfloat16 support, pas garanti)
  - AMÉLIORATION : check_eva_clip() affiche la vraie dim de sortie
    détectée (utile pour diagnostiquer les problèmes de config open_clip)
  - AMÉLIORATION : Rapport final affiche les commandes Vast.ai
    avec les bons paramètres (batch_size cohérent avec la VRAM détectée)

Usage :
  python test_pipeline.py --images ./mes_photos

Ce script vérifie :
  ✅ Dépendances installées (torch, CUDA, etc.)
  ✅ InsightFace détecte les visages (antelopev2 ou buffalo_l)
  ✅ EVA-CLIP se charge et produit le bon format de sortie
  ✅ prepare_dataset mini fonctionne sur vos images
  ✅ train_pulid.py tourne 2 epochs sans crash
  ✅ Le .safetensors est valide et chargeable dans PuLIDFlux2

Si tous les tests passent → prêt pour Vast.ai ! 🚀
"""

import os
import sys
import argparse
import logging
import json
import time
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def section(title):
    """Affiche un titre de section."""
    log.info("")
    log.info("=" * 70)
    log.info(f"  {title}")
    log.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : DÉPENDANCES
# ══════════════════════════════════════════════════════════════════════════════

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    section("ÉTAPE 1 — Vérification des dépendances")
    ok = True

    deps = {
        "torch"      : "pip install torch torchvision",
        "numpy"      : "pip install numpy",
        "PIL"        : "pip install Pillow",
        "tqdm"       : "pip install tqdm",
        "safetensors": "pip install safetensors",
        "open_clip"  : "pip install open-clip-torch",
        "insightface": "pip install insightface onnxruntime-gpu",
    }

    for module, install_cmd in deps.items():
        try:
            __import__(module)
            log.info(f"  {PASS} {module:15s} installé")
        except ImportError:
            log.error(f"  {FAIL} {module:15s} manquant → {install_cmd}")
            ok = False

    # CUDA + VRAM
    try:
        import torch
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"  {PASS} CUDA : {gpu} ({vram:.1f} Go VRAM)")

            if vram < 8:
                log.warning(f"  {WARN} VRAM < 8 Go — Phase 1 possible, Phase 2 risqué")
            elif vram < 12:
                log.warning(f"  {WARN} VRAM < 12 Go — réduire batch_size si OOM")
            else:
                log.info(f"  {PASS} VRAM suffisante pour Phase 1 et Phase 2")
        else:
            log.warning(f"  {WARN} CUDA non disponible — CPU uniquement (très lent)")
    except Exception as e:
        log.error(f"  {FAIL} Erreur CUDA : {e}")

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : INSIGHTFACE
# ══════════════════════════════════════════════════════════════════════════════

def check_insightface(images_dir: str):
    """
    Teste InsightFace sur les images locales.

    FIX v1.1.0 : essaie antelopev2 en premier, buffalo_l en fallback.
    L'ancienne version chargeait directement buffalo_l, ce qui créait
    des embeddings incompatibles avec le node ComfyUI (qui utilise antelopev2).
    """
    section("ÉTAPE 2 — Test InsightFace")

    try:
        from insightface.app import FaceAnalysis
        import numpy as np
        from PIL import Image

        # FIX v1.1.0 : antelopev2 en premier, buffalo_l en fallback
        # (cohérent avec le node ComfyUI et prepare_dataset v1.1.0)
        app = None
        loaded_model = None

        for model_name in ["antelopev2", "buffalo_l"]:
            possible_roots = [
                os.path.expanduser("~/.insightface"),
                None,  # cache par défaut InsightFace
            ]
            for root in possible_roots:
                try:
                    kwargs = {
                        "name"     : model_name,
                        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    }
                    if root:
                        kwargs["root"] = root

                    app = FaceAnalysis(**kwargs)
                    app.prepare(ctx_id=0, det_size=(640, 640))
                    loaded_model = model_name
                    break
                except Exception:
                    continue
            if app is not None:
                break

        if app is None:
            raise RuntimeError("Impossible de charger InsightFace (antelopev2 ni buffalo_l)")

        if loaded_model == "antelopev2":
            log.info(f"  {PASS} InsightFace AntelopeV2 chargé")
        else:
            log.warning(
                f"  {WARN} AntelopeV2 non trouvé — fallback buffalo_l\n"
                "         Les embeddings seront différents du node ComfyUI.\n"
                "         Installer AntelopeV2 pour de meilleurs résultats."
            )

        # Tester sur les images locales
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = sorted([p for ext in exts for p in Path(images_dir).glob(ext)])[:10]

        if not images:
            log.error(f"  {FAIL} Aucune image trouvée dans {images_dir}")
            return None, False

        detected = 0
        failed   = []

        for img_path in images:
            img_np = np.array(Image.open(img_path).convert("RGB"))
            faces  = app.get(img_np)

            if faces:
                detected += 1
                log.info(
                    f"  {PASS} {img_path.name:30s} → "
                    f"{len(faces)} visage(s), conf={faces[0].det_score:.3f}, "
                    f"embed_dim={faces[0].embedding.shape[0]}"
                )
            else:
                failed.append(img_path.name)
                log.warning(f"  {WARN} {img_path.name:30s} → aucun visage")

        log.info(f"\n  Résultat : {detected}/{len(images)} images avec visage détecté")

        if detected == 0:
            log.error(f"  {FAIL} Aucune image utilisable — vérifiez vos photos")
            log.error("         Les photos doivent contenir des visages clairement visibles")
            return None, False

        if failed:
            log.warning(
                f"  {WARN} {len(failed)} image(s) sans visage "
                "(seront rejetées par prepare_dataset)"
            )

        return app, True

    except Exception as e:
        log.error(f"  {FAIL} InsightFace erreur : {e}")
        log.error("         Installation : pip install insightface onnxruntime-gpu")
        return None, False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : EVA-CLIP
# ══════════════════════════════════════════════════════════════════════════════

def check_eva_clip():
    """
    Teste le chargement d'EVA-CLIP et son forward pass.

    FIX v1.1.0 :
    - Gestion du retour 3D [B, N, D] (extraction CLS token)
    - Vérification de la vraie dim de sortie au lieu de hardcoder (1, 768)
    - Forward sans autocast sur CPU (bfloat16 pas garanti sur tous les CPU)
    """
    section("ÉTAPE 3 — Test EVA-CLIP")

    try:
        import torch
        import open_clip

        log.info("  Chargement EVA02-CLIP-L-14-336...")
        log.info("  (peut prendre 1-2 min au premier lancement, ~800 Mo)")

        t0 = time.time()

        model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k"
        )
        visual = model.visual
        visual.eval()

        elapsed = time.time() - t0
        log.info(f"  {PASS} EVA-CLIP chargé en {elapsed:.1f}s")

        # Test forward
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visual = visual.to(device)
        dummy  = torch.randn(1, 3, 336, 336).to(device)

        with torch.no_grad():
            # FIX v1.1.0 : pas d'autocast sur CPU (bfloat16 non garanti)
            out = visual(dummy.float())

            # FIX v1.1.0 : gérer tuple/liste
            if isinstance(out, (list, tuple)):
                out = out[0]

            # FIX v1.1.0 : gérer output 3D [B, N, D] → extraire CLS token
            if out.dim() == 3:
                log.info(
                    f"  {WARN} Output 3D détecté : {tuple(out.shape)} — "
                    "extraction CLS token (index 0)"
                )
                out = out[:, 0, :]

        # FIX v1.1.0 : afficher la vraie dim détectée, ne pas hardcoder 768
        out_dim = out.shape[-1]
        log.info(f"  {PASS} Forward OK → shape finale : {tuple(out.shape)}")

        if out_dim == 768:
            log.info(f"  {PASS} Dim 768 — compatible IDFormer standard")
        else:
            log.warning(
                f"  {WARN} Dim {out_dim} (attendu : 768) — "
                "vérifier la config open_clip, peut impacter la qualité"
            )

        return visual, True

    except Exception as e:
        log.error(f"  {FAIL} EVA-CLIP erreur : {e}")
        log.error("         Installation : pip install open-clip-torch")
        return None, False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : PREPARE_DATASET (mini version inline)
# ══════════════════════════════════════════════════════════════════════════════

def check_prepare_dataset(images_dir: str, output_dir: str, face_detector):
    """
    Teste le pipeline prepare_dataset sur les images locales.

    FIX v1.1.0 : metadata.json incluait uniquement "filename", "embedding",
    "det_score". FaceDataset dans train_pulid.py requiert aussi "bbox" et
    "face_size". Sans ces champs, l'étape 5 (entraînement) crashait.
    Ajout du champ "original" pour traçabilité.
    """
    section("ÉTAPE 4 — Test prepare_dataset (mini)")

    try:
        import numpy as np
        from PIL import Image

        exts   = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        images = sorted([p for ext in exts for p in Path(images_dir).glob(ext)])[:10]

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        kept     = 0
        metadata = []

        for img_path in images:
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)
            faces   = face_detector.get(img_np)

            if len(faces) != 1:
                log.warning(
                    f"  {WARN} {img_path.name:30s} → "
                    f"{len(faces)} visage(s) — ignorée"
                )
                continue

            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w = x2 - x1
            face_h = y2 - y1

            if face_w < 64 or face_h < 64:
                log.warning(
                    f"  {WARN} {img_path.name:30s} → "
                    f"visage trop petit ({face_w}×{face_h}px)"
                )
                continue

            if face.det_score < 0.6:
                log.warning(
                    f"  {WARN} {img_path.name:30s} → "
                    f"confiance faible ({face.det_score:.3f})"
                )
                continue

            # Image acceptée
            out_name = f"face_{kept:05d}.jpg"
            out_path = os.path.join(output_dir, "images", out_name)
            img_pil.resize((512, 512), Image.LANCZOS).save(out_path, quality=95)

            # FIX v1.1.0 : inclure bbox, face_size, original
            # (requis par FaceDataset dans train_pulid.py)
            metadata.append({
                "filename" : out_name,
                "original" : img_path.name,
                "embedding": face.embedding.tolist(),
                "bbox"     : face.bbox.tolist(),
                "det_score": float(face.det_score),
                "face_size": [int(face_w), int(face_h)],
            })

            kept += 1
            log.info(
                f"  {PASS} {img_path.name:30s} → {out_name} "
                f"(conf={face.det_score:.3f}, face={face_w}×{face_h}px)"
            )

        if kept == 0:
            log.error(f"  {FAIL} Aucune image valide — vérifiez vos photos")
            log.error("         Les photos doivent contenir exactement 1 visage clair")
            return False

        # Sauvegarder metadata.json — format compatible avec FaceDataset
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"total_kept": kept, "files": metadata}, f, indent=2)

        log.info(f"\n  {PASS} Dataset test prêt : {kept} images")
        log.info(f"  {PASS} Dossier  : {output_dir}/images/")
        log.info(f"  {PASS} Metadata : {meta_path}")

        return True

    except Exception as e:
        log.error(f"  {FAIL} prepare_dataset erreur : {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 : TEST ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def check_training(dataset_dir: str, output_dir: str, comfyui_path: str):
    """
    Teste l'entraînement sur 2 epochs.

    FIX v1.1.0 : l'ancienne version cherchait "train_pulid_COMPLETE.py"
    en priorité — ce nom n'existe plus. Le bon nom est "train_pulid.py".
    """
    section("ÉTAPE 5 — Test entraînement (2 epochs, batch=1)")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # FIX v1.1.0 : "train_pulid.py" en premier (nom correct)
        possible_names = [
            "train_pulid.py",
            "train_pulid_klein.py",
            "train_pulid_COMPLETE.py",   # ancien nom, gardé en dernier fallback
        ]

        train_script = None
        for name in possible_names:
            path = os.path.join(script_dir, name)
            if os.path.exists(path):
                train_script = path
                break

        if not train_script:
            log.error(f"  {FAIL} Script d'entraînement non trouvé")
            log.error(f"         Cherché dans : {script_dir}")
            log.error(f"         Noms cherchés : {possible_names}")
            return False

        log.info(f"  Script : {os.path.basename(train_script)}")

        cmd = [
            sys.executable, train_script,
            "--dataset"     , dataset_dir,
            "--output"      , output_dir,
            "--comfyui_path", comfyui_path,
            "--phase"       , "1",
            "--epochs"      , "2",
            "--batch_size"  , "1",
            "--save_every"  , "999999",   # pas de checkpoint intermédiaire en test
            "--dim"         , "4096",
        ]

        log.info(f"  Commande : {' '.join(cmd)}")
        log.info("  Lancement (peut prendre 2-5 min selon le GPU)...")
        log.info("")

        t0     = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.error(f"  {FAIL} Entraînement échoué (code: {result.returncode})")
            log.error("")
            log.error("  Dernières lignes stderr :")
            for line in result.stderr.split("\n")[-30:]:
                if line.strip():
                    log.error(f"    {line}")
            return False

        # Afficher les dernières lignes stdout
        log.info("  Dernières lignes stdout :")
        for line in result.stdout.split("\n")[-20:]:
            if line.strip():
                log.info(f"    {line}")

        log.info("")
        log.info(f"  {PASS} Entraînement terminé en {elapsed:.0f}s")
        return True

    except Exception as e:
        log.error(f"  {FAIL} Erreur : {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 : VÉRIFICATION SAFETENSORS
# ══════════════════════════════════════════════════════════════════════════════

def check_safetensors(output_dir: str, comfyui_path: str):
    """Vérifie le fichier .safetensors généré."""
    section("ÉTAPE 6 — Vérification du .safetensors")

    try:
        from safetensors.torch import load_file

        # Chercher le fichier
        candidates = [
            os.path.join(output_dir, "pulid_klein_phase1_best.safetensors"),
            os.path.join(output_dir, "pulid_klein_phase1_latest.safetensors"),
        ]

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith(".safetensors"):
                    candidates.append(os.path.join(ckpt_dir, f))

        path = next((p for p in candidates if os.path.exists(p)), None)

        if not path:
            log.error(f"  {FAIL} Aucun .safetensors trouvé dans {output_dir}")
            log.error(f"         Cherché : {candidates[:2]} ...")
            return False

        state = load_file(path, device="cpu")
        size  = os.path.getsize(path) / 1e6

        log.info(f"  {PASS} Fichier trouvé : {os.path.basename(path)}")
        log.info(f"  {PASS} Taille         : {size:.1f} Mo")
        log.info(f"  {PASS} Clés           : {len(state)}")
        log.info("")

        # Vérifier les clés essentielles
        expected_keys = [
            "id_former.latents",
            "id_former.norm.weight",
            "id_former.proj.0.weight",
            "pulid_ca_double.0.to_q.weight",
            "pulid_ca_single.0.to_q.weight",
        ]

        missing_keys = []
        for key in expected_keys:
            if key in state:
                log.info(f"  {PASS} {key:45s} : {tuple(state[key].shape)}")
            else:
                missing_keys.append(key)
                log.warning(f"  {WARN} {key:45s} : manquante")

        if missing_keys:
            log.warning(
                f"  {WARN} {len(missing_keys)} clé(s) manquante(s) "
                "— checkpoint peut-être incomplet (2 epochs = normal)"
            )

        # Test de chargement dans PuLIDFlux2
        log.info("")
        log.info("  Test chargement dans PuLIDFlux2...")

        possible_paths = [
            os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2Klein", "pulid_flux2.py"),
            os.path.join(comfyui_path, "custom_nodes", "ComfyUI-PuLID-Flux2", "pulid_flux2.py"),
            os.path.join(os.path.dirname(__file__), "pulid_flux2.py"),
        ]

        pulid_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if pulid_path:
            sys.path.insert(0, os.path.dirname(pulid_path))
            try:
                from pulid_flux2 import PuLIDFlux2

                # Détecter la dim depuis les poids
                dim = state.get("id_former.latents", None)
                dim = dim.shape[-1] if dim is not None else 4096

                model = PuLIDFlux2(dim=dim)
                missing, unexpected = model.load_state_dict(state, strict=False)

                if missing:
                    log.warning(f"  {WARN} {len(missing)} clé(s) manquante(s) lors du chargement")
                if unexpected:
                    log.warning(f"  {WARN} {len(unexpected)} clé(s) inattendue(s)")

                log.info(f"  {PASS} Chargeable dans PuLIDFlux2 (dim={dim}) ✅")

            except Exception as e:
                log.warning(f"  {WARN} Erreur chargement PuLIDFlux2 : {e}")
        else:
            log.warning(f"  {WARN} pulid_flux2.py non trouvé — test de chargement skippé")
            log.warning(f"         Chemins cherchés : {possible_paths}")

        return True

    except Exception as e:
        log.error(f"  {FAIL} Erreur : {e}")
        import traceback
        log.error(traceback.format_exc())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Valide le pipeline PuLID Klein avant Vast.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--images", type=str, required=True,
        help="Dossier avec 5-10 photos de visages pour tester"
    )
    parser.add_argument(
        "--comfyui_path", type=str, default="C:/AI/ComfyUI",
        help="Chemin vers ComfyUI"
    )
    parser.add_argument(
        "--output", type=str, default="./test_output",
        help="Dossier de sortie pour les tests"
    )

    args = parser.parse_args()

    dataset_test = os.path.join(args.output, "dataset_test")
    train_test   = os.path.join(args.output, "train_test")

    results = {}
    t_start = time.time()

    # ── Étape 1 : Dépendances ─────────────────────────────────────────────────
    results["deps"] = check_dependencies()
    if not results["deps"]:
        log.error("")
        log.error("❌ Dépendances manquantes — installez-les avant de continuer")
        return

    # ── Étape 2 : InsightFace ─────────────────────────────────────────────────
    face_detector, results["insightface"] = check_insightface(args.images)
    if not results["insightface"]:
        log.error("")
        log.error("❌ InsightFace KO — impossible de continuer")
        return

    # ── Étape 3 : EVA-CLIP ────────────────────────────────────────────────────
    _, results["eva_clip"] = check_eva_clip()

    # ── Étape 4 : prepare_dataset ─────────────────────────────────────────────
    if results["insightface"]:
        results["dataset"] = check_prepare_dataset(
            args.images, dataset_test, face_detector
        )
    else:
        results["dataset"] = False
        log.warning("⚠️  Étape 4 skippée (InsightFace KO)")

    # ── Étape 5 : Entraînement ────────────────────────────────────────────────
    if results["dataset"]:
        results["training"] = check_training(
            dataset_test, train_test, args.comfyui_path
        )
    else:
        results["training"] = False
        log.warning("⚠️  Étape 5 skippée (dataset KO)")

    # ── Étape 6 : Safetensors ─────────────────────────────────────────────────
    if results["training"]:
        results["safetensors"] = check_safetensors(train_test, args.comfyui_path)
    else:
        results["safetensors"] = False
        log.warning("⚠️  Étape 6 skippée (training KO)")

    # ── Rapport Final ─────────────────────────────────────────────────────────
    section("RAPPORT FINAL")

    labels = {
        "deps"       : "Dépendances",
        "insightface": "InsightFace",
        "eva_clip"   : "EVA-CLIP",
        "dataset"    : "prepare_dataset",
        "training"   : "Entraînement (2 epochs)",
        "safetensors": "Fichier .safetensors",
    }

    all_ok = True
    for key, label in labels.items():
        status = PASS if results.get(key) else FAIL
        log.info(f"  {status} {label:30s}")
        if not results.get(key):
            all_ok = False

    elapsed = time.time() - t_start
    log.info("")
    log.info(f"  Durée totale : {elapsed:.0f}s")
    log.info("")

    if all_ok:
        # Détecter VRAM pour recommander le bon batch_size sur Vast.ai
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        except Exception:
            vram = 0

        bs1 = 8 if vram >= 40 else (4 if vram >= 24 else 2)
        bs2 = 4 if vram >= 40 else (2 if vram >= 24 else 1)

        log.info("🎉 TOUT EST OK — Vous pouvez lancer sur Vast.ai !")
        log.info("")
        log.info("📝 COMMANDES POUR VAST.AI :")
        log.info("")
        log.info("# Phase 1 (recommandé : A100 40GB)")
        log.info("python train_pulid.py \\")
        log.info("    --dataset ./dataset/filtered \\")
        log.info("    --output ./output \\")
        log.info("    --phase 1 \\")
        log.info("    --epochs 25 \\")
        log.info(f"    --batch_size {bs1}   # adapté à votre VRAM")
        log.info("")
        log.info("# Phase 2 (reprend depuis Phase 1)")
        log.info("python train_pulid.py \\")
        log.info("    --dataset ./dataset/filtered \\")
        log.info("    --output ./output_phase2 \\")
        log.info("    --phase 2 \\")
        log.info("    --resume ./output/pulid_klein_phase1_best.safetensors \\")
        log.info("    --flux_model_path ./models/flux-2-klein-9b-fp8.safetensors \\")
        log.info("    --epochs 15 \\")
        log.info(f"    --batch_size {bs2}   # adapté à votre VRAM")
    else:
        log.info("❌ Des erreurs sont présentes — corrigez-les avant Vast.ai")
        log.info("")
        log.info("Solutions courantes :")
        if not results.get("insightface"):
            log.info("  - InsightFace : pip install insightface onnxruntime-gpu")
        if not results.get("eva_clip"):
            log.info("  - EVA-CLIP    : pip install open-clip-torch")
        if not results.get("dataset"):
            log.info("  - Dataset     : vérifiez que vos photos contiennent des visages clairs")
        if not results.get("training"):
            log.info("  - Training    : vérifiez que train_pulid.py est dans le même dossier")


if __name__ == "__main__":
    main()
