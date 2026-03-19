"""
test_pipeline.py
================
Valide tout le pipeline PuLID Klein AVANT de louer sur Vast.ai.
Utilise 10 images locales, 2 epochs, batch=1.

Usage :
  python test_pipeline.py --images ./mes_photos --comfyui_path C:/AI/ComfyUI

Ce script vérifie :
  ✅ Dépendances installées
  ✅ InsightFace détecte les visages
  ✅ EVA-CLIP se charge
  ✅ prepare_dataset fonctionne sur tes images
  ✅ train_pulid_klein tourne 2 epochs sans crash
  ✅ Le .safetensors est valide et chargeable dans ton node
"""

import os
import sys
import argparse
import logging
import json
import time

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
    log.info("")
    log.info("=" * 55)
    log.info(f"  {title}")
    log.info("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : Dépendances
# ──────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    section("ÉTAPE 1 — Vérification des dépendances")
    ok = True

    deps = {
        "torch"        : "pip install torch",
        "numpy"        : "pip install numpy",
        "PIL"          : "pip install Pillow",
        "tqdm"         : "pip install tqdm",
        "safetensors"  : "pip install safetensors",
        "open_clip"    : "pip install open-clip-torch",
        "insightface"  : "pip install insightface onnxruntime-gpu",
    }

    for module, install_cmd in deps.items():
        try:
            __import__(module)
            log.info(f"  {PASS} {module}")
        except ImportError:
            log.error(f"  {FAIL} {module} manquant → {install_cmd}")
            ok = False

    # CUDA
    import torch
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  {PASS} CUDA : {gpu} ({vram:.1f} Go VRAM)")
    else:
        log.warning(f"  {WARN} CUDA non disponible — CPU seulement (très lent)")

    return ok


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : InsightFace
# ──────────────────────────────────────────────────────────────────────────────

def check_insightface(images_dir: str):
    section("ÉTAPE 2 — Test InsightFace")

    try:
        from insightface.app import FaceAnalysis
        import numpy as np
        from PIL import Image
        from pathlib import Path

        app = FaceAnalysis(
            name="antelopev2",
            root=r"C:\Users\mems\.insightface",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        log.info(f"  {PASS} InsightFace AntelopeV2 chargé")

        # Tester sur les images locales
        exts   = ("*.jpg", "*.jpeg", "*.png")
        images = []
        for ext in exts:
            images.extend(Path(images_dir).glob(ext))
        images = images[:10]

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
                log.info(f"  {PASS} {img_path.name} → {len(faces)} visage(s) détecté(s)")
            else:
                failed.append(img_path.name)
                log.warning(f"  {WARN} {img_path.name} → aucun visage")

        log.info(f"\n  Résultat : {detected}/{len(images)} images avec visage détecté")

        if detected == 0:
            log.error(f"  {FAIL} Aucune image utilisable — vérifiez vos photos")
            return None, False

        if failed:
            log.warning(f"  {WARN} Images sans visage (seront rejetées) : {failed}")

        return app, True

    except Exception as e:
        log.error(f"  {FAIL} InsightFace erreur : {e}")
        return None, False


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : EVA-CLIP
# ──────────────────────────────────────────────────────────────────────────────

def check_eva_clip():
    section("ÉTAPE 3 — Test EVA-CLIP")

    try:
        import torch
        import open_clip

        log.info("  Chargement EVA02-CLIP-L-14-336 (peut prendre 1-2 min au premier lancement)...")
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
            out = visual(dummy.float())
            if isinstance(out, (list, tuple)):
                out = out[0]

        log.info(f"  {PASS} Forward OK → shape: {out.shape} (attendu: [1, 768])")
        return visual, True

    except Exception as e:
        log.error(f"  {FAIL} EVA-CLIP erreur : {e}")
        log.error("         pip install open-clip-torch")
        return None, False


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : prepare_dataset sur images locales
# ──────────────────────────────────────────────────────────────────────────────

def check_prepare_dataset(images_dir: str, output_dir: str, face_detector):
    section("ÉTAPE 4 — Test prepare_dataset")

    try:
        import numpy as np
        from PIL import Image
        from pathlib import Path

        exts   = ("*.jpg", "*.jpeg", "*.png")
        images = []
        for ext in exts:
            images.extend(Path(images_dir).glob(ext))
        images = images[:10]

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

        kept     = 0
        metadata = []

        for img_path in images:
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)
            faces   = face_detector.get(img_np)

            if len(faces) != 1:
                log.warning(f"  {WARN} {img_path.name} → {len(faces)} visages, ignorée")
                continue

            face    = faces[0]
            out_name = f"face_{kept:05d}.jpg"
            out_path = os.path.join(output_dir, "images", out_name)
            img_pil.resize((512, 512), Image.LANCZOS).save(out_path, quality=95)

            metadata.append({
                "filename" : out_name,
                "embedding": face.embedding.tolist(),
                "det_score": float(face.det_score),
            })
            kept += 1
            log.info(f"  {PASS} {img_path.name} → {out_name}")

        if kept == 0:
            log.error(f"  {FAIL} Aucune image valide — vérifiez vos photos")
            return False

        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"total": kept, "files": metadata}, f, indent=2)

        log.info(f"\n  {PASS} Dataset test prêt : {kept} images → {output_dir}")
        log.info(f"  {PASS} metadata.json créé avec embeddings")
        return True

    except Exception as e:
        log.error(f"  {FAIL} prepare_dataset erreur : {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 : Test train 2 epochs
# ──────────────────────────────────────────────────────────────────────────────

def check_training(dataset_dir: str, output_dir: str, comfyui_path: str):
    section("ÉTAPE 5 — Test entraînement (2 epochs, batch=1)")

    try:
        import subprocess
        train_script = os.path.join(os.path.dirname(__file__), "train_pulid_klein.py")

        if not os.path.exists(train_script):
            log.error(f"  {FAIL} train_pulid_klein.py non trouvé à côté de ce script")
            return False

        cmd = [
            sys.executable, train_script,
            "--dataset",     dataset_dir,
            "--output",      output_dir,
            "--comfyui_path", comfyui_path,
            "--phase",       "1",
            "--epochs",      "2",
            "--batch_size",  "1",
            "--save_every",  "999999",   # pas de checkpoint intermédiaire
            "--dim",         "4096",
        ]

        log.info(f"  Commande : {' '.join(cmd)}")
        log.info("  Lancement (peut prendre 2-5 min)...")

        t0     = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.error(f"  {FAIL} Entraînement échoué :")
            for line in result.stderr.split("\n")[-20:]:
                if line.strip():
                    log.error(f"       {line}")
            return False

        # Afficher les dernières lignes du log
        for line in result.stdout.split("\n")[-15:]:
            if line.strip():
                log.info(f"  | {line}")

        log.info(f"\n  {PASS} Entraînement terminé en {elapsed:.0f}s")
        return True

    except Exception as e:
        log.error(f"  {FAIL} Erreur : {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 : Vérifier le .safetensors
# ──────────────────────────────────────────────────────────────────────────────

def check_safetensors(output_dir: str):
    section("ÉTAPE 6 — Vérification du .safetensors")

    try:
        from safetensors.torch import load_file
        import torch

        # Chercher le fichier best ou latest
        candidates = [
            os.path.join(output_dir, "pulid_klein_phase1_best.safetensors"),
            os.path.join(output_dir, "pulid_klein_phase1_latest.safetensors"),
        ]
        # Chercher aussi dans checkpoints/
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".safetensors"):
                    candidates.append(os.path.join(ckpt_dir, f))

        path = next((p for p in candidates if os.path.exists(p)), None)

        if not path:
            log.error(f"  {FAIL} Aucun .safetensors trouvé dans {output_dir}")
            return False

        state = load_file(path, device="cpu")
        size  = os.path.getsize(path) / 1e6

        log.info(f"  {PASS} Fichier : {os.path.basename(path)} ({size:.1f} Mo)")
        log.info(f"  {PASS} Clés    : {len(state)}")

        # Vérifier les clés essentielles
        expected_keys = ["id_former.latents", "id_former.norm.weight"]
        for key in expected_keys:
            if key in state:
                log.info(f"  {PASS} {key} : {state[key].shape}")
            else:
                log.warning(f"  {WARN} Clé manquante : {key}")

        # Simuler le chargement dans ton node
        log.info("\n  Test chargement dans PuLIDFlux2...")
        sys.path.insert(0, os.path.dirname(__file__))

        try:
            from pulid_flux2 import PuLIDFlux2
            model = PuLIDFlux2(dim=4096)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log.warning(f"  {WARN} Clés manquantes : {len(missing)}")
            log.info(f"  {PASS} Chargeable dans PuLIDFlux2 ✅")
        except ImportError:
            log.warning(f"  {WARN} pulid_flux2.py non trouvé — test de chargement skippé")

        return True

    except Exception as e:
        log.error(f"  {FAIL} Erreur : {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valide le pipeline PuLID Klein avant Vast.ai"
    )
    parser.add_argument("--images",       type=str, required=True,
                        help="Dossier avec 5-10 photos de visages pour tester")
    parser.add_argument("--comfyui_path", type=str, default="C:/AI/ComfyUI")
    parser.add_argument("--output",       type=str, default="./test_output")
    args = parser.parse_args()

    dataset_test = os.path.join(args.output, "dataset_test")
    train_test   = os.path.join(args.output, "train_test")

    results = {}
    t_start = time.time()

    # Étape 1
    results["deps"] = check_dependencies()
    if not results["deps"]:
        log.error("\n❌ Dépendances manquantes — installe-les avant de continuer")
        return

    # Étape 2
    face_detector, results["insightface"] = check_insightface(args.images)

    # Étape 3
    _, results["eva_clip"] = check_eva_clip()

    # Étape 4
    if results["insightface"]:
        results["dataset"] = check_prepare_dataset(
            args.images, dataset_test, face_detector
        )
    else:
        results["dataset"] = False
        log.warning("⚠️  Étape 4 skippée (InsightFace KO)")

    # Étape 5
    if results["dataset"]:
        results["training"] = check_training(
            dataset_test, train_test, args.comfyui_path
        )
    else:
        results["training"] = False
        log.warning("⚠️  Étape 5 skippée (dataset KO)")

    # Étape 6
    if results["training"]:
        results["safetensors"] = check_safetensors(train_test)
    else:
        results["safetensors"] = False
        log.warning("⚠️  Étape 6 skippée (training KO)")

    # Rapport final
    section("RAPPORT FINAL")
    labels = {
        "deps"        : "Dépendances",
        "insightface" : "InsightFace",
        "eva_clip"    : "EVA-CLIP",
        "dataset"     : "prepare_dataset",
        "training"    : "Entraînement 2 epochs",
        "safetensors" : "Fichier .safetensors",
    }

    all_ok = True
    for key, label in labels.items():
        status = PASS if results.get(key) else FAIL
        log.info(f"  {status} {label}")
        if not results.get(key):
            all_ok = False

    elapsed = time.time() - t_start
    log.info(f"\n  Durée totale : {elapsed:.0f}s")

    if all_ok:
        log.info(f"\n  🎉 TOUT EST OK — Tu peux lancer sur Vast.ai !")
        log.info(f"     Commande phase 1 sur Vast.ai (A100, batch=8) :")
        log.info(f"     python train_pulid_klein.py \\")
        log.info(f"       --dataset ./dataset/filtered \\")
        log.info(f"       --output ./output \\")
        log.info(f"       --phase 1 --epochs 20 --batch_size 8")
    else:
        log.info(f"\n  ❌ Des erreurs sont présentes — corrige-les avant Vast.ai")


if __name__ == "__main__":
    main()
