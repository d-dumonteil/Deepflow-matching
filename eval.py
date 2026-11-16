import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from scipy import linalg
from cleanfid import fid

from models.unet import UNetSmall
from solvers.ode import sample_ode
from utils.gpu import setup_device_and_env

# --- Paramètres (surchargeables par variables d'environnement) ---
NUM_SAMPLES = int(os.environ.get("FLOW_EVAL_N", 50000))
BATCH_SIZE = int(os.environ.get("FLOW_EVAL_BATCH", 512))
STEPS = int(os.environ.get("FLOW_EVAL_STEPS", 20))
METHOD = os.environ.get("FLOW_EVAL_METHOD", "euler")
OUTPUT_DIR = os.environ.get("FLOW_EVAL_DIR", "generated_cifar10_eval")
CACHE_DIR = os.environ.get("FLOW_CACHE_DIR", os.getcwd())   # stats de référence persistantes
DINO_REF_N = int(os.environ.get("FLOW_DINO_REF_N", 10000))  # taille d'échantillon de référence DINOv2
CLASSIFIER_PATH = os.environ.get("FLOW_CLASSIFIER", "")     # ResNet-18 CIFAR-10 entraîné (optionnel)

DINO_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # [0, 1]
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

CIFAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Distance de Fréchet entre deux gaussiennes multivariées (calcul analytique)."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def _patch_cleanfid_inception():
    """Torch >= 2.14 : Module.to() échoue sur le ScriptModule Inception de clean-fid
    ('param must be a Parameter'), y compris vers CPU. Le TorchScript est déjà chargé
    sur CPU : neutraliser .to() pour cette classe uniquement suffit."""
    import cleanfid.features as cff
    if not getattr(cff.InceptionV3W, "_to_neutralise", False):
        cff.InceptionV3W.to = lambda self, *args, **kwargs: self
        cff.InceptionV3W._to_neutralise = True


@torch.inference_mode()
def _dinov2_features(loader, device, desc):
    """Embeddings DINOv2 (ViT-B/14) d'un DataLoader ; retourne mu et sigma des features."""
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2 = dinov2.to(device)
    dinov2.eval()

    is_cuda = (device.type == "cuda")
    feats = []
    for imgs, _ in tqdm(loader, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        # fp16 sur CUDA ; fp32 sur TPU (ViT eager, compatible plugin torch_tpu)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
            f = dinov2(imgs)
        feats.append(f.float().cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def dinov2_stats_images(image_dir, device, batch_size=256):
    """Stats DINOv2 d'un dossier d'images (structure ImageFolder attendue)."""
    dataset = torchvision.datasets.ImageFolder(image_dir, transform=DINO_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    return _dinov2_features(loader, device, desc="Extraction DINOv2 (générées)")


def dinov2_stats_reference(device, cache_path, n_ref):
    """Stats DINOv2 de référence sur le train CIFAR-10, calculées une fois puis cachées."""
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        print(f"Stats de référence DINOv2 chargées depuis {cache_path}.")
        return data["mu"], data["sigma"]

    dataset = torchvision.datasets.CIFAR10(
        root="./data_cifar", train=True, download=True, transform=DINO_TRANSFORM
    )
    indices = np.random.RandomState(0).choice(len(dataset), size=n_ref, replace=False)
    sous_ensemble = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        sous_ensemble, batch_size=256, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    mu, sigma = _dinov2_features(loader, device, desc="Extraction DINOv2 (référence CIFAR-10)")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, mu=mu, sigma=sigma)
    print(f"Stats de référence DINOv2 sauvegardées dans {cache_path}.")
    return mu, sigma


def _save_batch_worker(cpu_imgs, start_idx, target_dir):
    """Écriture PNG parallélisée sur disque."""
    for i in range(cpu_imgs.shape[0]):
        vutils.save_image(cpu_imgs[i], os.path.join(target_dir, f"{start_idx + i:05d}.png"))


@torch.inference_mode()
def compute_cace_and_is(image_dir, device, clf_path, num_classes=10, batch_size=256):
    """IS et CACE à partir d'un ResNet-18 entraîné sur CIFAR-10."""
    classifier = torchvision.models.resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, num_classes)
    classifier.load_state_dict(torch.load(clf_path, map_location="cpu", weights_only=True))
    classifier = classifier.to(device)
    classifier.eval()

    dataset = torchvision.datasets.ImageFolder(image_dir, transform=CIFAR_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    preds = []
    is_cuda = (device.type == "cuda")
    for imgs, _ in tqdm(loader, desc="Audit CACE / IS"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
            probs = torch.softmax(classifier(imgs), dim=1)
        preds.append(probs.float().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    py = preds.mean(axis=0)

    # Inception Score : E_x [ KL( p(y|x) || p(y) ) ]
    log_p = np.log(preds + 1e-10)
    log_py = np.log(py + 1e-10)
    scores = np.sum(preds * (log_p - log_py[None, :]), axis=1)
    inception_score = float(np.exp(scores.mean()))

    # CACE : écart L1 entre la distribution marginale prédite et l'uniforme
    cace = float(np.sum(np.abs(py - 1.0 / num_classes)))
    return inception_score, cace


def trouver_modele():
    save_root = os.environ.get("FLOW_SAVE_DIR", "")
    candidats = []
    if save_root:
        candidats.append(os.path.join(save_root, "model_baseline.pt"))
    candidats.append("model_baseline.pt")
    for p in candidats:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "model_baseline.pt introuvable. Définissez FLOW_SAVE_DIR (dossier checkpoints "
        "du Drive) ou créez un symlink dans le répertoire du dépôt."
    )


def generer_images(model, device, sub_img_dir, is_cuda, is_tpu):
    torch.manual_seed(0)
    img_counter = 0
    amp_dtype = torch.float16 if is_cuda else torch.bfloat16
    amp_enabled = is_cuda or is_tpu

    print(f"Génération de {NUM_SAMPLES} images via {METHOD} (NFE={STEPS})...")
    with ThreadPoolExecutor(max_workers=8) as io_pool:
        futures = []
        with torch.inference_mode():
            pbar = tqdm(total=NUM_SAMPLES, desc="Génération d'échantillons")
            while img_counter < NUM_SAMPLES:
                current_batch = min(BATCH_SIZE, NUM_SAMPLES - img_counter)
                with torch.autocast(device_type=str(device.type), dtype=amp_dtype, enabled=amp_enabled):
                    imgs = sample_ode(
                        model, shape=(current_batch, 3, 32, 32),
                        method=METHOD, steps=STEPS, device=device,
                    )
                    imgs = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5
                imgs_cpu = imgs.float().cpu()
                futures.append(io_pool.submit(_save_batch_worker, imgs_cpu, img_counter, sub_img_dir))
                img_counter += current_batch
                pbar.update(current_batch)
            pbar.close()
        for f in futures:
            f.result()
    print("Écriture disque des PNG finalisée.")


def main():
    device, is_xla = setup_device_and_env()
    is_tpu = is_xla or (str(getattr(device, "type", "")) == "tpu")
    is_cuda = (str(getattr(device, "type", "")) == "cuda")

    # --- Modèle ---
    model_path = trouver_modele()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model = UNetSmall().to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Modèle chargé depuis : {model_path}")

    # --- Génération (ignorée si les images existent déjà) ---
    sub_img_dir = os.path.join(OUTPUT_DIR, "images")
    n_existants = len(os.listdir(sub_img_dir)) if os.path.isdir(sub_img_dir) else 0
    if n_existants >= NUM_SAMPLES:
        print(f"{n_existants} images déjà présentes dans {sub_img_dir} : génération ignorée.")
    else:
        if n_existants > 0:
            print(f"{n_existants} images incomplètes détectées : régénération totale.")
            shutil.rmtree(sub_img_dir)
        os.makedirs(sub_img_dir, exist_ok=True)
        generer_images(model, device, sub_img_dir, is_cuda, is_tpu)

    # --- 1. Clean-FID (Inception-v3) ---
    # Torch >= 2.14 : le ScriptModule Inception échoue sur Module.to(),
    # même vers CPU. Patch + exécution CPU hors CUDA.
    try:
        fid_device = "cuda" if is_cuda else "cpu"
        if not is_cuda:
            _patch_cleanfid_inception() 
            print("Patch clean-fid appliqué (conversion .to neutralisée) : Clean-FID sur CPU.")
        print("\n--- 1. Évaluation Clean-FID (Inception-v3) ---")
        clean_fid_score = fid.compute_fid(
            fdir1=sub_img_dir,
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split="train",
            mode="clean",
            device=fid_device,
            use_dataparallel=False,
        )
        print(f"Score Clean-FID (Inception-v3) : {clean_fid_score:.2f}")
    except Exception as e:
        print(f"Clean-FID échoué : {type(e).__name__}: {e}")

    # --- 2. FID DINOv2 (ViT-B/14), référence CIFAR-10 mise en cache ---
    # Métrique relative : comparer vos variantes entre elles, pas à la littérature.
    try:
        print("\n--- 2. Évaluation FID DINOv2 (ViT-B/14) ---")
        cache_path = os.path.join(CACHE_DIR, "dinov2_ref_cifar10.npz")
        mu_gen, sigma_gen = dinov2_stats_images(OUTPUT_DIR, device)
        mu_ref, sigma_ref = dinov2_stats_reference(device, cache_path, DINO_REF_N)
        dinov2_fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
        print(f"Score FID (DINOv2) : {dinov2_fid:.2f}")
    except Exception as e:
        print(f"FID DINOv2 échouée : {type(e).__name__}: {e}")

    # --- 3. IS et CACE (nécessitent un classificateur CIFAR-10 entraîné) ---
    if CLASSIFIER_PATH and os.path.exists(CLASSIFIER_PATH):
        try:
            print("\n--- 3. Évaluation IS et CACE ---")
            is_score, cace_score = compute_cace_and_is(OUTPUT_DIR, device, CLASSIFIER_PATH)
            print(f"Inception Score : {is_score:.2f}")
            print(f"Erreur de Calibration (CACE) : {cace_score:.4f}")
        except Exception as e:
            print(f"IS/CACE échoué : {type(e).__name__}: {e}")
    else:
        print("\n--- 3. IS et CACE ignorés ---")
        print("Renseignez FLOW_CLASSIFIER (checkpoint ResNet-18 entraîné sur CIFAR-10) "
              "pour activer ces métriques. Sans classificateur entraîné, elles seraient "
              "dénuées de sens.")


if __name__ == "__main__":
    main()