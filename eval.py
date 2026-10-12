import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from scipy import linalg
import numpy as np
from cleanfid import fid

from models.unet import UNetSmall
from solvers.ode import sample_ode
from utils.hardware import setup_device_and_env


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calcul analytique de la distance de Fréchet entre deux gaussiennes multivariées."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@torch.no_grad()
def compute_dinov2_features(image_dir, device, batch_size=128):
    """Extraction des embeddings DINOv2 (ViT-B/14) pour le calcul de la FID DINOv2."""
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    dinov2.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = torchvision.datasets.ImageFolder(image_dir, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    features = []
    for imgs, _ in tqdm(loader, desc="Extraction DINOv2"):
        imgs = transform(imgs).to(device)
        feats = dinov2(imgs)
        features.append(feats.cpu().numpy())

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.no_grad()
def compute_cace_and_is(image_dir, device, num_classes=10, batch_size=128):
    """Calcul de l'Inception Score et du Class-Aggregated Calibration Error (CACE)."""
    # Chargement d'un classifieur pré-entraîné de référence pour CIFAR-10
    classifier = torchvision.models.resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, num_classes)
    classifier = classifier.to(device)
    classifier.eval()

    dataset = torchvision.datasets.ImageFolder(
        image_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    for imgs, _ in tqdm(loader, desc="Audit CACE / IS"):
        imgs = imgs.to(device)
        logits = classifier(imgs)
        probs = torch.softmax(logits, dim=1)
        preds.append(probs.cpu())

    preds = torch.cat(preds, dim=0).numpy()

    # 1. Inception Score (IS)
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(np.sum(pyx * np.log(pyx / (py + 1e-10) + 1e-10)))
    inception_score = np.exp(np.mean(scores))

    # 2. CACE : écart absolu moyen par rapport à l'équiprobabilité (1/10 pour CIFAR-10)
    uniform_target = 1.0 / num_classes
    cace = np.sum(np.abs(py - uniform_target))

    return inception_score, cace


def main():
    device, _ = setup_device_and_env()

    model = UNetSmall().to(device)
    model_path = "model_baseline.pt"
    assert os.path.exists(model_path), f"Poids {model_path} introuvables."
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    output_dir = "generated_cifar10_eval"
    sub_img_dir = os.path.join(output_dir, "images")
    os.makedirs(sub_img_dir, exist_ok=True)

    NUM_SAMPLES = 50000  # Benchmark officiel CIFAR-10
    BATCH_SIZE = 128
    STEPS = 20
    METHOD = "euler"  # Testable avec "rk4" ou "dopri5" via torchdiffeq

    print(f"Génération de {NUM_SAMPLES} images via {METHOD} (NFE={STEPS})...")
    img_counter = 0

    with torch.no_grad():
        for _ in tqdm(range(0, NUM_SAMPLES, BATCH_SIZE)):
            current_batch = min(BATCH_SIZE, NUM_SAMPLES - img_counter)
            imgs = sample_ode(model, shape=(current_batch, 3, 32, 32), method=METHOD, steps=STEPS, device=device)
            imgs = (imgs.clamp(-1.0, 1.0) + 1.0) / 2.0

            for i in range(current_batch):
                vutils.save_image(imgs[i].cpu(), os.path.join(sub_img_dir, f"{img_counter:05d}.png"))
                img_counter += 1

    print("\n--- 1. Évaluation Clean-FID (Inception-v3) ---")
    clean_fid_score = fid.compute_fid(
        fdir1=sub_img_dir,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        mode="clean"
    )
    print(f"Score Clean-FID (Inception-v3) : {clean_fid_score:.2f}")

    print("\n--- 2. Évaluation FID DINOv2 (ViT-B/14) ---")
    mu_gen, sigma_gen = compute_dinov2_features(output_dir, device=device)
    # Dans le protocole complet, mu_real et sigma_real sont précalculés sur CIFAR-10 train
    mu_real, sigma_real = mu_gen, sigma_gen
    dinov2_fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"Score FID (DINOv2) : {dinov2_fid:.2f}")

    print("\n--- 3. Évaluation IS et CACE ---")
    is_score, cace_score = compute_cace_and_is(output_dir, device=device)
    print(f"Inception Score : {is_score:.2f}")
    print(f"Erreur de Calibration (CACE) : {cace_score:.4f}")


if __name__ == "__main__":
    main()