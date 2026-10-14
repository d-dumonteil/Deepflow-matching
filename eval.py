import os
import shutil
from concurrent.futures import ThreadPoolExecutor
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
from utils.gpu import setup_device_and_env


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


@torch.inference_mode()
def compute_dinov2_features(image_dir, device, batch_size=256):
    """Extraction des embeddings DINOv2 (ViT-B/14) en FP16 pour la FID DINOv2."""
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    dinov2.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = torchvision.datasets.ImageFolder(image_dir, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    features = []
    is_cuda = device.type == "cuda"
    for imgs, _ in tqdm(loader, desc="Extraction DINOv2"):
        imgs = transform(imgs).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
            feats = dinov2(imgs)
        features.append(feats.float().cpu().numpy())

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.inference_mode()
def compute_cace_and_is(image_dir, device, num_classes=10, batch_size=256):
    """Calcul de l'Inception Score et du Class-Aggregated Calibration Error (CACE)."""
    classifier = torchvision.models.resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, num_classes)
    classifier = classifier.to(device)
    classifier.eval()

    dataset = torchvision.datasets.ImageFolder(
        image_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    preds = []
    is_cuda = device.type == "cuda"
    for imgs, _ in tqdm(loader, desc="Audit CACE / IS"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
            logits = classifier(imgs)
            probs = torch.softmax(logits, dim=1)
        preds.append(probs.float().cpu())

    preds = torch.cat(preds, dim=0).numpy()

    # Inception Score
    py = np.mean(preds, axis=0)
    scores = [
        np.sum(preds[i, :] * np.log(preds[i, :] / (py + 1e-10) + 1e-10))
        for i in range(preds.shape[0])
    ]
    inception_score = np.exp(np.mean(scores))

    # CACE
    uniform_target = 1.0 / num_classes
    cace = np.sum(np.abs(py - uniform_target))

    return inception_score, cace


def _save_batch_worker(cpu_imgs, start_idx, target_dir):
    """Tâche worker pour paralléliser l'écriture PNG sur disque."""
    for i in range(cpu_imgs.shape[0]):
        vutils.save_image(cpu_imgs[i], os.path.join(target_dir, f"{start_idx + i:05d}.png"))


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
    if os.path.exists(sub_img_dir):
        shutil.rmtree(sub_img_dir)
    os.makedirs(sub_img_dir, exist_ok=True)

    NUM_SAMPLES = 50000
    BATCH_SIZE = 512  # Adapté aux 16 Go de VRAM de la T4 sur CIFAR-10
    STEPS = 20
    METHOD = "euler"
    is_cuda = (device.type == "cuda")

    print(f"Génération de {NUM_SAMPLES} images via {METHOD} (NFE={STEPS}) en FP16...")
    img_counter = 0

    # Pool de threads pour éviter que l'I/O disque ne bloque le calcul GPU
    with ThreadPoolExecutor(max_workers=8) as io_pool:
        futures = []
        with torch.inference_mode():
            pbar = tqdm(total=NUM_SAMPLES, desc="Génération d'échantillons")
            while img_counter < NUM_SAMPLES:
                current_batch = min(BATCH_SIZE, NUM_SAMPLES - img_counter)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                    imgs = sample_ode(
                        model,
                        shape=(current_batch, 3, 32, 32),
                        method=METHOD,
                        steps=STEPS,
                        device=device,
                    )
                    imgs = (imgs.clamp(-1.0, 1.0) + 1.0) * 0.5

                imgs_cpu = imgs.float().cpu()
                futures.append(io_pool.submit(_save_batch_worker, imgs_cpu, img_counter, sub_img_dir))

                img_counter += current_batch
                pbar.update(current_batch)
            pbar.close()

        print("Finalisation de l'écriture disque des images PNG...")
        for f in futures:
            f.result()

    print("\n--- 1. Évaluation Clean-FID (Inception-v3) ---")
    clean_fid_score = fid.compute_fid(
        fdir1=sub_img_dir,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        mode="clean",
        device=device,
    )
    print(f"Score Clean-FID (Inception-v3) : {clean_fid_score:.2f}")

    print("\n--- 2. Évaluation FID DINOv2 (ViT-B/14) ---")
    mu_gen, sigma_gen = compute_dinov2_features(output_dir, device=device)
    mu_real, sigma_real = mu_gen, sigma_gen  # À remplacer par les stats de référence précalculées
    dinov2_fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"Score FID (DINOv2) : {dinov2_fid:.2f}")

    print("\n--- 3. Évaluation IS et CACE ---")
    is_score, cace_score = compute_cace_and_is(output_dir, device=device)
    print(f"Inception Score : {is_score:.2f}")
    print(f"Erreur de Calibration (CACE) : {cace_score:.4f}")


if __name__ == "__main__":
    main()