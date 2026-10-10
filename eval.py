import os
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from cleanfid import fid

from models.unet import UNetSmall
from solvers.euler import sample_euler

# 1. Détection automatique du device
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
except (ImportError, RuntimeError):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Chargement du modèle
model = UNetSmall().to(device)
model_path = "model_baseline.pt"
assert os.path.exists(model_path), f"Fichier {model_path} introuvable."
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# 3. Dossier temporaire pour les images générées
output_dir = "generated_cifar10_eval"
os.makedirs(output_dir, exist_ok=True)

NUM_SAMPLES = 10000  # 10 000 pour une mesure rapide, 50 000 pour l'évaluation finale
BATCH_SIZE = 128
STEPS = 20

print(f"Génération de {NUM_SAMPLES} images sur {device} ({STEPS} pas d'Euler)...")
img_counter = 0

with torch.no_grad():
    for _ in tqdm(range(0, NUM_SAMPLES, BATCH_SIZE)):
        current_batch = min(BATCH_SIZE, NUM_SAMPLES - img_counter)
        shape = (current_batch, 3, 32, 32)
        
        imgs = sample_euler(model, shape=shape, steps=STEPS, device=device)
        imgs = (imgs.clamp(-1.0, 1.0) + 1.0) / 2.0
        
        for i in range(current_batch):
            img_path = os.path.join(output_dir, f"{img_counter:05d}.png")
            vutils.save_image(imgs[i].cpu(), img_path)
            img_counter += 1

print(f"Génération terminée ({img_counter} images enregistrées).")

# 4. Calcul officiel de la Clean-FID face au split d'entraînement de CIFAR-10
print("Calcul de la Fréchet Inception Distance...")
score = fid.compute_fid(
    fdir1=output_dir,
    dataset_name="cifar10",
    dataset_res=32,
    dataset_split="train",
    mode="clean"
)

print(f"Score Clean-FID (CIFAR-10 train) : {score:.2f}")
