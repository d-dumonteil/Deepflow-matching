import os
import torch
import torchvision.utils as vutils
from models.unet import UNetSmall
from solvers.euler import sample_euler
from utils.gpu import setup_device_and_env


def trouver_modele():
    candidats = []
    env = os.environ.get("FLOW_MODEL", "")
    if env:
        candidats.append(env)
    save_root = os.environ.get("FLOW_SAVE_DIR", "")
    if save_root:
        candidats.append(os.path.join(save_root, "model_baseline.pt"))
    candidats.append("model_baseline.pt")
    for p in candidats:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Modèle introuvable : FLOW_MODEL, FLOW_SAVE_DIR ou symlink local.")


device, _ = setup_device_and_env()

model = UNetSmall().to(device)
checkpoint = torch.load(trouver_modele(), map_location=device, weights_only=True)
state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
model.load_state_dict(state)
model.eval()

print(f"Génération de 16 échantillons sur {device}...")
with torch.no_grad():
    x = sample_euler(model, shape=(16, 3, 32, 32), steps=20, device=device)
    images = (x.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]

vutils.save_image(images.cpu(), "samples_generated.png", nrow=4)
print("Images générées enregistrées dans samples_generated.png")